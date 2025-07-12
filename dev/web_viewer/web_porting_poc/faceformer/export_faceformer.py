import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import copy

# Mock Wav2Vec2Model for ONNX export
class MockWav2Vec2Model(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, audio, dataset, frame_num=None):
        print("  MockWav2Vec2Model: Forward pass started.")
        if dataset == "vocaset":
            seq_len = frame_num if frame_num else 250 # Default to 250 frames for vocaset
        elif dataset == "BIWI":
            seq_len = frame_num * 2 if frame_num else 500 # Default to 500 frames for BIWI
        else:
            raise ValueError("Unknown dataset")
        print(f"  MockWav2Vec2Model: Generating dummy hidden_states with shape ({audio.shape[0]}, {seq_len}, 768)")
        return type('obj', (object,), {'last_hidden_state': torch.randn(audio.shape[0], seq_len, 768)})()

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    print("  init_biased_mask: Started.")
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    print("  init_biased_mask: Finished.")
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    print("  enc_dec_mask: Started.")
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    print("  enc_dec_mask: Finished.")
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        print("  PeriodicPositionalEncoding: Initializing.")
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
        print("  PeriodicPositionalEncoding: Initialized.")
    def forward(self, x):
        print("  PeriodicPositionalEncoding: Forward pass started.")
        x = x + self.pe[:, :x.size(1), :]
        print("  PeriodicPositionalEncoding: Forward pass finished.")
        return self.dropout(x)

class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        print("Faceformer: Initializing.")
        self.dataset = args.dataset
        self.audio_encoder = MockWav2Vec2Model(args.feature_dim) # Using mock model
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        print("Faceformer: Initialized.")

    def forward(self, audio, template, vertice, one_hot, criterion,teacher_forcing=True):
        raise NotImplementedError("This model is for inference only. Use the predict method.")

    def predict(self, audio, template, one_hot):
        print("Predict method: Started.")
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        print("Predict method: After obj_embedding.")
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        print("Predict method: After audio_encoder.")
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        print(f"Predict method: frame_num = {frame_num}.")
        hidden_states = self.audio_feature_map(hidden_states)
        print("Predict method: After audio_feature_map.")

        vertice_emb = None # Initialize vertice_emb outside the loop

        for i in range(frame_num):
            print(f"Predict method: Loop iteration {i}/{frame_num-1}.")
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            print(f"Predict method: Before transformer_decoder in iteration {i}.")
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            print(f"Predict method: After transformer_decoder in iteration {i}.")
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        print("Predict method: Finished.")
        return vertice_out

# Dummy args class to mimic argparse
class Args:
    def __init__(self):
        self.dataset = "vocaset"
        self.vertice_dim = 5023 * 3
        self.feature_dim = 64
        self.period = 30
        self.train_subjects = "subject1 subject2 subject3"
        self.device = "cpu" # Use CPU for export

class FaceformerPredictWrapper(nn.Module):
    def __init__(self, faceformer_model):
        super().__init__()
        self.faceformer_model = faceformer_model

    def forward(self, audio, template, one_hot):
        print("FaceformerPredictWrapper: Forward pass started.")
        result = self.faceformer_model.predict(audio, template, one_hot)
        print("FaceformerPredictWrapper: Forward pass finished.")
        return result

print("Script started.")

# Instantiate args
print("Instantiating Args class...")
args = Args()
print(f"Args instantiated: dataset={args.dataset}, vertice_dim={args.vertice_dim}, feature_dim={args.feature_dim}, period={args.period}, train_subjects={args.train_subjects}, device={args.device}")

# Instantiate the Faceformer model
print("Instantiating Faceformer model...")
faceformer_model = Faceformer(args)
faceformer_model.eval() # Set to evaluation mode
print("Faceformer model instantiated and set to eval mode.")

# Wrap the predict method for ONNX export
print("Wrapping predict method for ONNX export...")
model_to_export = FaceformerPredictWrapper(faceformer_model)
print("Predict method wrapped.")

# Create dummy input tensors for the predict method
print("Creating dummy input tensors...")
batch_size = 1
template_input = torch.randn(batch_size, args.vertice_dim)
audio_input = torch.randn(batch_size, 16000 * 1) 
num_train_subjects = len(args.train_subjects.split())
one_hot_input = torch.zeros(batch_size, num_train_subjects)
one_hot_input[0, 0] = 1.0
print(f"Dummy input tensors created: audio_input.shape={audio_input.shape}, template_input.shape={template_input.shape}, one_hot_input.shape={one_hot_input.shape}")

# Perform a forward pass using the predict method
print("Performing forward pass with predict method...")
with torch.no_grad():
    output = model_to_export(audio_input, template_input, one_hot_input)
print(f"Forward pass complete. Output shape: {output.shape}")

# Export to ONNX
onnx_path = "faceformer.onnx"
print(f"Starting ONNX export to {onnx_path}...")
try:
    torch.onnx.export(
        model_to_export,
        (audio_input, template_input, one_hot_input),
        onnx_path,
        input_names=['audio_input', 'template_input', 'one_hot_input'],
        output_names=['output'],
        dynamic_axes={
            'audio_input': {0: 'batch_size', 1: 'audio_length'},
            'template_input': {0: 'batch_size'},
            'one_hot_input': {0: 'batch_size'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=12 # Use opset 12 or higher for einsum
    )
    print(f"FaceFormer model exported to {onnx_path}")
    print("Saving input and output data for verification...")
    data = {
        "audio_input": audio_input.numpy().tolist(),
        "template_input": template_input.numpy().tolist(),
        "one_hot_input": one_hot_input.numpy().tolist(),
        "output": output.numpy().tolist()
    }

    with open("faceformer_sample_data.json", "w") as f:
        json.dump(data, f)
    print("FaceFormer sample input/output data saved to faceformer_sample_data.json")
    print("Script finished.")
except Exception as e:
    print(f"Error during ONNX export: {e}")
    import traceback
    traceback.print_exc()

# Save input and output data for verification
print("Saving input and output data for verification...")
data = {
    "audio_input": audio_input.numpy().tolist(),
    "template_input": template_input.numpy().tolist(),
    "one_hot_input": one_hot_input.numpy().tolist(),
    "output": output.numpy().tolist()
}

with open("faceformer_sample_data.json", "w") as f:
    json.dump(data, f)
print("FaceFormer sample input/output data saved to faceformer_sample_data.json")
print("Script finished.")