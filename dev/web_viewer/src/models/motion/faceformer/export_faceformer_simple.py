import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import copy

# Simplified version that avoids dynamic sequence length issues
class FaceformerCoreStepSimple(nn.Module):
    """Simplified core step that processes fixed sequence lengths"""
    def __init__(self, args, max_seq_len=20):
        super().__init__()
        print("FaceformerCoreStepSimple: Initializing.")
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.vertice_dim = args.vertice_dim
        self.max_seq_len = max_seq_len
        
        # Core components
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period, max_seq_len=max_seq_len)
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=max_seq_len, period=args.period)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.feature_dim, 
            nhead=4, 
            dim_feedforward=2*args.feature_dim, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        print("FaceformerCoreStepSimple: Initialized.")

    def forward(self, audio_features, vertice_sequence, current_length, one_hot, template):
        """
        Process with fixed-size inputs and current length indicator
        
        Args:
            audio_features: Pre-extracted audio features [batch, audio_seq_len, 768]
            vertice_sequence: Full sequence buffer [batch, max_seq_len, feature_dim] (padded)
            current_length: Current valid length in sequence [batch, 1] (scalar)
            one_hot: Subject one-hot [batch, num_subjects]
            template: Template vertices [batch, 1, vertice_dim]
            
        Returns:
            new_vertice_out: Next predicted vertices [batch, 1, vertice_dim]
            updated_sequence: Updated sequence buffer [batch, max_seq_len, feature_dim]
            new_length: Updated length [batch, 1]
        """
        print("FaceformerCoreStepSimple: Forward pass started.")
        batch_size = audio_features.shape[0]
        
        # Map audio features
        hidden_states = self.audio_feature_map(audio_features)
        
        # Style embedding
        obj_embedding = self.obj_vector(one_hot)  # [batch, feature_dim]
        style_emb = obj_embedding.unsqueeze(1)    # [batch, 1, feature_dim]
        
        # Extract current valid sequence based on length
        current_len = int(current_length[0, 0].item())  # Get scalar length
        current_seq = vertice_sequence[:, :current_len, :]  # [batch, current_len, feature_dim]
        
        # Apply positional encoding to current sequence
        vertice_input = self.PPE(current_seq)
        
        # Create masks for current length
        tgt_mask = self.biased_mask[:, :current_len, :current_len].clone().detach()
        memory_mask = enc_dec_mask_static(self.dataset, current_len, hidden_states.shape[1])
        
        # Transformer decoder step
        vertice_out = self.transformer_decoder(
            vertice_input, 
            hidden_states, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask
        )
        
        # Get output for this step
        current_out = self.vertice_map_r(vertice_out[:, -1:, :])  # [batch, 1, vertice_dim]
        
        # Prepare next step embedding
        new_output = self.vertice_map(current_out)  # [batch, 1, feature_dim]
        new_output = new_output + style_emb
        
        # Update sequence buffer
        updated_sequence = vertice_sequence.clone()
        if current_len < self.max_seq_len:
            updated_sequence[:, current_len:current_len+1, :] = new_output
        
        # Update length
        new_length = torch.clamp(current_length + 1, max=self.max_seq_len)
        
        # Add template
        new_vertice_out = current_out + template
        
        print("FaceformerCoreStepSimple: Forward pass finished.")
        return new_vertice_out, updated_sequence, new_length


# Simplified static mask function for ONNX compatibility
def enc_dec_mask_static(dataset, T, S):
    """Static version of enc_dec_mask for ONNX export"""
    mask = torch.ones(T, S, dtype=torch.bool)
    if dataset == "BIWI":
        for i in range(T):
            start_idx = min(i*2, S-1)
            end_idx = min(i*2+2, S)
            if start_idx < end_idx:
                mask[i, start_idx:end_idx] = False
    elif dataset == "vocaset":
        for i in range(min(T, S)):
            mask[i, i] = False
    return mask

# Simplified biased mask initialization
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

# Periodic Positional Encoding (same as original)
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

# Args class
class Args:
    def __init__(self):
        self.dataset = "vocaset"
        self.vertice_dim = 5023 * 3
        self.feature_dim = 64
        self.period = 30
        self.train_subjects = "subject1 subject2 subject3"
        self.device = "cpu"

def export_simplified_faceformer():
    print("Starting simplified FaceFormer export...")
    
    args = Args()
    max_seq_len = 20
    
    # Create simplified model
    print("\n=== Creating Simplified Model ===")
    simple_model = FaceformerCoreStepSimple(args, max_seq_len=max_seq_len)
    simple_model.eval()
    
    # Create dummy inputs for simplified model
    batch_size = 1
    audio_seq_len = 100
    
    audio_features = torch.randn(batch_size, audio_seq_len, 768)
    # Fixed-size sequence buffer (padded)
    vertice_sequence = torch.randn(batch_size, max_seq_len, args.feature_dim)
    current_length = torch.tensor([[5]], dtype=torch.long)  # Current length is 5
    one_hot = torch.zeros(batch_size, len(args.train_subjects.split()))
    one_hot[0, 0] = 1.0
    template = torch.randn(batch_size, 1, args.vertice_dim)
    
    # Test simplified model
    with torch.no_grad():
        new_out, updated_seq, new_len = simple_model(
            audio_features, vertice_sequence, current_length, one_hot, template
        )
        print(f"Simplified model output shapes:")
        print(f"  new_out={new_out.shape}")
        print(f"  updated_seq={updated_seq.shape}")
        print(f"  new_len={new_len.shape}, value={new_len.item()}")
    
    # Export simplified model
    print("\n=== Exporting Simplified Model ===")
    torch.onnx.export(
        simple_model,
        (audio_features, vertice_sequence, current_length, one_hot, template),
        "faceformer_simple_step.onnx",
        input_names=['audio_features', 'vertice_sequence', 'current_length', 'one_hot', 'template'],
        output_names=['new_vertice_out', 'updated_sequence', 'new_length'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'audio_seq_len'},
            'vertice_sequence': {0: 'batch_size'},
            'current_length': {0: 'batch_size'},
            'one_hot': {0: 'batch_size'},
            'template': {0: 'batch_size'},
            'new_vertice_out': {0: 'batch_size'},
            'updated_sequence': {0: 'batch_size'},
            'new_length': {0: 'batch_size'}
        },
        opset_version=16,
        do_constant_folding=False,
        verbose=False
    )
    print("Simplified model exported to faceformer_simple_step.onnx")
    
    # Save sample data for testing
    sample_data = {
        "audio_features": audio_features.numpy().tolist(),
        "vertice_sequence": vertice_sequence.numpy().tolist(),
        "current_length": current_length.numpy().tolist(),
        "one_hot": one_hot.numpy().tolist(),
        "template": template.numpy().tolist(),
        "new_vertice_out": new_out.numpy().tolist(),
        "updated_sequence": updated_seq.numpy().tolist(),
        "new_length": new_len.numpy().tolist(),
        "max_seq_len": max_seq_len
    }
    
    with open("faceformer_simple_sample_data.json", "w") as f:
        json.dump(sample_data, f)
    
    print("Sample data saved to faceformer_simple_sample_data.json")
    print("Export complete!")

if __name__ == "__main__":
    export_simplified_faceformer()
