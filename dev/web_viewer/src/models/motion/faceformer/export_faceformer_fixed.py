import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import copy

# Option 1: Export only the core transformer step (single iteration)
class FaceformerCoreStep(nn.Module):
    """Export only a single step of the autoregressive generation"""
    def __init__(self, args):
        super().__init__()
        print("FaceformerCoreStep: Initializing.")
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.vertice_dim = args.vertice_dim
        
        # Core components (no wav2vec2)
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        
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
        print("FaceformerCoreStep: Initialized.")

    def forward(self, audio_features, vertice_emb, one_hot, template):
        """
        Single step of autoregressive generation
        
        Args:
            audio_features: Pre-extracted audio features [batch, seq_len, 768]
            vertice_emb: Current vertex embeddings [batch, current_len, feature_dim] 
            one_hot: Subject one-hot [batch, num_subjects]
            template: Template vertices [batch, 1, vertice_dim]
            
        Returns:
            new_vertice_out: Next predicted vertices [batch, 1, vertice_dim]
            updated_vertice_emb: Updated embeddings for next iteration [batch, current_len+1, feature_dim]
        """
        print("FaceformerCoreStep: Forward pass started.")
        
        # Map audio features
        hidden_states = self.audio_feature_map(audio_features)
        
        # Style embedding
        obj_embedding = self.obj_vector(one_hot)  # [batch, feature_dim]
        style_emb = obj_embedding.unsqueeze(1)    # [batch, 1, feature_dim]
        
        # Apply positional encoding
        vertice_input = self.PPE(vertice_emb)
        
        # Create masks (fixed sizes for ONNX)
        seq_len = vertice_input.shape[1]
        tgt_mask = self.biased_mask[:, :seq_len, :seq_len].clone().detach()
        memory_mask = enc_dec_mask_static(self.dataset, seq_len, hidden_states.shape[1])
        
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
        updated_vertice_emb = torch.cat((vertice_emb, new_output), 1)  # [batch, seq_len+1, feature_dim]
        
        # Add template
        new_vertice_out = current_out + template
        
        print("FaceformerCoreStep: Forward pass finished.")
        return new_vertice_out, updated_vertice_emb

# Option 2: Fixed-length generation (teacher forcing style)
class FaceformerFixedLength(nn.Module):
    """Export with fixed maximum sequence length"""
    def __init__(self, args, max_frames=100):
        super().__init__()
        print("FaceformerFixedLength: Initializing.")
        self.dataset = args.dataset
        self.max_frames = max_frames
        self.feature_dim = args.feature_dim
        
        # Core components
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period, max_seq_len=max_frames)
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=max_frames, period=args.period)
        
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
        print("FaceformerFixedLength: Initialized.")

    def forward(self, audio_features, template, one_hot, sequence_length):
        """
        Fixed-length generation
        
        Args:
            audio_features: Pre-extracted audio features [batch, audio_seq_len, 768]
            template: Template vertices [batch, vertice_dim]
            one_hot: Subject one-hot [batch, num_subjects]
            sequence_length: Desired output length (scalar tensor)
            
        Returns:
            vertice_out: Generated vertices [batch, max_frames, vertice_dim]
            valid_mask: Mask indicating valid frames [batch, max_frames]
        """
        print("FaceformerFixedLength: Forward pass started.")
        batch_size = audio_features.shape[0]
        
        # Map audio features
        hidden_states = self.audio_feature_map(audio_features)
        
        # Style embedding
        obj_embedding = self.obj_vector(one_hot)  # [batch, feature_dim]
        style_emb = obj_embedding.unsqueeze(1)    # [batch, 1, feature_dim]
        template_expanded = template.unsqueeze(1)  # [batch, 1, vertice_dim]
        
        # Initialize sequence with style embedding
        vertice_emb = style_emb.repeat(1, self.max_frames, 1)  # [batch, max_frames, feature_dim]
        
        # Apply positional encoding
        vertice_input = self.PPE(vertice_emb)
        
        # Create masks
        tgt_mask = self.biased_mask[:, :self.max_frames, :self.max_frames].clone().detach()
        memory_mask = enc_dec_mask_static(self.dataset, self.max_frames, hidden_states.shape[1])
        
        # Single transformer pass
        vertice_out = self.transformer_decoder(
            vertice_input, 
            hidden_states, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask
        )
        
        # Map to vertices
        vertice_out = self.vertice_map_r(vertice_out)  # [batch, max_frames, vertice_dim]
        vertice_out = vertice_out + template_expanded  # Add template
        
        # Create validity mask
        frame_indices = torch.arange(self.max_frames, device=vertice_out.device).unsqueeze(0)  # [1, max_frames]
        sequence_length_expanded = sequence_length.unsqueeze(1)  # [batch, 1]
        valid_mask = frame_indices < sequence_length_expanded  # [batch, max_frames]
        
        print("FaceformerFixedLength: Forward pass finished.")
        return vertice_out, valid_mask


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

def export_faceformer_models():
    print("Starting FaceFormer export with multiple approaches...")
    
    args = Args()
    
    # Approach 1: Export core step for iterative generation in JavaScript
    print("\n=== Exporting Core Step Model ===")
    core_step_model = FaceformerCoreStep(args)
    core_step_model.eval()
    
    # Create dummy inputs for core step
    batch_size = 1
    audio_seq_len = 100
    current_seq_len = 5  # Current sequence length in generation
    
    audio_features = torch.randn(batch_size, audio_seq_len, 768)
    vertice_emb = torch.randn(batch_size, current_seq_len, args.feature_dim)
    one_hot = torch.zeros(batch_size, len(args.train_subjects.split()))
    one_hot[0, 0] = 1.0
    template = torch.randn(batch_size, 1, args.vertice_dim)
    
    # Test core step
    with torch.no_grad():
        new_out, updated_emb = core_step_model(audio_features, vertice_emb, one_hot, template)
        print(f"Core step output shapes: new_out={new_out.shape}, updated_emb={updated_emb.shape}")
    
    # Export core step with better dynamic axis handling
    torch.onnx.export(
        core_step_model,
        (audio_features, vertice_emb, one_hot, template),
        "faceformer_core_step.onnx",
        input_names=['audio_features', 'vertice_emb', 'one_hot', 'template'],
        output_names=['new_vertice_out', 'updated_vertice_emb'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'audio_seq_len'},
            'vertice_emb': {0: 'batch_size', 1: 'current_seq_len'},
            'one_hot': {0: 'batch_size'},
            'template': {0: 'batch_size'},
            'new_vertice_out': {0: 'batch_size'},
            'updated_vertice_emb': {0: 'batch_size', 1: 'current_seq_len_plus_1'}
        },
        opset_version=16,  # Use newer opset for better dynamic shape support
        do_constant_folding=False,  # Avoid constant folding that might break dynamic shapes
        verbose=False
    )
    print("Core step model exported to faceformer_core_step.onnx")
    
    # Approach 2: Export fixed-length model
    print("\n=== Exporting Fixed-Length Model ===")
    max_frames = 100
    fixed_model = FaceformerFixedLength(args, max_frames=max_frames)
    fixed_model.eval()
    
    # Create dummy inputs for fixed-length
    audio_features_fixed = torch.randn(batch_size, audio_seq_len, 768)
    template_fixed = torch.randn(batch_size, args.vertice_dim)
    one_hot_fixed = torch.zeros(batch_size, len(args.train_subjects.split()))
    one_hot_fixed[0, 0] = 1.0
    sequence_length = torch.tensor([50], dtype=torch.long)  # Desired output length
    
    # Test fixed-length
    with torch.no_grad():
        vertices_out, valid_mask = fixed_model(audio_features_fixed, template_fixed, one_hot_fixed, sequence_length)
        print(f"Fixed-length output shapes: vertices_out={vertices_out.shape}, valid_mask={valid_mask.shape}")
    
    # Export fixed-length (this approach may still have issues due to complexity)
    try:
        torch.onnx.export(
            fixed_model,
            (audio_features_fixed, template_fixed, one_hot_fixed, sequence_length),
            "faceformer_fixed_length.onnx",
            input_names=['audio_features', 'template', 'one_hot', 'sequence_length'],
            output_names=['vertices_out', 'valid_mask'],
            dynamic_axes={
                'audio_features': {0: 'batch_size', 1: 'audio_seq_len'},
                'template': {0: 'batch_size'},
                'one_hot': {0: 'batch_size'},
                'vertices_out': {0: 'batch_size'},
                'valid_mask': {0: 'batch_size'}
            },
            opset_version=15
        )
        print("Fixed-length model exported to faceformer_fixed_length.onnx")
    except Exception as e:
        print(f"Fixed-length export failed (expected due to complexity): {e}")
    
    # Save sample data for testing
    sample_data = {
        "core_step": {
            "audio_features": audio_features.numpy().tolist(),
            "vertice_emb": vertice_emb.numpy().tolist(),
            "one_hot": one_hot.numpy().tolist(),
            "template": template.numpy().tolist(),
            "new_vertice_out": new_out.numpy().tolist(),
            "updated_vertice_emb": updated_emb.numpy().tolist()
        },
        "fixed_length": {
            "audio_features": audio_features_fixed.numpy().tolist(),
            "template": template_fixed.numpy().tolist(),
            "one_hot": one_hot_fixed.numpy().tolist(),
            "sequence_length": sequence_length.numpy().tolist(),
            "vertices_out": vertices_out.numpy().tolist(),
            "valid_mask": valid_mask.numpy().tolist()
        }
    }
    
    with open("faceformer_sample_data.json", "w") as f:
        json.dump(sample_data, f)
    
    print("Sample data saved to faceformer_sample_data.json")
    print("Export complete!")

if __name__ == "__main__":
    export_faceformer_models()
