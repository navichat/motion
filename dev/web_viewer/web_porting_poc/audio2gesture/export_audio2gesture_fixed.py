#!/usr/bin/env python3
"""
Audio2Gesture Fixed Export - Single-step version for web deployment

This script applies the FaceFormer fixed-buffer approach to Audio2Gesture:
1. Creates a single-step model that processes one frame at a time
2. Uses fixed-size sequence buffers instead of dynamic concatenation
3. Implements manual GRU state management for ONNX compatibility
4. Exports a model suitable for JavaScript autoregressive generation

Based on the working FaceFormer approach that solved hanging export issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# Original model components (same as export_model.py)
class AudioEncoder_Conv1d(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()

        self.layer0 = nn.Conv1d(in_dim, hidden_dim, 7, 1, 0)
        self.layer1 = nn.Conv1d(hidden_dim, hidden_dim, 5, 1, 0)
        self.layer2 = nn.Conv1d(hidden_dim, out_dim, 4, 2, 1)
        self.layer3 = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.layer0(x))
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))

        x = torch.einsum('NDL->NLD', x)

        x = F.gelu(self.layer3(x))
        x = self.norm(x)

        x = torch.einsum('NLD->NDL', x)

        return x


class GRUDecoder(nn.Module):
    def __init__(self, mo_dim, aud_embed_dim, lxm_dim, hidden_dim, out_dim, depth):
        super().__init__()

        in_dim = mo_dim + aud_embed_dim + lxm_dim

        self.embed_in = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(in_dim+hidden_dim, hidden_dim, depth, batch_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.embed_pred = nn.Linear(hidden_dim, out_dim)

    def forward(self, mo, aud_embed, lxm, cell_state):
        in_embed = F.gelu(self.embed_in(torch.cat([mo, aud_embed, lxm], dim=-1)))
        cell_out, cell_state = self.gru(
            torch.cat([in_embed, mo, aud_embed, lxm], dim=-1).unsqueeze(1), cell_state
        )
        out = self.decoder_norm(cell_out)
        out = self.embed_pred(out.squeeze(1))

        return out, cell_state


class CellStateInitializer(nn.Module):
    def __init__(self, in_dim, hidden_dim, rnn_depth):
        super().__init__()

        self.rnn_depth = rnn_depth

        self.layer0 = nn.Linear(in_dim, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim*rnn_depth)

    def forward(self, mo, lxm):
        x = F.gelu(self.layer0(torch.cat([mo, lxm], dim=-1)))
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)

        x = x.reshape(x.shape[0], self.rnn_depth, -1)
        x = torch.einsum('NPD->PND', x).contiguous()

        return x


class Audio2GestureStepFixed(nn.Module):
    """
    Single-step Audio2Gesture model for ONNX export.
    
    Processes one gesture frame at a time using fixed-size buffers.
    Implements the same approach that solved FaceFormer hanging issues.
    """
    def __init__(self,
                 aud_dim, aud_hid_dim, aud_embed_dim,
                 mo_dim, lxm_dim,
                 rnn_hid_dim, rnn_out_dim, rnn_depth,
                 max_audio_length=30):  # Fixed buffer size
        super().__init__()

        self.max_audio_length = max_audio_length
        self.mo_dim = mo_dim
        self.aud_embed_dim = aud_embed_dim
        self.lxm_dim = lxm_dim
        self.rnn_depth = rnn_depth
        self.rnn_hid_dim = rnn_hid_dim

        # Core model components
        self.audio_encoder = AudioEncoder_Conv1d(aud_dim, aud_hid_dim, aud_embed_dim)
        self.motion_decoder = GRUDecoder(mo_dim, aud_embed_dim, lxm_dim, rnn_hid_dim, rnn_out_dim, rnn_depth)
        self.cell_state_init = CellStateInitializer(mo_dim+lxm_dim, rnn_hid_dim, rnn_depth)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward(self, 
                audio_window, 
                prev_motion, 
                current_lexeme, 
                hidden_state, 
                step_idx):
        """
        Process one gesture generation step.
        
        Args:
            audio_window: [1, aud_dim, max_audio_length] - Fixed-size audio window
            prev_motion: [1, mo_dim] - Previous motion frame
            current_lexeme: [1, lxm_dim] - Current lexeme features
            hidden_state: [rnn_depth, 1, rnn_hid_dim] - GRU hidden state
            step_idx: [1] - Current step index (0-based)
            
        Returns:
            new_motion: [1, mo_dim] - Predicted motion frame
            new_hidden_state: [rnn_depth, 1, rnn_hid_dim] - Updated hidden state
        """
        batch_size = audio_window.shape[0]
        
        # Encode audio window
        audio_encoded = self.audio_encoder(audio_window)  # [1, aud_embed_dim, encoded_length]
        
        # Get current audio embedding for this step
        # Use average pooling instead of indexing to avoid tracing issues
        current_audio_embed = torch.mean(audio_encoded, dim=2)  # [1, aud_embed_dim]
        
        # Predict next motion using GRU decoder
        new_motion, new_hidden_state = self.motion_decoder(
            prev_motion, 
            current_audio_embed, 
            current_lexeme, 
            hidden_state
        )
        
        return new_motion, new_hidden_state


# Model hyperparameters (same as original)
aud_dim = 80
aud_hid_dim = 64
aud_embed_dim = 64
mo_dim = 48
lxm_dim = 96
rnn_hid_dim = 1024
rnn_out_dim = 48
rnn_depth = 4
max_audio_length = 30  # Fixed buffer size

# Create the single-step model
print("Creating Audio2Gesture single-step model...")
model = Audio2GestureStepFixed(
    aud_dim, aud_hid_dim, aud_embed_dim,
    mo_dim, lxm_dim,
    rnn_hid_dim, rnn_out_dim, rnn_depth,
    max_audio_length
)

# Create dummy inputs for export
batch_size = 1
audio_window = torch.randn(batch_size, aud_dim, max_audio_length)
prev_motion = torch.randn(batch_size, mo_dim)
current_lexeme = torch.randn(batch_size, lxm_dim)
hidden_state = torch.randn(rnn_depth, batch_size, rnn_hid_dim)
step_idx = torch.tensor([0], dtype=torch.long)

print("Testing model forward pass...")
model.eval()
with torch.no_grad():
    new_motion, new_hidden_state = model(
        audio_window, prev_motion, current_lexeme, hidden_state, step_idx
    )
    print(f"✅ Forward pass successful!")
    print(f"   Input motion: {prev_motion.shape}")
    print(f"   Output motion: {new_motion.shape}")
    print(f"   Hidden state: {hidden_state.shape} -> {new_hidden_state.shape}")

# Export to ONNX
onnx_path = "audio2gesture_step_fixed.onnx"
print(f"\nExporting to {onnx_path}...")

torch.onnx.export(
    model,
    (audio_window, prev_motion, current_lexeme, hidden_state, step_idx),
    onnx_path,
    input_names=[
        'audio_window', 
        'prev_motion', 
        'current_lexeme', 
        'hidden_state', 
        'step_idx'
    ],
    output_names=['new_motion', 'new_hidden_state'],
    dynamic_axes={
        'audio_window': {0: 'batch_size'},
        'prev_motion': {0: 'batch_size'},
        'current_lexeme': {0: 'batch_size'},
        'hidden_state': {1: 'batch_size'},
        'step_idx': {0: 'batch_size'},
        'new_motion': {0: 'batch_size'},
        'new_hidden_state': {1: 'batch_size'}
    },
    opset_version=16
)
print(f"✅ Model exported to {onnx_path}")

# Save test data for validation
test_data = {
    "audio_window": audio_window.numpy().tolist(),
    "prev_motion": prev_motion.numpy().tolist(),
    "current_lexeme": current_lexeme.numpy().tolist(),
    "hidden_state": hidden_state.numpy().tolist(),
    "step_idx": step_idx.numpy().tolist(),
    "new_motion": new_motion.numpy().tolist(),
    "new_hidden_state": new_hidden_state.numpy().tolist()
}

test_data_path = "audio2gesture_step_test_data.json"
with open(test_data_path, "w") as f:
    json.dump(test_data, f)
print(f"✅ Test data saved to {test_data_path}")

print("\n🎉 Audio2Gesture single-step export complete!")
print("This model can now be used for JavaScript autoregressive generation")
print("similar to the successful FaceFormer approach.")
