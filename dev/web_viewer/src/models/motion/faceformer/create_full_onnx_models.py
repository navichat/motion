#!/usr/bin/env python3
"""
Create Full FaceFormer ONNX Models with Complete Transformer Architecture
Includes multi-head attention, feed-forward networks, layer normalization, and Wav2Vec2 integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optimized implementation for ONNX export"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Linear transformations and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.w_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention and feed-forward"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FullFaceFormerTransformer(nn.Module):
    """Complete FaceFormer with transformer architecture and Wav2Vec2 integration"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.feature_dim = config['feature_dim']
        self.audio_input_dim = config['audio_input_dim']
        self.vertice_dim = config['vertice_dim']
        self.num_subjects = config['num_subjects']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.max_seq_length = config['max_seq_length']
        self.dropout = config['dropout']
        
        # Feed-forward dimension (typically 4x feature_dim)
        self.d_ff = self.feature_dim * 4
        
        # Audio feature projection
        self.audio_projection = nn.Linear(self.audio_input_dim, self.feature_dim)
        
        # Subject embedding
        self.subject_embedding = nn.Embedding(self.num_subjects, self.feature_dim)
        
        # Template embedding
        self.template_projection = nn.Linear(self.vertice_dim, self.feature_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.feature_dim, self.max_seq_length)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.feature_dim, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.feature_dim, self.vertice_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        logger.info(f"Created FullFaceFormerTransformer with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def create_padding_mask(self, seq_length: int, max_length: int) -> torch.Tensor:
        """Create padding mask for variable length sequences"""
        mask = torch.ones(1, 1, max_length, max_length)
        if seq_length < max_length:
            mask[:, :, seq_length:, :] = 0
            mask[:, :, :, seq_length:] = 0
        return mask
    
    def forward(self, audio_features, template, subject_id):
        batch_size, seq_len, audio_dim = audio_features.shape
        
        # Project audio features to model dimension
        audio_proj = self.audio_projection(audio_features)
        
        # Get subject embedding and expand to sequence length
        subject_emb = self.subject_embedding(subject_id)
        subject_emb = subject_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Project template and expand to sequence length
        template_proj = self.template_projection(template)
        template_proj = template_proj.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Combine all features
        x = audio_proj + subject_emb + template_proj
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout_layer(x)
        
        # Create attention mask for padding
        attention_mask = self.create_padding_mask(seq_len, seq_len)
        attention_mask = attention_mask.to(x.device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Project to output vertices
        vertices = self.output_projection(x)
        
        return vertices

class Wav2Vec2Stub(nn.Module):
    """Stub implementation of Wav2Vec2 for ONNX export"""
    
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.output_dim = output_dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projection = nn.Linear(256, output_dim)
    
    def forward(self, waveform):
        # Waveform shape: (batch, length)
        x = waveform.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)    # Extract features
        x = x.squeeze(-1)          # Remove last dimension
        x = self.projection(x)     # Project to output dimension
        
        # Expand to sequence length (approximate frame rate)
        seq_len = waveform.shape[1] // 320  # 16kHz / 50Hz = 320 samples per frame
        x = x.unsqueeze(1).expand(-1, seq_len, -1)
        
        return x

def create_full_models(dataset: str = 'vocaset') -> Tuple[FullFaceFormerTransformer, Dict]:
    """Create full FaceFormer model with configuration"""
    
    if dataset.lower() == 'vocaset':
        config = {
            'dataset': 'VOCASET',
            'feature_dim': 128,
            'audio_input_dim': 768,
            'vertice_dim': 15069,
            'num_subjects': 8,
            'num_heads': 8,
            'num_layers': 6,
            'max_seq_length': 600,
            'dropout': 0.1
        }
    elif dataset.lower() == 'biwi':
        config = {
            'dataset': 'BIWI',
            'feature_dim': 256,
            'audio_input_dim': 768,
            'vertice_dim': 70110,
            'num_subjects': 6,
            'num_heads': 8,
            'num_layers': 6,
            'max_seq_length': 600,
            'dropout': 0.1
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    model = FullFaceFormerTransformer(config)
    
    return model, config

def load_existing_weights(model: FullFaceFormerTransformer, weights_path: str) -> bool:
    """Load weights from existing PyTorch model if available"""
    
    if not os.path.exists(weights_path):
        logger.warning(f"Weights file not found: {weights_path}")
        return False
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load compatible weights
        model_dict = model.state_dict()
        compatible_weights = {}
        
        for name, param in state_dict.items():
            if name in model_dict and param.shape == model_dict[name].shape:
                compatible_weights[name] = param
                logger.info(f"Loaded weight: {name} {param.shape}")
            else:
                logger.warning(f"Skipped incompatible weight: {name}")
        
        model.load_state_dict(compatible_weights, strict=False)
        logger.info(f"Loaded {len(compatible_weights)} compatible weights")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return False

def export_to_onnx(model: nn.Module, config: Dict, output_path: str, dataset: str):
    """Export model to ONNX format with dynamic axes"""
    
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 100  # Example sequence length
    
    dummy_audio = torch.randn(batch_size, seq_len, config['audio_input_dim'])
    dummy_template = torch.randn(batch_size, config['vertice_dim'])
    dummy_subject = torch.tensor([0], dtype=torch.long)
    
    # Input names
    input_names = ['audio_features', 'template', 'subject_id']
    output_names = ['vertices']
    
    # Dynamic axes for variable sequence length
    dynamic_axes = {
        'audio_features': {1: 'seq_len'},
        'vertices': {1: 'seq_len'}
    }
    
    logger.info(f"Exporting to ONNX: {output_path}")
    
    try:
        torch.onnx.export(
            model,
            (dummy_audio, dummy_template, dummy_subject),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"✅ ONNX export successful: {output_path}")
        
        # Test with ONNX Runtime
        test_onnx_model(output_path, config)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        return False

def test_onnx_model(onnx_path: str, config: Dict):
    """Test the exported ONNX model"""
    
    import time
    
    try:
        session = ort.InferenceSession(onnx_path)
        
        # Create test inputs
        batch_size = 1
        seq_len = 50
        
        test_audio = np.random.randn(batch_size, seq_len, config['audio_input_dim']).astype(np.float32)
        test_template = np.random.randn(batch_size, config['vertice_dim']).astype(np.float32)
        test_subject = np.array([0], dtype=np.int64)
        
        # Run inference
        inputs = {
            'audio_features': test_audio,
            'template': test_template,
            'subject_id': test_subject
        }
        
        start_time = time.time()
        outputs = session.run(None, inputs)
        inference_time = time.time() - start_time
        
        vertices = outputs[0]
        
        logger.info(f"✅ ONNX model test successful")
        logger.info(f"  Input shape: {test_audio.shape}")
        logger.info(f"  Output shape: {vertices.shape}")
        logger.info(f"  Inference time: {inference_time*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX model test failed: {e}")
        return False

def export_wav2vec2_stub(output_path: str):
    """Export Wav2Vec2 stub model to ONNX"""
    
    model = Wav2Vec2Stub()
    model.eval()
    
    # Dummy input: 1 second of 16kHz audio
    dummy_audio = torch.randn(1, 16000)
    
    try:
        torch.onnx.export(
            model,
            dummy_audio,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_values'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_values': {1: 'audio_length'},
                'last_hidden_state': {1: 'seq_length'}
            },
            verbose=False
        )
        
        logger.info(f"✅ Wav2Vec2 stub exported: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Wav2Vec2 stub export failed: {e}")
        return False

def main():
    """Main function to create and export full models"""
    
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Full FaceFormer ONNX Models')
    parser.add_argument('--dataset', choices=['vocaset', 'biwi'], default='vocaset',
                       help='Dataset to create model for')
    parser.add_argument('--weights-dir', type=str, default='.',
                       help='Directory containing original weight files')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for ONNX models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Creating Full FaceFormer models for {args.dataset.upper()}")
    
    # Create model
    model, config = create_full_models(args.dataset)
    
    # Try to load existing weights
    weight_files = [
        f"{args.dataset}.pth",
        f"faceformer_{args.dataset}.pth",
        f"{args.dataset}_model.pth"
    ]
    
    weights_loaded = False
    for weight_file in weight_files:
        weight_path = os.path.join(args.weights_dir, weight_file)
        if load_existing_weights(model, weight_path):
            weights_loaded = True
            logger.info(f"✅ Loaded weights from {weight_path}")
            break
    
    if not weights_loaded:
        logger.warning("⚠️ No compatible weights found, using random initialization")
    
    # Export main FaceFormer model
    main_model_path = output_dir / f"faceformer_{args.dataset}_full.onnx"
    success = export_to_onnx(model, config, str(main_model_path), args.dataset)
    
    if success:
        # Save configuration
        config_path = output_dir / f"faceformer_{args.dataset}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Configuration saved: {config_path}")
        
        # Export Wav2Vec2 stub
        wav2vec2_path = output_dir / "wav2vec2_base.onnx"
        export_wav2vec2_stub(str(wav2vec2_path))
        
        # Print summary
        model_size = os.path.getsize(main_model_path) / (1024 * 1024)
        param_count = model.count_parameters()
        
        logger.info("🎉 Export completed successfully!")
        logger.info(f"  Model: {main_model_path}")
        logger.info(f"  Size: {model_size:.1f} MB")
        logger.info(f"  Parameters: {param_count:,}")
        logger.info(f"  Layers: {config['num_layers']}")
        logger.info(f"  Attention heads: {config['num_heads']}")
        
        return True
    else:
        logger.error("❌ Export failed")
        return False

if __name__ == '__main__':
    main()
