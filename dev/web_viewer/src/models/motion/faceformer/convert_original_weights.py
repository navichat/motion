#!/usr/bin/env python3
"""
FaceFormer Original Weights to ONNX Converter
Converts the original .pth FaceFormer weights to ONNX format for web deployment
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
from collections import OrderedDict
import json

# Add the FaceFormer repo to path
sys.path.insert(0, '/home/barberb/motion/engine/FaceFormer_repo')

try:
    from faceformer import Faceformer
    from wav2vec import Wav2Vec2Model
except ImportError as e:
    print(f"❌ Could not import FaceFormer modules: {e}")
    print("💡 Make sure FaceFormer repo is available at /home/barberb/motion/engine/FaceFormer_repo")
    sys.exit(1)

class FaceFormerConfig:
    """Configuration class for FaceFormer models"""
    
    def __init__(self, dataset="BIWI"):
        self.dataset = dataset
        self.device = "cpu"  # Use CPU for conversion
        
        if dataset == "BIWI":
            self.feature_dim = 128
            self.period = 25
            self.vertice_dim = 23370 * 3  # 70110
            self.train_subjects = "F2 F3 F4 M3 M4 M5"
            self.fps = 25
        elif dataset == "vocaset":
            self.feature_dim = 64
            self.period = 30
            self.vertice_dim = 5023 * 3  # 15069
            self.train_subjects = "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA"
            self.fps = 30
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

def load_original_model(weight_path, dataset="BIWI"):
    """Load the original FaceFormer model with weights"""
    
    print(f"📥 Loading original FaceFormer model...")
    print(f"  Dataset: {dataset}")
    print(f"  Weights: {weight_path}")
    
    # Create configuration
    args = FaceFormerConfig(dataset)
    
    # Build model
    model = Faceformer(args)
    
    # Load weights
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Successfully loaded weights from {weight_path}")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return None, None
    
    model.eval()
    return model, args

def analyze_model_structure(model, args):
    """Analyze the model structure and extract key information"""
    
    print(f"\n🔍 Analyzing FaceFormer model structure...")
    
    info = {
        "dataset": args.dataset,
        "architecture": "FaceFormer",
        "components": {},
        "dimensions": {
            "feature_dim": args.feature_dim,
            "vertice_dim": args.vertice_dim,
            "period": args.period,
            "num_subjects": len(args.train_subjects.split())
        }
    }
    
    # Analyze components
    components = {}
    
    # Audio encoder (Wav2Vec2)
    if hasattr(model, 'audio_encoder'):
        components['audio_encoder'] = {
            "type": "Wav2Vec2Model",
            "output_dim": 768,  # Wav2Vec2 output dimension
            "frozen": True
        }
    
    # Audio feature mapping
    if hasattr(model, 'audio_feature_map'):
        components['audio_feature_map'] = {
            "type": "Linear",
            "input_dim": 768,
            "output_dim": args.feature_dim,
            "params": sum(p.numel() for p in model.audio_feature_map.parameters())
        }
    
    # Vertex mapping
    if hasattr(model, 'vertice_map'):
        components['vertice_map'] = {
            "type": "Linear", 
            "input_dim": args.vertice_dim,
            "output_dim": args.feature_dim,
            "params": sum(p.numel() for p in model.vertice_map.parameters())
        }
    
    # Positional encoding
    if hasattr(model, 'PPE'):
        components['PPE'] = {
            "type": "PeriodicPositionalEncoding",
            "d_model": args.feature_dim,
            "period": args.period,
            "params": sum(p.numel() for p in model.PPE.parameters())
        }
    
    # Transformer decoder
    if hasattr(model, 'transformer_decoder'):
        components['transformer_decoder'] = {
            "type": "TransformerDecoder",
            "d_model": args.feature_dim,
            "nhead": 4,
            "num_layers": 1,
            "params": sum(p.numel() for p in model.transformer_decoder.parameters())
        }
    
    # Vertex reconstruction
    if hasattr(model, 'vertice_map_r'):
        components['vertice_map_r'] = {
            "type": "Linear",
            "input_dim": args.feature_dim,
            "output_dim": args.vertice_dim,
            "params": sum(p.numel() for p in model.vertice_map_r.parameters())
        }
    
    # Style embedding
    if hasattr(model, 'obj_vector'):
        components['obj_vector'] = {
            "type": "Linear",
            "input_dim": len(args.train_subjects.split()),
            "output_dim": args.feature_dim,
            "bias": False,
            "params": sum(p.numel() for p in model.obj_vector.parameters())
        }
    
    info["components"] = components
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info["parameters"] = {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }
    
    print(f"✅ Model analysis complete:")
    print(f"  Components: {len(components)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return info

def extract_core_weights(model, args):
    """Extract the core trainable weights (excluding Wav2Vec2)"""
    
    print(f"\n🔧 Extracting core FaceFormer weights...")
    
    core_weights = OrderedDict()
    excluded_prefixes = ['audio_encoder.']  # Exclude Wav2Vec2 weights
    
    for name, param in model.named_parameters():
        # Skip frozen Wav2Vec2 weights
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
            
        # Convert to numpy for JSON serialization
        weight_data = param.detach().cpu().numpy().tolist()
        core_weights[name] = weight_data
        
        print(f"  ✅ {name}: {param.shape}")
    
    print(f"✅ Extracted {len(core_weights)} core weight tensors")
    return core_weights

def create_simplified_onnx_model(model, args, output_path):
    """Create a simplified ONNX model for the core FaceFormer functionality"""
    
    print(f"\n🔄 Creating simplified ONNX model...")
    
    # Define a simplified model that excludes Wav2Vec2
    class SimplifiedFaceFormer(nn.Module):
        def __init__(self, original_model, args):
            super().__init__()
            
            # Copy the core components (excluding audio_encoder)
            self.audio_feature_map = original_model.audio_feature_map
            self.vertice_map = original_model.vertice_map
            self.PPE = original_model.PPE
            self.transformer_decoder = original_model.transformer_decoder
            self.vertice_map_r = original_model.vertice_map_r
            self.obj_vector = original_model.obj_vector
            
            # Store configuration
            self.feature_dim = args.feature_dim
            self.vertice_dim = args.vertice_dim
            self.num_subjects = len(args.train_subjects.split())
            
            # Copy biased mask
            self.register_buffer('biased_mask', original_model.biased_mask)
        
        def forward(self, audio_features, template, one_hot):
            """
            Simplified forward pass that takes pre-computed audio features
            
            Args:
                audio_features: [batch_size, seq_len, 768] - Pre-computed Wav2Vec2 features
                template: [batch_size, vertice_dim] - Template vertices
                one_hot: [batch_size, num_subjects] - Subject one-hot encoding
            
            Returns:
                vertices: [batch_size, seq_len, vertice_dim] - Predicted vertices
            """
            batch_size = audio_features.shape[0]
            seq_len = audio_features.shape[1]
            
            # Map audio features to model dimension
            hidden_states = self.audio_feature_map(audio_features)  # [B, S, feature_dim]
            
            # Get style embedding
            obj_embedding = self.obj_vector(one_hot)  # [B, feature_dim]
            style_emb = obj_embedding.unsqueeze(1)  # [B, 1, feature_dim]
            
            # Prepare template
            template = template.unsqueeze(1)  # [B, 1, vertice_dim]
            
            # Initialize output list
            vertices_list = []
            
            # Auto-regressive generation
            vertice_emb = style_emb  # Start with style embedding
            
            for i in range(seq_len):
                # Apply positional encoding
                vertice_input = self.PPE(vertice_emb)
                
                # Create masks
                tgt_len = vertice_input.shape[1]
                tgt_mask = self.biased_mask[:, :tgt_len, :tgt_len].clone()
                
                # Transformer decoder
                vertice_out = self.transformer_decoder(
                    vertice_input, 
                    hidden_states,
                    tgt_mask=tgt_mask
                )
                
                # Map back to vertex space
                vertice_delta = self.vertice_map_r(vertice_out)
                
                # Add to template
                vertices = vertice_delta + template
                
                # Store the last generated vertex
                vertices_list.append(vertices[:, -1:, :])  # [B, 1, vertice_dim]
                
                # Prepare next input
                if i < seq_len - 1:
                    # Map last vertex back to feature space
                    next_input = self.vertice_map(vertices[:, -1:, :] - template)
                    next_input = next_input + style_emb
                    vertice_emb = torch.cat([vertice_emb, next_input], dim=1)
            
            # Concatenate all generated vertices
            output_vertices = torch.cat(vertices_list, dim=1)  # [B, seq_len, vertice_dim]
            
            return output_vertices
    
    # Create simplified model
    simplified_model = SimplifiedFaceFormer(model, args)
    simplified_model.eval()
    
    # Create example inputs
    batch_size = 1
    seq_len = 10  # Example sequence length
    
    audio_features = torch.randn(batch_size, seq_len, 768)
    template = torch.randn(batch_size, args.vertice_dim)
    one_hot = torch.zeros(batch_size, len(args.train_subjects.split()))
    one_hot[0, 0] = 1.0  # Select first subject
    
    # Test the model
    print(f"🧪 Testing simplified model...")
    with torch.no_grad():
        try:
            output = simplified_model(audio_features, template, one_hot)
            print(f"  ✅ Model test successful: {output.shape}")
        except Exception as e:
            print(f"  ❌ Model test failed: {e}")
            return False
    
    # Export to ONNX
    try:
        torch.onnx.export(
            simplified_model,
            (audio_features, template, one_hot),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio_features', 'template', 'one_hot'],
            output_names=['vertices'],
            dynamic_axes={
                'audio_features': {1: 'seq_len'},
                'vertices': {1: 'seq_len'}
            }
        )
        print(f"✅ ONNX export successful: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def convert_faceformer_weights(weight_path, dataset, output_dir):
    """Main conversion function"""
    
    print(f"🚀 Converting FaceFormer weights to ONNX...")
    print(f"📁 Input: {weight_path}")
    print(f"📊 Dataset: {dataset}")
    print(f"📂 Output: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load original model
    model, args = load_original_model(weight_path, dataset)
    if model is None:
        return False
    
    # Analyze model structure
    model_info = analyze_model_structure(model, args)
    
    # Extract core weights
    core_weights = extract_core_weights(model, args)
    
    # Save weights as JSON
    weights_json_path = os.path.join(output_dir, f"faceformer_{dataset.lower()}_weights.json")
    weights_data = {
        "model_info": model_info,
        "weights": core_weights,
        "export_info": {
            "source_file": weight_path,
            "dataset": dataset,
            "export_timestamp": "2025-07-13",
            "converter_version": "1.0.0"
        }
    }
    
    with open(weights_json_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    print(f"💾 Weights saved to: {weights_json_path}")
    
    # Create simplified ONNX model
    onnx_path = os.path.join(output_dir, f"faceformer_{dataset.lower()}_simplified.onnx")
    onnx_success = create_simplified_onnx_model(model, args, onnx_path)
    
    # Save model info
    info_path = os.path.join(output_dir, f"faceformer_{dataset.lower()}_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"📊 Model info saved to: {info_path}")
    
    # Summary
    print(f"\n🎯 Conversion Summary:")
    print(f"  ✅ Weights extracted: {len(core_weights)} tensors")
    print(f"  ✅ JSON export: {weights_json_path}")
    print(f"  {'✅' if onnx_success else '❌'} ONNX export: {onnx_path}")
    print(f"  ✅ Model info: {info_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert FaceFormer .pth weights to ONNX format')
    parser.add_argument('--biwi_weights', type=str, default='weights/biwi.pth',
                       help='Path to BIWI weights file')
    parser.add_argument('--vocaset_weights', type=str, default='weights/vocaset.pth', 
                       help='Path to VOCASET weights file')
    parser.add_argument('--output_dir', type=str, default='converted_weights',
                       help='Output directory for converted files')
    parser.add_argument('--datasets', nargs='+', default=['BIWI', 'vocaset'],
                       help='Datasets to convert (BIWI, vocaset)')
    
    args = parser.parse_args()
    
    success_count = 0
    total_count = 0
    
    for dataset in args.datasets:
        total_count += 1
        
        if dataset == "BIWI":
            weight_path = args.biwi_weights
        elif dataset == "vocaset":
            weight_path = args.vocaset_weights
        else:
            print(f"❌ Unknown dataset: {dataset}")
            continue
        
        if not os.path.exists(weight_path):
            print(f"❌ Weight file not found: {weight_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting {dataset} weights...")
        print(f"{'='*60}")
        
        success = convert_faceformer_weights(weight_path, dataset, args.output_dir)
        if success:
            success_count += 1
    
    print(f"\n🏁 Conversion Complete: {success_count}/{total_count} successful")
    
    if success_count > 0:
        print(f"\n💡 Next steps:")
        print(f"  1. Test the converted ONNX models")
        print(f"  2. Integrate with your web FaceFormer implementation")
        print(f"  3. Use the JSON weights for exact weight transfer")

if __name__ == "__main__":
    main()
