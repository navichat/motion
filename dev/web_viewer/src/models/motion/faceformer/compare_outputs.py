#!/usr/bin/env python3
"""
Compare outputs between Python FaceFormer and JavaScript FaceFormer
This script runs the Python model and saves outputs for comparison with JS model
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
import os

# Add the FaceFormer repo to the path
sys.path.append('/home/barberb/motion/engine/FaceFormer_repo')

try:
    from faceformer import Faceformer
    from wav2vec import Wav2Vec2Model
except ImportError as e:
    print(f"❌ Could not import FaceFormer modules: {e}")
    print("Make sure the FaceFormer repository is available")
    sys.exit(1)

class Args:
    """Mock args class for FaceFormer initialization"""
    def __init__(self):
        self.dataset = "vocaset"  # or "BIWI"
        self.feature_dim = 64
        self.vertice_dim = 15069  # Number of vertices * 3
        self.period = 25
        self.device = "cpu"  # Use CPU for compatibility
        self.train_subjects = "F1 F2 F3"  # Mock subjects

def load_test_data():
    """Load the test data from JSON file"""
    test_data_path = "/home/barberb/motion/engine/web_porting_poc/faceformer/faceformer_minimal_test_data.json"
    
    try:
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Could not load test data: {e}")
        return None

def run_python_model_comparison():
    """Run the Python FaceFormer model with test data and save outputs"""
    print("🐍 Running Python FaceFormer Model Comparison...")
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return None
    
    # Extract inputs
    inputs = test_data["inputs"]
    audio_features = torch.tensor(inputs["audio_features"], dtype=torch.float32)
    vertice_emb = torch.tensor(inputs["vertice_emb"], dtype=torch.float32) 
    one_hot = torch.tensor(inputs["one_hot"], dtype=torch.float32)
    template = torch.tensor(inputs["template"], dtype=torch.float32)
    
    print(f"📊 Input shapes:")
    print(f"  Audio features: {audio_features.shape}")
    print(f"  Vertice embedding: {vertice_emb.shape}")
    print(f"  One-hot: {one_hot.shape}")
    print(f"  Template: {template.shape}")
    
    try:
        # Initialize the model
        args = Args()
        model = Faceformer(args)
        model.eval()
        
        print("✅ Python FaceFormer model initialized successfully")
        
        # Run a single inference step (mimicking the minimal model)
        with torch.no_grad():
            # Use the template as input (like our minimal model)
            template_reshaped = template  # Already in [1, 1, 15069] format
            
            # Create mock audio (since we can't run full wav2vec easily)
            # Use the provided audio features directly
            mock_hidden_states = audio_features  # [1, 1, 768]
            mock_hidden_states = model.audio_feature_map(mock_hidden_states)  # Map to feature_dim
            
            # Get object embedding
            obj_embedding = model.obj_vector(one_hot)  # [1, feature_dim]
            
            # Create vertice input (template difference)
            vertice_input = template_reshaped - template_reshaped  # Zero difference for first step
            vertice_input = model.vertice_map(vertice_input)  # Map to feature_dim
            
            # Add style embedding
            style_emb = obj_embedding.unsqueeze(1)  # [1, 1, feature_dim] 
            vertice_input = vertice_input + style_emb
            
            # Apply positional encoding
            vertice_input_pe = model.PPE(vertice_input)
            
            # Create masks
            tgt_mask = model.biased_mask[:, :1, :1].clone().detach()  # [heads, 1, 1]
            memory_mask = torch.zeros(1, mock_hidden_states.shape[1], dtype=torch.bool)  # [1, seq_len]
            
            # Run transformer decoder
            decoder_output = model.transformer_decoder(
                vertice_input_pe, 
                mock_hidden_states, 
                tgt_mask=tgt_mask, 
                memory_mask=memory_mask
            )
            
            # Map back to vertex space
            vertex_delta = model.vertice_map_r(decoder_output)
            
            # Add template to get final vertices
            final_vertices = vertex_delta + template_reshaped
            
            # Prepare results
            results = {
                "python_model_outputs": {
                    "new_vertice_out": final_vertices.squeeze().tolist(),
                    "updated_vertice_emb": decoder_output.squeeze().tolist(),  # Use decoder output as embedding
                    "vertex_delta": vertex_delta.squeeze().tolist(),
                    "decoder_output": decoder_output.squeeze().tolist(),
                    "style_embedding": style_emb.squeeze().tolist(),
                    "obj_embedding": obj_embedding.squeeze().tolist()
                },
                "shapes": {
                    "final_vertices": list(final_vertices.shape),
                    "decoder_output": list(decoder_output.shape),
                    "vertex_delta": list(vertex_delta.shape),
                    "style_embedding": list(style_emb.shape),
                    "obj_embedding": list(obj_embedding.shape)
                },
                "model_info": {
                    "feature_dim": args.feature_dim,
                    "vertice_dim": args.vertice_dim,
                    "period": args.period,
                    "dataset": args.dataset
                }
            }
            
            print("✅ Python model inference completed successfully")
            print(f"📐 Output shapes:")
            for key, shape in results["shapes"].items():
                print(f"  {key}: {shape}")
            
            return results
            
    except Exception as e:
        print(f"❌ Error running Python model: {e}")
        import traceback
        traceback.print_exc()
        return None

def simplified_python_model():
    """Run a simplified version that matches our minimal ONNX model"""
    print("🔬 Running Simplified Python Model (matching ONNX minimal)...")
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return None
    
    # Extract inputs
    inputs = test_data["inputs"]
    audio_features = torch.tensor(inputs["audio_features"], dtype=torch.float32)
    vertice_emb = torch.tensor(inputs["vertice_emb"], dtype=torch.float32) 
    one_hot = torch.tensor(inputs["one_hot"], dtype=torch.float32)
    template = torch.tensor(inputs["template"], dtype=torch.float32)
    
    # Define simplified model components matching the minimal ONNX export
    class MinimalFaceFormer(nn.Module):
        def __init__(self, audio_dim=768, embedding_dim=64, vertex_dim=15069, num_subjects=3):
            super().__init__()
            
            # Audio processing
            self.audio_proj = nn.Linear(audio_dim, embedding_dim)
            
            # Embedding processing  
            self.emb_proj = nn.Linear(embedding_dim, embedding_dim)
            
            # Subject conditioning
            self.subject_embedding = nn.Linear(num_subjects, embedding_dim)
            
            # Fusion layer
            self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)
            
            # Output projections
            self.vertex_proj = nn.Linear(embedding_dim, vertex_dim)
            self.emb_update = nn.Linear(embedding_dim, embedding_dim)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        def forward(self, audio_features, vertice_emb, one_hot, template):
            batch_size = audio_features.shape[0]
            
            # Process inputs
            audio_proj = self.audio_proj(audio_features)  # [B, 1, emb_dim]
            emb_proj = self.emb_proj(vertice_emb)         # [B, 1, emb_dim]
            subject_emb = self.subject_embedding(one_hot) # [B, emb_dim]
            subject_emb = subject_emb.unsqueeze(1)        # [B, 1, emb_dim]
            
            # Fuse all features
            fused = torch.cat([audio_proj, emb_proj, subject_emb], dim=-1)  # [B, 1, emb_dim*3]
            fused = self.fusion(fused)  # [B, 1, emb_dim]
            
            # Generate outputs
            vertex_delta = self.vertex_proj(fused)    # [B, 1, vertex_dim]
            new_vertices = template + vertex_delta    # [B, 1, vertex_dim]
            updated_emb = self.emb_update(fused)      # [B, 1, emb_dim]
            
            return new_vertices, updated_emb
    
    # Create and run the model
    model = MinimalFaceFormer()
    model.eval()
    
    with torch.no_grad():
        new_vertices, updated_emb = model(audio_features, vertice_emb, one_hot, template)
        
        results = {
            "simplified_python_outputs": {
                "new_vertice_out": new_vertices.squeeze().tolist(),
                "updated_vertice_emb": updated_emb.squeeze().tolist()
            },
            "shapes": {
                "new_vertices": list(new_vertices.shape),
                "updated_emb": list(updated_emb.shape)
            }
        }
        
        print("✅ Simplified Python model completed")
        print(f"📐 Output shapes:")
        for key, shape in results["shapes"].items():
            print(f"  {key}: {shape}")
        
        return results

def save_comparison_data():
    """Save both model outputs for comparison"""
    print("💾 Saving comparison data...")
    
    # Run both models
    python_results = run_python_model_comparison()
    simplified_results = simplified_python_model()
    
    # Combine results
    comparison_data = {
        "timestamp": "2025-07-13",
        "test_inputs": load_test_data()["inputs"] if load_test_data() else None,
        "python_results": python_results,
        "simplified_results": simplified_results,
        "comparison_notes": {
            "python_model": "Full FaceFormer model (may have issues due to wav2vec complexity)",
            "simplified_model": "Minimal implementation matching ONNX export structure",
            "js_model": "To be compared - should match simplified_model outputs"
        }
    }
    
    # Save to file
    output_path = "/home/barberb/motion/dev/web_viewer/faceformer/python_model_outputs.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"✅ Comparison data saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Failed to save comparison data: {e}")
        return None

if __name__ == "__main__":
    print("🔍 FaceFormer Python vs JavaScript Output Comparison")
    print("=" * 60)
    
    output_file = save_comparison_data()
    
    if output_file:
        print("\n📋 Next steps:")
        print("1. Run the JavaScript model with the same test inputs")
        print("2. Compare the outputs using the comparison script")
        print("3. Analyze differences and adjust the models accordingly")
        print(f"\n📄 Python model outputs saved to: {output_file}")
    else:
        print("\n❌ Failed to generate comparison data")
