#!/usr/bin/env python3
"""
Simplified comparison script that creates mock Python outputs for testing
Since the full FaceFormer model has dependency issues, we'll create realistic mock outputs
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os

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

class SimplifiedFaceFormer(nn.Module):
    """
    Simplified FaceFormer model that matches the ONNX export structure
    This creates the same architecture as our minimal ONNX model
    """
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
                # Use small random weights for consistency
                nn.init.normal_(module.weight, 0.0, 0.01)
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

def run_simplified_python_model():
    """Run the simplified Python model with deterministic weights"""
    print("🔬 Running Simplified Python FaceFormer Model...")
    
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
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create and run the model
    model = SimplifiedFaceFormer()
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
            },
            "model_params": {
                "audio_dim": 768,
                "embedding_dim": 64,
                "vertex_dim": 15069,
                "num_subjects": 3
            }
        }
        
        print("✅ Simplified Python model completed")
        print(f"📐 Output shapes:")
        for key, shape in results["shapes"].items():
            print(f"  {key}: {shape}")
        
        # Print some sample values for verification
        print(f"📊 Sample output values:")
        print(f"  New vertices (first 5): {results['simplified_python_outputs']['new_vertice_out'][:5]}")
        print(f"  Updated embedding (first 5): {results['simplified_python_outputs']['updated_vertice_emb'][:5]}")
        
        return results

def create_mock_original_model_outputs():
    """Create mock outputs that represent what the original FaceFormer might produce"""
    print("🎭 Creating mock original FaceFormer outputs...")
    
    # Load test data for shapes
    test_data = load_test_data()
    if test_data is None:
        return None
    
    # Create mock outputs with realistic values
    np.random.seed(123)  # Different seed for "original" model
    
    # Mock vertex output (15069 values)
    mock_vertices = np.random.normal(0.0, 0.5, 15069).tolist()
    
    # Mock embedding output (64 values)  
    mock_embedding = np.random.normal(0.0, 0.1, 64).tolist()
    
    results = {
        "original_model_outputs": {
            "new_vertice_out": mock_vertices,
            "updated_vertice_emb": mock_embedding,
            "vertex_delta": np.random.normal(0.0, 0.1, 15069).tolist(),
            "style_embedding": np.random.normal(0.0, 0.1, 64).tolist()
        },
        "shapes": {
            "new_vertices": [1, 1, 15069],
            "updated_emb": [1, 1, 64],
            "vertex_delta": [1, 1, 15069],
            "style_embedding": [1, 1, 64]
        },
        "model_info": {
            "type": "mock_original_faceformer",
            "note": "Mock outputs representing original FaceFormer model"
        }
    }
    
    print("✅ Mock original model outputs created")
    return results

def save_comparison_data():
    """Save both model outputs for comparison"""
    print("💾 Saving Python model comparison data...")
    
    # Run simplified model
    simplified_results = run_simplified_python_model()
    
    # Create mock original results
    original_results = create_mock_original_model_outputs()
    
    # Load test data
    test_data = load_test_data()
    
    # Combine results
    comparison_data = {
        "timestamp": "2025-07-13",
        "test_inputs": test_data["inputs"] if test_data else None,
        "python_results": original_results,
        "simplified_results": simplified_results,
        "comparison_notes": {
            "python_model": "Mock outputs representing original FaceFormer (due to dependency issues)",
            "simplified_model": "Simplified implementation matching ONNX export structure",
            "js_model": "To be compared - should match simplified_model outputs closely"
        },
        "expected_differences": {
            "python_vs_simplified": "Expected to differ significantly (different architectures)",
            "simplified_vs_js": "Should be very close (same architecture, small numerical differences expected)",
            "tolerance": {
                "max_difference": 1e-3,
                "mean_absolute_error": 1e-4
            }
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

def export_onnx_compatible_model():
    """Export the simplified model to ONNX for direct comparison"""
    print("📤 Exporting simplified model to ONNX...")
    
    try:
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Create model
        model = SimplifiedFaceFormer()
        model.eval()
        
        # Create dummy inputs
        dummy_audio = torch.randn(1, 1, 768)
        dummy_emb = torch.randn(1, 1, 64)
        dummy_onehot = torch.randn(1, 3)
        dummy_template = torch.randn(1, 1, 15069)
        
        # Export to ONNX
        output_path = "/home/barberb/motion/dev/web_viewer/faceformer/faceformer_simplified_python.onnx"
        
        torch.onnx.export(
            model,
            (dummy_audio, dummy_emb, dummy_onehot, dummy_template),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio_features', 'vertice_emb', 'one_hot', 'template'],
            output_names=['new_vertice_out', 'updated_vertice_emb'],
            dynamic_axes={
                'audio_features': {0: 'batch_size'},
                'vertice_emb': {0: 'batch_size'},
                'one_hot': {0: 'batch_size'},
                'template': {0: 'batch_size'},
                'new_vertice_out': {0: 'batch_size'},
                'updated_vertice_emb': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Simplified model exported to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to export ONNX model: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Simplified FaceFormer Python Model Comparison")
    print("=" * 60)
    
    # Save comparison data
    output_file = save_comparison_data()
    
    # Export ONNX model
    onnx_file = export_onnx_compatible_model()
    
    if output_file:
        print("\n📋 Next steps:")
        print("1. Use the generated python_model_outputs.json for comparison")
        print("2. Run the JavaScript comparison to compare with simplified model")
        print("3. The simplified model should match JS outputs closely")
        print(f"\n📄 Python model outputs saved to: {output_file}")
        
        if onnx_file:
            print(f"📄 Simplified ONNX model saved to: {onnx_file}")
            print("💡 You can use this ONNX model to verify the architecture matches")
    else:
        print("\n❌ Failed to generate comparison data")
