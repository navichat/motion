
#!/usr/bin/env python3
"""
Output Verification: Python vs Web Implementation
This script ensures the web implementation produces identical outputs to Python
"""

import torch
import numpy as np
import json
import os
import onnxruntime as ort
from export_python_weights import SimplifiedFaceFormer

class OutputVerifier:
    def __init__(self):
        self.python_model = None
        self.onnx_session = None
        self.tolerance = 1e-5
        
    def setup_python_model(self):
        """Setup the Python reference model"""
        print("🐍 Setting up Python reference model...")
        torch.manual_seed(42)
        self.python_model = SimplifiedFaceFormer()
        self.python_model.eval()
        print("✅ Python model ready")
        
    def setup_onnx_model(self, onnx_path):
        """Setup the ONNX model for comparison"""
        print(f"📦 Setting up ONNX model: {onnx_path}")
        self.onnx_session = ort.InferenceSession(onnx_path)
        print("✅ ONNX model ready")
        
    def generate_test_data(self, batch_size=1, seq_len=5):
        """Generate consistent test data"""
        print(f"🎲 Generating test data (batch={batch_size}, seq_len={seq_len})...")
        
        # Set seed for reproducible test data
        torch.manual_seed(123)
        np.random.seed(123)
        
        test_data = {
            'audio_features': torch.randn(batch_size, seq_len, 768),
            'vertice_emb': torch.randn(batch_size, seq_len, 64),
            'one_hot': torch.randn(batch_size, 3),
            'template': torch.randn(batch_size, seq_len, 15069)
        }
        
        print("✅ Test data generated")
        return test_data
        
    def run_python_inference(self, test_data):
        """Run inference with Python model"""
        print("🐍 Running Python inference...")
        
        with torch.no_grad():
            vertices, embeddings = self.python_model(
                test_data['audio_features'],
                test_data['vertice_emb'], 
                test_data['one_hot'],
                test_data['template']
            )
        
        python_results = {
            'vertices': vertices.numpy(),
            'embeddings': embeddings.numpy()
        }
        
        print(f"✅ Python inference complete: {vertices.shape} vertices")
        return python_results
        
    def run_onnx_inference(self, test_data):
        """Run inference with ONNX model"""
        print("📦 Running ONNX inference...")
        
        # Convert to numpy and correct shapes for ONNX
        onnx_inputs = {
            'audio_features': test_data['audio_features'].numpy(),
            'vertice_emb': test_data['vertice_emb'].numpy(),
            'one_hot': test_data['one_hot'].numpy(),
            'template': test_data['template'].numpy()
        }
        
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)
        
        onnx_results = {
            'vertices': onnx_outputs[0],
            'embeddings': onnx_outputs[1]
        }
        
        print(f"✅ ONNX inference complete: {onnx_outputs[0].shape} vertices")
        return onnx_results
        
    def compare_outputs(self, python_results, onnx_results):
        """Compare Python and ONNX outputs"""
        print("🔍 Comparing outputs...")
        
        # Compare vertices
        vertex_diff = np.max(np.abs(python_results['vertices'] - onnx_results['vertices']))
        vertex_rel_diff = vertex_diff / (np.max(np.abs(python_results['vertices'])) + 1e-8)
        
        # Compare embeddings
        emb_diff = np.max(np.abs(python_results['embeddings'] - onnx_results['embeddings']))
        emb_rel_diff = emb_diff / (np.max(np.abs(python_results['embeddings'])) + 1e-8)
        
        print(f"📊 Comparison Results:")
        print(f"  Vertex max difference: {vertex_diff:.2e}")
        print(f"  Vertex relative difference: {vertex_rel_diff:.2e}")
        print(f"  Embedding max difference: {emb_diff:.2e}")
        print(f"  Embedding relative difference: {emb_rel_diff:.2e}")
        
        # Check if within tolerance
        vertex_ok = vertex_diff < self.tolerance
        emb_ok = emb_diff < self.tolerance
        
        if vertex_ok and emb_ok:
            print("✅ OUTPUT VERIFICATION PASSED - Identical results!")
            return True
        else:
            print("❌ OUTPUT VERIFICATION FAILED - Significant differences!")
            return False
            
    def verify_equivalence(self, onnx_path):
        """Full verification pipeline"""
        print("🧪 FULL OUTPUT VERIFICATION")
        print("=" * 40)
        
        # Setup models
        self.setup_python_model()
        self.setup_onnx_model(onnx_path)
        
        # Generate test data
        test_data = self.generate_test_data()
        
        # Run inference on both
        python_results = self.run_python_inference(test_data)
        onnx_results = self.run_onnx_inference(test_data)
        
        # Compare results
        success = self.compare_outputs(python_results, onnx_results)
        
        return success

def main():
    # Check if we have the ONNX model
    onnx_path = "faceformer_python_weights.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        print("Run export_python_weights.py first to create the ONNX model")
        return False
    
    verifier = OutputVerifier()
    success = verifier.verify_equivalence(onnx_path)
    
    print(f"\n🎯 Final Result: {'✅ VERIFIED' if success else '❌ FAILED'}")
    return success

if __name__ == "__main__":
    main()
