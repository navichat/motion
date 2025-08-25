#!/usr/bin/env python3
"""
Minimal FaceFormer Export for Web
Creates a super simple, working ONNX model for facial animation
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os

class MinimalFaceFormer(nn.Module):
    """
    Extremely simplified FaceFormer that focuses on working in ONNX
    """
    def __init__(self, audio_dim=768, embedding_dim=64, vertex_dim=15069, num_subjects=3):
        super().__init__()
        self.audio_dim = audio_dim
        self.embedding_dim = embedding_dim
        self.vertex_dim = vertex_dim
        self.num_subjects = num_subjects
        
        # Simple linear layers only - no complex transformers or attention
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.subject_embedding = nn.Linear(num_subjects, embedding_dim, bias=False)
        
        self.vertex_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        
        # Core generation layer
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),  # audio + subject + vertex
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, vertex_dim)
        )
        
        # Next embedding predictor
        self.embedding_updater = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # current + new
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, audio_features, vertice_emb, one_hot, template):
        """
        Single step forward pass for autoregressive generation
        
        Args:
            audio_features: [batch, 1, 768] - single frame audio features
            vertice_emb: [batch, 1, 64] - current vertex embedding  
            one_hot: [batch, 3] - subject identity
            template: [batch, 1, 15069] - template vertices
            
        Returns:
            new_vertice_out: [batch, 1, 15069] - predicted vertex displacement
            updated_vertice_emb: [batch, 1, 64] - updated embedding for next step
        """
        batch_size = audio_features.shape[0]
        
        # Process inputs
        audio_emb = self.audio_processor(audio_features)  # [batch, 1, 64]
        subject_emb = self.subject_embedding(one_hot).unsqueeze(1)  # [batch, 1, 64]
        vertex_emb = self.vertex_processor(vertice_emb)  # [batch, 1, 64]
        
        # Flatten for concatenation
        audio_flat = audio_emb.view(batch_size, -1)  # [batch, 64]
        subject_flat = subject_emb.view(batch_size, -1)  # [batch, 64] 
        vertex_flat = vertex_emb.view(batch_size, -1)  # [batch, 64]
        
        # Combine all features
        combined = torch.cat([audio_flat, subject_flat, vertex_flat], dim=1)  # [batch, 192]
        
        # Generate vertex displacement
        vertex_delta = self.generator(combined)  # [batch, 15069]
        vertex_delta = vertex_delta.unsqueeze(1)  # [batch, 1, 15069]
        
        # Add to template
        new_vertice_out = template + vertex_delta
        
        # Update embedding for next step
        current_emb_flat = vertex_flat  # [batch, 64]
        new_emb_flat = audio_flat  # Use audio as next embedding base
        
        emb_input = torch.cat([current_emb_flat, new_emb_flat], dim=1)  # [batch, 128]
        updated_emb = self.embedding_updater(emb_input)  # [batch, 64]
        updated_vertice_emb = updated_emb.unsqueeze(1)  # [batch, 1, 64]
        
        return new_vertice_out, updated_vertice_emb

def test_model():
    """Test the model before export"""
    print("Testing MinimalFaceFormer...")
    
    model = MinimalFaceFormer()
    model.eval()
    
    # Create test inputs
    batch_size = 1
    audio_features = torch.randn(batch_size, 1, 768)
    vertice_emb = torch.randn(batch_size, 1, 64) 
    one_hot = torch.zeros(batch_size, 3)
    one_hot[0, 0] = 1.0  # Select first subject
    template = torch.randn(batch_size, 1, 15069)
    
    print(f"Input shapes:")
    print(f"  audio_features: {audio_features.shape}")
    print(f"  vertice_emb: {vertice_emb.shape}")
    print(f"  one_hot: {one_hot.shape}")
    print(f"  template: {template.shape}")
    
    # Test forward pass
    with torch.no_grad():
        new_out, updated_emb = model(audio_features, vertice_emb, one_hot, template)
        
    print(f"Output shapes:")
    print(f"  new_vertice_out: {new_out.shape}")
    print(f"  updated_vertice_emb: {updated_emb.shape}")
    
    # Test multiple steps
    print("\nTesting autoregressive generation...")
    current_emb = vertice_emb.clone()
    generated_vertices = []
    
    for i in range(3):
        with torch.no_grad():
            new_out, current_emb = model(audio_features, current_emb, one_hot, template)
            generated_vertices.append(new_out.clone())
            print(f"  Step {i+1}: generated shape {new_out.shape}")
    
    print("✅ Model test successful!")
    return model, (audio_features, vertice_emb, one_hot, template)

def export_minimal_model():
    """Export the minimal model to ONNX"""
    print("Exporting MinimalFaceFormer to ONNX...")
    
    # Test first
    model, test_inputs = test_model()
    
    # Export to ONNX
    output_path = "faceformer_minimal.onnx"
    
    torch.onnx.export(
        model,
        test_inputs,
        output_path,
        input_names=['audio_features', 'vertice_emb', 'one_hot', 'template'],
        output_names=['new_vertice_out', 'updated_vertice_emb'],
        dynamic_axes={
            'audio_features': {0: 'batch_size'},
            'vertice_emb': {0: 'batch_size'},
            'one_hot': {0: 'batch_size'},
            'template': {0: 'batch_size'},
            'new_vertice_out': {0: 'batch_size'},
            'updated_vertice_emb': {0: 'batch_size'}
        },
        opset_version=14,  # Use stable opset
        do_constant_folding=True,
        verbose=False,
        export_params=True
    )
    
    print(f"✅ Model exported to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Print model info
        print("\n📊 ONNX Model Info:")
        print(f"  IR Version: {onnx_model.ir_version}")
        print(f"  Opset Version: {onnx_model.opset_import[0].version}")
        print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except ImportError:
        print("⚠️ ONNX not available for verification, but export completed")
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
    
    # Save test data
    audio_features, vertice_emb, one_hot, template = test_inputs
    
    with torch.no_grad():
        new_out, updated_emb = model(*test_inputs)
    
    test_data = {
        "inputs": {
            "audio_features": audio_features.numpy().tolist(),
            "vertice_emb": vertice_emb.numpy().tolist(), 
            "one_hot": one_hot.numpy().tolist(),
            "template": template.numpy().tolist()
        },
        "outputs": {
            "new_vertice_out": new_out.numpy().tolist(),
            "updated_vertice_emb": updated_emb.numpy().tolist()
        },
        "shapes": {
            "audio_features": list(audio_features.shape),
            "vertice_emb": list(vertice_emb.shape),
            "one_hot": list(one_hot.shape),
            "template": list(template.shape),
            "new_vertice_out": list(new_out.shape),
            "updated_vertice_emb": list(updated_emb.shape)
        }
    }
    
    with open("faceformer_minimal_test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("✅ Test data saved to faceformer_minimal_test_data.json")
    
    return output_path

if __name__ == "__main__":
    export_minimal_model()
