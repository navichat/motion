#!/usr/bin/env python3
"""
RSMT PyTorch to ONNX Conversion Script

This script converts the trained RSMT PyTorch models to ONNX format
for use in the JavaScript inference engine.

The RSMT system consists of three models:
1. DeepPhase: Skeleton → Phase vectors (autoencoder)
2. Manifold VAE: Phase vectors → Latent manifold (variational autoencoder)
3. Transition Sampler: Manifold interpolation (sequence generator)
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add the repository root to the path
sys.path.append('.')
sys.path.append('../..')

def convert_deephase_to_onnx(pytorch_model_path, output_path, input_dim=92, hidden_dim=512, latent_dim=32):
    """Convert DeepPhase model to ONNX format"""
    print(f"Converting DeepPhase model: {pytorch_model_path} -> {output_path}")
    
    try:
        # Define the DeepPhase model architecture
        class DeepPhaseModel(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim=512, latent_dim=32):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.latent_dim = latent_dim
                
                # Encoder layers
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(hidden_dim, hidden_dim // 2),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(hidden_dim // 2, latent_dim)
                )
                
                # Decoder layers
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, hidden_dim // 2),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(hidden_dim // 2, hidden_dim),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Linear(hidden_dim, input_dim)
                )
                
            def forward(self, x):
                z = self.encoder(x)
                recon_x = self.decoder(z)
                return recon_x, z
        
        # Create model instance
        model = DeepPhaseModel(input_dim, hidden_dim, latent_dim)
        
        # Load trained weights
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_dim)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['skeleton_input'],
            output_names=['reconstructed_skeleton', 'phase_vectors'],
            dynamic_axes={
                'skeleton_input': {0: 'batch_size'},
                'reconstructed_skeleton': {0: 'batch_size'},
                'phase_vectors': {0: 'batch_size'}
            }
        )
        
        print(f"✅ DeepPhase model exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert DeepPhase model: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_manifold_to_onnx(pytorch_model_path, output_path, input_dim=32, hidden_dim=128, latent_dim=8):
    """Convert Manifold VAE model to ONNX format"""
    print(f"Converting Manifold VAE model: {pytorch_model_path} -> {output_path}")
    
    try:
        # Define the Manifold VAE model architecture
        class ManifoldVAE(torch.nn.Module):
            def __init__(self, input_dim=32, hidden_dim=128, latent_dim=8):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.latent_dim = latent_dim
                
                # Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                )
                
                # Use actual layer names from the trained model
                self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
                self.fc_var = torch.nn.Linear(hidden_dim, latent_dim)
                
                # Decoder (5 layers based on state_dict)
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, input_dim)
                )
            
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_var(h)
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon_x = self.decode(z)
                return recon_x, mu, logvar
        
        # Create model instance
        model = ManifoldVAE(input_dim, hidden_dim, latent_dim)
        
        # Load trained weights
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_dim)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['phase_input'],
            output_names=['reconstructed_phase', 'mu', 'logvar'],
            dynamic_axes={
                'phase_input': {0: 'batch_size'},
                'reconstructed_phase': {0: 'batch_size'},
                'mu': {0: 'batch_size'},
                'logvar': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Manifold VAE model exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert Manifold VAE model: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_sampler_to_onnx(pytorch_model_path, output_path, manifold_dim=8, hidden_dim=128, sequence_length=16):
    """Convert Transition Sampler model to ONNX format"""
    print(f"Converting Transition Sampler model: {pytorch_model_path} -> {output_path}")
    
    try:
        # Define the Transition Sampler model architecture  
        class TransitionSampler(torch.nn.Module):
            def __init__(self, manifold_dim=8, hidden_dim=128, sequence_length=16):
                super().__init__()
                self.manifold_dim = manifold_dim
                self.hidden_dim = hidden_dim
                self.sequence_length = sequence_length
                
                # LSTM encoder (2 layers) - input is 17D based on weight shape
                # Probably source (8D) + target (8D) + phase info (1D) = 17D
                self.lstm = torch.nn.LSTM(17, hidden_dim, 
                                         num_layers=2, batch_first=True)
                
                # Output layers - produces manifold_dim outputs (8D)
                self.output = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, manifold_dim)  # 8D output
                )
                
                # Refinement layers - takes 25D input (probably concatenated features)
                self.refine = torch.nn.Sequential(
                    torch.nn.Linear(25, hidden_dim),  # 25D input based on weight shape
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, manifold_dim)  # 8D output
                )
            
            def forward(self, source, target):
                batch_size = source.shape[0]
                
                # Create 17D input: source (8D) + target (8D) + phase (1D)
                # Adding a simple phase indicator
                phase = torch.zeros(batch_size, 1)
                combined = torch.cat([source, target, phase], dim=1)  # [batch, 17]
                
                # Add sequence dimension for LSTM
                input_seq = combined.unsqueeze(1)  # [batch, 1, 17]
                
                # LSTM processing
                lstm_out, (hidden, cell) = self.lstm(input_seq)
                
                # Use the last hidden state
                last_hidden = hidden[-1]  # [batch, hidden_dim]
                
                # Generate initial transition point
                transition_point = self.output(last_hidden)  # [batch, 8]
                
                # Create 25D input for refinement
                # Concatenate: source + target + transition_point = 8 + 8 + 9 = 25
                # Let's pad the transition_point to make it 9D
                transition_padded = torch.cat([transition_point, torch.zeros(batch_size, 1)], dim=1)
                refine_input = torch.cat([source, target, transition_padded], dim=1)  # [batch, 25]
                
                # Refine the transition
                refined_point = self.refine(refine_input)  # [batch, 8]
                
                # Create sequence by interpolating between source and refined point
                sequence = []
                for i in range(self.sequence_length):
                    alpha = i / (self.sequence_length - 1)
                    interpolated = (1 - alpha) * source + alpha * refined_point
                    sequence.append(interpolated)
                
                # Stack to create sequence
                return torch.stack(sequence, dim=1)  # [batch, sequence_length, manifold_dim]
        
        # Create model instance
        model = TransitionSampler(manifold_dim, hidden_dim, sequence_length)
        
        # Load trained weights
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create dummy inputs
        dummy_source = torch.randn(1, manifold_dim)
        dummy_target = torch.randn(1, manifold_dim)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_source, dummy_target),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['source_manifold', 'target_manifold'],
            output_names=['transition_sequence'],
            dynamic_axes={
                'source_manifold': {0: 'batch_size'},
                'target_manifold': {0: 'batch_size'},
                'transition_sequence': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Transition Sampler model exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert Transition Sampler model: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_all_models(args):
    """Convert all RSMT models to ONNX format"""
    print("=== RSMT PyTorch to ONNX Conversion ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Convert DeepPhase model
    if args.deephase_model and os.path.exists(args.deephase_model):
        total_count += 1
        deephase_output = os.path.join(args.output_dir, 'deephase.onnx')
        if convert_deephase_to_onnx(args.deephase_model, deephase_output, 
                                  args.input_dim, args.hidden_dim, args.latent_dim):
            success_count += 1
    else:
        print(f"⚠️  DeepPhase model not found: {args.deephase_model}")
    
    # Convert Manifold VAE model
    if args.manifold_model and os.path.exists(args.manifold_model):
        total_count += 1
        manifold_output = os.path.join(args.output_dir, 'manifold_vae.onnx')
        if convert_manifold_to_onnx(args.manifold_model, manifold_output,
                                  args.latent_dim, args.vae_hidden_dim, args.manifold_dim):
            success_count += 1
    else:
        print(f"⚠️  Manifold VAE model not found: {args.manifold_model}")
    
    # Convert Transition Sampler model
    if args.sampler_model and os.path.exists(args.sampler_model):
        total_count += 1
        sampler_output = os.path.join(args.output_dir, 'transition_sampler.onnx')
        if convert_sampler_to_onnx(args.sampler_model, sampler_output,
                                 args.manifold_dim, args.sampler_hidden_dim, args.sequence_length):
            success_count += 1
    else:
        print(f"⚠️  Transition Sampler model not found: {args.sampler_model}")
    
    # Summary
    print(f"\n=== Conversion Summary ===")
    print(f"Successfully converted: {success_count}/{total_count} models")
    print(f"Output directory: {args.output_dir}")
    
    if success_count > 0:
        print("\n📁 Generated ONNX files:")
        for file in os.listdir(args.output_dir):
            if file.endswith('.onnx'):
                filepath = os.path.join(args.output_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  • {file} ({size_mb:.2f} MB)")
    
    return success_count == total_count

def main():
    parser = argparse.ArgumentParser(description="Convert RSMT PyTorch models to ONNX")
    
    # Model paths
    parser.add_argument("--deephase-model", type=str,
                      default="../../output/deephase/run_20250509_140327/best_model.pt",
                      help="Path to trained DeepPhase PyTorch model")
    
    parser.add_argument("--manifold-model", type=str,
                      default="../../output/manifold/run_20250509_143611/best_model.pt",
                      help="Path to trained Manifold VAE PyTorch model")
    
    parser.add_argument("--sampler-model", type=str,
                      default="../../output/sampler/run_20250509_144132/best_model.pt",
                      help="Path to trained Transition Sampler PyTorch model")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="./onnx_models",
                      help="Directory to save ONNX models")
    
    # Model architecture parameters
    parser.add_argument("--input-dim", type=int, default=92,
                      help="Input dimension for DeepPhase model (skeleton pose)")
    
    parser.add_argument("--hidden-dim", type=int, default=512,
                      help="Hidden dimension for DeepPhase model")
    
    parser.add_argument("--latent-dim", type=int, default=32,
                      help="Latent dimension for DeepPhase model (phase vectors)")
    
    parser.add_argument("--vae-hidden-dim", type=int, default=128,
                      help="Hidden dimension for Manifold VAE model")
    
    parser.add_argument("--manifold-dim", type=int, default=8,
                      help="Manifold dimension for VAE latent space")
    
    parser.add_argument("--sampler-hidden-dim", type=int, default=128,
                      help="Hidden dimension for Transition Sampler model")
    
    parser.add_argument("--sequence-length", type=int, default=16,
                      help="Sequence length for Transition Sampler output")
    
    args = parser.parse_args()
    
    print("Starting RSMT model conversion with parameters:")
    print(f"  DeepPhase: {args.input_dim} → {args.latent_dim} (hidden: {args.hidden_dim})")
    print(f"  Manifold VAE: {args.latent_dim} → {args.manifold_dim} (hidden: {args.vae_hidden_dim})")
    print(f"  Sampler: {args.manifold_dim} → {args.sequence_length}×{args.manifold_dim} (hidden: {args.sampler_hidden_dim})")
    print()
    
    success = convert_all_models(args)
    
    if success:
        print("\n🎉 All models converted successfully!")
        print("\nNext steps:")
        print("1. Copy the ONNX files to your web application directory")
        print("2. Use the RSMTInference JavaScript class to load and run the models")
        print("3. Test the inference pipeline in your web browser")
    else:
        print("\n❌ Some models failed to convert. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
