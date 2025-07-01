#!/usr/bin/env python3
"""
PyTorch to ONNX Export Script

This script exports PyTorch models to ONNX format for conversion to MAX.
Starting with the DeepPhase model as it has the clearest architecture.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from pathlib import Path
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "RSMT-Realtime-Stylized-Motion-Transition"))

class DeepPhaseNetwork(nn.Module):
    """
    DeepPhase network implementation for ONNX export.
    Architecture: 132 -> 256 -> 128 -> 32 -> 2
    """
    
    def __init__(self):
        super(DeepPhaseNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

class StyleVAEEncoder(nn.Module):
    """
    StyleVAE Encoder network for style vector extraction.
    """
    
    def __init__(self, input_dim=60*73, latent_dim=256):  # Assuming 73 features per frame
        super(StyleVAEEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean and log variance
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        encoded = self.encoder(x)
        mu = encoded[:, :self.latent_dim]
        logvar = encoded[:, self.latent_dim:]
        
        return mu, logvar

class StyleVAEDecoder(nn.Module):
    """
    StyleVAE Decoder network for motion generation.
    """
    
    def __init__(self, latent_dim=256, output_dim=60*73):
        super(StyleVAEDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, z):
        decoded = self.decoder(z)
        return decoded.view(-1, 60, 73)  # Reshape to [batch, frames, features]

class DeepMimicActor(nn.Module):
    """
    DeepMimic Actor network for PPO policy.
    Architecture: state_size -> 1024 -> 512 -> action_size
    """
    
    def __init__(self, state_size=197, action_size=36):  # Typical humanoid sizes
        super(DeepMimicActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
            nn.Tanh()  # Action bounds
        )
    
    def forward(self, state):
        return self.network(state)

class DeepMimicCritic(nn.Module):
    """
    DeepMimic Critic network for value function.
    Architecture: state_size -> 1024 -> 512 -> 1
    """
    
    def __init__(self, state_size=197):
        super(DeepMimicCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class TransitionNetModel(nn.Module):
    """
    TransitionNet model for motion transition generation.
    Simplified architecture for better ONNX compatibility.
    """
    
    def __init__(self, motion_dim=132, style_dim=256):
        super(TransitionNetModel, self).__init__()
        
        # Input: source_motion + target_motion + style_vector
        input_dim = motion_dim + motion_dim + style_dim  # 132 + 132 + 256 = 520
        
        # Simplified attention mechanism using linear projections
        self.attention_q = nn.Linear(input_dim, input_dim)
        self.attention_k = nn.Linear(input_dim, input_dim)
        self.attention_v = nn.Linear(input_dim, input_dim)
        self.attention_out = nn.Linear(input_dim, input_dim)
        
        # MLP for transition generation
        self.transition_mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, motion_dim)  # Output motion
        )
    
    def forward(self, source_motion, target_motion, style_vector):
        # Concatenate inputs
        combined = torch.cat([source_motion, target_motion, style_vector], dim=-1)
        
        # Simplified self-attention
        q = self.attention_q(combined)
        k = self.attention_k(combined)
        v = self.attention_v(combined)
        
        # Attention weights (simplified for ONNX compatibility)
        attention_weights = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True), dim=-1)
        attended = v * attention_weights
        attended = self.attention_out(attended)
        
        # Generate transition motion
        transition = self.transition_mlp(attended)
        return transition

class ONNXExporter:
    """Handles ONNX export for all models."""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "models"
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = {}  # Initialize metadata dictionary
        
        # Create subdirectories
        (self.output_dir / "onnx").mkdir(exist_ok=True)
        (self.output_dir / "deephase").mkdir(exist_ok=True)
        (self.output_dir / "stylevae").mkdir(exist_ok=True)
        (self.output_dir / "deepmimic").mkdir(exist_ok=True)
        (self.output_dir / "transitionnet").mkdir(exist_ok=True)
    
    def export_deephase(self):
        """Export DeepPhase model to ONNX."""
        print("Exporting DeepPhase model to ONNX...")
        
        try:
            # Create model
            model = DeepPhaseNetwork()
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 132)
            
            # Export to ONNX
            onnx_path = self.output_dir / "onnx" / "deephase.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['motion_input'],
                output_names=['phase_output'],
                dynamic_axes={
                    'motion_input': {0: 'batch_size'},
                    'phase_output': {0: 'batch_size'}
                }
            )
            
            print(f"✓ DeepPhase exported to: {onnx_path}")
            
            # Test the exported model
            self._test_onnx_model(onnx_path, dummy_input)
            
        except Exception as e:
            print(f"✗ Error exporting DeepPhase: {e}")
    
    def export_stylevae(self):
        """Export StyleVAE encoder and decoder to ONNX."""
        print("Exporting StyleVAE models to ONNX...")
        
        try:
            # Export encoder
            encoder = StyleVAEEncoder()
            encoder.eval()
            
            encoder_input = torch.randn(1, 60 * 73)  # Flattened motion sequence
            encoder_path = self.output_dir / "onnx" / "stylevae_encoder.onnx"
            
            torch.onnx.export(
                encoder,
                encoder_input,
                str(encoder_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['motion_sequence'],
                output_names=['style_mean', 'style_logvar'],
                dynamic_axes={
                    'motion_sequence': {0: 'batch_size'},
                    'style_mean': {0: 'batch_size'},
                    'style_logvar': {0: 'batch_size'}
                }
            )
            
            print(f"✓ StyleVAE Encoder exported to: {encoder_path}")
            
            # Export decoder
            decoder = StyleVAEDecoder()
            decoder.eval()
            
            decoder_input = torch.randn(1, 256)  # Style vector
            decoder_path = self.output_dir / "onnx" / "stylevae_decoder.onnx"
            
            torch.onnx.export(
                decoder,
                decoder_input,
                str(decoder_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['style_vector'],
                output_names=['motion_output'],
                dynamic_axes={
                    'style_vector': {0: 'batch_size'},
                    'motion_output': {0: 'batch_size'}
                }
            )
            
            print(f"✓ StyleVAE Decoder exported to: {decoder_path}")
            
        except Exception as e:
            print(f"✗ Error exporting StyleVAE: {e}")
    
    def export_deepmimic(self):
        """Export DeepMimic actor and critic to ONNX."""
        print("Exporting DeepMimic models to ONNX...")
        
        try:
            # Export actor
            actor = DeepMimicActor()
            actor.eval()
            
            state_input = torch.randn(1, 197)  # State vector
            actor_path = self.output_dir / "onnx" / "deepmimic_actor.onnx"
            
            torch.onnx.export(
                actor,
                state_input,
                str(actor_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['state_input'],
                output_names=['action_output'],
                dynamic_axes={
                    'state_input': {0: 'batch_size'},
                    'action_output': {0: 'batch_size'}
                }
            )
            
            print(f"✓ DeepMimic Actor exported to: {actor_path}")
            
            # Export critic
            critic = DeepMimicCritic()
            critic.eval()
            
            critic_path = self.output_dir / "onnx" / "deepmimic_critic.onnx"
            
            torch.onnx.export(
                critic,
                state_input,
                str(critic_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['state_input'],
                output_names=['value_output'],
                dynamic_axes={
                    'state_input': {0: 'batch_size'},
                    'value_output': {0: 'batch_size'}
                }
            )
            
            print(f"✓ DeepMimic Critic exported to: {critic_path}")
            
        except Exception as e:
            print(f"✗ Error exporting DeepMimic: {e}")
    
    def export_transition_net(self):
        """Export TransitionNet model to ONNX format."""
        try:
            print("Exporting TransitionNet...")
            model = TransitionNetModel()
            model.eval()
            
            # Create example inputs
            batch_size = 1
            motion_dim = 132
            style_dim = 256
            
            source_motion = torch.randn(batch_size, motion_dim)
            target_motion = torch.randn(batch_size, motion_dim)
            style_vector = torch.randn(batch_size, style_dim)
            
            # Export to ONNX
            output_path = os.path.join(self.output_dir, "transition_net.onnx")
            torch.onnx.export(
                model,
                (source_motion, target_motion, style_vector),
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['source_motion', 'target_motion', 'style_vector'],
                output_names=['transition_motion'],
                dynamic_axes={
                    'source_motion': {0: 'batch_size'},
                    'target_motion': {0: 'batch_size'},
                    'style_vector': {0: 'batch_size'},
                    'transition_motion': {0: 'batch_size'}
                }
            )
            
            # Validate the exported model
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                is_valid = True
            except Exception as e:
                print(f"  ✗ ONNX model validation failed: {e}")
                is_valid = False
            
            if is_valid:
                metadata = {
                    'name': 'TransitionNet',
                    'type': 'motion_transition',
                    'project': 'RSMT',
                    'model_file': 'transition_net.onnx',
                    'input_shape': {
                        'source_motion': list(source_motion.shape),
                        'target_motion': list(target_motion.shape),
                        'style_vector': list(style_vector.shape)
                    },
                    'output_shape': list([batch_size, motion_dim]),
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'description': 'Neural network for generating smooth motion transitions'
                }
                
                self.metadata['transitionnet'] = metadata
                print(f"✓ TransitionNet exported successfully to {output_path}")
                return True
            else:
                print("✗ TransitionNet export validation failed")
                return False
                
        except Exception as e:
            print(f"✗ Error exporting TransitionNet: {e}")
            return False
    
    def _test_onnx_model(self, onnx_path, test_input):
        """Test ONNX model to ensure it works correctly."""
        try:
            import onnxruntime as ort
            
            # Create inference session
            session = ort.InferenceSession(str(onnx_path))
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            result = session.run(None, {input_name: test_input.numpy()})
            
            print(f"  ✓ ONNX model test passed - output shape: {result[0].shape}")
            
        except ImportError:
            print("  ⚠ ONNX Runtime not available for testing")
        except Exception as e:
            print(f"  ✗ ONNX model test failed: {e}")
    
    def export_all(self):
        """Export all models to ONNX format."""
        print("Starting ONNX export for all models...")
        print("="*60)
        
        # Export in priority order
        self.export_deephase()
        print()
        
        self.export_stylevae()
        print()
        
        self.export_deepmimic()
        print()
        
        self.export_transition_net()
        print()
        
        print("="*60)
        print("ONNX Export Summary:")
        print(f"Output directory: {self.output_dir / 'onnx'}")
        
        # List exported files
        onnx_files = list((self.output_dir / "onnx").glob("*.onnx"))
        for onnx_file in onnx_files:
            file_size = onnx_file.stat().st_size / 1024  # KB
            print(f"  • {onnx_file.name} ({file_size:.1f} KB)")
        
        print(f"\nTotal models exported: {len(onnx_files)}")
        
        # Save export metadata
        self._save_export_metadata(onnx_files)
    
    def _save_export_metadata(self, onnx_files):
        """Save metadata about exported models."""
        metadata = {
            "export_timestamp": str(torch.utils.data.get_worker_info()),
            "pytorch_version": torch.__version__,
            "onnx_opset_version": 11,
            "exported_models": {}
        }
        
        for onnx_file in onnx_files:
            model_name = onnx_file.stem
            metadata["exported_models"][model_name] = {
                "file_path": str(onnx_file),
                "file_size_kb": onnx_file.stat().st_size / 1024,
                "status": "exported"
            }
        
        metadata_path = self.output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Export metadata saved to: {metadata_path}")

def main():
    """Main export function."""
    exporter = ONNXExporter()
    
    try:
        exporter.export_all()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Install MAX: pip install modular")
        print("2. Convert ONNX to MAX: max convert models/onnx/*.onnx")
        print("3. Create Mojo wrappers for optimized inference")
        print("4. Implement performance benchmarking")
        
        return 0
        
    except Exception as e:
        print(f"Export failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
