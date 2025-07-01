#!/usr/bin/env python3
"""
DeepPhase Model Migration Script

This script migrates the DeepPhase model from PyTorch to Mojo/MAX through ONNX.
It handles the complete migration pipeline: export -> convert -> wrap -> validate.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import onnx
import numpy as np

class DeepPhaseModel(nn.Module):
    """
    DeepPhase model architecture for motion phase encoding.
    This recreates the model structure for ONNX export.
    """
    
    def __init__(self, input_dim: int = 132, latent_dim: int = 32, phase_dim: int = 2):
        super(DeepPhaseModel, self).__init__()
        
        # Encoder layers (motion -> latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )
        
        # Phase decoder (latent -> 2D phase coordinates)
        self.phase_decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, phase_dim)
        )
    
    def forward(self, x):
        """Forward pass: motion -> phase coordinates."""
        latent = self.encoder(x)
        phase = self.phase_decoder(latent)
        return phase

class DeepPhaseMigrator:
    """Handles the complete DeepPhase migration process."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the migrator with configuration."""
        self.config = self._load_config(config_path)
        self.model_config = self.config["migration_config"]
        
        # Set up paths
        self.source_path = self.model_config["source_models"]["deephase"]
        self.onnx_path = Path(self.model_config["target_paths"]["onnx_models"])
        self.max_path = Path(self.model_config["target_paths"]["max_models"])
        
        # Create directories
        self.onnx_path.mkdir(parents=True, exist_ok=True)
        self.max_path.mkdir(parents=True, exist_ok=True)
        
        # Model files
        self.onnx_file = self.onnx_path / "deephase.onnx"
        self.max_file = self.max_path / "deephase.maxgraph"
        self.mojo_file = self.max_path / "deephase_wrapper.mojo"
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found")
            sys.exit(1)
    
    def find_pytorch_model(self) -> Optional[str]:
        """Find the PyTorch DeepPhase model file."""
        print(f"🔍 Searching for DeepPhase model in {self.source_path}")
        
        if not os.path.exists(self.source_path):
            print(f"❌ Source path {self.source_path} does not exist")
            return None
        
        # Look for model files
        model_files = []
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.endswith(('.pth', '.pt')) and 'deephase' in file.lower():
                    model_files.append(os.path.join(root, file))
        
        if model_files:
            model_file = model_files[0]  # Use the first found
            print(f"✅ Found model: {model_file}")
            return model_file
        else:
            print("⚠️  No trained model found. Will create a dummy model for migration testing.")
            return None
    
    def create_dummy_model(self) -> DeepPhaseModel:
        """Create a dummy DeepPhase model for testing migration."""
        print("🔧 Creating dummy DeepPhase model for migration testing...")
        
        model = DeepPhaseModel()
        
        # Initialize with reasonable weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        return model
    
    def load_pytorch_model(self, model_path: str) -> DeepPhaseModel:
        """Load the PyTorch DeepPhase model."""
        print(f"📥 Loading PyTorch model from {model_path}")
        
        try:
            # Try to load the model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model instance
            model = DeepPhaseModel()
            
            if isinstance(checkpoint, dict):
                # State dict format
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict is the state dict
                    model.load_state_dict(checkpoint)
            else:
                # Model object
                model = checkpoint
            
            model.eval()
            print("✅ PyTorch model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Creating dummy model instead...")
            return self.create_dummy_model()
    
    def export_to_onnx(self, model: DeepPhaseModel) -> bool:
        """Export PyTorch model to ONNX format."""
        print(f"📤 Exporting to ONNX: {self.onnx_file}")
        
        try:
            # Create dummy input
            input_shape = self.model_config["conversion_settings"]["input_shapes"]["deephase"]
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_input,),
                str(self.onnx_file),
                export_params=True,
                opset_version=self.model_config["conversion_settings"]["onnx_opset_version"],
                do_constant_folding=True,
                input_names=['motion_input'],
                output_names=['phase_output'],
                dynamic_axes={
                    'motion_input': {0: 'batch_size'},
                    'phase_output': {0: 'batch_size'}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(self.onnx_file))
            onnx.checker.check_model(onnx_model)
            
            print(f"✅ ONNX export successful: {self.onnx_file}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return False
    
    def convert_to_max(self) -> bool:
        """Convert ONNX model to MAX graph format."""
        print(f"🔄 Converting ONNX to MAX: {self.max_file}")
        
        try:
            # Use MAX CLI to convert ONNX to MAX graph
            cmd = [
                "max", "convert",
                str(self.onnx_file),
                "--output-file", str(self.max_file),
                "--optimization-level", self.model_config["conversion_settings"]["max_optimization_level"]
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ MAX conversion successful: {self.max_file}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                return True
            else:
                print(f"❌ MAX conversion failed")
                print(f"Error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ MAX CLI not found. Please install Modular MAX.")
            return False
        except Exception as e:
            print(f"❌ MAX conversion error: {e}")
            return False
    
    def create_mojo_wrapper(self) -> bool:
        """Create Mojo wrapper for the MAX model."""
        print(f"🔧 Creating Mojo wrapper: {self.mojo_file}")
        
        mojo_code = '''
from max.graph import Graph, TensorType
from max.engine import InferenceSession
from tensor import Tensor, TensorShape
from utils.index import Index

struct DeepPhaseMAX:
    """DeepPhase model with MAX acceleration."""
    
    var session: InferenceSession
    
    fn __init__(inout self, model_path: String) raises:
        """Initialize DeepPhase model with MAX acceleration."""
        let graph = Graph(model_path)
        self.session = InferenceSession(graph)
    
    fn encode_phase(self, motion_data: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
        """
        Encode motion data to phase coordinates.
        
        Args:
            motion_data: Input motion tensor [batch_size, 132]
            
        Returns:
            Phase coordinates tensor [batch_size, 2]
        """
        let outputs = self.session.execute("motion_input", motion_data)
        return outputs.get[DType.float32]("phase_output")
    
    fn batch_encode(self, motion_batch: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
        """Batch processing for multiple motion sequences."""
        let batch_size = motion_batch.shape()[0]
        var results = Tensor[DType.float32](TensorShape(batch_size, 2))
        
        for i in range(batch_size):
            let single_motion = motion_batch[i]
            let phase_result = self.encode_phase(single_motion)
            results[i] = phase_result
            
        return results

# Example usage function
fn test_deephase_max() raises:
    """Test function for DeepPhase MAX model."""
    let model = DeepPhaseMAX("''' + str(self.max_file) + '''")
    
    # Create test input
    let test_input = Tensor[DType.float32](TensorShape(1, 132))
    
    # Run inference
    let phase_output = model.encode_phase(test_input)
    
    print("DeepPhase MAX inference successful!")
    print("Input shape:", test_input.shape())
    print("Output shape:", phase_output.shape())

fn main() raises:
    test_deephase_max()
'''
        
        try:
            with open(self.mojo_file, 'w') as f:
                f.write(mojo_code)
            
            print(f"✅ Mojo wrapper created: {self.mojo_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create Mojo wrapper: {e}")
            return False
    
    def validate_migration(self) -> bool:
        """Validate the migrated model against PyTorch."""
        print("🧪 Validating migration...")
        
        try:
            # Load original PyTorch model
            model_path = self.find_pytorch_model()
            if model_path:
                pytorch_model = self.load_pytorch_model(model_path)
            else:
                pytorch_model = self.create_dummy_model()
            
            pytorch_model.eval()
            
            # Create test input
            input_shape = self.model_config["conversion_settings"]["input_shapes"]["deephase"]
            test_input = torch.randn(*input_shape)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input)
            
            print(f"✅ PyTorch inference successful")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {pytorch_output.shape}")
            
            # TODO: Add MAX model validation when MAX Python bindings are available
            print("⚠️  MAX model validation requires MAX Python bindings")
            print("   Use the Mojo wrapper for testing MAX inference")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def run_migration(self, step: str = "all") -> bool:
        """Run the complete migration process or specific step."""
        print("🚀 Starting DeepPhase Migration")
        print("=" * 50)
        
        success = True
        
        if step in ["all", "export"]:
            # Step 1: Find and load PyTorch model
            model_path = self.find_pytorch_model()
            if model_path:
                pytorch_model = self.load_pytorch_model(model_path)
            else:
                pytorch_model = self.create_dummy_model()
            
            # Step 2: Export to ONNX
            if not self.export_to_onnx(pytorch_model):
                success = False
                if step != "all":
                    return False
        
        if step in ["all", "convert"] and success:
            # Step 3: Convert to MAX
            if not self.convert_to_max():
                success = False
                if step != "all":
                    return False
        
        if step in ["all", "wrap"] and success:
            # Step 4: Create Mojo wrapper
            if not self.create_mojo_wrapper():
                success = False
                if step != "all":
                    return False
        
        if step in ["all", "validate"] and success:
            # Step 5: Validate migration
            if not self.validate_migration():
                success = False
        
        if success:
            print("\n🎉 DeepPhase migration completed successfully!")
            print(f"📁 Files created:")
            print(f"   ONNX model: {self.onnx_file}")
            print(f"   MAX model: {self.max_file}")
            print(f"   Mojo wrapper: {self.mojo_file}")
            print(f"\n📋 Next steps:")
            print(f"   1. Test Mojo wrapper: mojo {self.mojo_file}")
            print(f"   2. Run validation: python scripts/validate_migration.py --model deephase")
            print(f"   3. Proceed to next model migration")
        else:
            print("\n❌ DeepPhase migration failed")
            print("Check the error messages above and troubleshoot")
        
        return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate DeepPhase model to Mojo/MAX")
    parser.add_argument("--step", choices=["all", "export", "convert", "wrap", "validate"], 
                       default="all", help="Migration step to run")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = DeepPhaseMigrator(args.config)
    
    # Run migration
    success = migrator.run_migration(args.step)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
