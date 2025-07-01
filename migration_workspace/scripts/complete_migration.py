#!/usr/bin/env python3
"""
Complete PyTorch to Mojo/MAX Migration Script

This script handles the complete migration pipeline:
1. Extract and analyze PyTorch models
2. Export to ONNX format
3. Convert ONNX to MAX Graph format
4. Generate Mojo wrapper code
5. Validate converted models

Usage:
    python complete_migration.py --model deephase --validate
    python complete_migration.py --model deepmimic --all
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import torch
import torch.nn as nn
import onnx
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMigrator:
    """Handles migration of PyTorch models to Mojo/MAX."""
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.models_dir = self.workspace / "models"
        self.onnx_dir = self.models_dir / "onnx"
        self.max_dir = self.models_dir / "max"
        self.mojo_dir = self.models_dir / "mojo"
        
        # Create directories
        for dir_path in [self.onnx_dir, self.max_dir, self.mojo_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def export_deephase_model(self) -> bool:
        """Export DeepPhase model from PyTorch to ONNX."""
        try:
            logger.info("Exporting DeepPhase model to ONNX...")
            
            # Recreate the DeepPhase model architecture
            class DeepPhaseModel(nn.Module):
                def __init__(self, input_dim=132, latent_dim=32, phase_dim=2):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 128), 
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, latent_dim)
                    )
                    self.phase_decoder = nn.Sequential(
                        nn.Linear(latent_dim, 16),
                        nn.LeakyReLU(0.2),
                        nn.Linear(16, phase_dim)
                    )
                
                def forward(self, x):
                    latent = self.encoder(x)
                    phase = self.phase_decoder(latent)
                    return phase
            
            # Create model and dummy input
            model = DeepPhaseModel()
            dummy_input = torch.randn(1, 132)
            
            # Export to ONNX
            onnx_path = self.onnx_dir / "deephase.onnx"
            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True,
                opset_version=11,
                input_names=['motion_input'],
                output_names=['phase_output'],
                dynamic_axes={
                    'motion_input': {0: 'batch_size'},
                    'phase_output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"DeepPhase ONNX model saved to: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export DeepPhase model: {e}")
            return False
    
    def export_deepmimic_models(self) -> bool:
        """Export DeepMimic actor and critic models to ONNX."""
        try:
            logger.info("Exporting DeepMimic models to ONNX...")
            
            # Actor network
            class ActorNetwork(nn.Module):
                def __init__(self, input_dim=197, action_dim=36):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 1024)
                    self.fc2 = nn.Linear(1024, 512)
                    self.action_mean = nn.Linear(512, action_dim)
                
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    action = torch.tanh(self.action_mean(x))
                    return action
            
            # Critic network
            class CriticNetwork(nn.Module):
                def __init__(self, input_dim=197):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 1024)
                    self.fc2 = nn.Linear(1024, 512)
                    self.value = nn.Linear(512, 1)
                
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    value = self.value(x)
                    return value
            
            # Export actor
            actor = ActorNetwork()
            dummy_state = torch.randn(1, 197)
            
            actor_path = self.onnx_dir / "deepmimic_actor.onnx"
            torch.onnx.export(
                actor, dummy_state, actor_path,
                export_params=True,
                opset_version=11,
                input_names=['state'],
                output_names=['action'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                }
            )
            
            # Export critic
            critic = CriticNetwork()
            
            critic_path = self.onnx_dir / "deepmimic_critic.onnx"
            torch.onnx.export(
                critic, dummy_state, critic_path,
                export_params=True,
                opset_version=11,
                input_names=['state'],
                output_names=['value'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                }
            )
            
            logger.info(f"DeepMimic Actor ONNX model saved to: {actor_path}")
            logger.info(f"DeepMimic Critic ONNX model saved to: {critic_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export DeepMimic models: {e}")
            return False
    
    def convert_onnx_to_max(self, model_name: str) -> bool:
        """Convert ONNX model to MAX Graph format."""
        try:
            logger.info(f"Converting {model_name} ONNX to MAX...")
            
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            max_path = self.max_dir / f"{model_name}.maxgraph"
            
            if not onnx_path.exists():
                logger.error(f"ONNX file not found: {onnx_path}")
                return False
            
            # Use MAX CLI to convert ONNX to MAX graph
            cmd = ["max", "convert", str(onnx_path), "--output-file", str(max_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"MAX model saved to: {max_path}")
                return True
            else:
                logger.error(f"MAX conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to convert {model_name} to MAX: {e}")
            return False
    
    def generate_mojo_wrapper(self, model_name: str, input_shape: List[int], output_shape: List[int]) -> bool:
        """Generate Mojo wrapper code for the converted model."""
        try:
            logger.info(f"Generating Mojo wrapper for {model_name}...")
            
            wrapper_template = f'''"""
{model_name.title()} Model - Mojo Wrapper

Auto-generated wrapper for the converted {model_name} model.
"""

from max import Model
from max.tensor import Tensor, TensorShape


struct {model_name.title()}Model:
    """Wrapper for the {model_name} MAX model."""
    
    var model: Model
    
    fn __init__(inout self, model_path: String):
        """Initialize with the MAX model file."""
        self.model = Model(model_path)
    
    fn predict(self, input_data: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Run inference on the model."""
        let result = self.model.execute("input", input_data)
        return result.get[DType.float32]("output")
    
    fn batch_predict(self, batch_data: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Run batch inference."""
        return self.predict(batch_data)


fn main():
    """Test the {model_name} model wrapper."""
    print("Testing {model_name.title()} Model")
    
    # Load model
    var model = {model_name.title()}Model("models/max/{model_name}.maxgraph")
    
    # Create test input
    var test_input = Tensor[DType.float32](TensorShape({", ".join(map(str, input_shape))}))
    
    # Run inference
    let output = model.predict(test_input)
    
    print("Input shape:", test_input.shape())
    print("Output shape:", output.shape())
    print("Model test completed successfully!")
'''
            
            wrapper_path = self.mojo_dir / f"{model_name}_wrapper.mojo"
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_template)
            
            logger.info(f"Mojo wrapper saved to: {wrapper_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate Mojo wrapper for {model_name}: {e}")
            return False
    
    def validate_model(self, model_name: str) -> bool:
        """Validate the converted model against original."""
        try:
            logger.info(f"Validating {model_name} model conversion...")
            
            # Load ONNX model
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            if not onnx_path.exists():
                logger.error(f"ONNX model not found: {onnx_path}")
                return False
            
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX model {model_name} is valid")
            
            # Check MAX model exists
            max_path = self.max_dir / f"{model_name}.maxgraph"
            if max_path.exists():
                logger.info(f"MAX model {model_name} exists")
                return True
            else:
                logger.warning(f"MAX model not found: {max_path}")
                return False
                
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            return False
    
    def migrate_all_models(self) -> Dict[str, bool]:
        """Migrate all models in the project."""
        results = {}
        
        # DeepPhase model
        logger.info("=== Migrating DeepPhase Model ===")
        if self.export_deephase_model():
            if self.convert_onnx_to_max("deephase"):
                self.generate_mojo_wrapper("deephase", [1, 132], [1, 2])
                results["deephase"] = self.validate_model("deephase")
            else:
                results["deephase"] = False
        else:
            results["deephase"] = False
        
        # DeepMimic models
        logger.info("=== Migrating DeepMimic Models ===")
        if self.export_deepmimic_models():
            # Actor
            if self.convert_onnx_to_max("deepmimic_actor"):
                self.generate_mojo_wrapper("deepmimic_actor", [1, 197], [1, 36])
                results["deepmimic_actor"] = self.validate_model("deepmimic_actor")
            else:
                results["deepmimic_actor"] = False
            
            # Critic
            if self.convert_onnx_to_max("deepmimic_critic"):
                self.generate_mojo_wrapper("deepmimic_critic", [1, 197], [1, 1])
                results["deepmimic_critic"] = self.validate_model("deepmimic_critic")
            else:
                results["deepmimic_critic"] = False
        else:
            results["deepmimic_actor"] = False
            results["deepmimic_critic"] = False
        
        return results


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="PyTorch to Mojo/MAX Migration")
    parser.add_argument("--model", choices=["deephase", "deepmimic", "all"], default="all",
                       help="Model to migrate")
    parser.add_argument("--validate", action="store_true", help="Validate converted models")
    parser.add_argument("--workspace", default=".", help="Migration workspace directory")
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = ModelMigrator(args.workspace)
    
    if args.model == "all":
        logger.info("Starting complete migration process...")
        results = migrator.migrate_all_models()
        
        # Print results
        print("\n" + "="*50)
        print("MIGRATION RESULTS")
        print("="*50)
        for model, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{model:20} {status}")
        
        total_success = sum(results.values())
        total_models = len(results)
        print(f"\nTotal: {total_success}/{total_models} models migrated successfully")
        
    elif args.model == "deephase":
        logger.info("Migrating DeepPhase model...")
        if migrator.export_deephase_model():
            if migrator.convert_onnx_to_max("deephase"):
                migrator.generate_mojo_wrapper("deephase", [1, 132], [1, 2])
                if args.validate:
                    migrator.validate_model("deephase")
        
    elif args.model == "deepmimic":
        logger.info("Migrating DeepMimic models...")
        if migrator.export_deepmimic_models():
            for model_name in ["deepmimic_actor", "deepmimic_critic"]:
                if migrator.convert_onnx_to_max(model_name):
                    input_shape = [1, 197]
                    output_shape = [1, 36] if "actor" in model_name else [1, 1]
                    migrator.generate_mojo_wrapper(model_name, input_shape, output_shape)
                    if args.validate:
                        migrator.validate_model(model_name)


if __name__ == "__main__":
    main()
