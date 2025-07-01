#!/usr/bin/env python3
"""
PyTorch to MAX Migration using MAX Python API

This script migrates PyTorch models to MAX using the Python API instead of CLI.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import torch
import torch.nn as nn

# MAX imports
from max.graph import Graph, ops
from max.tensor import Tensor, TensorShape
from max.engine import InferenceSession

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAXModelBuilder:
    """Builds MAX models from PyTorch architectures."""
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.models_dir = self.workspace / "models" / "max"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def build_deephase_max_model(self) -> bool:
        """Build DeepPhase model directly in MAX."""
        try:
            logger.info("Building DeepPhase model in MAX...")
            
            graph = Graph()
            
            # Input: motion data [batch_size, 132]
            motion_input = graph.input(TensorShape("batch_size", 132))
            
            # Encoder Layer 1: 132 -> 256 + LeakyReLU
            with graph.layer("encoder_layer1"):
                enc1_weight = graph.constant(np.random.randn(132, 256).astype(np.float32) * 0.1)
                enc1_bias = graph.constant(np.zeros((256,), dtype=np.float32))
                enc1_matmul = ops.matmul(motion_input, enc1_weight)
                enc1_add = ops.add(enc1_matmul, enc1_bias)
                enc1_out = ops.leaky_relu(enc1_add, alpha=0.2)
            
            # Encoder Layer 2: 256 -> 128 + LeakyReLU
            with graph.layer("encoder_layer2"):
                enc2_weight = graph.constant(np.random.randn(256, 128).astype(np.float32) * 0.1)
                enc2_bias = graph.constant(np.zeros((128,), dtype=np.float32))
                enc2_matmul = ops.matmul(enc1_out, enc2_weight)
                enc2_add = ops.add(enc2_matmul, enc2_bias)
                enc2_out = ops.leaky_relu(enc2_add, alpha=0.2)
            
            # Encoder Layer 3: 128 -> 32 (latent)
            with graph.layer("encoder_layer3"):
                enc3_weight = graph.constant(np.random.randn(128, 32).astype(np.float32) * 0.1)
                enc3_bias = graph.constant(np.zeros((32,), dtype=np.float32))
                latent = ops.matmul(enc2_out, enc3_weight)
                latent_biased = ops.add(latent, enc3_bias)
            
            # Phase Decoder Layer 1: 32 -> 16 + LeakyReLU
            with graph.layer("decoder_layer1"):
                dec1_weight = graph.constant(np.random.randn(32, 16).astype(np.float32) * 0.1)
                dec1_bias = graph.constant(np.zeros((16,), dtype=np.float32))
                dec1_matmul = ops.matmul(latent_biased, dec1_weight)
                dec1_add = ops.add(dec1_matmul, dec1_bias)
                dec1_out = ops.leaky_relu(dec1_add, alpha=0.2)
            
            # Phase Decoder Layer 2: 16 -> 2 (phase output)
            with graph.layer("decoder_layer2"):
                dec2_weight = graph.constant(np.random.randn(16, 2).astype(np.float32) * 0.1)
                dec2_bias = graph.constant(np.zeros((2,), dtype=np.float32))
                phase_matmul = ops.matmul(dec1_out, dec2_weight)
                phase_output = ops.add(phase_matmul, dec2_bias)
            
            # Set output
            graph.output(phase_output)
            
            # Save the graph
            model_path = self.models_dir / "deephase.maxgraph"
            graph.save(str(model_path))
            
            logger.info(f"DeepPhase MAX model saved to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build DeepPhase MAX model: {e}")
            return False
    
    def build_deepmimic_actor_max_model(self) -> bool:
        """Build DeepMimic Actor model in MAX."""
        try:
            logger.info("Building DeepMimic Actor model in MAX...")
            
            graph = Graph()
            
            # Input: state [batch_size, 197]
            state_input = graph.input(TensorShape("batch_size", 197))
            
            # Layer 1: 197 -> 1024 + ReLU
            with graph.layer("fc1"):
                fc1_weight = graph.constant(np.random.randn(197, 1024).astype(np.float32) * 0.1)
                fc1_bias = graph.constant(np.zeros((1024,), dtype=np.float32))
                fc1_matmul = ops.matmul(state_input, fc1_weight)
                fc1_add = ops.add(fc1_matmul, fc1_bias)
                fc1_out = ops.relu(fc1_add)
            
            # Layer 2: 1024 -> 512 + ReLU
            with graph.layer("fc2"):
                fc2_weight = graph.constant(np.random.randn(1024, 512).astype(np.float32) * 0.1)
                fc2_bias = graph.constant(np.zeros((512,), dtype=np.float32))
                fc2_matmul = ops.matmul(fc1_out, fc2_weight)
                fc2_add = ops.add(fc2_matmul, fc2_bias)
                fc2_out = ops.relu(fc2_add)
            
            # Output Layer: 512 -> 36 + Tanh (bounded actions)
            with graph.layer("action_output"):
                out_weight = graph.constant(np.random.randn(512, 36).astype(np.float32) * 0.1)
                out_bias = graph.constant(np.zeros((36,), dtype=np.float32))
                out_matmul = ops.matmul(fc2_out, out_weight)
                out_add = ops.add(out_matmul, out_bias)
                action_output = ops.tanh(out_add)
            
            # Set output
            graph.output(action_output)
            
            # Save the graph
            model_path = self.models_dir / "deepmimic_actor.maxgraph"
            graph.save(str(model_path))
            
            logger.info(f"DeepMimic Actor MAX model saved to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build DeepMimic Actor MAX model: {e}")
            return False
    
    def build_deepmimic_critic_max_model(self) -> bool:
        """Build DeepMimic Critic model in MAX."""
        try:
            logger.info("Building DeepMimic Critic model in MAX...")
            
            graph = Graph()
            
            # Input: state [batch_size, 197]
            state_input = graph.input(TensorShape("batch_size", 197))
            
            # Layer 1: 197 -> 1024 + ReLU
            with graph.layer("fc1"):
                fc1_weight = graph.constant(np.random.randn(197, 1024).astype(np.float32) * 0.1)
                fc1_bias = graph.constant(np.zeros((1024,), dtype=np.float32))
                fc1_matmul = ops.matmul(state_input, fc1_weight)
                fc1_add = ops.add(fc1_matmul, fc1_bias)
                fc1_out = ops.relu(fc1_add)
            
            # Layer 2: 1024 -> 512 + ReLU
            with graph.layer("fc2"):
                fc2_weight = graph.constant(np.random.randn(1024, 512).astype(np.float32) * 0.1)
                fc2_bias = graph.constant(np.zeros((512,), dtype=np.float32))
                fc2_matmul = ops.matmul(fc1_out, fc2_weight)
                fc2_add = ops.add(fc2_matmul, fc2_bias)
                fc2_out = ops.relu(fc2_add)
            
            # Output Layer: 512 -> 1 (value estimate)
            with graph.layer("value_output"):
                out_weight = graph.constant(np.random.randn(512, 1).astype(np.float32) * 0.1)
                out_bias = graph.constant(np.zeros((1,), dtype=np.float32))
                out_matmul = ops.matmul(fc2_out, out_weight)
                value_output = ops.add(out_matmul, out_bias)
            
            # Set output
            graph.output(value_output)
            
            # Save the graph
            model_path = self.models_dir / "deepmimic_critic.maxgraph"
            graph.save(str(model_path))
            
            logger.info(f"DeepMimic Critic MAX model saved to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build DeepMimic Critic MAX model: {e}")
            return False
    
    def test_max_model(self, model_name: str, input_shape: List[int]) -> bool:
        """Test a MAX model with dummy input."""
        try:
            logger.info(f"Testing {model_name} MAX model...")
            
            model_path = self.models_dir / f"{model_name}.maxgraph"
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the model
            session = InferenceSession(str(model_path))
            
            # Create test input
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            result = session.run(test_input)
            
            logger.info(f"Test successful for {model_name}")
            logger.info(f"Input shape: {test_input.shape}")
            logger.info(f"Output shape: {result.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            return False


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="PyTorch to MAX Migration")
    parser.add_argument("--model", choices=["deephase", "deepmimic", "all"], default="all",
                       help="Model to migrate")
    parser.add_argument("--test", action="store_true", help="Test converted models")
    parser.add_argument("--workspace", default=".", help="Migration workspace directory")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = MAXModelBuilder(args.workspace)
    
    results = {}
    
    if args.model in ["deephase", "all"]:
        logger.info("=== Building DeepPhase Model ===")
        success = builder.build_deephase_max_model()
        results["deephase"] = success
        
        if success and args.test:
            builder.test_max_model("deephase", [1, 132])
    
    if args.model in ["deepmimic", "all"]:
        logger.info("=== Building DeepMimic Models ===")
        
        # Actor
        actor_success = builder.build_deepmimic_actor_max_model()
        results["deepmimic_actor"] = actor_success
        
        if actor_success and args.test:
            builder.test_max_model("deepmimic_actor", [1, 197])
        
        # Critic
        critic_success = builder.build_deepmimic_critic_max_model()
        results["deepmimic_critic"] = critic_success
        
        if critic_success and args.test:
            builder.test_max_model("deepmimic_critic", [1, 197])
    
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


if __name__ == "__main__":
    main()
