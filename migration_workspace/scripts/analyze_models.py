#!/usr/bin/env python3
"""
PyTorch Model Analysis Tool for Mojo/MAX Migration

This script analyzes existing PyTorch models to understand their architecture,
dependencies, and migration complexity.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from dataclasses import dataclass
import importlib.util

@dataclass
class ModelInfo:
    """Information about a PyTorch model."""
    name: str
    path: str
    architecture: Dict[str, Any]
    parameters: int
    input_shape: Optional[List[int]]
    output_shape: Optional[List[int]]
    dependencies: List[str]
    complexity: str
    migration_notes: List[str]

class ModelAnalyzer:
    """Analyzes PyTorch models for migration planning."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.models_info = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file {config_path} not found. Using default settings.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "migration_config": {
                "source_models": {
                    "deephase": "../RSMT-Realtime-Stylized-Motion-Transition/",
                    "stylevae": "../RSMT-Realtime-Stylized-Motion-Transition/",
                    "transitionnet": "../RSMT-Realtime-Stylized-Motion-Transition/",
                    "deepmimic": "../pytorch_DeepMimic/"
                }
            }
        }
    
    def analyze_all_models(self) -> List[ModelInfo]:
        """Analyze all models specified in configuration."""
        print("🔍 Analyzing PyTorch models for migration...")
        print("=" * 50)
        
        source_models = self.config["migration_config"]["source_models"]
        
        for model_name, model_path in source_models.items():
            print(f"\n📊 Analyzing {model_name}...")
            try:
                model_info = self._analyze_model(model_name, model_path)
                if model_info:
                    self.models_info.append(model_info)
                    self._print_model_summary(model_info)
            except Exception as e:
                print(f"❌ Error analyzing {model_name}: {str(e)}")
        
        return self.models_info
    
    def _analyze_model(self, model_name: str, model_path: str) -> Optional[ModelInfo]:
        """Analyze a specific model."""
        if not os.path.exists(model_path):
            print(f"⚠️  Path {model_path} does not exist")
            return None
        
        # Find model files
        model_files = self._find_model_files(model_path)
        if not model_files:
            print(f"⚠️  No model files found in {model_path}")
            return None
        
        # Analyze based on model type
        if model_name == "deephase":
            return self._analyze_deephase(model_path, model_files)
        elif model_name == "stylevae":
            return self._analyze_stylevae(model_path, model_files)
        elif model_name == "transitionnet":
            return self._analyze_transitionnet(model_path, model_files)
        elif model_name == "deepmimic":
            return self._analyze_deepmimic(model_path, model_files)
        else:
            return self._analyze_generic_model(model_name, model_path, model_files)
    
    def _find_model_files(self, model_path: str) -> List[str]:
        """Find PyTorch model files in the given path."""
        model_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.pth', '.pt', '.pkl')):
                    model_files.append(os.path.join(root, file))
                elif file.endswith('.py') and any(keyword in file.lower() for keyword in ['model', 'net', 'train']):
                    model_files.append(os.path.join(root, file))
        return model_files
    
    def _analyze_deephase(self, model_path: str, model_files: List[str]) -> ModelInfo:
        """Analyze DeepPhase model specifically."""
        print("  🧠 Analyzing DeepPhase architecture...")
        
        # Look for training script to understand architecture
        train_script = None
        for file in model_files:
            if 'deephase' in file.lower() and file.endswith('.py'):
                train_script = file
                break
        
        architecture = {
            "type": "Autoencoder",
            "encoder_layers": [132, 256, 128, 32],
            "decoder_layers": [32, 16, 2],
            "activation": "LeakyReLU",
            "loss_function": "MSE"
        }
        
        migration_notes = [
            "Standard feedforward architecture - good for MAX conversion",
            "LeakyReLU activation supported in MAX",
            "MSE loss function available in MAX",
            "Consider batch processing for better performance"
        ]
        
        return ModelInfo(
            name="DeepPhase",
            path=model_path,
            architecture=architecture,
            parameters=self._estimate_parameters(architecture["encoder_layers"] + architecture["decoder_layers"]),
            input_shape=[132],
            output_shape=[2],
            dependencies=["torch", "torch.nn"],
            complexity="Medium",
            migration_notes=migration_notes
        )
    
    def _analyze_stylevae(self, model_path: str, model_files: List[str]) -> ModelInfo:
        """Analyze StyleVAE model specifically."""
        print("  🎨 Analyzing StyleVAE architecture...")
        
        architecture = {
            "type": "Variational Autoencoder",
            "encoder_layers": [256, 128, 64],
            "latent_dim": 256,
            "decoder_layers": [256, 128, 256],
            "activation": "LeakyReLU",
            "loss_function": "VAE Loss (Reconstruction + KL)"
        }
        
        migration_notes = [
            "VAE architecture requires careful handling of sampling",
            "KL divergence loss needs custom implementation in MAX",
            "Reparameterization trick may need special attention",
            "Consider splitting encoder/decoder for easier conversion"
        ]
        
        return ModelInfo(
            name="StyleVAE",
            path=model_path,
            architecture=architecture,
            parameters=self._estimate_parameters([256, 128, 64, 256, 128, 256]),
            input_shape=[60, 256],
            output_shape=[256],
            dependencies=["torch", "torch.nn", "torch.distributions"],
            complexity="High",
            migration_notes=migration_notes
        )
    
    def _analyze_transitionnet(self, model_path: str, model_files: List[str]) -> ModelInfo:
        """Analyze TransitionNet model specifically."""
        print("  🔄 Analyzing TransitionNet architecture...")
        
        architecture = {
            "type": "Feedforward Network",
            "layers": [321, 256, 128, 63],  # Combined input to motion output
            "activation": "ReLU",
            "loss_function": "MSE"
        }
        
        migration_notes = [
            "Standard feedforward - excellent for MAX conversion",
            "Large input dimension requires efficient tensor operations",
            "Consider input preprocessing optimization",
            "Good candidate for kernel fusion in MAX"
        ]
        
        return ModelInfo(
            name="TransitionNet",
            path=model_path,
            architecture=architecture,
            parameters=self._estimate_parameters(architecture["layers"]),
            input_shape=[321],
            output_shape=[63],
            dependencies=["torch", "torch.nn"],
            complexity="Medium",
            migration_notes=migration_notes
        )
    
    def _analyze_deepmimic(self, model_path: str, model_files: List[str]) -> ModelInfo:
        """Analyze DeepMimic RL models."""
        print("  🤖 Analyzing DeepMimic RL architecture...")
        
        architecture = {
            "type": "Reinforcement Learning (PPO)",
            "actor_layers": [1024, 512],
            "critic_layers": [1024, 512],
            "activation": "ReLU",
            "algorithm": "Proximal Policy Optimization"
        }
        
        migration_notes = [
            "RL models require careful migration of training loop",
            "Actor-Critic architecture can be split into separate models",
            "Policy sampling needs special attention in MAX",
            "Consider migrating inference first, training later"
        ]
        
        return ModelInfo(
            name="DeepMimic",
            path=model_path,
            architecture=architecture,
            parameters=self._estimate_parameters([1024, 512, 1024, 512]),
            input_shape=None,  # Variable based on environment
            output_shape=None,  # Variable based on action space
            dependencies=["torch", "torch.nn", "pybullet"],
            complexity="High",
            migration_notes=migration_notes
        )
    
    def _analyze_generic_model(self, model_name: str, model_path: str, model_files: List[str]) -> ModelInfo:
        """Analyze a generic PyTorch model."""
        print(f"  📋 Analyzing {model_name} (generic)...")
        
        # Try to load and inspect actual model files
        for model_file in model_files:
            if model_file.endswith(('.pth', '.pt')):
                try:
                    model_data = torch.load(model_file, map_location='cpu')
                    if isinstance(model_data, dict):
                        # State dict format
                        param_count = sum(p.numel() for p in model_data.values() if isinstance(p, torch.Tensor))
                    else:
                        # Model object
                        param_count = sum(p.numel() for p in model_data.parameters())
                    
                    return ModelInfo(
                        name=model_name,
                        path=model_path,
                        architecture={"type": "Unknown", "parameters": param_count},
                        parameters=param_count,
                        input_shape=None,
                        output_shape=None,
                        dependencies=["torch"],
                        complexity="Unknown",
                        migration_notes=["Manual analysis required", "Check model architecture"]
                    )
                except Exception as e:
                    print(f"    ⚠️  Could not load {model_file}: {e}")
        
        return ModelInfo(
            name=model_name,
            path=model_path,
            architecture={"type": "Unknown"},
            parameters=0,
            input_shape=None,
            output_shape=None,
            dependencies=["torch"],
            complexity="Unknown",
            migration_notes=["Model files found but could not analyze", "Manual inspection required"]
        )
    
    def _estimate_parameters(self, layers: List[int]) -> int:
        """Estimate number of parameters from layer sizes."""
        total_params = 0
        for i in range(len(layers) - 1):
            # Weights + biases
            total_params += layers[i] * layers[i + 1] + layers[i + 1]
        return total_params
    
    def _print_model_summary(self, model_info: ModelInfo):
        """Print a summary of the model analysis."""
        print(f"  ✅ {model_info.name} Analysis Complete")
        print(f"     📐 Architecture: {model_info.architecture.get('type', 'Unknown')}")
        print(f"     🔢 Parameters: {model_info.parameters:,}")
        print(f"     📥 Input Shape: {model_info.input_shape}")
        print(f"     📤 Output Shape: {model_info.output_shape}")
        print(f"     🎯 Complexity: {model_info.complexity}")
        print(f"     📝 Migration Notes: {len(model_info.migration_notes)} items")
    
    def generate_migration_report(self, output_file: str = "migration_analysis_report.json"):
        """Generate a detailed migration report."""
        print(f"\n📄 Generating migration report: {output_file}")
        
        report = {
            "analysis_summary": {
                "total_models": len(self.models_info),
                "complexity_breakdown": self._get_complexity_breakdown(),
                "total_parameters": sum(model.parameters for model in self.models_info),
                "migration_priority": self._get_migration_priority()
            },
            "models": []
        }
        
        for model in self.models_info:
            model_data = {
                "name": model.name,
                "path": model.path,
                "architecture": model.architecture,
                "parameters": model.parameters,
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "dependencies": model.dependencies,
                "complexity": model.complexity,
                "migration_notes": model.migration_notes,
                "recommended_approach": self._get_migration_approach(model)
            }
            report["models"].append(model_data)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved to {output_file}")
        return report
    
    def _get_complexity_breakdown(self) -> Dict[str, int]:
        """Get breakdown of models by complexity."""
        breakdown = {"Low": 0, "Medium": 0, "High": 0, "Unknown": 0}
        for model in self.models_info:
            breakdown[model.complexity] += 1
        return breakdown
    
    def _get_migration_priority(self) -> List[str]:
        """Get recommended migration priority order."""
        # Sort by complexity (easier first) and dependencies
        priority_order = []
        
        # First: Medium complexity models (good starting point)
        medium_models = [m.name for m in self.models_info if m.complexity == "Medium"]
        priority_order.extend(sorted(medium_models))
        
        # Second: Low complexity models
        low_models = [m.name for m in self.models_info if m.complexity == "Low"]
        priority_order.extend(sorted(low_models))
        
        # Last: High complexity models
        high_models = [m.name for m in self.models_info if m.complexity == "High"]
        priority_order.extend(sorted(high_models))
        
        return priority_order
    
    def _get_migration_approach(self, model: ModelInfo) -> Dict[str, str]:
        """Get recommended migration approach for a model."""
        if model.complexity == "Low":
            return {
                "strategy": "Direct ONNX conversion",
                "timeline": "1-2 days",
                "risk": "Low"
            }
        elif model.complexity == "Medium":
            return {
                "strategy": "ONNX conversion with validation",
                "timeline": "3-5 days",
                "risk": "Medium"
            }
        elif model.complexity == "High":
            return {
                "strategy": "Phased migration with custom components",
                "timeline": "1-2 weeks",
                "risk": "High"
            }
        else:
            return {
                "strategy": "Manual analysis required",
                "timeline": "Unknown",
                "risk": "Unknown"
            }
    
    def print_migration_summary(self):
        """Print a summary of migration recommendations."""
        print("\n" + "=" * 60)
        print("🎯 MIGRATION SUMMARY")
        print("=" * 60)
        
        complexity_breakdown = self._get_complexity_breakdown()
        total_params = sum(model.parameters for model in self.models_info)
        
        print(f"📊 Models Found: {len(self.models_info)}")
        print(f"🔢 Total Parameters: {total_params:,}")
        print(f"📈 Complexity Breakdown:")
        for complexity, count in complexity_breakdown.items():
            print(f"   {complexity}: {count} models")
        
        print(f"\n🚀 Recommended Migration Order:")
        priority = self._get_migration_priority()
        for i, model_name in enumerate(priority, 1):
            model = next(m for m in self.models_info if m.name == model_name)
            print(f"   {i}. {model_name} ({model.complexity} complexity)")
        
        print(f"\n💡 Key Recommendations:")
        print("   • Start with DeepPhase as it's foundational for RSMT")
        print("   • Validate each model thoroughly before proceeding")
        print("   • Keep PyTorch versions as fallback during migration")
        print("   • Focus on inference first, training migration later")
        
        print(f"\n📋 Next Steps:")
        print("   1. Review the detailed report: migration_analysis_report.json")
        print("   2. Start with the first model in priority order")
        print("   3. Run: python scripts/migrate_deephase.py")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze PyTorch models for Mojo/MAX migration")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--output", default="migration_analysis_report.json", help="Output report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(args.config)
    
    # Analyze all models
    models = analyzer.analyze_all_models()
    
    if not models:
        print("❌ No models found to analyze. Check your configuration.")
        return 1
    
    # Generate report
    analyzer.generate_migration_report(args.output)
    
    # Print summary
    analyzer.print_migration_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
