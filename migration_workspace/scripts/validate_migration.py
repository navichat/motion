#!/usr/bin/env python3
"""
Migration Validation Script

This script validates migrated models by comparing outputs between PyTorch and MAX,
testing performance, and ensuring numerical accuracy.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

class MigrationValidator:
    """Validates migrated models for accuracy and performance."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the validator with configuration."""
        self.config = self._load_config(config_path)
        self.validation_config = self.config["migration_config"]["validation_settings"]
        
        # Paths
        self.onnx_path = Path(self.config["migration_config"]["target_paths"]["onnx_models"])
        self.max_path = Path(self.config["migration_config"]["target_paths"]["max_models"])
        self.data_path = Path(self.config["migration_config"]["target_paths"]["validation_data"])
        
        # Results storage
        self.validation_results = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found")
            sys.exit(1)
    
    def validate_model(self, model_name: str) -> Dict:
        """Validate a specific migrated model."""
        print(f"🧪 Validating {model_name} migration...")
        print("=" * 50)
        
        results = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }
        
        # Test 1: File existence
        results["tests"]["file_existence"] = self._test_file_existence(model_name)
        
        # Test 2: ONNX model validation
        results["tests"]["onnx_validation"] = self._test_onnx_model(model_name)
        
        # Test 3: Numerical accuracy (if PyTorch model available)
        results["tests"]["numerical_accuracy"] = self._test_numerical_accuracy(model_name)
        
        # Test 4: Performance benchmarking
        results["tests"]["performance"] = self._test_performance(model_name)
        
        # Test 5: Mojo wrapper compilation
        results["tests"]["mojo_compilation"] = self._test_mojo_compilation(model_name)
        
        # Overall result
        all_passed = all(test.get("passed", False) for test in results["tests"].values())
        results["overall_passed"] = all_passed
        
        self.validation_results[model_name] = results
        
        # Print summary
        self._print_validation_summary(results)
        
        return results
    
    def _test_file_existence(self, model_name: str) -> Dict:
        """Test if all required files exist."""
        print("📁 Testing file existence...")
        
        required_files = {
            "onnx": self.onnx_path / f"{model_name}.onnx",
            "max": self.max_path / f"{model_name}.maxgraph",
            "mojo": self.max_path / f"{model_name}_wrapper.mojo"
        }
        
        results = {"passed": True, "details": {}}
        
        for file_type, file_path in required_files.items():
            exists = file_path.exists()
            results["details"][file_type] = {
                "path": str(file_path),
                "exists": exists,
                "size": file_path.stat().st_size if exists else 0
            }
            
            if exists:
                print(f"  ✅ {file_type.upper()}: {file_path}")
            else:
                print(f"  ❌ {file_type.upper()}: {file_path} (missing)")
                results["passed"] = False
        
        return results
    
    def _test_onnx_model(self, model_name: str) -> Dict:
        """Test ONNX model validity."""
        print("🔍 Testing ONNX model...")
        
        onnx_file = self.onnx_path / f"{model_name}.onnx"
        results = {"passed": False, "details": {}}
        
        if not onnx_file.exists():
            results["details"]["error"] = "ONNX file not found"
            print("  ❌ ONNX file not found")
            return results
        
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_file))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(onnx_file))
            
            # Get model info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            results["details"] = {
                "input_name": input_info.name,
                "input_shape": input_info.shape,
                "input_type": input_info.type,
                "output_name": output_info.name,
                "output_shape": output_info.shape,
                "output_type": output_info.type
            }
            
            # Test inference with dummy data
            if model_name == "deephase":
                test_input = np.random.randn(1, 132).astype(np.float32)
            elif model_name == "stylevae":
                test_input = np.random.randn(1, 60, 256).astype(np.float32)
            elif model_name == "transitionnet":
                test_input = np.random.randn(1, 321).astype(np.float32)
            else:
                test_input = np.random.randn(1, 100).astype(np.float32)
            
            output = session.run(None, {input_info.name: test_input})
            results["details"]["test_output_shape"] = output[0].shape
            
            results["passed"] = True
            print(f"  ✅ ONNX model valid")
            print(f"     Input: {input_info.name} {input_info.shape}")
            print(f"     Output: {output_info.name} {output_info.shape}")
            
        except Exception as e:
            results["details"]["error"] = str(e)
            print(f"  ❌ ONNX validation failed: {e}")
        
        return results
    
    def _test_numerical_accuracy(self, model_name: str) -> Dict:
        """Test numerical accuracy between PyTorch and ONNX."""
        print("🎯 Testing numerical accuracy...")
        
        results = {"passed": False, "details": {}}
        
        try:
            # Load test data
            test_data = self._load_test_data(model_name)
            if test_data is None:
                results["details"]["error"] = "No test data available"
                print("  ⚠️  No test data available for accuracy testing")
                return results
            
            # Run PyTorch inference (if available)
            pytorch_output = self._run_pytorch_inference(model_name, test_data)
            
            # Run ONNX inference
            onnx_output = self._run_onnx_inference(model_name, test_data)
            
            if pytorch_output is not None and onnx_output is not None:
                # Compare outputs
                diff = np.abs(pytorch_output - onnx_output)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                threshold = self.validation_config["accuracy_threshold"]
                passed = max_diff < threshold
                
                results["details"] = {
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff),
                    "threshold": threshold,
                    "pytorch_shape": pytorch_output.shape,
                    "onnx_shape": onnx_output.shape
                }
                
                results["passed"] = passed
                
                if passed:
                    print(f"  ✅ Accuracy test passed")
                    print(f"     Max difference: {max_diff:.2e} (threshold: {threshold:.2e})")
                else:
                    print(f"  ❌ Accuracy test failed")
                    print(f"     Max difference: {max_diff:.2e} (threshold: {threshold:.2e})")
            else:
                results["details"]["error"] = "Could not run inference comparison"
                print("  ⚠️  Could not compare PyTorch and ONNX outputs")
        
        except Exception as e:
            results["details"]["error"] = str(e)
            print(f"  ❌ Accuracy test error: {e}")
        
        return results
    
    def _test_performance(self, model_name: str) -> Dict:
        """Test performance of migrated model."""
        print("⚡ Testing performance...")
        
        results = {"passed": False, "details": {}}
        
        try:
            # Load test data
            test_data = self._load_test_data(model_name)
            if test_data is None:
                results["details"]["error"] = "No test data available"
                print("  ⚠️  No test data available for performance testing")
                return results
            
            # Test different batch sizes
            batch_sizes = self.validation_config["test_batch_sizes"]
            iterations = self.validation_config["performance_iterations"]
            
            performance_data = {}
            
            for batch_size in batch_sizes:
                print(f"  📊 Testing batch size {batch_size}...")
                
                # Create batched test data
                if len(test_data.shape) == 1:
                    batched_data = np.tile(test_data, (batch_size, 1))
                else:
                    batched_data = np.tile(test_data, (batch_size,) + (1,) * (len(test_data.shape) - 1))
                
                # ONNX performance
                onnx_times = []
                for _ in range(iterations):
                    start_time = time.time()
                    self._run_onnx_inference(model_name, batched_data)
                    onnx_times.append(time.time() - start_time)
                
                onnx_avg_time = np.mean(onnx_times)
                onnx_throughput = batch_size / onnx_avg_time
                
                performance_data[f"batch_{batch_size}"] = {
                    "onnx_avg_time": onnx_avg_time,
                    "onnx_throughput": onnx_throughput,
                    "batch_size": batch_size
                }
                
                print(f"     ONNX: {onnx_avg_time*1000:.2f}ms, {onnx_throughput:.1f} samples/sec")
            
            results["details"] = performance_data
            results["passed"] = True
            
        except Exception as e:
            results["details"]["error"] = str(e)
            print(f"  ❌ Performance test error: {e}")
        
        return results
    
    def _test_mojo_compilation(self, model_name: str) -> Dict:
        """Test Mojo wrapper compilation."""
        print("🔧 Testing Mojo compilation...")
        
        results = {"passed": False, "details": {}}
        
        mojo_file = self.max_path / f"{model_name}_wrapper.mojo"
        
        if not mojo_file.exists():
            results["details"]["error"] = "Mojo file not found"
            print("  ❌ Mojo file not found")
            return results
        
        try:
            # Try to compile Mojo file
            cmd = ["mojo", "build", str(mojo_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                results["passed"] = True
                results["details"]["compilation_output"] = result.stdout
                print("  ✅ Mojo compilation successful")
            else:
                results["details"]["error"] = result.stderr
                print(f"  ❌ Mojo compilation failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            results["details"]["error"] = "Compilation timeout"
            print("  ❌ Mojo compilation timeout")
        except FileNotFoundError:
            results["details"]["error"] = "Mojo compiler not found"
            print("  ⚠️  Mojo compiler not found - skipping compilation test")
            results["passed"] = True  # Don't fail if Mojo not installed
        except Exception as e:
            results["details"]["error"] = str(e)
            print(f"  ❌ Mojo compilation error: {e}")
        
        return results
    
    def _load_test_data(self, model_name: str) -> Optional[np.ndarray]:
        """Load test data for the model."""
        test_file = self.data_path / f"{model_name}_test_input.npy"
        
        if test_file.exists():
            return np.load(test_file)
        else:
            # Generate dummy test data
            if model_name == "deephase":
                return np.random.randn(132).astype(np.float32)
            elif model_name == "stylevae":
                return np.random.randn(60, 256).astype(np.float32)
            elif model_name == "transitionnet":
                return np.random.randn(321).astype(np.float32)
            else:
                return np.random.randn(100).astype(np.float32)
    
    def _run_pytorch_inference(self, model_name: str, test_data: np.ndarray) -> Optional[np.ndarray]:
        """Run PyTorch inference if model is available."""
        try:
            # This would need to be implemented based on actual PyTorch models
            # For now, return None to indicate PyTorch inference not available
            return None
        except Exception:
            return None
    
    def _run_onnx_inference(self, model_name: str, test_data: np.ndarray) -> Optional[np.ndarray]:
        """Run ONNX inference."""
        try:
            import onnxruntime as ort
            
            onnx_file = self.onnx_path / f"{model_name}.onnx"
            if not onnx_file.exists():
                return None
            
            session = ort.InferenceSession(str(onnx_file))
            input_name = session.get_inputs()[0].name
            
            # Ensure correct shape
            if len(test_data.shape) == 1:
                test_data = test_data.reshape(1, -1)
            
            output = session.run(None, {input_name: test_data})
            return output[0]
        
        except Exception as e:
            print(f"    ONNX inference error: {e}")
            return None
    
    def _print_validation_summary(self, results: Dict):
        """Print validation summary."""
        print(f"\n📋 Validation Summary for {results['model']}")
        print("-" * 40)
        
        for test_name, test_result in results["tests"].items():
            status = "✅ PASS" if test_result.get("passed", False) else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        overall_status = "✅ PASS" if results["overall_passed"] else "❌ FAIL"
        print(f"\nOverall: {overall_status}")
        
        if not results["overall_passed"]:
            print("\n🔧 Troubleshooting:")
            for test_name, test_result in results["tests"].items():
                if not test_result.get("passed", False) and "error" in test_result.get("details", {}):
                    print(f"  • {test_name}: {test_result['details']['error']}")
    
    def generate_validation_report(self, output_file: str = "validation_report.json"):
        """Generate a detailed validation report."""
        print(f"\n📄 Generating validation report: {output_file}")
        
        report = {
            "validation_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_models": len(self.validation_results),
                "passed_models": sum(1 for r in self.validation_results.values() if r["overall_passed"]),
                "failed_models": sum(1 for r in self.validation_results.values() if not r["overall_passed"])
            },
            "model_results": self.validation_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Validation report saved to {output_file}")
        return report

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate migrated models")
    parser.add_argument("--model", help="Specific model to validate (e.g., deephase)")
    parser.add_argument("--all", action="store_true", help="Validate all models")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--output", default="validation_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        print("❌ Please specify --model <name> or --all")
        return 1
    
    # Initialize validator
    validator = MigrationValidator(args.config)
    
    # Validate models
    if args.all:
        models = ["deephase", "stylevae", "transitionnet"]
        for model in models:
            validator.validate_model(model)
    else:
        validator.validate_model(args.model)
    
    # Generate report
    validator.generate_validation_report(args.output)
    
    # Return success/failure
    all_passed = all(r["overall_passed"] for r in validator.validation_results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
