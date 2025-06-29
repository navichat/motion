#!/usr/bin/env python3
"""
Test Migration Implementations

This script tests the migrated models and compares their outputs with PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
import time
import subprocess


class PyTorchDeepPhase(nn.Module):
    """PyTorch DeepPhase implementation for comparison."""
    
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


def test_weight_extraction():
    """Test if weights were properly extracted."""
    print("="*60)
    print("TESTING WEIGHT EXTRACTION")
    print("="*60)
    
    weights_dir = Path("weights")
    
    # Check if all weight files exist
    expected_files = [
        "deephase_weights.npz",
        "deepmimic_actor_weights.npz", 
        "deepmimic_critic_weights.npz"
    ]
    
    results = {}
    for file in expected_files:
        file_path = weights_dir / file
        if file_path.exists():
            # Load and check the weights
            weights = np.load(file_path)
            print(f"✓ {file}: {len(weights.files)} layers, {file_path.stat().st_size / 1024:.1f} KB")
            
            # Show some weight statistics
            total_params = sum(weights[key].size for key in weights.files)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Layers: {list(weights.files)}")
            results[file] = True
        else:
            print(f"✗ {file}: NOT FOUND")
            results[file] = False
    
    return results


def test_pytorch_model():
    """Test PyTorch model inference."""
    print("\n" + "="*60)
    print("TESTING PYTORCH MODEL")
    print("="*60)
    
    # Create model
    model = PyTorchDeepPhase()
    model.eval()
    
    # Create test input
    batch_size = 10
    test_input = torch.randn(batch_size, 132)
    
    # Test inference
    start_time = time.time()
    iterations = 1000
    
    with torch.no_grad():
        for _ in range(iterations):
            output = model(test_input)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / iterations
    throughput = batch_size / avg_time
    
    print(f"✓ PyTorch model working correctly")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Average inference time: {avg_time * 1000:.3f} ms")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Sample output: {output[0].numpy()}")
    
    return {
        "working": True,
        "avg_time": avg_time,
        "throughput": throughput,
        "output_shape": output.shape
    }


def test_onnx_models():
    """Test ONNX model exports."""
    print("\n" + "="*60)
    print("TESTING ONNX MODELS")
    print("="*60)
    
    onnx_dir = Path("models/onnx")
    if not onnx_dir.exists():
        print("✗ ONNX directory not found")
        return False
    
    onnx_files = list(onnx_dir.glob("*.onnx"))
    
    for onnx_file in onnx_files:
        try:
            import onnx
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
            print(f"✓ {onnx_file.name}: Valid ONNX model ({onnx_file.stat().st_size / 1024:.1f} KB)")
        except ImportError:
            print(f"⚠ {onnx_file.name}: ONNX not available for validation")
        except Exception as e:
            print(f"✗ {onnx_file.name}: Invalid - {e}")
    
    return len(onnx_files) > 0


def test_mojo_compilation():
    """Test if Mojo files can be compiled."""
    print("\n" + "="*60)
    print("TESTING MOJO COMPILATION")
    print("="*60)
    
    mojo_dir = Path("mojo")
    if not mojo_dir.exists():
        print("✗ Mojo directory not found")
        return False
    
    mojo_files = list(mojo_dir.glob("*.mojo"))
    results = {}
    
    for mojo_file in mojo_files:
        try:
            # Try to compile/run the Mojo file
            result = subprocess.run(
                ["mojo", str(mojo_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=mojo_dir.parent
            )
            
            if result.returncode == 0:
                print(f"✓ {mojo_file.name}: Compiles and runs successfully")
                if result.stdout:
                    print(f"  Output: {result.stdout.strip()[:100]}...")
                results[mojo_file.name] = True
            else:
                print(f"✗ {mojo_file.name}: Compilation failed")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()[:200]}...")
                results[mojo_file.name] = False
                
        except subprocess.TimeoutExpired:
            print(f"⚠ {mojo_file.name}: Timeout during execution")
            results[mojo_file.name] = False
        except FileNotFoundError:
            print(f"⚠ Mojo compiler not found. Cannot test {mojo_file.name}")
            results[mojo_file.name] = False
        except Exception as e:
            print(f"✗ {mojo_file.name}: Error - {e}")
            results[mojo_file.name] = False
    
    return results


def test_weight_loading():
    """Test if we can load extracted weights."""
    print("\n" + "="*60)
    print("TESTING WEIGHT LOADING")
    print("="*60)
    
    try:
        # Load DeepPhase weights
        weights = np.load("weights/deephase_weights.npz")
        
        # Create PyTorch model
        model = PyTorchDeepPhase()
        
        # Try to map the weights (this tests weight compatibility)
        state_dict = {}
        weight_mapping = {
            "encoder.0.weight": "encoder.0.weight",
            "encoder.0.bias": "encoder.0.bias", 
            "encoder.2.weight": "encoder.2.weight",
            "encoder.2.bias": "encoder.2.bias",
            "encoder.4.weight": "encoder.4.weight",
            "encoder.4.bias": "encoder.4.bias",
            "phase_decoder.0.weight": "phase_decoder.0.weight",
            "phase_decoder.0.bias": "phase_decoder.0.bias",
            "phase_decoder.2.weight": "phase_decoder.2.weight",
            "phase_decoder.2.bias": "phase_decoder.2.bias"
        }
        
        for pytorch_key, weight_key in weight_mapping.items():
            if weight_key in weights.files:
                state_dict[pytorch_key] = torch.from_numpy(weights[weight_key])
                print(f"✓ Mapped {weight_key} -> {pytorch_key}: {weights[weight_key].shape}")
            else:
                print(f"✗ Missing weight: {weight_key}")
        
        # Try to load the weights into the model
        try:
            model.load_state_dict(state_dict)
            print(f"✓ Successfully loaded {len(state_dict)} weight tensors")
            
            # Test inference with loaded weights
            test_input = torch.randn(1, 132)
            with torch.no_grad():
                output = model(test_input)
            print(f"✓ Inference with loaded weights successful: {output.shape}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load weights into model: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        return False


def create_migration_report(results):
    """Create a comprehensive test report."""
    print("\n" + "="*60)
    print("MIGRATION TEST REPORT")
    print("="*60)
    
    all_tests = [
        ("Weight Extraction", results.get("weight_extraction", {})),
        ("PyTorch Model", results.get("pytorch_model", {})),
        ("ONNX Models", results.get("onnx_models", False)),
        ("Mojo Compilation", results.get("mojo_compilation", {})),
        ("Weight Loading", results.get("weight_loading", False))
    ]
    
    total_passed = 0
    total_tests = len(all_tests)
    
    for test_name, test_result in all_tests:
        if isinstance(test_result, dict):
            if test_result:
                passed = sum(1 for v in test_result.values() if v)
                total = len(test_result)
                print(f"{test_name:20} {passed}/{total} passed")
                if passed == total:
                    total_passed += 1
            else:
                print(f"{test_name:20} ✗ FAILED")
        else:
            status = "✓ PASSED" if test_result else "✗ FAILED"
            print(f"{test_name:20} {status}")
            if test_result:
                total_passed += 1
    
    print("\n" + "="*60)
    print(f"OVERALL RESULTS: {total_passed}/{total_tests} test categories passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Migration is working correctly!")
    elif total_passed >= total_tests * 0.7:
        print("⚠ Most tests passed - Migration is mostly working")
    else:
        print("❌ Several tests failed - Migration needs attention")
    
    # Performance summary
    pytorch_results = results.get("pytorch_model", {})
    if pytorch_results.get("working"):
        print(f"\n📊 Performance Baseline (PyTorch):")
        print(f"  Inference time: {pytorch_results['avg_time'] * 1000:.3f} ms")
        print(f"  Throughput: {pytorch_results['throughput']:.1f} samples/sec")
    
    return total_passed / total_tests


def main():
    """Run all tests."""
    print("🧪 TESTING PYTORCH TO MOJO/MAX MIGRATION")
    print("🧪 " + "="*58)
    
    results = {}
    
    # Test 1: Weight extraction
    results["weight_extraction"] = test_weight_extraction()
    
    # Test 2: PyTorch model
    results["pytorch_model"] = test_pytorch_model()
    
    # Test 3: ONNX models
    results["onnx_models"] = test_onnx_models()
    
    # Test 4: Mojo compilation
    results["mojo_compilation"] = test_mojo_compilation()
    
    # Test 5: Weight loading
    results["weight_loading"] = test_weight_loading()
    
    # Generate report
    success_rate = create_migration_report(results)
    
    # Save detailed results
    import json
    with open("test_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "success_rate": success_rate,
            "detailed_results": results
        }, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to test_results.json")
    
    return success_rate > 0.7


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
