#!/usr/bin/env python3
"""
RSMT ONNX Model Validation Script

This script tests the converted ONNX models to ensure they work correctly
and produces a test dataset for JavaScript validation.
"""

import os
import sys
import numpy as np
import json
import argparse
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️  ONNXRuntime not available. Install with: pip install onnxruntime")
    ONNX_AVAILABLE = False

def test_deephase_onnx(onnx_path, test_data_path=None):
    """Test the DeepPhase ONNX model"""
    print(f"Testing DeepPhase ONNX model: {onnx_path}")
    
    if not ONNX_AVAILABLE:
        return False, "ONNXRuntime not available"
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [out.name for out in session.get_outputs()]
        
        print(f"  Input: {input_name} {input_shape}")
        print(f"  Outputs: {output_names}")
        
        # Create test data
        batch_size = 4
        input_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        test_input = np.random.randn(batch_size, input_dim).astype(np.float32)
        
        # Run inference
        outputs = session.run(output_names, {input_name: test_input})
        
        # Validate outputs
        reconstructed, phase_vectors = outputs
        
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Phase vectors shape: {phase_vectors.shape}")
        
        # Check reconstruction quality (should be reasonable)
        mse = np.mean((test_input - reconstructed) ** 2)
        print(f"  Reconstruction MSE: {mse:.6f}")
        
        # Save test data for JavaScript validation
        if test_data_path:
            test_data = {
                'input': test_input.tolist(),
                'expected_reconstruction': reconstructed.tolist(),
                'expected_phase_vectors': phase_vectors.tolist(),
                'metadata': {
                    'input_dim': int(input_dim),
                    'phase_dim': int(phase_vectors.shape[1]),
                    'reconstruction_mse': float(mse)
                }
            }
            
            with open(test_data_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"  Test data saved to: {test_data_path}")
        
        return True, f"DeepPhase test passed (MSE: {mse:.6f})"
        
    except Exception as e:
        return False, f"DeepPhase test failed: {e}"

def test_manifold_onnx(onnx_path, test_data_path=None):
    """Test the Manifold VAE ONNX model"""
    print(f"Testing Manifold VAE ONNX model: {onnx_path}")
    
    if not ONNX_AVAILABLE:
        return False, "ONNXRuntime not available"
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [out.name for out in session.get_outputs()]
        
        print(f"  Input: {input_name} {input_shape}")
        print(f"  Outputs: {output_names}")
        
        # Create test data (phase vectors from DeepPhase)
        batch_size = 4
        input_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        test_input = np.random.randn(batch_size, input_dim).astype(np.float32)
        
        # Run inference
        outputs = session.run(output_names, {input_name: test_input})
        
        # Validate outputs
        reconstructed, mu, logvar = outputs
        
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        
        # Check reconstruction quality
        mse = np.mean((test_input - reconstructed) ** 2)
        print(f"  Reconstruction MSE: {mse:.6f}")
        
        # Check VAE properties
        std = np.exp(0.5 * logvar)
        print(f"  Mean mu: {np.mean(mu):.6f}, std: {np.std(mu):.6f}")
        print(f"  Mean std: {np.mean(std):.6f}")
        
        # Save test data for JavaScript validation
        if test_data_path:
            test_data = {
                'input': test_input.tolist(),
                'expected_reconstruction': reconstructed.tolist(),
                'expected_mu': mu.tolist(),
                'expected_logvar': logvar.tolist(),
                'metadata': {
                    'phase_dim': int(input_dim),
                    'manifold_dim': int(mu.shape[1]),
                    'reconstruction_mse': float(mse)
                }
            }
            
            with open(test_data_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"  Test data saved to: {test_data_path}")
        
        return True, f"Manifold VAE test passed (MSE: {mse:.6f})"
        
    except Exception as e:
        return False, f"Manifold VAE test failed: {e}"

def test_sampler_onnx(onnx_path, test_data_path=None):
    """Test the Transition Sampler ONNX model"""
    print(f"Testing Transition Sampler ONNX model: {onnx_path}")
    
    if not ONNX_AVAILABLE:
        return False, "ONNXRuntime not available"
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"  Inputs: {[(inp.name, inp.shape) for inp in inputs]}")
        print(f"  Outputs: {[(out.name, out.shape) for out in outputs]}")
        
        # Create test data (manifold points)
        batch_size = 4
        manifold_dim = inputs[0].shape[1] if len(inputs[0].shape) > 1 else inputs[0].shape[0]
        
        source_manifold = np.random.randn(batch_size, manifold_dim).astype(np.float32)
        target_manifold = np.random.randn(batch_size, manifold_dim).astype(np.float32)
        
        # Run inference
        input_dict = {
            inputs[0].name: source_manifold,
            inputs[1].name: target_manifold
        }
        
        outputs_result = session.run([out.name for out in outputs], input_dict)
        transition_sequence = outputs_result[0]
        
        print(f"  Source shape: {source_manifold.shape}")
        print(f"  Target shape: {target_manifold.shape}")
        print(f"  Transition sequence shape: {transition_sequence.shape}")
        
        # Validate transition properties
        seq_length = transition_sequence.shape[1]
        print(f"  Sequence length: {seq_length}")
        
        # Check that transition starts and ends reasonably
        start_distances = np.linalg.norm(transition_sequence[:, 0] - source_manifold, axis=1)
        end_distances = np.linalg.norm(transition_sequence[:, -1] - target_manifold, axis=1)
        
        print(f"  Mean start distance: {np.mean(start_distances):.6f}")
        print(f"  Mean end distance: {np.mean(end_distances):.6f}")
        
        # Save test data for JavaScript validation
        if test_data_path:
            test_data = {
                'source_manifold': source_manifold.tolist(),
                'target_manifold': target_manifold.tolist(),
                'expected_transition': transition_sequence.tolist(),
                'metadata': {
                    'manifold_dim': int(manifold_dim),
                    'sequence_length': int(seq_length),
                    'mean_start_distance': float(np.mean(start_distances)),
                    'mean_end_distance': float(np.mean(end_distances))
                }
            }
            
            with open(test_data_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"  Test data saved to: {test_data_path}")
        
        return True, f"Transition Sampler test passed"
        
    except Exception as e:
        return False, f"Transition Sampler test failed: {e}"

def create_integration_test_data(output_path):
    """Create test data for complete RSMT pipeline integration"""
    print(f"Creating integration test data: {output_path}")
    
    # Create realistic skeleton pose data
    # This simulates a walking motion with 23 joints * 4 values (quaternion) = 92 values
    batch_size = 8
    input_dim = 92
    
    # Generate several pose sequences
    test_poses = []
    for i in range(batch_size):
        # Simulate walking cycle poses
        t = i / batch_size * 2 * np.pi
        pose = np.zeros(input_dim)
        
        # Simulate some joint rotations for walking
        # Hip rotation
        pose[0:4] = [np.cos(t/2), 0, np.sin(t/2), 0]  # Hip sway
        
        # Leg joints (simplified)
        pose[8:12] = [np.cos(t), 0, np.sin(t), 0]    # Left hip
        pose[12:16] = [np.cos(t + np.pi), 0, np.sin(t + np.pi), 0]  # Right hip
        
        # Add some noise for realism
        pose += np.random.normal(0, 0.1, input_dim)
        
        # Normalize quaternions (every 4 values)
        for j in range(0, input_dim, 4):
            q = pose[j:j+4]
            norm = np.linalg.norm(q)
            if norm > 0:
                pose[j:j+4] = q / norm
        
        test_poses.append(pose.tolist())
    
    # Create test data structure
    integration_data = {
        'test_poses': test_poses,
        'metadata': {
            'num_poses': batch_size,
            'pose_dimension': input_dim,
            'joint_count': input_dim // 4,
            'description': 'Simulated walking cycle poses for RSMT integration testing'
        },
        'expected_pipeline': {
            'description': 'Expected data flow through RSMT pipeline',
            'steps': [
                '1. Input skeleton poses (92D) → DeepPhase encoder → Phase vectors (32D)',
                '2. Phase vectors (32D) → Manifold VAE encoder → Manifold points (8D)',
                '3. Two manifold points → Transition Sampler → Transition sequence (16×8D)',
                '4. Transition sequence → Manifold VAE decoder → Phase sequence (16×32D)',
                '5. Phase sequence → DeepPhase decoder → Skeleton sequence (16×92D)'
            ]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(integration_data, f, indent=2)
    
    print(f"✅ Integration test data saved: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate RSMT ONNX models")
    
    parser.add_argument("--onnx-dir", type=str, default="./onnx_models",
                      help="Directory containing ONNX models")
    
    parser.add_argument("--test-data-dir", type=str, default="./test_data",
                      help="Directory to save test data for JavaScript validation")
    
    parser.add_argument("--skip-deephase", action="store_true",
                      help="Skip testing DeepPhase model")
    
    parser.add_argument("--skip-manifold", action="store_true",
                      help="Skip testing Manifold VAE model")
    
    parser.add_argument("--skip-sampler", action="store_true",
                      help="Skip testing Transition Sampler model")
    
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        print("❌ ONNXRuntime is required for model validation")
        print("Install with: pip install onnxruntime")
        return 1
    
    print("=== RSMT ONNX Model Validation ===")
    
    # Create test data directory
    os.makedirs(args.test_data_dir, exist_ok=True)
    
    results = []
    
    # Test DeepPhase model
    if not args.skip_deephase:
        deephase_path = os.path.join(args.onnx_dir, 'deephase.onnx')
        if os.path.exists(deephase_path):
            test_data_path = os.path.join(args.test_data_dir, 'deephase_test.json')
            success, message = test_deephase_onnx(deephase_path, test_data_path)
            results.append(('DeepPhase', success, message))
        else:
            results.append(('DeepPhase', False, f'Model not found: {deephase_path}'))
    
    # Test Manifold VAE model
    if not args.skip_manifold:
        manifold_path = os.path.join(args.onnx_dir, 'manifold_vae.onnx')
        if os.path.exists(manifold_path):
            test_data_path = os.path.join(args.test_data_dir, 'manifold_test.json')
            success, message = test_manifold_onnx(manifold_path, test_data_path)
            results.append(('Manifold VAE', success, message))
        else:
            results.append(('Manifold VAE', False, f'Model not found: {manifold_path}'))
    
    # Test Transition Sampler model
    if not args.skip_sampler:
        sampler_path = os.path.join(args.onnx_dir, 'transition_sampler.onnx')
        if os.path.exists(sampler_path):
            test_data_path = os.path.join(args.test_data_dir, 'sampler_test.json')
            success, message = test_sampler_onnx(sampler_path, test_data_path)
            results.append(('Transition Sampler', success, message))
        else:
            results.append(('Transition Sampler', False, f'Model not found: {sampler_path}'))
    
    # Create integration test data
    integration_path = os.path.join(args.test_data_dir, 'integration_test.json')
    if create_integration_test_data(integration_path):
        results.append(('Integration Test Data', True, 'Created successfully'))
    
    # Print results
    print(f"\n=== Validation Results ===")
    success_count = 0
    for model_name, success, message in results:
        status = "✅" if success else "❌"
        print(f"{status} {model_name}: {message}")
        if success:
            success_count += 1
    
    print(f"\nSuccessfully validated: {success_count}/{len(results)} components")
    
    if success_count == len(results):
        print("\n🎉 All ONNX models validated successfully!")
        print(f"\nTest data saved in: {args.test_data_dir}")
        print("\nNext steps:")
        print("1. Copy ONNX models and test data to your web application")
        print("2. Use the RSMTInference JavaScript class to load the models")
        print("3. Run the integration tests in your web browser")
    else:
        print("\n❌ Some validations failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
