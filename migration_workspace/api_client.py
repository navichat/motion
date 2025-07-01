#!/usr/bin/env python3
"""
Motion Inference API Client

Test client for the deployed Motion Inference API server.
Demonstrates how to interact with all available endpoints.
"""

import requests
import numpy as np
import json
import time
import sys
from typing import Dict, Any

class MotionAPIClient:
    """Client for Motion Inference API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def encode_motion_phase(self, motion_features: np.ndarray) -> Dict[str, Any]:
        """Encode motion features to phase coordinates."""
        data = {"motion_features": motion_features.tolist()}
        response = self.session.post(f"{self.base_url}/api/motion/phase", json=data)
        response.raise_for_status()
        return response.json()
    
    def extract_motion_style(self, motion_sequence: np.ndarray) -> Dict[str, Any]:
        """Extract style vectors from motion sequence."""
        data = {"motion_sequence": motion_sequence.tolist()}
        response = self.session.post(f"{self.base_url}/api/motion/style", json=data)
        response.raise_for_status()
        return response.json()
    
    def generate_actions(self, state: np.ndarray) -> Dict[str, Any]:
        """Generate character control actions."""
        data = {"state": state.tolist()}
        response = self.session.post(f"{self.base_url}/api/deepmimic/actions", json=data)
        response.raise_for_status()
        return response.json()
    
    def estimate_state_value(self, state: np.ndarray) -> Dict[str, Any]:
        """Estimate state value."""
        data = {"state": state.tolist()}
        response = self.session.post(f"{self.base_url}/api/deepmimic/value", json=data)
        response.raise_for_status()
        return response.json()
    
    def process_motion_pipeline(self, motion_features: np.ndarray, character_state: np.ndarray) -> Dict[str, Any]:
        """Process full motion pipeline."""
        data = {
            "motion_features": motion_features.tolist(),
            "character_state": character_state.tolist()
        }
        response = self.session.post(f"{self.base_url}/api/motion/pipeline", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_process_motions(self, motion_batch: np.ndarray, state_batch: np.ndarray) -> Dict[str, Any]:
        """Process batch of motions."""
        data = {
            "motion_batch": motion_batch.tolist(),
            "state_batch": state_batch.tolist()
        }
        response = self.session.post(f"{self.base_url}/api/motion/batch", json=data)
        response.raise_for_status()
        return response.json()
    
    def run_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """Run performance benchmark."""
        data = {"iterations": iterations}
        response = self.session.post(f"{self.base_url}/api/benchmark", json=data)
        response.raise_for_status()
        return response.json()

def test_api_comprehensive(base_url: str = "http://localhost:5000"):
    """Run comprehensive API tests."""
    print("🧪 Motion Inference API Client Tests")
    print("=" * 50)
    
    # Initialize client
    client = MotionAPIClient(base_url)
    
    try:
        # Test health check
        print("🔍 Testing health check...")
        health = client.health_check()
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  Uptime: {health.get('uptime_seconds', 0):.1f}s")
        
        if health.get('status') != 'healthy':
            print("❌ API is not healthy")
            return False
        
        # Test model info
        print("\n📋 Testing model information...")
        models = client.get_models()
        print(f"  Loaded models: {models.get('total_models', 0)}")
        for model in models.get('loaded_models', []):
            print(f"    ✓ {model}")
        
        # Test motion phase encoding
        print("\n🎯 Testing motion phase encoding...")
        motion_features = np.random.randn(132).astype(np.float32)
        phase_result = client.encode_motion_phase(motion_features)
        print(f"  Input shape: {phase_result['input_shape']}")
        print(f"  Output shape: {phase_result['output_shape']}")
        print(f"  Inference time: {phase_result['inference_time_ms']:.3f}ms")
        print(f"  Phase coordinates: [{phase_result['phase_coordinates'][0]:.3f}, {phase_result['phase_coordinates'][1]:.3f}]")
        
        # Test style extraction
        print("\n🎨 Testing style extraction...")
        motion_sequence = np.random.randn(60 * 73).astype(np.float32)
        style_result = client.extract_motion_style(motion_sequence)
        print(f"  Input shape: {style_result['input_shape']}")
        print(f"  Output shapes: mu{style_result['output_shapes']['mu']}, logvar{style_result['output_shapes']['logvar']}")
        print(f"  Inference time: {style_result['inference_time_ms']:.3f}ms")
        
        # Test action generation
        print("\n🎮 Testing action generation...")
        state = np.random.randn(197).astype(np.float32)
        action_result = client.generate_actions(state)
        print(f"  Input shape: {action_result['input_shape']}")
        print(f"  Output shape: {action_result['output_shape']}")
        print(f"  Inference time: {action_result['inference_time_ms']:.3f}ms")
        print(f"  First 3 actions: {action_result['actions'][:3]}")
        
        # Test state value estimation
        print("\n💰 Testing state value estimation...")
        value_result = client.estimate_state_value(state)
        print(f"  Input shape: {value_result['input_shape']}")
        print(f"  Output shape: {value_result['output_shape']}")
        print(f"  Inference time: {value_result['inference_time_ms']:.3f}ms")
        print(f"  State value: {value_result['state_value'][0]:.3f}")
        
        # Test full pipeline
        print("\n🚀 Testing full motion pipeline...")
        pipeline_result = client.process_motion_pipeline(motion_features, state)
        print(f"  Inference time: {pipeline_result['inference_time_ms']:.3f}ms")
        print(f"  Phase coordinates: {len(pipeline_result.get('phase_coordinates', []))} values")
        print(f"  Actions: {len(pipeline_result.get('actions', []))} values")
        print(f"  State value: {pipeline_result.get('state_value', ['N/A'])[0]}")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        batch_size = 5
        motion_batch = np.random.randn(batch_size, 132).astype(np.float32)
        state_batch = np.random.randn(batch_size, 197).astype(np.float32)
        
        batch_result = client.batch_process_motions(motion_batch, state_batch)
        print(f"  Batch size: {batch_result['batch_size']}")
        print(f"  Total time: {batch_result['total_inference_time_ms']:.3f}ms")
        print(f"  Average time per item: {batch_result['avg_inference_time_ms']:.3f}ms")
        
        # Test benchmark
        print("\n⚡ Running performance benchmark...")
        benchmark_result = client.run_benchmark(50)
        print(f"  Iterations: {benchmark_result['iterations']}")
        print(f"  Total benchmark time: {benchmark_result['total_benchmark_time_ms']:.1f}ms")
        if 'deephase_avg_time_ms' in benchmark_result:
            fps = 1000 / benchmark_result['deephase_avg_time_ms']
            print(f"  DeepPhase performance: {fps:.1f} FPS")
        
        print("\n✅ All API tests passed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API server at {base_url}")
        print("   Make sure the server is running")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function for API testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Motion Inference API Client')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='API server URL (default: http://localhost:5000)')
    parser.add_argument('--test', action='store_true',
                       help='Run comprehensive tests')
    
    args = parser.parse_args()
    
    if args.test:
        success = test_api_comprehensive(args.url)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        client = MotionAPIClient(args.url)
        
        print(f"🔗 Connected to Motion Inference API at {args.url}")
        print("Available methods:")
        print("  client.health_check()")
        print("  client.get_models()")
        print("  client.encode_motion_phase(motion_features)")
        print("  client.extract_motion_style(motion_sequence)")
        print("  client.generate_actions(state)")
        print("  client.estimate_state_value(state)")
        print("  client.process_motion_pipeline(motion_features, character_state)")
        print("  client.batch_process_motions(motion_batch, state_batch)")
        print("  client.run_benchmark(iterations)")
        print("\nExample:")
        print("  motion_features = np.random.randn(132)")
        print("  result = client.encode_motion_phase(motion_features)")
        
        # Make client available in interactive session
        import code
        code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()
