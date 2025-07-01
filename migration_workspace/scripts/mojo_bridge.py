#!/usr/bin/env python3
"""
Mojo-Python Bridge for Motion Inference

This script provides a Python interface to the Mojo motion inference engine,
allowing seamless integration with existing Python codebases while leveraging
Mojo's performance benefits.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

class MojoMotionBridge:
    """
    Bridge between Python and Mojo for motion inference.
    Provides high-level interface while maintaining performance.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the motion inference bridge.
        
        Args:
            models_dir: Directory containing ONNX models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models" / "onnx"
        
        self.models_dir = Path(models_dir)
        self.onnx_sessions = {}
        self.performance_stats = {}
        
        # Load ONNX models
        self._load_onnx_models()
        
        print("🔗 Mojo-Python bridge initialized successfully!")
    
    def _load_onnx_models(self):
        """Load all available ONNX models."""
        print("📦 Loading ONNX models...")
        
        model_files = {
            'deephase': 'deephase.onnx',
            'stylevae_encoder': 'stylevae_encoder.onnx',
            'stylevae_decoder': 'stylevae_decoder.onnx',
            'deepmimic_actor': 'deepmimic_actor.onnx',
            'deepmimic_critic': 'deepmimic_critic.onnx'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            
            if model_path.exists():
                try:
                    # Create ONNX Runtime session with optimizations
                    providers = ['CPUExecutionProvider']
                    if ort.get_available_providers():
                        if 'CUDAExecutionProvider' in ort.get_available_providers():
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=providers
                    )
                    
                    self.onnx_sessions[model_name] = session
                    print(f"  ✅ {model_name}: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ {model_name}: Failed to load - {e}")
            else:
                print(f"  ⚠️  {model_name}: File not found - {filename}")
        
        print(f"📊 Loaded {len(self.onnx_sessions)} models successfully")
    
    def encode_motion_phase(self, motion_features: np.ndarray) -> np.ndarray:
        """
        Encode motion data to 2D phase coordinates using DeepPhase.
        
        Args:
            motion_features: Motion feature vector [132] or batch [N, 132]
            
        Returns:
            phase_coordinates: 2D phase coordinates [2] or [N, 2]
        """
        if 'deephase' not in self.onnx_sessions:
            raise RuntimeError("DeepPhase model not loaded")
        
        # Ensure correct input shape
        if motion_features.ndim == 1:
            motion_features = motion_features.reshape(1, -1)
        
        # Validate input dimensions
        if motion_features.shape[1] != 132:
            raise ValueError(f"Expected 132 motion features, got {motion_features.shape[1]}")
        
        # Run inference
        start_time = time.time()
        
        session = self.onnx_sessions['deephase']
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: motion_features.astype(np.float32)})
        
        inference_time = time.time() - start_time
        self._update_performance_stats('deephase', inference_time)
        
        phase_coords = result[0]
        
        # Return single vector if single input was provided
        if phase_coords.shape[0] == 1:
            return phase_coords[0]
        
        return phase_coords
    
    def extract_motion_style(self, motion_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract style vectors from motion sequence using StyleVAE encoder.
        
        Args:
            motion_sequence: Motion sequence [60, 73] or [N, 60*73]
            
        Returns:
            (mu, logvar): Style distribution parameters
        """
        if 'stylevae_encoder' not in self.onnx_sessions:
            raise RuntimeError("StyleVAE encoder not loaded")
        
        # Flatten if necessary
        if motion_sequence.ndim == 2 and motion_sequence.shape[0] == 1:
            motion_sequence = motion_sequence.flatten().reshape(1, -1)
        elif motion_sequence.ndim == 2 and motion_sequence.shape == (60, 73):
            motion_sequence = motion_sequence.flatten().reshape(1, -1)
        elif motion_sequence.ndim == 1:
            motion_sequence = motion_sequence.reshape(1, -1)
        
        # Validate input dimensions
        expected_size = 60 * 73  # 4380
        if motion_sequence.shape[1] != expected_size:
            raise ValueError(f"Expected {expected_size} flattened features, got {motion_sequence.shape[1]}")
        
        # Run inference
        start_time = time.time()
        
        session = self.onnx_sessions['stylevae_encoder']
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: motion_sequence.astype(np.float32)})
        
        inference_time = time.time() - start_time
        self._update_performance_stats('stylevae_encoder', inference_time)
        
        mu, logvar = result[0], result[1]
        
        # Return single vectors if single input was provided
        if mu.shape[0] == 1:
            return mu[0], logvar[0]
        
        return mu, logvar
    
    def generate_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Generate character control actions using DeepMimic actor.
        
        Args:
            state: Character state vector [197] or batch [N, 197]
            
        Returns:
            actions: Control actions [36] or [N, 36]
        """
        if 'deepmimic_actor' not in self.onnx_sessions:
            raise RuntimeError("DeepMimic actor not loaded")
        
        # Ensure correct input shape
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Validate input dimensions
        if state.shape[1] != 197:
            raise ValueError(f"Expected 197 state features, got {state.shape[1]}")
        
        # Run inference
        start_time = time.time()
        
        session = self.onnx_sessions['deepmimic_actor']
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: state.astype(np.float32)})
        
        inference_time = time.time() - start_time
        self._update_performance_stats('deepmimic_actor', inference_time)
        
        actions = result[0]
        
        # Return single vector if single input was provided
        if actions.shape[0] == 1:
            return actions[0]
        
        return actions
    
    def estimate_state_value(self, state: np.ndarray) -> np.ndarray:
        """
        Estimate state value using DeepMimic critic.
        
        Args:
            state: Character state vector [197] or batch [N, 197]
            
        Returns:
            values: State values [1] or [N, 1]
        """
        if 'deepmimic_critic' not in self.onnx_sessions:
            raise RuntimeError("DeepMimic critic not loaded")
        
        # Ensure correct input shape
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Validate input dimensions
        if state.shape[1] != 197:
            raise ValueError(f"Expected 197 state features, got {state.shape[1]}")
        
        # Run inference
        start_time = time.time()
        
        session = self.onnx_sessions['deepmimic_critic']
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: state.astype(np.float32)})
        
        inference_time = time.time() - start_time
        self._update_performance_stats('deepmimic_critic', inference_time)
        
        values = result[0]
        
        # Return single value if single input was provided
        if values.shape[0] == 1:
            return values[0]
        
        return values
    
    def process_motion_pipeline(self, motion_features: np.ndarray, character_state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full motion processing pipeline combining multiple models.
        
        Args:
            motion_features: Motion feature vector [132]
            character_state: Character state vector [197]
            
        Returns:
            results: Dictionary containing all inference results
        """
        results = {}
        
        # Phase encoding
        if 'deephase' in self.onnx_sessions:
            results['phase_coordinates'] = self.encode_motion_phase(motion_features)
        
        # Action generation
        if 'deepmimic_actor' in self.onnx_sessions:
            results['actions'] = self.generate_actions(character_state)
        
        # Value estimation
        if 'deepmimic_critic' in self.onnx_sessions:
            results['state_value'] = self.estimate_state_value(character_state)
        
        return results
    
    def batch_process_motions(self, motion_batch: np.ndarray, state_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a batch of motions efficiently.
        
        Args:
            motion_batch: Batch of motion features [N, 132]
            state_batch: Batch of character states [N, 197]
            
        Returns:
            results: Dictionary containing batched inference results
        """
        batch_size = motion_batch.shape[0]
        
        if state_batch.shape[0] != batch_size:
            raise ValueError("Motion and state batch sizes must match")
        
        print(f"🔄 Processing batch of {batch_size} motions...")
        
        start_time = time.time()
        
        results = {}
        
        # Batch phase encoding
        if 'deephase' in self.onnx_sessions:
            results['phase_coordinates'] = self.encode_motion_phase(motion_batch)
        
        # Batch action generation
        if 'deepmimic_actor' in self.onnx_sessions:
            results['actions'] = self.generate_actions(state_batch)
        
        # Batch value estimation
        if 'deepmimic_critic' in self.onnx_sessions:
            results['state_values'] = self.estimate_state_value(state_batch)
        
        total_time = time.time() - start_time
        throughput = batch_size / total_time
        
        print(f"✅ Processed {batch_size} motions in {total_time:.3f}s ({throughput:.1f} FPS)")
        
        return results
    
    def _update_performance_stats(self, model_name: str, inference_time: float):
        """Update performance statistics for a model."""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                'total_calls': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = self.performance_stats[model_name]
        stats['total_calls'] += 1
        stats['total_time'] += inference_time
        stats['min_time'] = min(stats['min_time'], inference_time)
        stats['max_time'] = max(stats['max_time'], inference_time)
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Get detailed performance statistics."""
        report = {}
        
        for model_name, stats in self.performance_stats.items():
            if stats['total_calls'] > 0:
                avg_time = stats['total_time'] / stats['total_calls']
                report[model_name] = {
                    'total_calls': stats['total_calls'],
                    'average_time_ms': avg_time * 1000,
                    'min_time_ms': stats['min_time'] * 1000,
                    'max_time_ms': stats['max_time'] * 1000,
                    'estimated_fps': 1.0 / avg_time if avg_time > 0 else 0
                }
        
        return report
    
    def benchmark_inference(self, num_iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark inference performance across all models.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            benchmark_results: Performance metrics for each model
        """
        print(f"🏃 Running performance benchmark ({num_iterations} iterations)...")
        
        # Create test data
        motion_features = np.random.randn(132).astype(np.float32)
        character_state = np.random.randn(197).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            self.process_motion_pipeline(motion_features, character_state)
        
        # Reset stats
        self.performance_stats = {}
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            self.process_motion_pipeline(motion_features, character_state)
        
        total_time = time.time() - start_time
        
        # Generate report
        report = self.get_performance_report()
        report['overall'] = {
            'total_time_s': total_time,
            'iterations': num_iterations,
            'avg_pipeline_time_ms': (total_time / num_iterations) * 1000,
            'estimated_pipeline_fps': num_iterations / total_time
        }
        
        return report

def main():
    """Main function for testing the bridge."""
    print("🚀 Testing Mojo-Python Motion Bridge")
    print("=" * 50)
    
    try:
        # Initialize bridge
        bridge = MojoMotionBridge()
        
        # Create test data
        motion_features = np.random.randn(132).astype(np.float32)
        character_state = np.random.randn(197).astype(np.float32)
        
        print("\n🧪 Running single inference test...")
        
        # Test individual functions
        if 'deephase' in bridge.onnx_sessions:
            phase_coords = bridge.encode_motion_phase(motion_features)
            print(f"  Phase coordinates: {phase_coords.shape} - {phase_coords[:2]}")
        
        if 'deepmimic_actor' in bridge.onnx_sessions:
            actions = bridge.generate_actions(character_state)
            print(f"  Actions: {actions.shape} - range [{actions.min():.3f}, {actions.max():.3f}]")
        
        if 'deepmimic_critic' in bridge.onnx_sessions:
            value = bridge.estimate_state_value(character_state)
            print(f"  State value: {value}")
        
        # Test full pipeline
        print("\n🔄 Testing full motion pipeline...")
        results = bridge.process_motion_pipeline(motion_features, character_state)
        
        for key, value in results.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        batch_size = 32
        motion_batch = np.random.randn(batch_size, 132).astype(np.float32)
        state_batch = np.random.randn(batch_size, 197).astype(np.float32)
        
        batch_results = bridge.batch_process_motions(motion_batch, state_batch)
        
        # Run performance benchmark
        print("\n⚡ Running performance benchmark...")
        benchmark_results = bridge.benchmark_inference(100)
        
        print("\n📊 Performance Report:")
        for model_name, stats in benchmark_results.items():
            if model_name == 'overall':
                print(f"  Overall Pipeline: {stats['avg_pipeline_time_ms']:.2f} ms/inference ({stats['estimated_pipeline_fps']:.1f} FPS)")
            else:
                print(f"  {model_name}: {stats['average_time_ms']:.2f} ms/inference ({stats['estimated_fps']:.1f} FPS)")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
