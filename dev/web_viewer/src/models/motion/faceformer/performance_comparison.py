#!/usr/bin/env python3
"""
Performance Comparison: Full Transformer vs Simplified FaceFormer
Compare inference times and capabilities between different implementations
"""

import onnxruntime as ort
import numpy as np
import time
import json
import os
from pathlib import Path
from typing import Dict

def test_model_performance(onnx_path: str, config: Dict, num_runs: int = 10) -> Dict:
    """Test performance of an ONNX model"""
    
    print(f"📊 Testing model: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Model not found: {onnx_path}")
        return None
    
    try:
        # Create session with different providers
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"  Providers: {session.get_providers()}")
        print(f"  Input names: {session.input_meta}")
        print(f"  Output names: {session.output_meta}")
        
        # Prepare test data
        batch_size = 1
        seq_lengths = [25, 50, 100, 200]  # Test different sequence lengths
        
        results = {
            'model_path': onnx_path,
            'providers': session.get_providers(),
            'performance': {}
        }
        
        for seq_len in seq_lengths:
            print(f"  Testing sequence length: {seq_len}")
            
            # Create inputs based on model requirements
            if 'audio_features' in [inp.name for inp in session.input_meta]:
                # Full transformer model
                test_audio = np.random.randn(batch_size, seq_len, config['audio_input_dim']).astype(np.float32)
                test_template = np.random.randn(batch_size, config['vertice_dim']).astype(np.float32)
                test_subject = np.array([0], dtype=np.int64)
                
                inputs = {
                    'audio_features': test_audio,
                    'template': test_template,
                    'subject_id': test_subject
                }
            else:
                # Simplified model (adjust based on your simplified model inputs)
                inputs = {}
                for inp in session.input_meta:
                    if 'audio' in inp.name.lower():
                        inputs[inp.name] = np.random.randn(batch_size, seq_len, config['audio_input_dim']).astype(np.float32)
                    elif 'template' in inp.name.lower():
                        inputs[inp.name] = np.random.randn(batch_size, config['vertice_dim']).astype(np.float32)
                    elif 'subject' in inp.name.lower():
                        inputs[inp.name] = np.array([0], dtype=np.int64)
            
            # Warmup
            for _ in range(3):
                _ = session.run(None, inputs)
            
            # Benchmark
            times = []
            for run in range(num_runs):
                start_time = time.time()
                outputs = session.run(None, inputs)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # Calculate throughput
            frames_per_second = seq_len / (avg_time / 1000)
            
            results['performance'][seq_len] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'frames_per_second': frames_per_second,
                'output_shape': [out.shape for out in outputs]
            }
            
            print(f"    Avg: {avg_time:.2f}ms ± {std_time:.2f}ms")
            print(f"    Range: {min_time:.2f}ms - {max_time:.2f}ms")
            print(f"    Throughput: {frames_per_second:.1f} frames/sec")
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return None

def compare_models():
    """Compare performance between different model implementations"""
    
    print("🚀 FaceFormer Performance Comparison")
    print("=" * 60)
    
    models_dir = Path("./models")
    configs = {}
    results = {}
    
    # Load configurations
    for config_file in models_dir.glob("*_config.json"):
        dataset = config_file.stem.replace("faceformer_", "").replace("_config", "")
        with open(config_file, 'r') as f:
            configs[dataset] = json.load(f)
    
    # Test each model
    for config_name, config in configs.items():
        print(f"\n📋 Testing {config_name.upper()} models...")
        
        # Test full transformer model
        full_model_path = models_dir / f"faceformer_{config_name}_full.onnx"
        if full_model_path.exists():
            print(f"\n🧠 Full Transformer Model ({config_name.upper()})")
            results[f"{config_name}_full"] = test_model_performance(str(full_model_path), config)
        
        # Test simplified model if it exists
        simple_model_path = models_dir.parent / f"faceformer_{config_name}_simple.onnx"
        if simple_model_path.exists():
            print(f"\n⚡ Simplified Model ({config_name.upper()})")
            results[f"{config_name}_simple"] = test_model_performance(str(simple_model_path), config)
    
    # Generate comparison report
    generate_comparison_report(results, configs)

def generate_comparison_report(results: Dict, configs: Dict):
    """Generate a detailed comparison report"""
    
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON REPORT")
    print("=" * 80)
    
    # Model sizes
    print("\n📏 Model Sizes:")
    for model_name, result in results.items():
        if result:
            model_path = result['model_path']
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"  {model_name:20}: {size_mb:8.1f} MB")
    
    # Performance comparison by sequence length
    seq_lengths = [25, 50, 100, 200]
    
    for seq_len in seq_lengths:
        print(f"\n⏱️  Performance at {seq_len} frames:")
        print(f"{'Model':<25} {'Avg Time (ms)':<15} {'Throughput (FPS)':<20} {'Efficiency':<15}")
        print("-" * 75)
        
        for model_name, result in results.items():
            if result and seq_len in result['performance']:
                perf = result['performance'][seq_len]
                avg_time = perf['avg_time_ms']
                fps = perf['frames_per_second']
                
                # Calculate efficiency (frames per second per MB)
                model_size_mb = os.path.getsize(result['model_path']) / (1024 * 1024)
                efficiency = fps / model_size_mb
                
                print(f"{model_name:<25} {avg_time:<15.2f} {fps:<20.1f} {efficiency:<15.2f}")
    
    # Architecture comparison
    print(f"\n🏗️  Architecture Comparison:")
    print(f"{'Dataset':<12} {'Model Type':<15} {'Parameters':<12} {'Layers':<8} {'Heads':<8} {'Vertices':<10}")
    print("-" * 80)
    
    for config_name, config in configs.items():
        # Estimate parameters for full model
        feature_dim = config['feature_dim']
        num_layers = config['num_layers']
        num_heads = config['num_heads']
        audio_dim = config['audio_input_dim']
        vertex_dim = config['vertice_dim']
        
        # Rough parameter estimation
        attention_params = num_layers * (4 * feature_dim * feature_dim + feature_dim)  # Q, K, V, O
        ffn_params = num_layers * (2 * feature_dim * feature_dim * 4)  # FFN layers
        projection_params = audio_dim * feature_dim + feature_dim * vertex_dim
        
        total_params = attention_params + ffn_params + projection_params
        
        print(f"{config_name.upper():<12} {'Full':<15} {total_params//1000:>8}K {num_layers:<8} {num_heads:<8} {vertex_dim//1000:>8}K")
    
    # Memory and computational requirements
    print(f"\n💾 Memory & Compute Requirements:")
    print(f"{'Model':<25} {'Memory (MB)':<15} {'Compute (GFLOPS)':<18}")
    print("-" * 60)
    
    for model_name, result in results.items():
        if result:
            model_size_mb = os.path.getsize(result['model_path']) / (1024 * 1024)
            
            # Estimate compute requirements (very rough)
            if 'full' in model_name:
                # Full transformer has much higher compute
                compute_gflops = model_size_mb * 0.5  # Rough estimate
            else:
                # Simplified model
                compute_gflops = model_size_mb * 0.1
            
            print(f"{model_name:<25} {model_size_mb:<15.1f} {compute_gflops:<18.2f}")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    print("  💻 For CPU/Mobile deployment:")
    print("     → Use simplified models for best performance")
    print("     → VOCASET simplified model recommended for real-time")
    print()
    print("  🎮 For GPU/WebGPU deployment:")
    print("     → Full transformer models provide best quality")
    print("     → WebGPU can handle larger models efficiently")
    print()
    print("  🌐 For Web deployment:")
    print("     → Start with simplified models, upgrade to full if needed")
    print("     → Use ONNX Runtime Web with WebGPU provider")
    print()
    print("  ⚡ For real-time applications:")
    print("     → Target <50ms inference time per frame")
    print("     → Use sequence length 25-50 for best latency")

def main():
    """Main function"""
    
    print("🎭 FaceFormer Model Performance Analysis")
    print("Testing full transformer models vs simplified implementations")
    print()
    
    # Change to the correct directory
    os.chdir("/home/barberb/motion/dev/web_viewer/faceformer")
    
    compare_models()
    
    print("\n✅ Performance analysis complete!")
    print("\nNext steps:")
    print("  1. Open full_faceformer_demo.html in a modern browser")
    print("  2. Test with WebGPU, WebNN, or ONNX Runtime Web")
    print("  3. Compare inference times between backends")

if __name__ == "__main__":
    main()
