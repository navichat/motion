#!/usr/bin/env python3
"""
Simple Model Test and Performance Analysis
Test the full transformer models we created
"""

import onnxruntime as ort
import numpy as np
import time
import json
import os
from pathlib import Path
from typing import Dict

def test_full_model(model_path: str, config: Dict) -> Dict:
    """Test the full transformer model"""
    
    print(f"🧠 Testing: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        # Create session
        session = ort.InferenceSession(model_path)
        
        print(f"  ✅ Model loaded successfully")
        print(f"  📊 Providers: {session.get_providers()}")
        
        # Get input/output info
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        print(f"  📥 Inputs: {input_names}")
        print(f"  📤 Outputs: {output_names}")
        
        # Test different sequence lengths
        seq_lengths = [25, 50, 100]
        results = {}
        
        for seq_len in seq_lengths:
            print(f"\n  🔬 Testing sequence length: {seq_len}")
            
            # Create test inputs
            batch_size = 1
            audio_features = np.random.randn(batch_size, seq_len, config['audio_input_dim']).astype(np.float32)
            template = np.random.randn(batch_size, config['vertice_dim']).astype(np.float32)
            subject_id = np.array([0], dtype=np.int64)
            
            inputs = {
                'audio_features': audio_features,
                'template': template,
                'subject_id': subject_id
            }
            
            # Warmup
            for _ in range(3):
                _ = session.run(None, inputs)
            
            # Benchmark
            num_runs = 5
            times = []
            
            for run in range(num_runs):
                start_time = time.time()
                outputs = session.run(None, inputs)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = seq_len / (avg_time / 1000)
            
            results[seq_len] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'fps': fps,
                'output_shape': [out.shape for out in outputs]
            }
            
            print(f"    ⏱️  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
            print(f"    🚀 Throughput: {fps:.1f} FPS")
            print(f"    📏 Output shape: {outputs[0].shape}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze_models():
    """Analyze all available models"""
    
    print("🎭 FaceFormer Full Transformer Analysis")
    print("=" * 60)
    
    models_dir = Path("./models")
    
    # Load configs and test models
    datasets = ['vocaset', 'biwi']
    all_results = {}
    
    for dataset in datasets:
        print(f"\n📋 Testing {dataset.upper()} model...")
        
        # Load config
        config_path = models_dir / f"faceformer_{dataset}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Test full model
            model_path = models_dir / f"faceformer_{dataset}_full.onnx"
            results = test_full_model(str(model_path), config)
            
            if results:
                all_results[dataset] = {
                    'config': config,
                    'results': results,
                    'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
                }
        else:
            print(f"❌ Config not found: {config_path}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Model comparison table
    print(f"\n📏 Model Specifications:")
    print(f"{'Dataset':<10} {'Size (MB)':<12} {'Parameters':<12} {'Vertices':<10} {'Layers':<8} {'Heads':<6}")
    print("-" * 70)
    
    for dataset, data in all_results.items():
        config = data['config']
        size_mb = data['model_size_mb']
        
        # Estimate parameters
        feature_dim = config['feature_dim']
        num_layers = config['num_layers']
        audio_dim = config['audio_input_dim']
        vertex_dim = config['vertice_dim']
        
        # Rough parameter count
        params_k = (
            num_layers * (4 * feature_dim * feature_dim) +  # Attention
            num_layers * (2 * feature_dim * feature_dim * 4) +  # FFN
            audio_dim * feature_dim + feature_dim * vertex_dim  # Projections
        ) // 1000
        
        print(f"{dataset.upper():<10} {size_mb:<12.1f} {params_k:<9}K {vertex_dim//1000:<7}K {num_layers:<8} {config['num_heads']:<6}")
    
    # Performance comparison
    seq_lengths = [25, 50, 100]
    for seq_len in seq_lengths:
        print(f"\n⏱️  Performance at {seq_len} frames:")
        print(f"{'Dataset':<10} {'Time (ms)':<12} {'FPS':<8} {'Efficiency':<12}")
        print("-" * 45)
        
        for dataset, data in all_results.items():
            if seq_len in data['results']:
                perf = data['results'][seq_len]
                avg_time = perf['avg_time_ms']
                fps = perf['fps']
                efficiency = fps / data['model_size_mb']  # FPS per MB
                
                print(f"{dataset.upper():<10} {avg_time:<12.2f} {fps:<8.1f} {efficiency:<12.2f}")
    
    # Architecture details
    print(f"\n🏗️  Architecture Details:")
    for dataset, data in all_results.items():
        config = data['config']
        print(f"\n  {dataset.upper()}:")
        print(f"    Feature dimension: {config['feature_dim']}")
        print(f"    Audio input dimension: {config['audio_input_dim']}")
        print(f"    Vertex dimension: {config['vertice_dim']}")
        print(f"    Transformer layers: {config['num_layers']}")
        print(f"    Attention heads: {config['num_heads']}")
        print(f"    Max sequence length: {config['max_seq_length']}")
        print(f"    Number of subjects: {config['num_subjects']}")
    
    # Recommendations
    print(f"\n🎯 Implementation Recommendations:")
    print("  🌐 Web Deployment:")
    print("     → Use ONNX Runtime Web with WebGPU provider")
    print("     → VOCASET model is lighter for web applications")
    print("     → Target 25-50 frame sequences for real-time")
    
    print("\n  ⚡ Performance Optimization:")
    print("     → Use batch processing for multiple sequences")
    print("     → Cache transformer states for auto-regressive generation")
    print("     → Consider dynamic quantization for smaller models")
    
    print("\n  🎮 Backend Selection:")
    print("     → WebGPU: Best for full transformer models")
    print("     → WebNN: Hardware-accelerated AI operations")
    print("     → ONNX Runtime: Cross-platform optimization")
    print("     → WASM: Reliable CPU fallback")

def main():
    """Main function"""
    
    print("🚀 Full FaceFormer Transformer Analysis")
    print("Testing complete implementation with multi-head attention")
    print()
    
    # Change to correct directory
    os.chdir("/home/barberb/motion/dev/web_viewer/faceformer")
    
    analyze_models()
    
    print(f"\n✅ Analysis complete!")
    print(f"\n🌐 Next steps:")
    print(f"  1. Open full_faceformer_demo.html in Chrome/Edge with WebGPU")
    print(f"  2. Test real-time inference with audio input")
    print(f"  3. Compare performance across different backends")
    print(f"  4. Integrate with 3D face rendering for complete pipeline")

if __name__ == "__main__":
    main()
