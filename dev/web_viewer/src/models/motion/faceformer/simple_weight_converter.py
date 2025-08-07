#!/usr/bin/env python3
"""
Simple FaceFormer Weight Analyzer and Converter
Directly analyzes .pth weight files and converts to JSON/ONNX without loading the full model
"""

import torch
import numpy as np
import json
import os
import argparse
from collections import OrderedDict

def analyze_weight_file(weight_path, dataset):
    """Analyze the structure of a .pth weight file"""
    
    print(f"🔍 Analyzing {dataset} weights: {weight_path}")
    
    try:
        # Load weights
        weights = torch.load(weight_path, map_location='cpu')
        
        if not isinstance(weights, dict):
            print(f"❌ Expected dict, got {type(weights)}")
            return None
        
        print(f"✅ Loaded {len(weights)} weight tensors")
        
        # Analyze structure
        analysis = {
            "dataset": dataset,
            "total_tensors": len(weights),
            "tensor_info": {},
            "component_groups": {},
            "dimensions": {},
            "statistics": {}
        }
        
        # Group tensors by component
        component_groups = {}
        total_params = 0
        
        for name, tensor in weights.items():
            # Get tensor info
            shape = list(tensor.shape)
            num_params = tensor.numel()
            total_params += num_params
            
            analysis["tensor_info"][name] = {
                "shape": shape,
                "num_params": num_params,
                "dtype": str(tensor.dtype)
            }
            
            # Group by component
            component = name.split('.')[0] if '.' in name else name
            if component not in component_groups:
                component_groups[component] = []
            component_groups[component].append({
                "name": name,
                "shape": shape,
                "params": num_params
            })
        
        analysis["component_groups"] = component_groups
        analysis["statistics"]["total_parameters"] = total_params
        
        # Print analysis
        print(f"\n📊 Component Analysis:")
        for component, tensors in component_groups.items():
            component_params = sum(t["params"] for t in tensors)
            print(f"  {component}: {len(tensors)} tensors, {component_params:,} params")
            for tensor in tensors[:3]:  # Show first 3 tensors
                print(f"    - {tensor['name']}: {tensor['shape']}")
            if len(tensors) > 3:
                print(f"    ... and {len(tensors)-3} more")
        
        print(f"\n📈 Total parameters: {total_params:,}")
        
        return analysis, weights
        
    except Exception as e:
        print(f"❌ Failed to analyze weights: {e}")
        return None, None

def extract_core_components(weights, dataset):
    """Extract core trainable components (excluding Wav2Vec2)"""
    
    print(f"\n🔧 Extracting core components...")
    
    core_weights = OrderedDict()
    excluded_prefixes = [
        'audio_encoder.encoder.',  # Wav2Vec2 encoder layers
        'audio_encoder.feature_extractor.',  # Wav2Vec2 feature extractor
        'audio_encoder.feature_projection.',  # Wav2Vec2 feature projection
    ]
    
    core_components = [
        'audio_feature_map',  # Linear: 768 -> feature_dim
        'vertice_map',        # Linear: vertice_dim -> feature_dim
        'PPE',               # Positional encoding
        'transformer_decoder', # Transformer decoder
        'vertice_map_r',     # Linear: feature_dim -> vertice_dim
        'obj_vector',        # Style embedding
        'biased_mask'        # Temporal bias (if present)
    ]
    
    extracted_count = 0
    total_core_params = 0
    
    for name, tensor in weights.items():
        # Check if it's a core component
        is_core = any(name.startswith(comp) for comp in core_components)
        is_excluded = any(name.startswith(prefix) for prefix in excluded_prefixes)
        
        if is_core and not is_excluded:
            # Convert to list for JSON serialization
            core_weights[name] = tensor.detach().cpu().numpy().tolist()
            extracted_count += 1
            total_core_params += tensor.numel()
            print(f"  ✅ {name}: {list(tensor.shape)}")
    
    print(f"✅ Extracted {extracted_count} core tensors ({total_core_params:,} parameters)")
    return core_weights

def detect_model_config(weights, dataset):
    """Detect model configuration from weight shapes"""
    
    config = {
        "dataset": dataset,
        "architecture": "FaceFormer"
    }
    
    # Detect feature dimension
    if 'audio_feature_map.weight' in weights:
        audio_map_shape = weights['audio_feature_map.weight'].shape
        config['feature_dim'] = audio_map_shape[0]  # Output dimension
        config['audio_input_dim'] = audio_map_shape[1]  # Should be 768 for Wav2Vec2
    
    # Detect vertex dimension
    if 'vertice_map.weight' in weights:
        vertice_map_shape = weights['vertice_map.weight'].shape
        config['vertice_dim'] = vertice_map_shape[1]  # Input dimension
    
    # Detect number of subjects
    if 'obj_vector.weight' in weights:
        obj_vector_shape = weights['obj_vector.weight'].shape
        config['num_subjects'] = obj_vector_shape[1]  # Input dimension
    
    # Set dataset-specific defaults
    if dataset == "BIWI":
        config.update({
            'period': 25,
            'fps': 25,
            'expected_vertice_dim': 23370 * 3,
            'expected_feature_dim': 128
        })
    elif dataset == "vocaset":
        config.update({
            'period': 30,
            'fps': 30,
            'expected_vertice_dim': 5023 * 3,
            'expected_feature_dim': 64
        })
    
    return config

def create_web_compatible_weights(core_weights, config, output_path):
    """Create web-compatible weight file"""
    
    print(f"\n🌐 Creating web-compatible weights...")
    
    web_weights = {
        "model_info": {
            "architecture": "FaceFormer",
            "dataset": config["dataset"],
            "feature_dim": config.get("feature_dim", "unknown"),
            "vertice_dim": config.get("vertice_dim", "unknown"),
            "num_subjects": config.get("num_subjects", "unknown"),
            "audio_input_dim": config.get("audio_input_dim", 768),
            "export_timestamp": "2025-07-13",
            "converter_version": "2.0.0"
        },
        "weights": core_weights,
        "config": config
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(web_weights, f, indent=2)
    
    print(f"💾 Web weights saved: {output_path}")
    
    # Calculate file size
    file_size = os.path.getsize(output_path)
    print(f"📦 File size: {file_size / (1024*1024):.1f} MB")
    
    return True

def create_simple_onnx_spec(config, output_path):
    """Create a simple ONNX model specification"""
    
    print(f"\n📋 Creating ONNX model specification...")
    
    onnx_spec = {
        "model_type": "FaceFormer",
        "dataset": config["dataset"],
        "inputs": {
            "audio_features": {
                "shape": ["batch_size", "sequence_length", config.get("audio_input_dim", 768)],
                "type": "float32",
                "description": "Pre-computed Wav2Vec2 audio features"
            },
            "template": {
                "shape": ["batch_size", config.get("vertice_dim", "unknown")],
                "type": "float32", 
                "description": "Template face vertices"
            },
            "one_hot": {
                "shape": ["batch_size", config.get("num_subjects", "unknown")],
                "type": "float32",
                "description": "Subject identity one-hot encoding"
            }
        },
        "outputs": {
            "vertices": {
                "shape": ["batch_size", "sequence_length", config.get("vertice_dim", "unknown")],
                "type": "float32",
                "description": "Generated face vertices"
            }
        },
        "implementation_notes": [
            "This model requires pre-computed Wav2Vec2 features as input",
            "The audio preprocessing should use facebook/wav2vec2-base-960h",
            "Auto-regressive generation is required for proper temporal consistency",
            "Positional encoding period: " + str(config.get("period", "unknown")),
            "Subject encoding based on training subjects from original dataset"
        ]
    }
    
    spec_path = output_path.replace('.onnx', '_spec.json')
    with open(spec_path, 'w') as f:
        json.dump(onnx_spec, f, indent=2)
    
    print(f"📋 ONNX spec saved: {spec_path}")
    return True

def convert_weights(weight_path, dataset, output_dir):
    """Main conversion function"""
    
    print(f"🚀 Converting {dataset} FaceFormer weights...")
    print(f"📁 Input: {weight_path}")
    print(f"📂 Output: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Analyze weights
    analysis, weights = analyze_weight_file(weight_path, dataset)
    if analysis is None:
        return False
    
    # Extract core components
    core_weights = extract_core_components(weights, dataset)
    
    # Detect configuration
    config = detect_model_config(weights, dataset)
    
    # Create web-compatible weights
    web_weights_path = os.path.join(output_dir, f"faceformer_{dataset.lower()}_weights.json")
    web_success = create_web_compatible_weights(core_weights, config, web_weights_path)
    
    # Create ONNX specification
    onnx_spec_path = os.path.join(output_dir, f"faceformer_{dataset.lower()}.onnx")
    onnx_spec_success = create_simple_onnx_spec(config, onnx_spec_path)
    
    # Save analysis
    analysis_path = os.path.join(output_dir, f"faceformer_{dataset.lower()}_analysis.json") 
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n🎯 Conversion Summary for {dataset}:")
    print(f"  ✅ Analysis: {analysis_path}")
    print(f"  {'✅' if web_success else '❌'} Web weights: {web_weights_path}")
    print(f"  {'✅' if onnx_spec_success else '❌'} ONNX spec: {onnx_spec_path}")
    print(f"  📊 Core components: {len(core_weights)}")
    print(f"  🎯 Model config: {config}")
    
    return web_success and onnx_spec_success

def main():
    parser = argparse.ArgumentParser(description='Convert FaceFormer .pth weights to web format')
    parser.add_argument('--weights_dir', type=str, default='weights',
                       help='Directory containing weight files')
    parser.add_argument('--output_dir', type=str, default='converted_weights',
                       help='Output directory for converted files')
    parser.add_argument('--datasets', nargs='+', default=['biwi', 'vocaset'],
                       help='Datasets to convert')
    
    args = parser.parse_args()
    
    success_count = 0
    total_count = 0
    
    for dataset in args.datasets:
        total_count += 1
        
        weight_file = f"{dataset}.pth"
        weight_path = os.path.join(args.weights_dir, weight_file)
        
        if not os.path.exists(weight_path):
            print(f"❌ Weight file not found: {weight_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting {dataset.upper()} weights...")
        print(f"{'='*60}")
        
        success = convert_weights(weight_path, dataset.upper(), args.output_dir)
        if success:
            success_count += 1
    
    print(f"\n🏁 Conversion Complete: {success_count}/{total_count} successful")
    
    if success_count > 0:
        print(f"\n💡 Next Steps:")
        print(f"  1. Use the JSON weight files in your web implementation")
        print(f"  2. Implement Wav2Vec2 preprocessing for audio features")
        print(f"  3. Create auto-regressive generation loop")
        print(f"  4. Test with the extracted model configurations")

if __name__ == "__main__":
    main()
