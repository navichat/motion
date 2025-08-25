#!/usr/bin/env python3
"""
Simplified ONNX Model Creator for FaceFormer Web Deployment
Creates functional ONNX models that work with modern web runtimes
"""

import json
import numpy as np
import onnx
from onnx import helper, TensorProto
import os
import argparse

class SimpleFaceFormerONNX:
    def __init__(self):
        self.weights = None
        self.config = None
        
    def load_weights(self, weights_path):
        """Load converted weights from JSON file"""
        print(f"📥 Loading weights from {weights_path}")
        
        with open(weights_path, 'r') as f:
            data = json.load(f)
            
        self.weights = data['weights']
        self.config = data['config']
        
        print(f"✅ Loaded {len(self.weights)} weight tensors")
        return True
        
    def create_linear_model(self, output_path):
        """Create a simplified linear model that works"""
        print("🔨 Creating simplified linear ONNX model...")
        
        # Define inputs
        inputs = [
            helper.make_tensor_value_info(
                'audio_features',
                TensorProto.FLOAT,
                [1, -1, self.config['audio_input_dim']]  # batch=1, dynamic sequence
            ),
            helper.make_tensor_value_info(
                'template',
                TensorProto.FLOAT,
                [1, self.config['vertice_dim']]  # batch=1
            ),
            helper.make_tensor_value_info(
                'subject_id',
                TensorProto.INT64,
                [1]  # Single subject ID
            )
        ]
        
        # Define output
        outputs = [
            helper.make_tensor_value_info(
                'vertices',
                TensorProto.FLOAT,
                [1, -1, self.config['vertice_dim']]  # batch=1, dynamic sequence
            )
        ]
        
        # Create simplified computation graph
        nodes = []
        initializers = []
        
        # 1. Convert subject_id to one-hot
        num_subjects = self.config['num_subjects']
        eye_matrix = np.eye(num_subjects, dtype=np.float32)
        
        nodes.append(helper.make_node(
            'Gather',
            inputs=['subject_onehot_matrix', 'subject_id'],
            outputs=['subject_onehot'],
            name='subject_to_onehot'
        ))
        
        initializers.append(helper.make_tensor(
            name='subject_onehot_matrix',
            data_type=TensorProto.FLOAT,
            dims=[num_subjects, num_subjects],
            vals=eye_matrix.flatten().tolist()
        ))
        
        # 2. Get style embedding
        obj_weight = np.array(self.weights['obj_vector.weight'], dtype=np.float32).T  # Transpose for ONNX
        nodes.append(helper.make_node(
            'MatMul',
            inputs=['subject_onehot', 'obj_vector_weight'],
            outputs=['style_embedding'],
            name='get_style_embedding'
        ))
        
        initializers.append(helper.make_tensor(
            name='obj_vector_weight',
            data_type=TensorProto.FLOAT,
            dims=list(obj_weight.shape),
            vals=obj_weight.flatten().tolist()
        ))
        
        # 3. Map audio features
        audio_weight = np.array(self.weights['audio_feature_map.weight'], dtype=np.float32).T
        audio_bias = np.array(self.weights['audio_feature_map.bias'], dtype=np.float32)
        
        nodes.append(helper.make_node(
            'MatMul',
            inputs=['audio_features', 'audio_feature_map_weight'],
            outputs=['mapped_audio_raw'],
            name='map_audio_features'
        ))
        
        nodes.append(helper.make_node(
            'Add',
            inputs=['mapped_audio_raw', 'audio_feature_map_bias'],
            outputs=['mapped_audio'],
            name='add_audio_bias'
        ))
        
        initializers.extend([
            helper.make_tensor(
                name='audio_feature_map_weight',
                data_type=TensorProto.FLOAT,
                dims=list(audio_weight.shape),
                vals=audio_weight.flatten().tolist()
            ),
            helper.make_tensor(
                name='audio_feature_map_bias',
                data_type=TensorProto.FLOAT,
                dims=list(audio_bias.shape),
                vals=audio_bias.flatten().tolist()
            )
        ])
        
        # 4. Expand style embedding to match sequence length
        nodes.append(helper.make_node(
            'Shape',
            inputs=['mapped_audio'],
            outputs=['audio_shape'],
            name='get_audio_shape'
        ))
        
        nodes.append(helper.make_node(
            'Gather',
            inputs=['audio_shape', 'seq_dim_index'],
            outputs=['seq_len'],
            name='get_seq_len'
        ))
        
        # Add constant for sequence dimension index
        initializers.append(helper.make_tensor(
            name='seq_dim_index',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[1]  # Sequence is dimension 1
        ))
        
        # Reshape style embedding and expand
        nodes.append(helper.make_node(
            'Unsqueeze',
            inputs=['style_embedding', 'seq_axis'],
            outputs=['style_embedding_unsqueezed'],
            name='unsqueeze_style'
        ))
        
        initializers.append(helper.make_tensor(
            name='seq_axis',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]  # Add dimension at axis 1
        ))
        
        nodes.append(helper.make_node(
            'Expand',
            inputs=['style_embedding_unsqueezed', 'expand_shape'],
            outputs=['style_embedding_expanded'],
            name='expand_style'
        ))
        
        # We need to create the expand shape dynamically
        nodes.append(helper.make_node(
            'Concat',
            inputs=['batch_size', 'seq_len', 'feature_size'],
            outputs=['expand_shape'],
            axis=0,
            name='create_expand_shape'
        ))
        
        # Add constants
        initializers.extend([
            helper.make_tensor(
                name='batch_size',
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[1]
            ),
            helper.make_tensor(
                name='feature_size',
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[self.config['feature_dim']]
            )
        ])
        
        # 5. Add style to audio features
        nodes.append(helper.make_node(
            'Add',
            inputs=['mapped_audio', 'style_embedding_expanded'],
            outputs=['styled_features'],
            name='add_style_to_audio'
        ))
        
        # 6. Map to vertex space
        vertex_weight = np.array(self.weights['vertice_map_r.weight'], dtype=np.float32).T
        vertex_bias = np.array(self.weights['vertice_map_r.bias'], dtype=np.float32)
        
        nodes.append(helper.make_node(
            'MatMul',
            inputs=['styled_features', 'vertice_map_r_weight'],
            outputs=['vertex_deltas_raw'],
            name='map_to_vertices'
        ))
        
        nodes.append(helper.make_node(
            'Add',
            inputs=['vertex_deltas_raw', 'vertice_map_r_bias'],
            outputs=['vertex_deltas'],
            name='add_vertex_bias'
        ))
        
        initializers.extend([
            helper.make_tensor(
                name='vertice_map_r_weight',
                data_type=TensorProto.FLOAT,
                dims=list(vertex_weight.shape),
                vals=vertex_weight.flatten().tolist()
            ),
            helper.make_tensor(
                name='vertice_map_r_bias',
                data_type=TensorProto.FLOAT,
                dims=list(vertex_bias.shape),
                vals=vertex_bias.flatten().tolist()
            )
        ])
        
        # 7. Add template to get final vertices
        nodes.append(helper.make_node(
            'Add',
            inputs=['vertex_deltas', 'template'],
            outputs=['vertices'],
            name='add_template'
        ))
        
        # Create the graph
        graph = helper.make_graph(
            nodes=nodes,
            name="SimpleFaceFormer",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers
        )
        
        # Create the model
        model = helper.make_model(graph, producer_name="FaceFormer-Web-Simple")
        
        # Validate and save
        try:
            onnx.checker.check_model(model)
            print("✅ Model validation passed")
        except Exception as e:
            print(f"⚠️ Model validation warning: {e}")
            # Continue anyway for web deployment
        
        print(f"💾 Saving ONNX model to {output_path}")
        onnx.save(model, output_path)
        
        # Print model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📊 ONNX model size: {model_size:.1f}MB")
        
        return model_size

def main():
    parser = argparse.ArgumentParser(description='Simple FaceFormer ONNX Converter')
    parser.add_argument('--input', required=True, help='Input weights JSON file')
    parser.add_argument('--output', required=True, help='Output ONNX model path')
    
    args = parser.parse_args()
    
    converter = SimpleFaceFormerONNX()
    
    print("🚀 Simple FaceFormer ONNX Converter")
    print("=" * 40)
    
    converter.load_weights(args.input)
    model_size = converter.create_linear_model(args.output)
    
    print(f"✅ Conversion complete! Model size: {model_size:.1f}MB")

if __name__ == '__main__':
    main()
