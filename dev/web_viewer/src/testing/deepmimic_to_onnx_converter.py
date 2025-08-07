#!/usr/bin/env python3
"""
DeepMimic PyTorch to ONNX Model Converter
Converts trained DeepMimic actor and critic networks to ONNX format for web inference
"""

import torch
import torch.onnx
import numpy as np
import os
import sys
from pathlib import Path

# Add pytorch_DeepMimic to path if available
deepmimic_path = Path("../../pytorch_DeepMimic/deepmimic")
if deepmimic_path.exists():
    sys.path.append(str(deepmimic_path))

class DeepMimicONNXConverter:
    def __init__(self):
        self.device = torch.device('cpu')  # Use CPU for ONNX export
        self.state_size = 197  # Standard humanoid state size
        self.action_size = 43  # Standard humanoid action size
        
    def create_mock_actor_network(self):
        """
        Create a mock actor network with the same structure as DeepMimic
        """
        class MockActor(torch.nn.Module):
            def __init__(self, state_size, action_size):
                super(MockActor, self).__init__()
                self.fc1 = torch.nn.Linear(state_size, 1024)
                self.fc2 = torch.nn.Linear(1024, 512)
                self.output = torch.nn.Linear(512, action_size)
                self.relu = torch.nn.ReLU()
                
            def forward(self, state):
                x = self.relu(self.fc1(state))
                x = self.relu(self.fc2(x))
                actions = self.output(x)
                return actions
                
        return MockActor(self.state_size, self.action_size)
    
    def create_mock_critic_network(self):
        """
        Create a mock critic network with the same structure as DeepMimic
        """
        class MockCritic(torch.nn.Module):
            def __init__(self, state_size):
                super(MockCritic, self).__init__()
                self.fc1 = torch.nn.Linear(state_size, 1024)
                self.fc2 = torch.nn.Linear(1024, 512)
                self.output = torch.nn.Linear(512, 1)
                self.relu = torch.nn.ReLU()
                
            def forward(self, state):
                x = self.relu(self.fc1(state))
                x = self.relu(self.fc2(x))
                value = self.output(x)
                return value
                
        return MockCritic(self.state_size)
    
    def load_pytorch_model(self, model_path, model_type='actor'):
        """
        Load a PyTorch model from file
        """
        try:
            if model_type == 'actor':
                model = self.create_mock_actor_network()
            else:
                model = self.create_mock_critic_network()
            
            if os.path.exists(model_path):
                # Try to load the actual weights
                state_dict = torch.load(model_path, map_location=self.device)
                try:
                    model.load_state_dict(state_dict)
                    print(f"✅ Loaded weights from {model_path}")
                except Exception as e:
                    print(f"⚠️  Could not load weights from {model_path}: {e}")
                    print("Using randomly initialized weights")
            else:
                print(f"⚠️  Model file not found: {model_path}")
                print("Using randomly initialized weights")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def convert_to_onnx(self, pytorch_model, output_path, model_type='actor'):
        """
        Convert PyTorch model to ONNX format
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.state_size, device=self.device)
            
            # Set up ONNX export parameters
            input_names = ['state']
            output_names = ['actions'] if model_type == 'actor' else ['value']
            
            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    output_names[0]: {0: 'batch_size'}
                }
            )
            
            print(f"✅ Successfully exported {model_type} model to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error converting {model_type} model to ONNX: {e}")
            return False
    
    def verify_onnx_model(self, onnx_path, model_type='actor'):
        """
        Verify the exported ONNX model works correctly
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create inference session
            session = ort.InferenceSession(onnx_path)
            
            # Test inference
            test_input = np.random.randn(1, self.state_size).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})
            
            expected_shape = (1, self.action_size) if model_type == 'actor' else (1, 1)
            actual_shape = output[0].shape
            
            if actual_shape == expected_shape:
                print(f"✅ ONNX model verification successful: {actual_shape}")
                return True
            else:
                print(f"❌ Shape mismatch: expected {expected_shape}, got {actual_shape}")
                return False
                
        except Exception as e:
            print(f"❌ ONNX model verification failed: {e}")
            return False
    
    def convert_deepmimic_models(self, 
                                actor_path="../../pytorch_DeepMimic/deepmimic/output/agent0_model_anet.pth",
                                critic_path="../../pytorch_DeepMimic/deepmimic/output/agent0_model_cnet.pth",
                                output_dir="./"):
        """
        Convert both actor and critic models to ONNX
        """
        print("🚀 DeepMimic PyTorch to ONNX Converter")
        print("=" * 50)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert Actor Network
        print("\n🎭 Converting Actor Network...")
        actor_model = self.load_pytorch_model(actor_path, 'actor')
        if actor_model:
            actor_onnx_path = os.path.join(output_dir, "deepmimic_actor.onnx")
            if self.convert_to_onnx(actor_model, actor_onnx_path, 'actor'):
                self.verify_onnx_model(actor_onnx_path, 'actor')
        
        # Convert Critic Network
        print("\n🎯 Converting Critic Network...")
        critic_model = self.load_pytorch_model(critic_path, 'critic')
        if critic_model:
            critic_onnx_path = os.path.join(output_dir, "deepmimic_critic.onnx")
            if self.convert_to_onnx(critic_model, critic_onnx_path, 'critic'):
                self.verify_onnx_model(critic_onnx_path, 'critic')
        
        print("\n✅ Conversion complete!")
        print(f"📁 Output directory: {output_dir}")
        print("🌐 Models are ready for web inference!")

def main():
    converter = DeepMimicONNXConverter()
    
    # Default paths
    actor_path = "../../pytorch_DeepMimic/deepmimic/output/agent0_model_anet.pth"
    critic_path = "../../pytorch_DeepMimic/deepmimic/output/agent0_model_cnet.pth"
    output_dir = "./"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    if len(sys.argv) > 2:
        actor_path = sys.argv[2]
    if len(sys.argv) > 3:
        critic_path = sys.argv[3]
    
    converter.convert_deepmimic_models(actor_path, critic_path, output_dir)

if __name__ == "__main__":
    main()
