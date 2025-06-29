"""
DeepPhase Model - Mojo Implementation

This implements the DeepPhase model for motion phase encoding in Mojo.
Migrated from PyTorch for better performance and deployment efficiency.
"""

from max import Model
from max.graph import Graph, ops
from max.tensor import Tensor, TensorShape
import math


struct DeepPhaseModel:
    """DeepPhase model for motion-to-phase encoding."""
    
    var graph: Graph
    var input_dim: Int
    var latent_dim: Int 
    var phase_dim: Int
    
    fn __init__(inout self, input_dim: Int = 132, latent_dim: Int = 32, phase_dim: Int = 2):
        """Initialize the DeepPhase model."""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.graph = Graph()
        self._build_graph()
    
    fn _build_graph(inout self):
        """Build the computational graph for DeepPhase."""
        # Input placeholder
        let input_shape = TensorShape(1, self.input_dim)
        let motion_input = self.graph.input(input_shape)
        
        # Encoder: motion -> latent representation
        # Layer 1: 132 -> 256
        let encoder_1_weights = self.graph.constant(TensorShape(self.input_dim, 256))
        let encoder_1_bias = self.graph.constant(TensorShape(256))
        let encoder_1 = ops.matmul(motion_input, encoder_1_weights)
        let encoder_1_biased = ops.add(encoder_1, encoder_1_bias)
        let encoder_1_activated = ops.leaky_relu(encoder_1_biased, alpha=0.2)
        
        # Layer 2: 256 -> 128  
        let encoder_2_weights = self.graph.constant(TensorShape(256, 128))
        let encoder_2_bias = self.graph.constant(TensorShape(128))
        let encoder_2 = ops.matmul(encoder_1_activated, encoder_2_weights)
        let encoder_2_biased = ops.add(encoder_2, encoder_2_bias)
        let encoder_2_activated = ops.leaky_relu(encoder_2_biased, alpha=0.2)
        
        # Layer 3: 128 -> 32 (latent)
        let encoder_3_weights = self.graph.constant(TensorShape(128, self.latent_dim))
        let encoder_3_bias = self.graph.constant(TensorShape(self.latent_dim))
        let latent = ops.matmul(encoder_2_activated, encoder_3_weights)
        let latent_biased = ops.add(latent, encoder_3_bias)
        
        # Phase decoder: latent -> 2D phase coordinates
        # Layer 1: 32 -> 16
        let decoder_1_weights = self.graph.constant(TensorShape(self.latent_dim, 16))
        let decoder_1_bias = self.graph.constant(TensorShape(16))
        let decoder_1 = ops.matmul(latent_biased, decoder_1_weights)
        let decoder_1_biased = ops.add(decoder_1, decoder_1_bias)
        let decoder_1_activated = ops.leaky_relu(decoder_1_biased, alpha=0.2)
        
        # Layer 2: 16 -> 2 (phase output)
        let decoder_2_weights = self.graph.constant(TensorShape(16, self.phase_dim))
        let decoder_2_bias = self.graph.constant(TensorShape(self.phase_dim))
        let phase_output = ops.matmul(decoder_1_activated, decoder_2_weights)
        let phase_final = ops.add(phase_output, decoder_2_bias)
        
        # Set output
        self.graph.output(phase_final)
    
    fn forward(self, motion_data: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: motion -> phase coordinates."""
        let model = Model(self.graph)
        let result = model.execute("motion_input", motion_data)
        return result.get[DType.float32]("phase_output")


fn main():
    """Test the DeepPhase model implementation."""
    print("DeepPhase Mojo Implementation Test")
    
    # Create model
    var model = DeepPhaseModel()
    
    # Create test input (batch_size=1, input_dim=132)
    var test_input = Tensor[DType.float32](TensorShape(1, 132))
    
    # Fill with random-like test data
    for i in range(132):
        test_input[i] = 0.1 * i
    
    # Forward pass
    let phase_output = model.forward(test_input)
    
    print("Input shape:", test_input.shape())
    print("Output shape:", phase_output.shape())
    print("DeepPhase model test completed successfully!")
