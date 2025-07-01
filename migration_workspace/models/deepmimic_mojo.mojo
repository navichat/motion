"""
DeepMimic Actor Network - Mojo Implementation

This implements the actor network for the DeepMimic reinforcement learning system.
Migrated from PyTorch for better performance and deployment efficiency.
"""

from max import Model
from max.graph import Graph, ops
from max.tensor import Tensor, TensorShape
import math


struct DeepMimicActor:
    """Actor network for DeepMimic RL system."""
    
    var graph: Graph
    var input_dim: Int
    var action_dim: Int
    var hidden_dim: Int
    
    fn __init__(inout self, input_dim: Int = 197, action_dim: Int = 36, hidden_dim: Int = 1024):
        """Initialize the DeepMimic actor network."""
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.graph = Graph()
        self._build_graph()
    
    fn _build_graph(inout self):
        """Build the computational graph for the actor network."""
        # Input placeholder (state)
        let input_shape = TensorShape(1, self.input_dim)
        let state_input = self.graph.input(input_shape)
        
        # Hidden layer 1: input_dim -> 1024
        let fc1_weights = self.graph.constant(TensorShape(self.input_dim, self.hidden_dim))
        let fc1_bias = self.graph.constant(TensorShape(self.hidden_dim))
        let fc1 = ops.matmul(state_input, fc1_weights)
        let fc1_biased = ops.add(fc1, fc1_bias)
        let fc1_activated = ops.relu(fc1_biased)
        
        # Hidden layer 2: 1024 -> 512
        let fc2_weights = self.graph.constant(TensorShape(self.hidden_dim, 512))
        let fc2_bias = self.graph.constant(TensorShape(512))
        let fc2 = ops.matmul(fc1_activated, fc2_weights)
        let fc2_biased = ops.add(fc2, fc2_bias)
        let fc2_activated = ops.relu(fc2_biased)
        
        # Output layer: 512 -> action_dim (mean)
        let out_weights = self.graph.constant(TensorShape(512, self.action_dim))
        let out_bias = self.graph.constant(TensorShape(self.action_dim))
        let action_mean = ops.matmul(fc2_activated, out_weights)
        let action_mean_biased = ops.add(action_mean, out_bias)
        
        # Action bounds - tanh activation for bounded actions
        let action_bounded = ops.tanh(action_mean_biased)
        
        # Set output
        self.graph.output(action_bounded)
    
    fn forward(self, state: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: state -> action."""
        let model = Model(self.graph)
        let result = model.execute("state_input", state)
        return result.get[DType.float32]("action_output")


struct DeepMimicCritic:
    """Critic network for DeepMimic RL system."""
    
    var graph: Graph
    var input_dim: Int
    var hidden_dim: Int
    
    fn __init__(inout self, input_dim: Int = 197, hidden_dim: Int = 1024):
        """Initialize the DeepMimic critic network."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.graph = Graph()
        self._build_graph()
    
    fn _build_graph(inout self):
        """Build the computational graph for the critic network."""
        # Input placeholder (state)
        let input_shape = TensorShape(1, self.input_dim)
        let state_input = self.graph.input(input_shape)
        
        # Hidden layer 1: input_dim -> 1024
        let fc1_weights = self.graph.constant(TensorShape(self.input_dim, self.hidden_dim))
        let fc1_bias = self.graph.constant(TensorShape(self.hidden_dim))
        let fc1 = ops.matmul(state_input, fc1_weights)
        let fc1_biased = ops.add(fc1, fc1_bias)
        let fc1_activated = ops.relu(fc1_biased)
        
        # Hidden layer 2: 1024 -> 512
        let fc2_weights = self.graph.constant(TensorShape(self.hidden_dim, 512))
        let fc2_bias = self.graph.constant(TensorShape(512))
        let fc2 = ops.matmul(fc1_activated, fc2_weights)
        let fc2_biased = ops.add(fc2, fc2_bias)
        let fc2_activated = ops.relu(fc2_biased)
        
        # Output layer: 512 -> 1 (value estimate)
        let out_weights = self.graph.constant(TensorShape(512, 1))
        let out_bias = self.graph.constant(TensorShape(1))
        let value_output = ops.matmul(fc2_activated, out_weights)
        let value_final = ops.add(value_output, out_bias)
        
        # Set output
        self.graph.output(value_final)
    
    fn forward(self, state: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: state -> value estimate."""
        let model = Model(self.graph)
        let result = model.execute("state_input", state)
        return result.get[DType.float32]("value_output")


fn main():
    """Test the DeepMimic actor and critic networks."""
    print("DeepMimic Mojo Implementation Test")
    
    # Create networks
    var actor = DeepMimicActor()
    var critic = DeepMimicCritic()
    
    # Create test state input (batch_size=1, state_dim=197)
    var test_state = Tensor[DType.float32](TensorShape(1, 197))
    
    # Fill with test data
    for i in range(197):
        test_state[i] = 0.01 * i
    
    # Test actor
    let action = actor.forward(test_state)
    print("Actor - Input shape:", test_state.shape())
    print("Actor - Output shape:", action.shape())
    
    # Test critic  
    let value = critic.forward(test_state)
    print("Critic - Input shape:", test_state.shape())
    print("Critic - Output shape:", value.shape())
    
    print("DeepMimic networks test completed successfully!")
