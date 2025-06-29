"""
Fixed Motion Inference Engine - Mojo Implementation

This is a corrected version addressing the issues found in static analysis.
"""

from tensor import Tensor, TensorShape
from utils.index import Index
from memory import memset_zero
from algorithm import vectorize, parallelize
from math import sqrt, exp, tanh
from time import now
from random import random_float64


struct LinearLayer:
    """High-performance linear layer with proper error handling."""
    
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var input_size: Int
    var output_size: Int
    
    fn __init__(inout self, input_size: Int, output_size: Int):
        """Initialize linear layer with Xavier initialization."""
        self.input_size = input_size
        self.output_size = output_size
        self.weights = Tensor[DType.float32](TensorShape(output_size, input_size))
        self.bias = Tensor[DType.float32](TensorShape(output_size))
        
        # Xavier initialization
        let std = sqrt(2.0 / Float32(input_size + output_size))
        self._initialize_weights(std)
    
    fn _initialize_weights(inout self, std: Float32):
        """Initialize weights with proper random values."""
        for i in range(self.weights.num_elements()):
            let random_val = random_float64().cast[DType.float32]()
            self.weights[i] = std * (random_val - 0.5) * 2.0
        
        # Zero bias initialization
        memset_zero(self.bias)
    
    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass with optimized matrix multiplication."""
        var output = Tensor[DType.float32](TensorShape(self.output_size))
        
        @parameter
        fn compute_row(i: Int):
            var sum: Float32 = 0.0
            
            @parameter
            fn vectorized_multiply(j: Int):
                sum += self.weights[i * self.input_size + j] * input[j]
            
            vectorize[vectorized_multiply, 16](self.input_size)
            output[i] = sum + self.bias[i]
        
        parallelize[compute_row](self.output_size)
        return output


struct ActivationFunctions:
    """Collection of activation functions."""
    
    @staticmethod
    fn relu(input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """ReLU activation function."""
        var output = Tensor[DType.float32](input.shape())
        
        @parameter
        fn relu_vectorized(i: Int):
            output[i] = max(Float32(0.0), input[i])
        
        vectorize[relu_vectorized, 16](input.num_elements())
        return output
    
    @staticmethod
    fn tanh_activation(input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Tanh activation function."""
        var output = Tensor[DType.float32](input.shape())
        
        @parameter
        fn tanh_vectorized(i: Int):
            output[i] = tanh(input[i])
        
        vectorize[tanh_vectorized, 16](input.num_elements())
        return output


struct DeepPhaseNetwork:
    """DeepPhase network for motion phase encoding."""
    
    var layer1: LinearLayer
    var layer2: LinearLayer  
    var layer3: LinearLayer
    var layer4: LinearLayer
    
    fn __init__(inout self):
        """Initialize DeepPhase network layers."""
        self.layer1 = LinearLayer(132, 256)
        self.layer2 = LinearLayer(256, 128)
        self.layer3 = LinearLayer(128, 32)
        self.layer4 = LinearLayer(32, 2)
    
    fn forward(self, motion_features: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: motion features -> phase coordinates."""
        # Layer 1: 132 -> 256 with ReLU
        var x1 = self.layer1.forward(motion_features)
        var x1_relu = ActivationFunctions.relu(x1)
        
        # Layer 2: 256 -> 128 with ReLU
        var x2 = self.layer2.forward(x1_relu)
        var x2_relu = ActivationFunctions.relu(x2)
        
        # Layer 3: 128 -> 32 with ReLU
        var x3 = self.layer3.forward(x2_relu)
        var x3_relu = ActivationFunctions.relu(x3)
        
        # Layer 4: 32 -> 2 (output)
        var phase_output = self.layer4.forward(x3_relu)
        
        return phase_output


struct DeepMimicActor:
    """DeepMimic actor network for character control."""
    
    var layer1: LinearLayer
    var layer2: LinearLayer
    var output_layer: LinearLayer
    
    fn __init__(inout self, state_size: Int = 197, action_size: Int = 36):
        """Initialize actor network layers."""
        self.layer1 = LinearLayer(state_size, 1024)
        self.layer2 = LinearLayer(1024, 512)
        self.output_layer = LinearLayer(512, action_size)
    
    fn forward(self, state: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: state -> actions."""
        # Hidden layers with ReLU
        var h1 = ActivationFunctions.relu(self.layer1.forward(state))
        var h2 = ActivationFunctions.relu(self.layer2.forward(h1))
        
        # Output with tanh activation for bounded actions
        var actions = ActivationFunctions.tanh_activation(self.output_layer.forward(h2))
        
        return actions


struct MotionInferenceEngine:
    """High-level inference engine for motion processing."""
    
    var deephase: DeepPhaseNetwork
    var deepmimic_actor: DeepMimicActor
    var initialized: Bool
    
    fn __init__(inout self):
        """Initialize the inference engine."""
        self.deephase = DeepPhaseNetwork()
        self.deepmimic_actor = DeepMimicActor()
        self.initialized = True
    
    fn encode_motion_phase(self, motion_features: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Encode motion to phase coordinates."""
        return self.deephase.forward(motion_features)
    
    fn generate_actions(self, state: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Generate control actions from state."""
        return self.deepmimic_actor.forward(state)
    
    fn process_motion_pipeline(self, motion_data: Tensor[DType.float32], state: Tensor[DType.float32]) -> (Tensor[DType.float32], Tensor[DType.float32]):
        """Full motion processing pipeline."""
        let phase_coords = self.encode_motion_phase(motion_data)
        let actions = self.generate_actions(state)
        return (phase_coords, actions)


fn benchmark_inference(iterations: Int = 1000) -> Float64:
    """Benchmark inference performance."""
    var engine = MotionInferenceEngine()
    
    # Create test inputs
    var motion_features = Tensor[DType.float32](TensorShape(132))
    var state = Tensor[DType.float32](TensorShape(197))
    
    # Fill with test data
    for i in range(132):
        motion_features[i] = Float32(i) * 0.01
    
    for i in range(197):
        state[i] = Float32(i) * 0.005
    
    # Warm up
    for i in range(10):
        _ = engine.process_motion_pipeline(motion_features, state)
    
    # Benchmark
    let start_time = now()
    
    for i in range(iterations):
        _ = engine.process_motion_pipeline(motion_features, state)
    
    let end_time = now()
    let total_time_ns = (end_time - start_time)
    let avg_time_ms = Float64(total_time_ns) / 1_000_000.0 / Float64(iterations)
    
    return avg_time_ms


fn main():
    """Main function for testing the motion inference engine."""
    print("🚀 Mojo Motion Inference Engine - Fixed Version")
    print("=" * 50)
    
    # Test basic functionality
    var engine = MotionInferenceEngine()
    print("✅ Engine initialized successfully")
    
    # Create test inputs
    var motion_features = Tensor[DType.float32](TensorShape(132))
    var state = Tensor[DType.float32](TensorShape(197))
    
    # Fill with test data
    for i in range(132):
        motion_features[i] = Float32(i) * 0.01
    
    for i in range(197):
        state[i] = Float32(i) * 0.005
    
    print("🧪 Running inference test...")
    
    # Test inference
    let (phase_coords, actions) = engine.process_motion_pipeline(motion_features, state)
    
    print("📊 Results:")
    print("  Phase coordinates shape: [" + str(phase_coords.shape()[0]) + "]")
    print("  Actions shape: [" + str(actions.shape()[0]) + "]")
    
    # Quick validation
    print("  Phase coord[0]:", phase_coords[0])
    print("  Phase coord[1]:", phase_coords[1])
    print("  Action[0]:", actions[0])
    
    # Performance benchmark
    print("⚡ Running performance benchmark...")
    let avg_time = benchmark_inference(100)
    let fps = 1000.0 / avg_time
    
    print("  Average inference time: " + str(avg_time) + " ms")
    print("  Estimated throughput: " + str(fps) + " FPS")
    
    if fps > 30.0:
        print("✅ Performance target achieved (>30 FPS)")
    else:
        print("⚠️ Performance below target (<30 FPS)")
    
    print("🎉 Motion inference engine testing complete!")
