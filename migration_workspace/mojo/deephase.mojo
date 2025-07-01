"""
DeepPhase Model - Pure Mojo Implementation
High-performance motion phase encoding for real-time applications.
"""

from tensor import Tensor, TensorShape
from algorithm import vectorize
import math


struct DeepPhaseModel:
    """High-performance DeepPhase model implementation in Mojo."""
    
    var input_dim: Int
    var latent_dim: Int
    var phase_dim: Int
    
    # Layer weights (loaded from file)
    var enc1_weight: Tensor[DType.float32]
    var enc1_bias: Tensor[DType.float32]
    var enc2_weight: Tensor[DType.float32]
    var enc2_bias: Tensor[DType.float32]
    var enc3_weight: Tensor[DType.float32]
    var enc3_bias: Tensor[DType.float32]
    var dec1_weight: Tensor[DType.float32]
    var dec1_bias: Tensor[DType.float32]
    var dec2_weight: Tensor[DType.float32]
    var dec2_bias: Tensor[DType.float32]
    
    fn __init__(inout self, input_dim: Int = 132, latent_dim: Int = 32, phase_dim: Int = 2):
        """Initialize the DeepPhase model."""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        
        # Initialize weights (will be loaded from file)
        self.enc1_weight = Tensor[DType.float32](TensorShape(132, 256))
        self.enc1_bias = Tensor[DType.float32](TensorShape(256))
        self.enc2_weight = Tensor[DType.float32](TensorShape(256, 128))
        self.enc2_bias = Tensor[DType.float32](TensorShape(128))
        self.enc3_weight = Tensor[DType.float32](TensorShape(128, 32))
        self.enc3_bias = Tensor[DType.float32](TensorShape(32))
        self.dec1_weight = Tensor[DType.float32](TensorShape(32, 16))
        self.dec1_bias = Tensor[DType.float32](TensorShape(16))
        self.dec2_weight = Tensor[DType.float32](TensorShape(16, 2))
        self.dec2_bias = Tensor[DType.float32](TensorShape(2))
    
    fn leaky_relu(self, x: Tensor[DType.float32], alpha: Float32 = 0.2) -> Tensor[DType.float32]:
        """Vectorized LeakyReLU activation."""
        var result = Tensor[DType.float32](x.shape())
        
        @parameter
        fn compute_leaky_relu[simd_width: Int](idx: Int):
            let vals = x.load[width=simd_width](idx)
            let zeros = SIMD[DType.float32, simd_width](0.0)
            let alphas = SIMD[DType.float32, simd_width](alpha)
            let negative_part = vals * alphas
            let positive_part = vals
            let result_vals = (vals > zeros).select(positive_part, negative_part)
            result.store[width=simd_width](idx, result_vals)
        
        vectorize[compute_leaky_relu, 8](x.num_elements())
        return result
    
    fn relu(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Vectorized ReLU activation."""
        var result = Tensor[DType.float32](x.shape())
        
        @parameter
        fn compute_relu[simd_width: Int](idx: Int):
            let vals = x.load[width=simd_width](idx)
            let zeros = SIMD[DType.float32, simd_width](0.0)
            let result_vals = (vals > zeros).select(vals, zeros)
            result.store[width=simd_width](idx, result_vals)
        
        vectorize[compute_relu, 8](x.num_elements())
        return result
    
    fn tanh_activation(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Vectorized Tanh activation."""
        var result = Tensor[DType.float32](x.shape())
        
        @parameter
        fn compute_tanh[simd_width: Int](idx: Int):
            let vals = x.load[width=simd_width](idx)
            let result_vals = math.tanh(vals)
            result.store[width=simd_width](idx, result_vals)
        
        vectorize[compute_tanh, 8](x.num_elements())
        return result
    
    fn linear_layer(self, input: Tensor[DType.float32], weight: Tensor[DType.float32], 
                   bias: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Optimized linear layer computation."""
        # Matrix multiplication: input @ weight + bias
        let batch_size = input.shape()[0]
        let output_dim = weight.shape()[1]
        var output = Tensor[DType.float32](TensorShape(batch_size, output_dim))
        
        # Vectorized matrix multiplication
        for b in range(batch_size):
            for o in range(output_dim):
                var sum_val: Float32 = 0.0
                
                @parameter
                fn compute_dot[simd_width: Int](i: Int):
                    let input_vals = input.load[width=simd_width](b * input.shape()[1] + i)
                    let weight_vals = weight.load[width=simd_width](i * output_dim + o)
                    sum_val += (input_vals * weight_vals).reduce_add()
                
                vectorize[compute_dot, 8](input.shape()[1])
                output[b * output_dim + o] = sum_val + bias[o]
        
        return output
    
    fn forward(self, motion_data: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass: motion -> phase coordinates."""
        # Encoder Layer 1: 132 -> 256 + LeakyReLU
        var enc1 = self.linear_layer(motion_data, self.enc1_weight, self.enc1_bias)
        enc1 = self.leaky_relu(enc1, 0.2)
        
        # Encoder Layer 2: 256 -> 128 + LeakyReLU
        var enc2 = self.linear_layer(enc1, self.enc2_weight, self.enc2_bias)
        enc2 = self.leaky_relu(enc2, 0.2)
        
        # Encoder Layer 3: 128 -> 32 (latent)
        var latent = self.linear_layer(enc2, self.enc3_weight, self.enc3_bias)
        
        # Decoder Layer 1: 32 -> 16 + LeakyReLU
        var dec1 = self.linear_layer(latent, self.dec1_weight, self.dec1_bias)
        dec1 = self.leaky_relu(dec1, 0.2)
        
        # Decoder Layer 2: 16 -> 2 (phase output)
        let phase_output = self.linear_layer(dec1, self.dec2_weight, self.dec2_bias)
        
        return phase_output
    
    fn batch_forward(self, batch_data: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Batch inference with optimized memory access."""
        return self.forward(batch_data)


fn benchmark_deephase():
    """Benchmark the DeepPhase model performance."""
    print("DeepPhase Mojo Benchmark")
    
    var model = DeepPhaseModel()
    
    # Create test data
    var test_input = Tensor[DType.float32](TensorShape(100, 132))  # Batch of 100
    
    # Fill with test data
    for i in range(100 * 132):
        test_input[i] = Float32(i % 1000) * 0.001
    
    # Warm-up
    for _ in range(10):
        _ = model.forward(test_input)
    
    # Benchmark
    let start_time = now()
    let iterations = 1000
    
    for _ in range(iterations):
        let output = model.forward(test_input)
    
    let end_time = now()
    let total_time = (end_time - start_time) / 1e9  # Convert to seconds
    let avg_time = total_time / iterations
    let throughput = 100.0 / avg_time  # samples per second
    
    print("Benchmark Results:")
    print("Average inference time:", avg_time * 1000, "ms")
    print("Throughput:", throughput, "samples/second")


fn main():
    """Test and benchmark the DeepPhase implementation."""
    benchmark_deephase()
