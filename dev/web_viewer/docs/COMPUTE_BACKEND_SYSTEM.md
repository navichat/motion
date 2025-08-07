# Compute Backend System Documentation

## Overview

The Compute Backend System provides unified access to WebGPU, WebNN, and WebAssembly (WASM) compute capabilities for the avatar platform. This system enables high-performance AI inference, matrix operations, and parallel processing across different hardware configurations and browser capabilities.

## Architecture

### Backend Types

#### WebGPU Backend (`src/compute/backends/webgpu/`)
- **GPU Acceleration**: Direct access to graphics hardware for parallel computation
- **Compute Shaders**: Custom WGSL shaders for specialized operations
- **Memory Management**: Efficient GPU memory allocation and buffer management
- **Pipeline Optimization**: Optimized compute pipelines for AI workloads

#### WebNN Backend (`src/compute/backends/webnn/`)
- **Neural Network Optimization**: Hardware-accelerated neural network inference
- **Cross-platform**: Unified API across different hardware (CPU, GPU, NPU)
- **Model Format Support**: ONNX, TensorFlow Lite, and custom model formats
- **Quantization**: Support for INT8, FP16, and other quantized formats

#### WebAssembly Backend (`src/compute/backends/wasm/`)
- **Universal Compatibility**: Runs on all modern browsers
- **SIMD Optimization**: Single Instruction, Multiple Data operations
- **Threading**: Multi-threaded execution using SharedArrayBuffer
- **Memory Efficiency**: Optimized memory layout for numerical computations

### Backend Selection Strategy

```javascript
// Automatic backend selection based on capabilities
class ComputeBackendSelector {
  async selectOptimalBackend(workloadType) {
    const capabilities = await this.detectCapabilities();
    
    // Prefer WebGPU for large parallel workloads
    if (capabilities.webgpu && workloadType.parallel && workloadType.size > 1000) {
      return new WebGPUBackend();
    }
    
    // Prefer WebNN for neural network inference
    if (capabilities.webnn && workloadType.type === 'neural-network') {
      return new WebNNBackend();
    }
    
    // Fall back to WASM for universal compatibility
    return new WASMBackend();
  }
}
```

## Features

### WebGPU Capabilities

#### Compute Shader Programming
```wgsl
// Example WGSL compute shader for matrix multiplication
@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= arrayLength(&result) || col >= arrayLength(&result)) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < 256u; k++) {
        sum += matrixA[row * 256u + k] * matrixB[k * 256u + col];
    }
    
    result[row * 256u + col] = sum;
}
```

#### Buffer Management
```javascript
// WebGPU buffer operations
class WebGPUBackend {
  async createBuffer(data, usage) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    
    return buffer;
  }
  
  async readBuffer(buffer) {
    const readBuffer = this.device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
    this.device.queue.submit([encoder.finish()]);
    
    await readBuffer.mapAsync(GPUMapMode.READ);
    return new Float32Array(readBuffer.getMappedRange().slice());
  }
}
```

### WebNN Capabilities

#### Model Loading and Inference
```javascript
// WebNN model inference
class WebNNBackend {
  async loadModel(modelUrl) {
    const modelBuffer = await fetch(modelUrl).then(r => r.arrayBuffer());
    
    const context = await navigator.ml.createContext();
    const graph = await context.createModel(modelBuffer);
    
    return { context, graph };
  }
  
  async inference(model, inputs) {
    const outputs = await model.graph.compute(inputs);
    return outputs;
  }
}
```

#### Optimization Features
- **Graph Optimization**: Automatic computation graph optimization
- **Operator Fusion**: Combine multiple operations for efficiency
- **Memory Planning**: Optimal memory allocation for inference
- **Quantization**: Automatic precision reduction for performance

### WASM Capabilities

#### SIMD Operations
```javascript
// WASM SIMD vector operations
const wasmModule = await WebAssembly.instantiateStreaming(
  fetch('/src/compute/backends/wasm/vector-ops.wasm')
);

// Vectorized addition using SIMD
function vectorAdd(a, b, result) {
  const addFunction = wasmModule.instance.exports.vector_add_simd;
  
  // Allocate memory in WASM
  const aPtr = wasmModule.instance.exports.malloc(a.length * 4);
  const bPtr = wasmModule.instance.exports.malloc(b.length * 4);
  const resultPtr = wasmModule.instance.exports.malloc(result.length * 4);
  
  // Copy data to WASM memory
  new Float32Array(wasmModule.instance.exports.memory.buffer, aPtr, a.length).set(a);
  new Float32Array(wasmModule.instance.exports.memory.buffer, bPtr, b.length).set(b);
  
  // Execute SIMD operation
  addFunction(aPtr, bPtr, resultPtr, a.length);
  
  // Copy result back
  result.set(new Float32Array(wasmModule.instance.exports.memory.buffer, resultPtr, result.length));
  
  // Free memory
  wasmModule.instance.exports.free(aPtr);
  wasmModule.instance.exports.free(bPtr);
  wasmModule.instance.exports.free(resultPtr);
}
```

#### Threading Support
```javascript
// Multi-threaded WASM execution
class WASMThreadPool {
  constructor(threadCount = navigator.hardwareConcurrency) {
    this.threads = [];
    this.initializeThreads(threadCount);
  }
  
  async initializeThreads(count) {
    for (let i = 0; i < count; i++) {
      const worker = new Worker('/src/compute/backends/wasm/thread-worker.js');
      await this.initializeWorker(worker);
      this.threads.push(worker);
    }
  }
  
  async executeParallel(operation, data) {
    const chunkSize = Math.ceil(data.length / this.threads.length);
    const promises = [];
    
    for (let i = 0; i < this.threads.length; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, data.length);
      const chunk = data.slice(start, end);
      
      promises.push(this.executeOnThread(this.threads[i], operation, chunk));
    }
    
    const results = await Promise.all(promises);
    return this.combineResults(results);
  }
}
```

## Performance Optimization

### Workload Distribution

#### Dynamic Load Balancing
```javascript
class WorkloadBalancer {
  constructor() {
    this.backends = new Map();
    this.performanceMetrics = new Map();
  }
  
  async distributeWorkload(tasks) {
    const sortedBackends = this.sortBackendsByPerformance();
    const distribution = this.calculateOptimalDistribution(tasks, sortedBackends);
    
    const promises = distribution.map(({ backend, tasks }) => 
      backend.executeBatch(tasks)
    );
    
    return Promise.all(promises);
  }
  
  updatePerformanceMetrics(backend, executionTime, throughput) {
    this.performanceMetrics.set(backend.type, {
      avgExecutionTime: executionTime,
      throughput,
      timestamp: Date.now()
    });
  }
}
```

#### Backend Fallback Strategy
```javascript
class RobustComputeManager {
  async executeWithFallback(operation, data) {
    const backendPriority = ['webgpu', 'webnn', 'wasm'];
    
    for (const backendType of backendPriority) {
      try {
        const backend = await this.getBackend(backendType);
        if (await backend.isCapable(operation)) {
          return await backend.execute(operation, data);
        }
      } catch (error) {
        console.warn(`${backendType} backend failed:`, error);
        continue;
      }
    }
    
    throw new Error('All compute backends failed');
  }
}
```

### Memory Management

#### Buffer Pooling
```javascript
class BufferPool {
  constructor() {
    this.pools = new Map(); // size -> buffer[]
    this.inUse = new Set();
  }
  
  acquire(size) {
    const pool = this.pools.get(size) || [];
    
    if (pool.length > 0) {
      const buffer = pool.pop();
      this.inUse.add(buffer);
      return buffer;
    }
    
    // Create new buffer if pool is empty
    const buffer = this.createBuffer(size);
    this.inUse.add(buffer);
    return buffer;
  }
  
  release(buffer) {
    if (this.inUse.has(buffer)) {
      this.inUse.delete(buffer);
      
      const size = buffer.size || buffer.byteLength;
      if (!this.pools.has(size)) {
        this.pools.set(size, []);
      }
      
      this.pools.get(size).push(buffer);
    }
  }
}
```

## Testing

### Unit Tests (`tests/unit/compute/`)

#### Backend Initialization
```javascript
test('should initialize WebGPU backend', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const backend = new WebGPUBackend();
    const initialized = await backend.initialize();
    
    return {
      initialized,
      hasAdapter: !!backend.adapter,
      hasDevice: !!backend.device,
      features: backend.device ? Array.from(backend.device.features) : []
    };
  });
  
  if (result.initialized) {
    expect(result.hasAdapter).toBe(true);
    expect(result.hasDevice).toBe(true);
  }
});
```

#### Performance Benchmarking
```javascript
test('should benchmark matrix multiplication across backends', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const matrixSize = 256;
    const matrixA = new Float32Array(matrixSize * matrixSize).map(() => Math.random());
    const matrixB = new Float32Array(matrixSize * matrixSize).map(() => Math.random());
    
    const backends = ['webgpu', 'webnn', 'wasm'];
    const results = {};
    
    for (const backendType of backends) {
      try {
        const backend = await createBackend(backendType);
        await backend.initialize();
        
        const startTime = performance.now();
        const result = await backend.matrixMultiply(matrixA, matrixB, matrixSize);
        const endTime = performance.now();
        
        results[backendType] = {
          success: true,
          executionTime: endTime - startTime,
          hasResult: !!result,
          resultSize: result.length
        };
      } catch (error) {
        results[backendType] = {
          success: false,
          error: error.message
        };
      }
    }
    
    return results;
  });
  
  // At least one backend should work
  const workingBackends = Object.keys(result).filter(backend => result[backend].success);
  expect(workingBackends.length).toBeGreaterThan(0);
  
  // Performance comparison
  workingBackends.forEach(backend => {
    expect(result[backend].executionTime).toBeLessThan(5000); // < 5 seconds
    expect(result[backend].hasResult).toBe(true);
  });
});
```

#### Memory Usage Testing
```javascript
test('should manage memory efficiently', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const backend = new WebGPUBackend();
    await backend.initialize();
    
    const initialMemory = performance.memory.usedJSHeapSize;
    
    // Allocate and deallocate buffers
    const buffers = [];
    for (let i = 0; i < 100; i++) {
      const data = new Float32Array(1024).map(() => Math.random());
      const buffer = await backend.createBuffer(data);
      buffers.push(buffer);
    }
    
    const peakMemory = performance.memory.usedJSHeapSize;
    
    // Clean up
    for (const buffer of buffers) {
      backend.destroyBuffer(buffer);
    }
    
    // Force garbage collection if available
    if (window.gc) window.gc();
    
    const finalMemory = performance.memory.usedJSHeapSize;
    
    return {
      initialMemory,
      peakMemory,
      finalMemory,
      memoryLeakDetected: finalMemory > initialMemory * 1.1 // 10% tolerance
    };
  });
  
  expect(result.memoryLeakDetected).toBe(false);
  expect(result.peakMemory).toBeGreaterThan(result.initialMemory);
});
```

## Integration with AI Models

### Model Inference Pipeline
```javascript
class AIInferencePipeline {
  constructor() {
    this.backendManager = new ComputeBackendManager();
  }
  
  async runInference(model, input) {
    // Select optimal backend for this model
    const backend = await this.backendManager.selectOptimal(model.requirements);
    
    // Prepare input data
    const inputTensor = await backend.createTensor(input, model.inputShape);
    
    // Run inference
    const outputTensor = await backend.inference(model, inputTensor);
    
    // Convert back to JavaScript
    const output = await backend.tensorToArray(outputTensor);
    
    // Clean up
    backend.destroyTensor(inputTensor);
    backend.destroyTensor(outputTensor);
    
    return output;
  }
}
```

### Real-time Processing
```javascript
// Streaming inference for real-time applications
class StreamingInference {
  constructor(model) {
    this.model = model;
    this.inputBuffer = new CircularBuffer(model.inputSize);
    this.outputBuffer = new CircularBuffer(model.outputSize);
  }
  
  async processStream(dataChunk) {
    this.inputBuffer.push(dataChunk);
    
    if (this.inputBuffer.isFull()) {
      const input = this.inputBuffer.getWindow();
      const output = await this.runInference(input);
      this.outputBuffer.push(output);
      
      this.inputBuffer.advance();
    }
    
    return this.outputBuffer.latest();
  }
}
```

## Configuration

### Backend Configuration
```javascript
const computeConfig = {
  webgpu: {
    preferredAdapterType: 'high-performance',
    powerPreference: 'high-performance',
    limits: {
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxBufferSize: 1024 * 1024 * 1024 // 1GB
    }
  },
  webnn: {
    deviceType: 'auto', // 'cpu', 'gpu', 'npu', 'auto'
    powerPreference: 'default',
    precision: 'fp32'
  },
  wasm: {
    threads: navigator.hardwareConcurrency,
    simd: true,
    memory: {
      initial: 16, // 16MB
      maximum: 256 // 256MB
    }
  }
};
```

### Performance Tuning
```javascript
const performanceConfig = {
  bufferPoolSize: 50,
  maxConcurrentOperations: 4,
  fallbackTimeout: 5000, // 5 seconds
  memoryCleanupInterval: 30000, // 30 seconds
  benchmarkInterval: 60000 // 1 minute
};
```

## Troubleshooting

### Common Issues

#### WebGPU Not Available
```javascript
// Check WebGPU support
if (!navigator.gpu) {
  console.warn('WebGPU not supported');
  // Fall back to other backends
}

// Check for specific features
const adapter = await navigator.gpu.requestAdapter();
if (!adapter.features.has('timestamp-query')) {
  console.warn('Timestamp queries not supported');
}
```

#### WebNN Compatibility
```javascript
// Check WebNN support
if (!navigator.ml) {
  console.warn('WebNN not available');
}

// Test model compatibility
try {
  const context = await navigator.ml.createContext();
  // Test with simple model
} catch (error) {
  console.error('WebNN context creation failed:', error);
}
```

#### WASM Loading Issues
```javascript
// Debug WASM module loading
try {
  const module = await WebAssembly.instantiateStreaming(
    fetch('/path/to/module.wasm')
  );
} catch (error) {
  if (error.message.includes('mime type')) {
    console.error('WASM MIME type not configured on server');
  } else if (error.message.includes('streaming')) {
    console.error('Streaming compilation not supported');
    // Fall back to non-streaming
  }
}
```

### Performance Debugging
```javascript
// Profile compute operations
class ComputeProfiler {
  async profile(operation, ...args) {
    const startTime = performance.now();
    const startMemory = performance.memory.usedJSHeapSize;
    
    const result = await operation(...args);
    
    const endTime = performance.now();
    const endMemory = performance.memory.usedJSHeapSize;
    
    console.log(`Operation took ${endTime - startTime}ms`);
    console.log(`Memory delta: ${endMemory - startMemory} bytes`);
    
    return result;
  }
}
```

For more information, see the main [REORGANIZATION_STATUS.md](../REORGANIZATION_STATUS.md) documentation.
