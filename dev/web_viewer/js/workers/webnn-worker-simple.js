/**
 * WebNN Worker for Neural Network inference tasks
 * Handles ML model execution and WebNN-specific benchmarks
 */

// Track active tasks and WebNN state
let activeTasks = new Map();
let webnnContext = null;
let webnnAdapter = null;

// Worker message handling
// Initialize WebNN if available
async function initWebNN() {
    if (!navigator.ml) {
        console.warn('WebNN not available in this worker');
        return false;
    }
    
    try {
        webnnContext = await navigator.ml.createContext();
        console.log('WebNN initialized successfully');
        return true;
    } catch (error) {
        console.error('Failed to initialize WebNN:', error);
        return false;
    }
}

// Handle messages from main thread
self.onmessage = function(event) {
    const { type, data } = event.data;
    
    switch (type) {
        case 'execute':
            executeTask(data);
            break;
        case 'cancel':
            cancelTask(data.taskId);
            break;
        case 'init_webnn':
            initWebNN().then(success => {
                self.postMessage({
                    type: 'webnn_init',
                    success: success
                });
            });
            break;
        default:
            console.warn('Unknown message type:', type);
    }
};

let currentTask = null;
let cancelled = false;
let webnnContext = null;

// Initialize WebNN if available
async function initializeWebNN() {
    try {
        if (typeof navigator !== 'undefined' && navigator.ml) {
            webnnContext = await navigator.ml.createContext();
            console.log('WebNN Worker: WebNN context initialized');
            return true;
        }
    } catch (error) {
        console.log('WebNN Worker: WebNN not available, using CPU fallback');
    }
    return false;
}

function executeTask(taskData) {
    const { taskId, jobType, duration = 1000, complexity = 1, shouldFail = false } = taskData;
    
    currentTask = taskId;
    cancelled = false;
    
    console.log(`WebNN Worker: Starting task ${taskId} (${jobType})`);
    
    // Handle test error case
    if (shouldFail || jobType === 'ErrorTestJob') {
        setTimeout(() => {
            self.postMessage({
                type: 'error',
                taskId: taskId,
                error: 'Intentional test error from WebNN worker'
            });
        }, 50);
        return;
    }
    
    // Handle WebNN-specific job types
    if (jobType === 'WebNNImageClassification' || jobType === 'WebNNTextProcessing' || jobType === 'WebNNAudioProcessing') {
        simulateWebNNSpecificWork(taskId, duration, complexity, jobType);
    } else {
        // Fallback to generic WebNN simulation
        simulateWebNNInference(taskId, duration, complexity);
    }
}

function cancelTask(taskId) {
    if (currentTask === taskId) {
        cancelled = true;
        self.postMessage({
            type: 'cancelled',
            taskId: taskId
        });
    }
}

async function simulateWebNNInference(taskId, duration, complexity) {
    const startTime = Date.now();
    const steps = Math.max(5, Math.floor(duration / 200)); // Fewer steps for inference
    const stepDuration = duration / steps;
    
    try {
        for (let i = 0; i < steps && !cancelled; i++) {
            // Simulate neural network inference
            if (webnnContext) {
                await simulateWebNNCompute(complexity);
            } else {
                // Fallback to CPU neural network simulation
                await simulateNeuralNetworkCPU(complexity);
            }
            
            // Check if cancelled
            if (cancelled) {
                return;
            }
            
            // Report progress
            const progress = Math.round(((i + 1) / steps) * 100);
            const elapsed = Date.now() - startTime;
            
            self.postMessage({
                type: 'progress',
                taskId: taskId,
                progress: progress,
                stats: {
                    step: i + 1,
                    totalSteps: steps,
                    elapsed: elapsed,
                    estimated: (elapsed / (i + 1)) * steps,
                    usingWebNN: !!webnnContext
                }
            });
            
            // Small delay between inference steps
            await new Promise(resolve => setTimeout(resolve, Math.max(10, stepDuration - 100)));
        }
        
        if (!cancelled) {
            // Task completed successfully
            const totalTime = Date.now() - startTime;
            
            self.postMessage({
                type: 'completed',
                taskId: taskId,
                result: {
                    success: true,
                    executionTime: totalTime,
                    workerType: 'webnn',
                    steps: steps,
                    complexity: complexity,
                    usingWebNN: !!webnnContext
                }
            });
        }
        
    } catch (error) {
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: error.message
        });
    }
}

// Simulate WebNN neural network inference
async function simulateWebNNCompute(complexity) {
    if (!webnnContext) {
        return simulateNeuralNetworkCPU(complexity);
    }
    
    return new Promise((resolve) => {
        // Simulate neural network inference using WebNN
        setTimeout(() => {
            // In a real implementation, this would:
            // 1. Load neural network model
            // 2. Prepare input tensors
            // 3. Execute inference
            // 4. Return output tensors
            resolve();
        }, 20 + complexity * 10);
    });
}

// Fallback CPU neural network simulation
async function simulateNeuralNetworkCPU(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const neurons = 100 + complexity * 50;
            const layers = 3 + complexity;
            let activation = 0;
            
            // Simulate forward pass through neural network
            for (let layer = 0; layer < layers; layer++) {
                for (let neuron = 0; neuron < neurons; neuron++) {
                    // Simulate activation function (ReLU)
                    activation = Math.max(0, Math.random() * 2 - 1 + activation * 0.1);
                }
                // Simulate layer normalization
                activation = activation / neurons;
            }
            
            resolve(activation);
        }, 0);
    });
}

// Enhanced WebNN work simulation for specific job types
async function simulateWebNNSpecificWork(taskId, duration, complexity, jobType) {
    const startTime = Date.now();
    const steps = Math.max(5, Math.floor(duration / 200)); // AI inference steps
    const stepDuration = duration / steps;
    
    try {
        for (let i = 0; i < steps && !cancelled; i++) {
            // Simulate job-type specific neural network inference
            if (webnnContext) {
                switch (jobType) {
                    case 'WebNNImageClassification':
                        await simulateImageClassification(complexity);
                        break;
                    case 'WebNNTextProcessing':
                        await simulateTextProcessing(complexity);
                        break;
                    case 'WebNNAudioProcessing':
                        await simulateAudioProcessing(complexity);
                        break;
                    default:
                        await simulateNeuralInference(complexity);
                }
            } else {
                // Fallback to CPU simulation
                await simulateNeuralInference(complexity);
            }
            
            // Check if cancelled
            if (cancelled) {
                return;
            }
            
            // Report progress with AI-specific stats
            const progress = Math.round(((i + 1) / steps) * 100);
            const elapsed = Date.now() - startTime;
            
            self.postMessage({
                type: 'progress',
                taskId: taskId,
                progress: progress,
                stats: {
                    step: i + 1,
                    totalSteps: steps,
                    elapsed: elapsed,
                    estimated: (elapsed / (i + 1)) * steps,
                    usingWebNN: !!webnnContext,
                    jobType: jobType,
                    processingType: webnnContext ? 'WebNN Inference' : 'CPU Fallback'
                }
            });
            
            // Inference timing
            await new Promise(resolve => setTimeout(resolve, Math.max(10, stepDuration - 100)));
        }
        
        if (!cancelled) {
            // Task completed successfully
            const totalTime = Date.now() - startTime;
            
            self.postMessage({
                type: 'completed',
                taskId: taskId,
                result: {
                    success: true,
                    executionTime: totalTime,
                    workerType: 'webnn',
                    steps: steps,
                    complexity: complexity,
                    jobType: jobType,
                    usingWebNN: !!webnnContext
                }
            });
        }
        
    } catch (error) {
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: error.message
        });
    }
}

// WebNN job-specific simulation functions
async function simulateImageClassification(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate CNN inference (convolution + pooling + classification)
            const features = 512 * complexity;
            let confidence = 0;
            for (let i = 0; i < features; i++) {
                confidence += Math.tanh(i * 0.01) * Math.sigmoid(i * 0.005);
            }
            resolve(confidence / features);
        }, 30);
    });
}

async function simulateTextProcessing(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate transformer attention mechanism
            const tokens = 128 * complexity;
            let attention = 0;
            for (let i = 0; i < tokens; i++) {
                for (let j = 0; j < tokens; j++) {
                    attention += Math.exp(-Math.abs(i - j) * 0.1);
                }
            }
            resolve(attention / (tokens * tokens));
        }, 40);
    });
}

async function simulateAudioProcessing(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate speech recognition (MFCC + RNN)
            const samples = 1024 * complexity;
            let energy = 0;
            for (let i = 0; i < samples; i++) {
                energy += Math.pow(Math.sin(i * Math.PI / samples), 2);
            }
            resolve(energy / samples);
        }, 35);
    });
}

// Initialize WebNN and notify ready
initializeWebNN().then((hasWebNN) => {
    self.postMessage({
        type: 'ready',
        workerType: 'webnn',
        capabilities: {
            webnn: hasWebNN
        }
    });
});

// WebNN Benchmark Functions
async function runWebNNInferenceBenchmark(complexity = 1) {
    if (!webnnContext) {
        throw new Error('WebNN context not available');
    }
    
    try {
        // Create a simple neural network for benchmarking
        const inputSize = 224 * 224 * 3; // Typical image input
        const hiddenSize = 1000 * complexity;
        const outputSize = 1000;
        
        const builder = new MLGraphBuilder(webnnContext);
        
        // Define network architecture
        const input = builder.input('input', { type: 'float32', dimensions: [1, inputSize] });
        
        // Hidden layer weights and bias
        const hiddenWeights = builder.constant({ 
            type: 'float32', 
            dimensions: [inputSize, hiddenSize] 
        }, new Float32Array(inputSize * hiddenSize).fill(0.1));
        
        const hiddenBias = builder.constant({ 
            type: 'float32', 
            dimensions: [hiddenSize] 
        }, new Float32Array(hiddenSize).fill(0.01));
        
        // Output layer weights and bias
        const outputWeights = builder.constant({ 
            type: 'float32', 
            dimensions: [hiddenSize, outputSize] 
        }, new Float32Array(hiddenSize * outputSize).fill(0.1));
        
        const outputBias = builder.constant({ 
            type: 'float32', 
            dimensions: [outputSize] 
        }, new Float32Array(outputSize).fill(0.01));
        
        // Build the network
        const hidden = builder.relu(builder.add(builder.matmul(input, hiddenWeights), hiddenBias));
        const output = builder.add(builder.matmul(hidden, outputWeights), outputBias);
        
        // Compile the graph
        const graph = await builder.build({ 'output': output });
        
        // Prepare input data
        const inputData = new Float32Array(inputSize);
        for (let i = 0; i < inputSize; i++) {
            inputData[i] = Math.random();
        }
        
        const startTime = performance.now();
        
        // Run inference multiple times
        const iterations = 10;
        for (let i = 0; i < iterations; i++) {
            const results = await webnnContext.compute(graph, { 'input': inputData });
            // Ensure computation completes
            await results.output.getData();
        }
        
        const endTime = performance.now();
        
        const duration = (endTime - startTime) / 1000; // seconds
        const inferenceTime = duration / iterations;
        const throughput = iterations / duration; // inferences per second
        
        // Calculate approximate FLOPS
        const flopsPerInference = (inputSize * hiddenSize + hiddenSize * outputSize) * 2; // MAC operations
        const totalFLOPS = flopsPerInference * iterations;
        const gflops = totalFLOPS / duration / 1e9;
        
        return {
            inferenceTime: inferenceTime,
            throughput: throughput,
            gflops: gflops,
            iterations: iterations,
            modelComplexity: complexity,
            webnn: true
        };
        
    } catch (error) {
        console.warn('WebNN inference benchmark failed:', error);
        throw error;
    }
}

async function runWebNNMemoryBenchmark(complexity = 1) {
    if (!webnnContext) {
        throw new Error('WebNN context not available');
    }
    
    try {
        const dataSize = 1024 * 1024 * complexity; // 1M elements per complexity
        
        const builder = new MLGraphBuilder(webnnContext);
        
        // Create tensors for memory bandwidth test
        const input = builder.input('input', { type: 'float32', dimensions: [1, dataSize] });
        
        // Simple memory copy operation using identity + small constant
        const constant = builder.constant({ 
            type: 'float32', 
            dimensions: [1, dataSize] 
        }, new Float32Array(dataSize).fill(1.001));
        
        const output = builder.mul(input, constant);
        
        const graph = await builder.build({ 'output': output });
        
        // Prepare test data
        const inputData = new Float32Array(dataSize);
        for (let i = 0; i < dataSize; i++) {
            inputData[i] = Math.random();
        }
        
        const startTime = performance.now();
        
        // Run memory operations
        const iterations = 20;
        for (let i = 0; i < iterations; i++) {
            const results = await webnnContext.compute(graph, { 'input': inputData });
            await results.output.getData();
        }
        
        const endTime = performance.now();
        
        const duration = (endTime - startTime) / 1000; // seconds
        const bytesTransferred = dataSize * 4 * 2 * iterations; // float32 * read/write * iterations
        const bandwidth = (bytesTransferred / 1024 / 1024) / duration; // MB/s
        
        return {
            bandwidth: bandwidth,
            duration: duration,
            dataSize: dataSize,
            iterations: iterations,
            webnn: true
        };
        
    } catch (error) {
        console.warn('WebNN memory benchmark failed:', error);
        throw error;
    }
}
