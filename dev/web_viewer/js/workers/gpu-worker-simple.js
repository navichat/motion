/**
 * GPU Worker for WebGPU-based task execution
 * Handles GPU computation for high-performance tasks with WebGPU benchmarks
 */

// Track active tasks and WebGPU state
let activeTasks = new Map();
let currentTask = null;
let cancelled = false;
let gpuDevice = null;
let gpuAdapter = null;

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
        case 'init_webgpu':
            initializeGPU().then(success => {
                self.postMessage({
                    type: 'webgpu_init',
                    success: success
                });
            });
            break;
        default:
            console.warn('Unknown message type:', type);
    }
};

// Initialize GPU if available
async function initializeGPU() {
    try {
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                gpuDevice = await adapter.requestDevice();
                console.log('GPU Worker: WebGPU device initialized');
                return true;
            }
        }
    } catch (error) {
        console.log('GPU Worker: WebGPU not available, using CPU fallback');
    }
    return false;
}

function executeTask(taskData) {
    const { taskId, jobType, duration = 1000, complexity = 1, shouldFail = false } = taskData;
    
    currentTask = taskId;
    cancelled = false;
    
    console.log(`GPU Worker: Starting task ${taskId} (${jobType})`);
    
    // Handle test error case
    if (shouldFail || jobType === 'ErrorTestJob') {
        setTimeout(() => {
            self.postMessage({
                type: 'error',
                taskId: taskId,
                error: 'Intentional test error from GPU worker'
            });
            
            // Reset current task status
            currentTask = null;
        }, 50);
        return;
    }
    
    // Handle GPU-specific job types
    if (jobType === 'WebGPUMatrix' || jobType === 'WebGPUImage' || jobType === 'WebGPUParticle') {
        simulateWebGPUWork(taskId, duration, complexity, jobType);
    } else {
        // Fallback to generic GPU simulation
        simulateGPUWork(taskId, duration, complexity);
    }
}

function cancelTask(taskId) {
    if (currentTask === taskId) {
        cancelled = true;
        currentTask = null; // Reset current task when cancelled
        self.postMessage({
            type: 'cancelled',
            taskId: taskId
        });
    }
}

async function simulateGPUWork(taskId, duration, complexity) {
    const startTime = Date.now();
    const steps = Math.max(8, Math.floor(duration / 125)); // Fewer steps for GPU
    const stepDuration = duration / steps;
    
    try {
        for (let i = 0; i < steps && !cancelled; i++) {
            // Simulate GPU computation
            if (gpuDevice) {
                await simulateWebGPUCompute(complexity);
            } else {
                // Fallback to CPU simulation
                await simulateMatrixOperations(complexity);
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
                    usingGPU: !!gpuDevice
                }
            });
            
            // Small delay between steps
            await new Promise(resolve => setTimeout(resolve, Math.max(5, stepDuration - 50)));
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
                    workerType: 'gpu',
                    steps: steps,
                    complexity: complexity,
                    usingGPU: !!gpuDevice
                }
            });
            
            // Reset current task status
            currentTask = null;
        }
        
    } catch (error) {
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: error.message
        });
        
        // Reset current task status on error
        currentTask = null;
    }
}

// Simulate WebGPU compute operations
async function simulateWebGPUCompute(complexity) {
    if (!gpuDevice) {
        return simulateMatrixOperations(complexity);
    }
    
    return new Promise((resolve) => {
        // Simulate GPU compute pipeline execution
        setTimeout(() => {
            // In a real implementation, this would set up compute shaders
            // and execute them on the GPU
            resolve();
        }, 10 + complexity * 5);
    });
}

// Fallback matrix operations simulation
async function simulateMatrixOperations(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const size = 50 + complexity * 10;
            let result = 0;
            
            // Simulate matrix multiplication
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    result += Math.sin(i) * Math.cos(j);
                }
            }
            
            resolve(result);
        }, 0);
    });
}

// Enhanced WebGPU work simulation for specific job types
async function simulateWebGPUWork(taskId, duration, complexity, jobType) {
    const startTime = Date.now();
    const steps = Math.max(6, Math.floor(duration / 150)); // Optimized for GPU
    const stepDuration = duration / steps;
    
    try {
        for (let i = 0; i < steps && !cancelled; i++) {
            // Simulate job-type specific GPU computation
            if (gpuDevice) {
                switch (jobType) {
                    case 'WebGPUMatrix':
                        await simulateGPUMatrixOps(complexity);
                        break;
                    case 'WebGPUImage':
                        await simulateGPUImageProcessing(complexity);
                        break;
                    case 'WebGPUParticle':
                        await simulateGPUParticleSystem(complexity);
                        break;
                    default:
                        await simulateWebGPUCompute(complexity);
                }
            } else {
                // Fallback to CPU simulation
                await simulateMatrixOperations(complexity);
            }
            
            // Check if cancelled
            if (cancelled) {
                return;
            }
            
            // Report progress with job-specific stats
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
                    usingGPU: !!gpuDevice,
                    jobType: jobType,
                    processingType: gpuDevice ? 'WebGPU Compute' : 'CPU Fallback'
                }
            });
            
            // Shorter delay for GPU work
            await new Promise(resolve => setTimeout(resolve, Math.max(5, stepDuration - 80)));
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
                    workerType: 'gpu',
                    steps: steps,
                    complexity: complexity,
                    jobType: jobType,
                    usingWebGPU: !!gpuDevice
                }
            });
            
            // Reset current task status
            currentTask = null;
        }
        
    } catch (error) {
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: error.message
        });
        
        // Reset current task status on error
        currentTask = null;
    }
}

// GPU job-specific simulation functions
async function simulateGPUMatrixOps(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate parallel matrix operations
            const size = 32 * complexity;
            let result = 0;
            for (let i = 0; i < size; i++) {
                result += Math.sqrt(i) * Math.log(i + 1);
            }
            resolve(result);
        }, 10);
    });
}

async function simulateGPUImageProcessing(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate image convolution/filtering
            const pixels = 256 * complexity;
            let result = 0;
            for (let i = 0; i < pixels; i++) {
                result += Math.sin(i * 0.1) * Math.cos(i * 0.1);
            }
            resolve(result);
        }, 15);
    });
}

async function simulateGPUParticleSystem(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate particle physics calculations
            const particles = 64 * complexity;
            let result = 0;
            for (let i = 0; i < particles; i++) {
                result += Math.tan(i * 0.01) * Math.atan(i * 0.01);
            }
            resolve(result);
        }, 20);
    });
}

// Initialize GPU and notify ready
initializeGPU().then((hasGPU) => {
    self.postMessage({
        type: 'ready',
        workerType: 'gpu',
        capabilities: {
            webgpu: hasGPU
        }
    });
});

// WebGPU Benchmark Functions
async function runWebGPUMemoryBandwidthTest(complexity = 1) {
    if (!gpuDevice) {
        throw new Error('WebGPU device not available');
    }
    
    const bufferSize = 1024 * 1024 * complexity; // 1MB per complexity
    
    // Create buffers
    const inputBuffer = gpuDevice.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const outputBuffer = gpuDevice.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create compute shader for memory copy
    const shaderModule = gpuDevice.createShaderModule({
        code: `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let index = id.x;
                if (index >= arrayLength(&input)) {
                    return;
                }
                output[index] = input[index] * 1.1 + 0.1;
            }
        `
    });
    
    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
    
    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } }
        ]
    });
    
    // Fill input buffer with test data
    const inputData = new Float32Array(bufferSize / 4);
    for (let i = 0; i < inputData.length; i++) {
        inputData[i] = Math.random();
    }
    
    gpuDevice.queue.writeBuffer(inputBuffer, 0, inputData);
    
    const startTime = performance.now();
    
    // Run compute shader multiple times for better measurement
    for (let iter = 0; iter < 10; iter++) {
        const commandEncoder = gpuDevice.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(inputData.length / 64));
        passEncoder.end();
        
        gpuDevice.queue.submit([commandEncoder.finish()]);
        await gpuDevice.queue.onSubmittedWorkDone();
    }
    
    const endTime = performance.now();
    
    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    
    const duration = (endTime - startTime) / 1000; // seconds
    const bytesTransferred = bufferSize * 2 * 10; // Read + Write * iterations
    const bandwidth = (bytesTransferred / 1024 / 1024) / duration; // MB/s
    
    return {
        bandwidth: bandwidth,
        duration: duration,
        bufferSize: bufferSize,
        iterations: 10,
        webgpu: true
    };
}

async function runWebGPUFLOPSTest(complexity = 1) {
    if (!gpuDevice) {
        throw new Error('WebGPU device not available');
    }
    
    const arraySize = 1024 * 1024 * complexity; // 1M elements per complexity
    const bufferSize = arraySize * 4; // 4 bytes per float32
    
    // Create buffers
    const inputBufferA = gpuDevice.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const inputBufferB = gpuDevice.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const outputBuffer = gpuDevice.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create compute shader for FLOPS test
    const shaderModule = gpuDevice.createShaderModule({
        code: `
            @group(0) @binding(0) var<storage, read> inputA: array<f32>;
            @group(0) @binding(1) var<storage, read> inputB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let index = id.x;
                if (index >= arrayLength(&inputA)) {
                    return;
                }
                
                let a = inputA[index];
                let b = inputB[index];
                
                // Multiple FMA operations (2 FLOPs each)
                var result = a * b + a;  // 2 FLOPs
                result = result * b + a; // 2 FLOPs
                result = result * a + b; // 2 FLOPs
                result = result * b + a; // 2 FLOPs
                result = result * a + b; // 2 FLOPs
                // Total: 10 FLOPs per element
                
                output[index] = result;
            }
        `
    });
    
    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
    
    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBufferA } },
            { binding: 1, resource: { buffer: inputBufferB } },
            { binding: 2, resource: { buffer: outputBuffer } }
        ]
    });
    
    // Fill input buffers
    const inputDataA = new Float32Array(arraySize);
    const inputDataB = new Float32Array(arraySize);
    for (let i = 0; i < arraySize; i++) {
        inputDataA[i] = Math.random();
        inputDataB[i] = Math.random();
    }
    
    gpuDevice.queue.writeBuffer(inputBufferA, 0, inputDataA);
    gpuDevice.queue.writeBuffer(inputBufferB, 0, inputDataB);
    
    const startTime = performance.now();
    
    // Run compute shader
    const commandEncoder = gpuDevice.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(arraySize / 64));
    passEncoder.end();
    
    gpuDevice.queue.submit([commandEncoder.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    
    const endTime = performance.now();
    
    // Cleanup
    inputBufferA.destroy();
    inputBufferB.destroy();
    outputBuffer.destroy();
    
    const duration = (endTime - startTime) / 1000; // seconds
    const operations = arraySize * 10; // 10 FLOPs per element
    const flops = operations / duration;
    const gflops = flops / 1e9;
    
    return {
        gflops: gflops,
        duration: duration,
        operations: operations,
        elementsProcessed: arraySize,
        webgpu: true
    };
}
