/**
 * WASM Worker for WASM-based task execution
 * Handles WASM module loading and execution for high-performance tasks
 */

// Track active tasks
let activeTasks = new Map();

// Handle messages from main thread
self.onmessage = function(event) {
    const { type, data, capabilities } = event.data;
    
    switch (type) {
        case 'init':
            // WASM workers are always ready
            self.postMessage({
                type: 'ready',
                workerType: 'wasm',
                capabilities: { wasm: true, onnx: false } 
            });
            break;
        case 'execute':
            executeTask(data);
            break;
        case 'cancel':
            cancelTask(data.taskId);
            break;
        default:
            console.warn('Unknown message type:', type);
    }
};

async function executeTask(taskData) {
    const { taskId, jobType, duration, complexity } = taskData;
    
    console.log(`[WASM Worker] Received task: ${taskId} (${jobType})`);

    try {
        // Store active task
        activeTasks.set(taskId, { cancelled: false });
        
        // Send progress updates
        const startTime = Date.now();
        const progressInterval = setInterval(() => {
            if (activeTasks.has(taskId) && !activeTasks.get(taskId).cancelled) {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1.0);
                
                self.postMessage({
                    type: 'progress',
                    taskId: taskId,
                    progress: progress,
                    stats: {
                        elapsed: elapsed,
                        estimatedTotal: duration,
                        workerType: 'wasm'
                    }
                });
                
                if (progress >= 1.0) {
                    clearInterval(progressInterval);
                }
            }
        }, 100);
        
        // Simulate WASM computation with WebAssembly-style operations
        const result = await simulateWasmComputation(taskData);
        
        // Check if task was cancelled
        if (activeTasks.has(taskId) && activeTasks.get(taskId).cancelled) {
            clearInterval(progressInterval);
            activeTasks.delete(taskId);
            self.postMessage({
                type: 'cancelled',
                taskId: taskId
            });
            return;
        }
        
        clearInterval(progressInterval);
        activeTasks.delete(taskId);
        
        // Send completion
        console.log(`[WASM Worker] Task ${taskId} completed with result:`, result);
        self.postMessage({
            type: 'completed',
            taskId: taskId,
            result: {
                executionTime: Date.now() - startTime,
                workerType: 'wasm',
                jobType: jobType,
                wasmOptimized: true,
                memoryUsage: getMemoryUsage(),
                computeIntensity: complexity || 1,
                simulationResult: result
            }
        });
        
    } catch (error) {
        activeTasks.delete(taskId);
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: error.message
        });
    }
}

async function simulateWasmComputation(taskData) {
    const { duration = 1000, complexity = 1, jobType } = taskData;
    
    // Simulate different types of WASM computations
    const computeSteps = Math.floor(duration / 50) * complexity;
    
    for (let i = 0; i < computeSteps; i++) {
        // Check for cancellation
        if (activeTasks.has(taskData.taskId) && activeTasks.get(taskData.taskId).cancelled) {
            return;
        }
        
        // Simulate WASM-style computations based on job type
        switch (jobType) {
            case 'matrix_multiply':
                await simulateMatrixMultiply(complexity);
                break;
            case 'image_processing':
                await simulateImageProcessing(complexity);
                break;
            case 'crypto_hash':
                await simulateCryptoHash(complexity);
                break;
            case 'physics_simulation':
                await simulatePhysics(complexity);
                break;
            default:
                await simulateGenericComputation(complexity);
        }
        
        // Small delay to prevent blocking
        await new Promise(resolve => setTimeout(resolve, 10));
    }
}

async function simulateMatrixMultiply(complexity) {
    // Simulate matrix multiplication with WASM-level performance
    const size = 50 * complexity;
    const a = new Float32Array(size);
    const b = new Float32Array(size);
    const result = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
        a[i] = Math.random();
        b[i] = Math.random();
        result[i] = a[i] * b[i] + Math.sin(a[i]) * Math.cos(b[i]);
    }
    
    return result;
}

async function simulateImageProcessing(complexity) {
    // Simulate image processing operations
    const pixels = 1000 * complexity;
    const imageData = new Uint8ClampedArray(pixels * 4); // RGBA
    
    for (let i = 0; i < pixels * 4; i += 4) {
        // Apply some filters
        imageData[i] = Math.floor(Math.random() * 255);     // R
        imageData[i + 1] = Math.floor(Math.random() * 255); // G
        imageData[i + 2] = Math.floor(Math.random() * 255); // B
        imageData[i + 3] = 255; // A
        
        // Apply convolution-like operations
        const brightness = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
        imageData[i] = Math.min(255, brightness * 1.2);
        imageData[i + 1] = Math.min(255, brightness * 1.1);
        imageData[i + 2] = Math.min(255, brightness * 1.3);
    }
    
    return imageData;
}

async function simulateCryptoHash(complexity) {
    // Simulate cryptographic operations
    const data = new Uint8Array(1000 * complexity);
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.floor(Math.random() * 256);
    }
    
    // Simple hash-like operation
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
        hash = ((hash << 5) - hash + data[i]) & 0xffffffff;
        hash = hash ^ (hash >>> 16);
    }
    
    return hash;
}

async function simulatePhysics(complexity) {
    // Simulate physics calculations
    const particles = 100 * complexity;
    const positions = new Float32Array(particles * 3);
    const velocities = new Float32Array(particles * 3);
    
    for (let i = 0; i < particles * 3; i++) {
        positions[i] = Math.random() * 100;
        velocities[i] = (Math.random() - 0.5) * 10;
    }
    
    // Simulate one physics step
    const dt = 0.016; // 60 FPS
    for (let i = 0; i < particles * 3; i++) {
        positions[i] += velocities[i] * dt;
        velocities[i] *= 0.99; // Friction
    }
    
    return { positions, velocities };
}

async function simulateGenericComputation(complexity) {
    // Generic computation simulation
    let result = 0;
    const iterations = 1000 * complexity;
    
    for (let i = 0; i < iterations; i++) {
        result += Math.sin(i) * Math.cos(i * 0.1) + Math.sqrt(i + 1);
    }
    
    return result;
}

function cancelTask(taskId) {
    if (activeTasks.has(taskId)) {
        activeTasks.get(taskId).cancelled = true;
    }
}

function getMemoryUsage() {
    // Estimate memory usage (in browsers that support it)
    if (typeof performance !== 'undefined' && performance.memory) {
        return {
            used: performance.memory.usedJSHeapSize,
            total: performance.memory.totalJSHeapSize,
            limit: performance.memory.jsHeapSizeLimit
        };
    }
    
    return {
        used: 'unknown',
        total: 'unknown',
        limit: 'unknown'
    };
}

// Handle worker termination
self.addEventListener('beforeunload', () => {
    // Clean up active tasks
    activeTasks.clear();
});
