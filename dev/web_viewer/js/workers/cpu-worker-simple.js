/**
 * Simple CPU Worker - Non-module version for direct Worker instantiation
 */

// Worker message handling
self.addEventListener('message', function(event) {
    const { type, data, capabilities } = event.data;
    
    switch (type) {
        case 'init':
            // CPU workers are always ready
            self.postMessage({
                type: 'ready',
                workerType: 'cpu',
                capabilities: { 
                    cpu: true, 
                    webgpu: false,  // Explicitly set to false for CPU workers
                    onnx: false 
                } 
            });
            break;
        case 'execute':
            executeTask(data);
            break;
        case 'cancel':
            cancelTask(data.taskId);
            break;
        default:
            console.log('Unknown message type:', type);
    }
});

let currentTask = null;
let cancelled = false;

function executeTask(taskData) {
    const { taskId, jobType, duration = 1000, complexity = 1, shouldFail = false } = taskData;
    
    currentTask = taskId;
    cancelled = false; // Reset cancelled flag for new task
    
    console.log(`[CPU Worker] Received task: ${taskId} (${jobType})`);
    
    // Handle test error case
    if (shouldFail || jobType === 'ErrorTestJob') {
        setTimeout(() => {
            console.log(`[CPU Worker] Task ${taskId} is a test error, failing intentionally.`);
            self.postMessage({
                type: 'error',
                taskId: taskId,
                error: 'Intentional test error from worker'
            });
            
            // Reset current task status
            currentTask = null;
        }, 50);
        return;
    }
    
    // Handle WASM-specific job types and AI models
    if (jobType === 'WASMMatrix' || jobType === 'WASMPrime' || jobType === 'WASMFractal' || jobType === 'VAD') {
        simulateWASMWork(taskId, duration, complexity, jobType);
    } else {
        // Fallback to generic CPU simulation
        simulateCPUWork(taskId, duration, complexity);
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

async function simulateCPUWork(taskId, duration, complexity) {
    const startTime = Date.now();
    const steps = Math.max(10, Math.floor(duration / 100)); // At least 10 steps
    const stepDuration = duration / steps;
    
    console.log(`[CPU Worker] ⚙️  Starting CPU compute for task ${taskId} - Type: SIMULATED CPU, Duration: ${duration}ms, Complexity: ${complexity}`);
    
    try {
        for (let i = 0; i < steps && !cancelled; i++) {
            // Simulate CPU work based on complexity
            const workAmount = 1000 * complexity;
            const result = await simulateWork(workAmount);
            
            console.log(`[CPU Worker] 🖥️  SIMULATED CPU compute step ${i+1}/${steps} for task ${taskId}, work amount: ${workAmount}`);
            
            // Check if cancelled
            if (cancelled) {
                currentTask = null; // Reset current task when cancelled
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
                    estimated: (elapsed / (i + 1)) * steps
                }
            });
            
            // Small delay between steps
            await new Promise(resolve => setTimeout(resolve, Math.max(10, stepDuration - workAmount / 1000)));
        }
        
        if (!cancelled) {
            // Task completed successfully
            const totalTime = Date.now() - startTime;
            
            console.log(`[CPU Worker] ✅ CPU compute COMPLETED for task ${taskId} in ${totalTime}ms - Type: SIMULATED CPU`);
            
            self.postMessage({
                type: 'completed',
                taskId: taskId,
                result: {
                    success: true,
                    executionTime: totalTime,
                    workerType: 'cpu',
                    steps: steps,
                    complexity: complexity,
                    inferenceType: 'SIMULATED_CPU'
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

// Enhanced WASM work simulation for specific job types
async function simulateWASMWork(taskId, duration, complexity, jobType) {
    const startTime = Date.now();
    const steps = Math.max(8, Math.floor(duration / 125)); // WASM optimized steps
    const stepDuration = duration / steps;
    
    console.log(`[CPU Worker] 🔧 Starting WASM/AI model ${jobType} for task ${taskId} - Type: SIMULATED, Duration: ${duration}ms`);
    
    try {
        for (let i = 0; i < steps && !cancelled; i++) {
            // Simulate job-type specific WASM computation
            let workAmount;
            switch (jobType) {
                case 'WASMMatrix':
                    workAmount = await simulateMatrixMultiplication(complexity);
                    console.log(`[CPU Worker] 📊 SIMULATED WASM Matrix step ${i+1}/${steps}, operations: ${workAmount}`);
                    break;
                case 'WASMPrime':
                    workAmount = await simulatePrimeComputation(complexity);
                    console.log(`[CPU Worker] 🔢 SIMULATED WASM Prime step ${i+1}/${steps}, primes found: ${workAmount}`);
                    break;
                case 'WASMFractal':
                    workAmount = await simulateFractalGeneration(complexity);
                    console.log(`[CPU Worker] 🌀 SIMULATED WASM Fractal step ${i+1}/${steps}, iterations: ${workAmount}`);
                    break;
                case 'VAD':
                    workAmount = await simulateVAD(complexity);
                    console.log(`[CPU Worker] 🎙️  SIMULATED VAD step ${i+1}/${steps}, voice activity: ${workAmount ? workAmount.toFixed(4) : 'N/A'}`);
                    break;
                default:
                    workAmount = await simulateWork(1000 * complexity);
                    console.log(`[CPU Worker] ⚙️  SIMULATED CPU work step ${i+1}/${steps}, work amount: ${workAmount}`);
            }
            
            // Check if cancelled
            if (cancelled) {
                currentTask = null; // Reset current task when cancelled
                return;
            }
            
            // Report progress with WASM-specific stats
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
                    jobType: jobType,
                    workAmount: workAmount,
                    processingType: 'WASM CPU'
                }
            });
            
            // Small delay between steps
            await new Promise(resolve => setTimeout(resolve, Math.max(10, stepDuration - 50)));
        }
        
        if (!cancelled) {
            // Task completed successfully
            const totalTime = Date.now() - startTime;
            
            console.log(`[CPU Worker] ✅ WASM/AI Model ${jobType} COMPLETED for task ${taskId} in ${totalTime}ms - Type: SIMULATED`);
            
            self.postMessage({
                type: 'completed',
                taskId: taskId,
                result: {
                    success: true,
                    executionTime: totalTime,
                    workerType: 'cpu',
                    steps: steps,
                    complexity: complexity,
                    jobType: jobType,
                    wasmOptimized: true,
                    inferenceType: 'SIMULATED_WASM'
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

// WASM job-specific simulation functions
async function simulateMatrixMultiplication(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const size = 32 * complexity;
            let result = 0;
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    result += Math.sqrt(i * j + 1);
                }
            }
            resolve(result);
        }, 15);
    });
}

async function simulatePrimeComputation(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const limit = 1000 * complexity;
            let primeCount = 0;
            for (let n = 2; n < limit; n++) {
                let isPrime = true;
                for (let i = 2; i <= Math.sqrt(n); i++) {
                    if (n % i === 0) {
                        isPrime = false;
                        break;
                    }
                }
                if (isPrime) primeCount++;
            }
            resolve(primeCount);
        }, 20);
    });
}

async function simulateFractalGeneration(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const iterations = 100 * complexity;
            let fractalPoints = 0;
            for (let i = 0; i < iterations; i++) {
                let x = (i % 50) / 25.0 - 1.0;
                let y = Math.floor(i / 50) / 25.0 - 1.0;
                let zx = 0, zy = 0;
                let iter = 0;
                while (zx * zx + zy * zy < 4 && iter < 100) {
                    let tmp = zx * zx - zy * zy + x;
                    zy = 2 * zx * zy + y;
                    zx = tmp;
                    iter++;
                }
                if (iter === 100) fractalPoints++;
            }
            resolve(fractalPoints);
        }, 25);
    });
}

async function simulateVAD(complexity) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate Voice Activity Detection processing
            const audioFrames = 1000 * complexity; // Number of audio frames
            const frameSize = 160; // Samples per frame (10ms at 16kHz)
            let voiceActivityScore = 0;
            
            for (let frame = 0; frame < audioFrames; frame++) {
                let energy = 0;
                let zeroCrossings = 0;
                let prevSample = 0;
                
                // Simulate frame-level VAD features
                for (let sample = 0; sample < frameSize; sample++) {
                    // Simulate audio sample
                    const audioSample = Math.sin(frame * 0.1 + sample * 0.01) + 
                                       Math.random() * 0.1 - 0.05; // Add noise
                    
                    // Energy calculation
                    energy += audioSample * audioSample;
                    
                    // Zero crossing rate
                    if ((audioSample > 0 && prevSample <= 0) || 
                        (audioSample <= 0 && prevSample > 0)) {
                        zeroCrossings++;
                    }
                    prevSample = audioSample;
                }
                
                // Combine features for VAD decision
                const energyNorm = energy / frameSize;
                const zcrNorm = zeroCrossings / frameSize;
                
                // Simple VAD threshold logic
                if (energyNorm > 0.01 && zcrNorm < 0.3) {
                    voiceActivityScore += 1.0;
                } else {
                    voiceActivityScore += 0.1; // Background noise
                }
            }
            
            resolve(voiceActivityScore / audioFrames);
        }, 30 + complexity * 20);
    });
}

// Simulate CPU-intensive work
async function simulateWork(iterations) {
    return new Promise((resolve) => {
        // Use setTimeout to avoid blocking the event loop completely
        setTimeout(() => {
            let sum = 0;
            for (let i = 0; i < iterations; i++) {
                sum += Math.sqrt(i) * Math.sin(i);
            }
            resolve(sum);
        }, 0);
    });
}


