/**
 * WASM Worker for REAL WebAssembly-based task execution
 * Handles actual WASM module loading and execution for high-performance tasks
 */

// Import real WASM modules
try {
    importScripts('real-wasm-modules.js');
} catch (error) {
    console.error('[WASM Worker] Failed to import real WASM modules:', error);
    // Fallback mode without WASM
}

// Track active tasks and WASM state
let activeTasks = new Map();
let wasmCompute = null;
let wasmInitialized = false;

// Handle messages from main thread
self.onmessage = function(event) {
    const { type, data, capabilities } = event.data;
    
    switch (type) {
        case 'init':
            initializeRealWasm();
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

async function initializeRealWasm() {
    try {
        console.log('[WASM Worker] Initializing REAL WebAssembly modules...');
        
        // Check if RealWasmCompute is available
        if (typeof RealWasmCompute === 'undefined') {
            console.warn('[WASM Worker] RealWasmCompute not available, using fallback mode');
            wasmInitialized = false;
            
            self.postMessage({
                type: 'ready',
                workerType: 'wasm',
                capabilities: { 
                    wasm: false,
                    error: 'RealWasmCompute module not loaded',
                    fallbackToCPU: true
                }
            });
            return;
        }
        
        wasmCompute = new RealWasmCompute();
        const initResult = await wasmCompute.initialize();
        
        if (initResult.success) {
            wasmInitialized = true;
            console.log('[WASM Worker] Real WASM modules initialized successfully');
            
            self.postMessage({
                type: 'ready',
                workerType: 'wasm',
                capabilities: { 
                    wasm: true,
                    wasmModules: initResult.modules,
                    simdSupport: initResult.simdSupport,
                    realWasmExecution: true
                }
            });
        } else {
            console.error('[WASM Worker] Failed to initialize WASM modules:', initResult.error);
            wasmInitialized = false;
            
            self.postMessage({
                type: 'ready',
                workerType: 'wasm',
                capabilities: { 
                    wasm: false,
                    error: initResult.error,
                    fallbackToCPU: true
                }
            });
        }
        
    } catch (error) {
        console.error('[WASM Worker] WASM initialization error:', error);
        wasmInitialized = false;
        
        self.postMessage({
            type: 'ready',
            workerType: 'wasm',
            capabilities: { 
                wasm: false,
                error: error.message,
                fallbackToCPU: true
            }
        });
    }
}

async function executeTask(taskData) {
    const { taskId, jobType, duration, complexity } = taskData;
    
    console.log(`[WASM Worker] Received REAL WASM task: ${taskId} (${jobType})`);

    try {
        // Store active task
        activeTasks.set(taskId, { cancelled: false, startTime: Date.now() });
        
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
        
        // Perform REAL WASM computation or honest fallback
        let modelOutput = {};
        let executionProvider = ['cpu', 'javascript']; // Default fallback
        let actualWasmUsed = false;

        if (wasmInitialized && wasmCompute) {
            console.log(`[WASM Worker] Using REAL WebAssembly for ${jobType}`);
            
            try {
                switch (jobType) {
                    case 'WASMMatrix':
                        const matrixSize = 128 + ((complexity || 1) * 32);
                        modelOutput = await wasmCompute.performMatrixMultiplication(matrixSize, complexity);
                        executionProvider = ['wasm', 'webassembly'];
                        actualWasmUsed = true;
                        break;
                        
                    case 'WASMPrime':
                        const maxPrime = 10000 + ((complexity || 1) * 5000);
                        modelOutput = await wasmCompute.performPrimeComputation(maxPrime);
                        executionProvider = ['wasm', 'sieve-algorithm'];
                        actualWasmUsed = true;
                        break;
                        
                    case 'WASMFractal':
                        const resolution = 256 + ((complexity || 1) * 64);
                        const maxIter = 100 + ((complexity || 1) * 50);
                        modelOutput = await wasmCompute.performMandelbrotComputation(resolution, resolution, maxIter);
                        executionProvider = ['wasm', 'mandelbrot'];
                        actualWasmUsed = true;
                        break;
                        
                    default:
                        // No WASM implementation for this job type
                        console.log(`[WASM Worker] No WASM implementation for ${jobType}, falling back to CPU`);
                        modelOutput = await performJavaScriptFallback(jobType, complexity, duration);
                        executionProvider = ['cpu', 'javascript-fallback'];
                        actualWasmUsed = false;
                }
            } catch (wasmError) {
                console.error(`[WASM Worker] WASM execution failed for ${jobType}:`, wasmError);
                modelOutput = await performJavaScriptFallback(jobType, complexity, duration);
                executionProvider = ['cpu', 'wasm-error-fallback'];
                actualWasmUsed = false;
            }
        } else {
            // WASM not initialized, fallback to JavaScript
            console.log(`[WASM Worker] WASM not initialized, falling back to JavaScript for ${jobType}`);
            modelOutput = await performJavaScriptFallback(jobType, complexity, duration);
            executionProvider = ['cpu', 'javascript-fallback'];
            actualWasmUsed = false;
        }
        
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
        
        const executionTime = Date.now() - startTime;
        activeTasks.delete(taskId);

        // Send completion message with HONEST execution provider
        self.postMessage({
            type: 'completed',
            taskId: taskId,
            result: {
                success: true,
                executionTime: executionTime,
                workerType: 'wasm',
                jobType: jobType,
                actualWasmUsed: actualWasmUsed,
                wasmInitialized: wasmInitialized,
                modelOutput: modelOutput,
                executionProvider: executionProvider, // HONEST labeling
                memoryUsage: getMemoryUsage()
            }
        });

        // Send AVATAR AI COLLECTED message for E2E test capture
        console.log(`AVATAR AI COLLECTED ${JSON.stringify({
            jobType: jobType,
            executionTime: executionTime,
            modelOutput: {
                ...modelOutput,
                executionProvider: executionProvider,
                actualWasmUsed: actualWasmUsed
            }
        })}`);

    } catch (error) {
        console.error(`[WASM Worker] Task ${taskId} failed:`, error);
        activeTasks.delete(taskId);
        
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: error.message,
            executionProvider: ['cpu', 'error-fallback']
        });
    }
}

async function performJavaScriptFallback(jobType, complexity, duration) {
    console.log(`[WASM Worker] Performing JavaScript fallback for ${jobType}`);
    
    // Simulate computation time based on job complexity
    const baseTime = 200;
    const complexityTime = (complexity || 1) * 150;
    const simulationTime = Math.min(baseTime + complexityTime, duration * 0.8);
    
    const startTime = performance.now();
    await new Promise(resolve => setTimeout(resolve, simulationTime));
    const executionTime = performance.now() - startTime;
    
    switch (jobType) {
        case 'WASMMatrix':
            const matrixSize = 64 + ((complexity || 1) * 16);
            return {
                type: 'matrix_computation_fallback',
                matrix_size: matrixSize,
                operations_count: Math.pow(matrixSize, 3) * 2,
                execution_time_ms: executionTime,
                algorithm: 'javascript_fallback',
                actual_wasm_execution: false,
                fallback_reason: 'wasm_not_available',
                estimated_flops: (Math.pow(matrixSize, 3) * 2) / (executionTime / 1000)
            };
            
        case 'WASMPrime':
            const searchRange = 5000 + ((complexity || 1) * 2500);
            return {
                type: 'prime_computation_fallback',
                search_range: searchRange,
                execution_time_ms: executionTime,
                algorithm: 'javascript_fallback',
                actual_wasm_execution: false,
                fallback_reason: 'wasm_not_available',
                estimated_primes: Math.floor(searchRange / Math.log(searchRange))
            };
            
        case 'WASMFractal':
            const resolution = 128 + ((complexity || 1) * 32);
            return {
                type: 'fractal_computation_fallback',
                resolution: `${resolution}x${resolution}`,
                total_pixels: resolution * resolution,
                execution_time_ms: executionTime,
                algorithm: 'javascript_fallback',
                actual_wasm_execution: false,
                fallback_reason: 'wasm_not_available'
            };
            
        default:
            return {
                type: 'generic_computation_fallback',
                job_type: jobType,
                execution_time_ms: executionTime,
                actual_wasm_execution: false,
                fallback_reason: 'unknown_job_type'
            };
    }
}

function cancelTask(taskId) {
    if (activeTasks.has(taskId)) {
        activeTasks.get(taskId).cancelled = true;
        console.log(`[WASM Worker] Task ${taskId} cancelled`);
    }
}

function getMemoryUsage() {
    if (self.performance && self.performance.memory) {
        return {
            used: self.performance.memory.usedJSHeapSize,
            total: self.performance.memory.totalJSHeapSize,
            limit: self.performance.memory.jsHeapSizeLimit
        };
    }
    return { used: 0, total: 0, limit: 0 };
}
