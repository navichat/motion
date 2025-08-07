/**
 * GPU Worker for Task Management System
 * Handles WebGPU computations
 */

// Worker thread context
const isWorkerContext = typeof importScripts === 'function';

if (isWorkerContext) {
    // Running in Web Worker context
    let currentJob = null;
    let shouldStop = false;
    let gpuDevice = null;

    self.addEventListener('message', async function(e) {
        const { type, data } = e.data;

        switch (type) {
            case 'execute':
                await executeJob(data);
                break;
            case 'stop':
                stopCurrentJob();
                break;
            case 'ping':
                self.postMessage({ type: 'pong', workerId: data.workerId });
                break;
            case 'init-gpu':
                await initializeGPU();
                break;
            default:
                console.warn('Unknown message type:', type);
        }
    });

    async function initializeGPU() {
        try {
            if (!navigator.gpu) {
                throw new Error('WebGPU is not supported');
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error('Failed to get WebGPU adapter');
            }

            gpuDevice = await adapter.requestDevice();
            
            self.postMessage({
                type: 'gpu-initialized',
                success: true,
                info: {
                    limits: gpuDevice.limits,
                    features: Array.from(gpuDevice.features)
                }
            });
        } catch (error) {
            self.postMessage({
                type: 'gpu-initialized',
                success: false,
                error: error.message
            });
        }
    }

    async function executeJob(jobData) {
        try {
            shouldStop = false;
            const { jobType, duration, complexity, taskId } = jobData;
            
            self.postMessage({
                type: 'started',
                taskId: taskId,
                timestamp: Date.now()
            });

            // Create and execute the job
            const job = createGPUJob(jobType, duration, complexity);
            currentJob = job;

            const progressCallback = (progress, stats) => {
                self.postMessage({
                    type: 'progress',
                    taskId: taskId,
                    progress: progress,
                    stats: stats
                });
            };

            const shouldStopCallback = () => shouldStop;

            const result = await job.execute(progressCallback, shouldStopCallback);

            if (!shouldStop) {
                self.postMessage({
                    type: 'completed',
                    taskId: taskId,
                    result: result,
                    timestamp: Date.now()
                });
            } else {
                self.postMessage({
                    type: 'cancelled',
                    taskId: taskId,
                    timestamp: Date.now()
                });
            }

        } catch (error) {
            self.postMessage({
                type: 'error',
                taskId: jobData.taskId,
                error: error.message,
                timestamp: Date.now()
            });
        } finally {
            currentJob = null;
        }
    }

    function stopCurrentJob() {
        shouldStop = true;
        if (currentJob) {
            currentJob.interrupt();
        }
    }

    function createGPUJob(type, duration, complexity) {
        return {
            type: type,
            duration: duration,
            complexity: complexity,
            progress: 0,
            
            async execute(progressCallback, shouldStopCallback) {
                const startTime = performance.now();
                const totalSteps = Math.max(5, complexity * 5); // Fewer steps for GPU
                const stepDuration = duration / totalSteps;

                for (let step = 0; step < totalSteps; step++) {
                    if (shouldStopCallback && shouldStopCallback()) {
                        return null;
                    }

                    // Simulate GPU work
                    await this.simulateGPUWork(stepDuration, type, complexity);
                    
                    this.progress = Math.round((step + 1) / totalSteps * 100);
                    
                    if (progressCallback) {
                        progressCallback(this.progress, {
                            step: step + 1,
                            totalSteps: totalSteps,
                            type: type,
                            gpuUtilization: Math.random() * 0.3 + 0.7 // 70-100%
                        });
                    }
                }

                return {
                    type: type,
                    executionTime: performance.now() - startTime,
                    complexity: complexity,
                    gpuMemoryUsed: Math.floor(Math.random() * 1000) + 500, // MB
                    computeUnitsUsed: complexity * 100,
                    success: true
                };
            },

            async simulateGPUWork(duration, jobType, complexity) {
                const startTime = performance.now();
                
                if (gpuDevice) {
                    // Use actual WebGPU if available
                    await this.performWebGPUComputation(jobType, complexity);
                } else {
                    // Fallback to CPU simulation
                    switch (jobType) {
                        case 'JobA':
                            await this.simulateParallelMatrixOps(duration, complexity);
                            break;
                        case 'JobB':
                            await this.simulateGPUNeuralNetwork(duration, complexity);
                            break;
                        case 'JobC':
                            await this.simulateShaderProcessing(duration, complexity);
                            break;
                        default:
                            await this.sleep(duration);
                    }
                }
                
                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async performWebGPUComputation(jobType, complexity) {
                try {
                    // Simple WebGPU compute shader example
                    const shaderCode = `
                        @group(0) @binding(0) var<storage, read> input: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                        
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                            let index = global_id.x;
                            if (index >= arrayLength(&input)) {
                                return;
                            }
                            
                            let value = input[index];
                            output[index] = value * value + sin(value) * cos(value);
                        }
                    `;

                    const computeShader = gpuDevice.createShaderModule({
                        code: shaderCode
                    });

                    // Create buffers
                    const dataSize = complexity * 1000;
                    const inputData = new Float32Array(dataSize);
                    for (let i = 0; i < dataSize; i++) {
                        inputData[i] = Math.random() * 2 - 1;
                    }

                    const inputBuffer = gpuDevice.createBuffer({
                        size: inputData.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    });

                    const outputBuffer = gpuDevice.createBuffer({
                        size: inputData.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                    });

                    // Write input data
                    gpuDevice.queue.writeBuffer(inputBuffer, 0, inputData);

                    // Create compute pipeline
                    const computePipeline = gpuDevice.createComputePipeline({
                        layout: 'auto',
                        compute: {
                            module: computeShader,
                            entryPoint: 'main',
                        },
                    });

                    // Create bind group
                    const bindGroup = gpuDevice.createBindGroup({
                        layout: computePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: inputBuffer } },
                            { binding: 1, resource: { buffer: outputBuffer } },
                        ],
                    });

                    // Dispatch compute
                    const commandEncoder = gpuDevice.createCommandEncoder();
                    const passEncoder = commandEncoder.beginComputePass();
                    passEncoder.setPipeline(computePipeline);
                    passEncoder.setBindGroup(0, bindGroup);
                    passEncoder.dispatchWorkgroups(Math.ceil(dataSize / 64));
                    passEncoder.end();

                    gpuDevice.queue.submit([commandEncoder.finish()]);
                    await gpuDevice.queue.onSubmittedWorkDone();

                    // Cleanup
                    inputBuffer.destroy();
                    outputBuffer.destroy();

                } catch (error) {
                    console.warn('WebGPU computation failed, falling back to CPU:', error);
                    await this.simulateParallelMatrixOps(100, complexity);
                }
            },

            async simulateParallelMatrixOps(duration, complexity) {
                const startTime = performance.now();
                const size = Math.min(100, complexity * 20);
                
                // Simulate parallel matrix operations
                const promises = [];
                const numBatches = Math.min(8, complexity);
                
                for (let batch = 0; batch < numBatches; batch++) {
                    promises.push(this.matrixBatch(size, batch));
                }
                
                await Promise.all(promises);
                
                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async matrixBatch(size, batchId) {
                return new Promise(resolve => {
                    setTimeout(() => {
                        const matrixA = new Float32Array(size * size);
                        const matrixB = new Float32Array(size * size);
                        const result = new Float32Array(size * size);
                        
                        // Fill with random data
                        for (let i = 0; i < size * size; i++) {
                            matrixA[i] = Math.random() * 2 - 1;
                            matrixB[i] = Math.random() * 2 - 1;
                        }
                        
                        // Compute
                        for (let i = 0; i < size; i++) {
                            for (let j = 0; j < size; j++) {
                                let sum = 0;
                                for (let k = 0; k < size; k++) {
                                    sum += matrixA[i * size + k] * matrixB[k * size + j];
                                }
                                result[i * size + j] = sum;
                            }
                        }
                        
                        resolve(result);
                    }, 1);
                });
            },

            async simulateGPUNeuralNetwork(duration, complexity) {
                const startTime = performance.now();
                const layers = complexity;
                const batchSize = 32;
                const hiddenSize = 256;
                
                // Simulate parallel neural network processing
                const promises = [];
                
                for (let batch = 0; batch < batchSize; batch++) {
                    promises.push(this.processBatch(layers, hiddenSize, batch));
                }
                
                await Promise.all(promises);
                
                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async processBatch(layers, hiddenSize, batchId) {
                return new Promise(resolve => {
                    setTimeout(() => {
                        let input = new Float32Array(hiddenSize);
                        for (let i = 0; i < hiddenSize; i++) {
                            input[i] = Math.random() * 2 - 1;
                        }
                        
                        for (let layer = 0; layer < layers; layer++) {
                            const output = new Float32Array(hiddenSize);
                            for (let i = 0; i < hiddenSize; i++) {
                                let sum = 0;
                                for (let j = 0; j < hiddenSize; j++) {
                                    sum += input[j] * (Math.random() - 0.5);
                                }
                                output[i] = Math.tanh(sum);
                            }
                            input = output;
                        }
                        
                        resolve(input);
                    }, 1);
                });
            },

            async simulateShaderProcessing(duration, complexity) {
                const startTime = performance.now();
                const frameCount = complexity * 5;
                const textureSize = 256;
                
                // Simulate fragment shader processing
                const promises = [];
                
                for (let frame = 0; frame < frameCount; frame++) {
                    promises.push(this.processFrame(textureSize, frame));
                }
                
                await Promise.all(promises);
                
                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async processFrame(textureSize, frameId) {
                return new Promise(resolve => {
                    setTimeout(() => {
                        const pixels = new Uint8Array(textureSize * textureSize * 4);
                        
                        for (let y = 0; y < textureSize; y++) {
                            for (let x = 0; x < textureSize; x++) {
                                const idx = (y * textureSize + x) * 4;
                                const u = x / textureSize;
                                const v = y / textureSize;
                                const time = frameId * 0.01;
                                
                                // Simple shader effect
                                pixels[idx] = Math.floor(128 + 127 * Math.sin(u * 10 + time));
                                pixels[idx + 1] = Math.floor(128 + 127 * Math.cos(v * 10 + time));
                                pixels[idx + 2] = Math.floor(128 + 127 * Math.sin(u * v * 20 + time));
                                pixels[idx + 3] = 255;
                            }
                        }
                        
                        resolve(pixels);
                    }, 1);
                });
            },

            async sleep(ms) {
                return new Promise(resolve => setTimeout(resolve, ms));
            },

            interrupt() {
                this.interrupted = true;
            }
        };
    }

    // Initialize GPU on worker startup
    initializeGPU();

} else {
    // Running in main thread context - provide worker wrapper
    class GPUWorker {
        constructor(workerId = 'gpu-worker') {
            this.workerId = workerId;
            this.worker = null;
            this.busy = false;
            this.currentTask = null;
            this.callbacks = {};
            this.gpuInitialized = false;
            this._init();
        }

        _init() {
            try {
                // Create the worker using a blob URL for self-contained execution
                const workerCode = `
                    // GPU Worker Implementation (WebGL simulation)
                    self.addEventListener('message', function(e) {
                        const { taskId, taskData, command } = e.data;
                        
                        try {
                            let result;
                            switch (command) {
                                case 'execute':
                                    result = executeGPUTask(taskData);
                                    break;
                                case 'cancel':
                                    self.postMessage({ taskId, status: 'cancelled' });
                                    return;
                                default:
                                    throw new Error('Unknown command: ' + command);
                            }
                            
                            self.postMessage({ 
                                taskId, 
                                status: 'completed', 
                                result 
                            });
                            
                        } catch (error) {
                            self.postMessage({ 
                                taskId, 
                                status: 'error', 
                                error: error.message 
                            });
                        }
                    });
                    
                    function executeGPUTask(taskData) {
                        // Simulate GPU-like parallel computation
                        const width = taskData.width || 512;
                        const height = taskData.height || 512;
                        const channels = taskData.channels || 4;
                        
                        // Simulate matrix operations
                        const dataSize = width * height * channels;
                        const buffer = new Float32Array(dataSize);
                        
                        for (let i = 0; i < dataSize; i += 4) {
                            // Simulate RGBA processing
                            buffer[i] = Math.random();     // R
                            buffer[i + 1] = Math.random(); // G  
                            buffer[i + 2] = Math.random(); // B
                            buffer[i + 3] = 1.0;           // A
                        }
                        
                        return { 
                            width, 
                            height, 
                            channels,
                            processedPixels: dataSize / 4,
                            timestamp: Date.now() 
                        };
                    }
                `;
                
                const blob = new Blob([workerCode], { type: 'application/javascript' });
                this.worker = new Worker(URL.createObjectURL(blob));
                
                this.worker.addEventListener('message', (e) => {
                    this._handleMessage(e.data);
                });

                this.worker.addEventListener('error', (error) => {
                    console.error('GPU Worker error:', error);
                    if (this.callbacks.onError) {
                        this.callbacks.onError(error);
                    }
                });

                // Test worker responsiveness
                this.ping();

            } catch (error) {
                console.error('Failed to create GPU worker:', error);
                throw error;
            }
        }

        _handleMessage(data) {
            const { type, taskId } = data;

            switch (type) {
                case 'gpu-initialized':
                    this.gpuInitialized = data.success;
                    if (data.success) {
                        console.log('GPU initialized in worker:', data.info);
                    } else {
                        console.warn('GPU initialization failed:', data.error);
                    }
                    break;
                case 'started':
                    if (this.callbacks.onStarted) {
                        this.callbacks.onStarted(data);
                    }
                    break;
                case 'progress':
                    if (this.callbacks.onProgress) {
                        this.callbacks.onProgress(data.progress, data.stats);
                    }
                    break;
                case 'completed':
                    this.busy = false;
                    this.currentTask = null;
                    if (this.callbacks.onCompleted) {
                        this.callbacks.onCompleted(data.result);
                    }
                    break;
                case 'cancelled':
                    this.busy = false;
                    this.currentTask = null;
                    if (this.callbacks.onCancelled) {
                        this.callbacks.onCancelled();
                    }
                    break;
                case 'error':
                    this.busy = false;
                    this.currentTask = null;
                    if (this.callbacks.onError) {
                        this.callbacks.onError(new Error(data.error));
                    }
                    break;
                case 'pong':
                    console.log(`GPU Worker ${this.workerId} is responsive`);
                    break;
            }
        }

        async execute(jobData, callbacks = {}) {
            if (this.busy) {
                throw new Error('Worker is busy');
            }

            this.busy = true;
            this.currentTask = jobData;
            this.callbacks = callbacks;

            return new Promise((resolve, reject) => {
                const originalCallbacks = { ...callbacks };
                
                this.callbacks.onCompleted = (result) => {
                    if (originalCallbacks.onCompleted) {
                        originalCallbacks.onCompleted(result);
                    }
                    resolve(result);
                };

                this.callbacks.onError = (error) => {
                    if (originalCallbacks.onError) {
                        originalCallbacks.onError(error);
                    }
                    reject(error);
                };

                this.callbacks.onCancelled = () => {
                    if (originalCallbacks.onCancelled) {
                        originalCallbacks.onCancelled();
                    }
                    resolve(null);
                };

                this.worker.postMessage({
                    type: 'execute',
                    data: jobData
                });
            });
        }

        stop() {
            if (this.busy) {
                this.worker.postMessage({ type: 'stop' });
            }
        }

        ping() {
            this.worker.postMessage({
                type: 'ping',
                data: { workerId: this.workerId }
            });
        }

        terminate() {
            if (this.worker) {
                this.worker.terminate();
                this.worker = null;
            }
            this.busy = false;
            this.currentTask = null;
        }

        getStatus() {
            return {
                workerId: this.workerId,
                busy: this.busy,
                currentTask: this.currentTask ? this.currentTask.taskId : null,
                gpuInitialized: this.gpuInitialized
            };
        }
    }

    // Export for browser use
    if (typeof window !== 'undefined') {
        window.GPUWorker = GPUWorker;
    }
}
