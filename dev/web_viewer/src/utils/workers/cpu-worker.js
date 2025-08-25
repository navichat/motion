/**
 * CPU Worker for Task Management System
 * Handles CPU-intensive computations
 */

// Worker thread context
const isWorkerContext = typeof importScripts === 'function';

if (isWorkerContext) {
    // Running in Web Worker context
    let currentJob = null;
    let shouldStop = false;

    // Import dependencies if needed
    // importScripts('../MockGPUJobs.js');

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
            default:
                console.warn('Unknown message type:', type);
        }
    });

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
            const job = createMockJob(jobType, duration, complexity);
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

    function createMockJob(type, duration, complexity) {
        // Simple mock job implementation for worker context
        return {
            type: type,
            duration: duration,
            complexity: complexity,
            progress: 0,
            
            async execute(progressCallback, shouldStopCallback) {
                const startTime = performance.now();
                const totalSteps = Math.max(10, complexity * 10);
                const stepDuration = duration / totalSteps;

                for (let step = 0; step < totalSteps; step++) {
                    if (shouldStopCallback && shouldStopCallback()) {
                        return null;
                    }

                    // Simulate work based on job type
                    await this.simulateWork(stepDuration, type, complexity);
                    
                    this.progress = Math.round((step + 1) / totalSteps * 100);
                    
                    if (progressCallback) {
                        progressCallback(this.progress, {
                            step: step + 1,
                            totalSteps: totalSteps,
                            type: type
                        });
                    }
                }

                return {
                    type: type,
                    executionTime: performance.now() - startTime,
                    complexity: complexity,
                    success: true
                };
            },

            async simulateWork(duration, jobType, complexity) {
                const startTime = performance.now();
                
                switch (jobType) {
                    case 'JobA':
                        await this.simulateMatrixOperations(duration, complexity);
                        break;
                    case 'JobB':
                        await this.simulateNeuralNetwork(duration, complexity);
                        break;
                    case 'JobC':
                        await this.simulateMediaProcessing(duration, complexity);
                        break;
                    default:
                        await this.sleep(duration);
                }
            },

            async simulateMatrixOperations(duration, complexity) {
                const startTime = performance.now();
                const size = Math.min(50, complexity * 10); // Smaller for worker
                
                // Generate matrices
                const matrixA = [];
                const matrixB = [];
                
                for (let i = 0; i < size; i++) {
                    matrixA[i] = new Float32Array(size);
                    matrixB[i] = new Float32Array(size);
                    for (let j = 0; j < size; j++) {
                        matrixA[i][j] = Math.random() * 2 - 1;
                        matrixB[i][j] = Math.random() * 2 - 1;
                    }
                }

                // Multiply matrices
                const result = [];
                for (let i = 0; i < size; i++) {
                    result[i] = new Float32Array(size);
                    for (let j = 0; j < size; j++) {
                        let sum = 0;
                        for (let k = 0; k < size; k++) {
                            sum += matrixA[i][k] * matrixB[k][j];
                        }
                        result[i][j] = sum;
                    }
                }

                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async simulateNeuralNetwork(duration, complexity) {
                const startTime = performance.now();
                const layers = complexity;
                const neuronCount = 32;
                
                let input = new Float32Array(neuronCount);
                for (let i = 0; i < neuronCount; i++) {
                    input[i] = Math.random() * 2 - 1;
                }

                for (let layer = 0; layer < layers; layer++) {
                    const weights = [];
                    for (let i = 0; i < neuronCount; i++) {
                        weights[i] = new Float32Array(neuronCount);
                        for (let j = 0; j < neuronCount; j++) {
                            weights[i][j] = (Math.random() - 0.5) * 2;
                        }
                    }

                    const output = new Float32Array(neuronCount);
                    for (let i = 0; i < neuronCount; i++) {
                        let sum = 0;
                        for (let j = 0; j < neuronCount; j++) {
                            sum += input[j] * weights[i][j];
                        }
                        output[i] = Math.tanh(sum); // Activation function
                    }
                    input = output;
                }

                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async simulateMediaProcessing(duration, complexity) {
                const startTime = performance.now();
                const frameCount = complexity * 10;
                const frameSize = 512;
                
                for (let frame = 0; frame < frameCount; frame++) {
                    const frameData = new Float32Array(frameSize);
                    for (let i = 0; i < frameSize; i++) {
                        frameData[i] = Math.sin(frame * i * 0.001) * Math.cos(i * 0.01);
                    }

                    // Apply simple filter
                    for (let i = 1; i < frameSize - 1; i++) {
                        frameData[i] = (frameData[i-1] + frameData[i] + frameData[i+1]) / 3;
                    }
                }

                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async sleep(ms) {
                return new Promise(resolve => setTimeout(resolve, ms));
            },

            interrupt() {
                // Handle interruption
                this.interrupted = true;
            }
        };
    }

} else {
    // Running in main thread context - provide worker wrapper
    class CPUWorker {
        constructor(workerId = 'cpu-worker') {
            this.workerId = workerId;
            this.worker = null;
            this.busy = false;
            this.currentTask = null;
            this.callbacks = {};
            this._init();
        }

        _init() {
            try {
                // Create the worker using a blob URL for self-contained execution
                const workerCode = `
                    // CPU Worker Implementation
                    self.addEventListener('message', function(e) {
                        const { taskId, taskData, command } = e.data;
                        
                        try {
                            let result;
                            switch (command) {
                                case 'execute':
                                    result = executeCPUTask(taskData);
                                    break;
                                case 'cancel':
                                    // Handle cancellation
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
                    
                    function executeCPUTask(taskData) {
                        // Simulate CPU-intensive work
                        const iterations = taskData.iterations || 1000000;
                        let result = 0;
                        
                        for (let i = 0; i < iterations; i++) {
                            result += Math.sqrt(i) * Math.sin(i);
                        }
                        
                        return { 
                            iterations, 
                            result, 
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
                    console.error('CPU Worker error:', error);
                    if (this.callbacks.onError) {
                        this.callbacks.onError(error);
                    }
                });

                // Test worker responsiveness
                this.ping();

            } catch (error) {
                console.error('Failed to create CPU worker:', error);
                throw error;
            }
        }

        _handleMessage(data) {
            const { type, taskId } = data;

            switch (type) {
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
                    console.log(`CPU Worker ${this.workerId} is responsive`);
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
                currentTask: this.currentTask ? this.currentTask.taskId : null
            };
        }
    }

    // Export for browser use
    if (typeof window !== 'undefined') {
        window.CPUWorker = CPUWorker;
    }
}
