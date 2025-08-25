/**
 * WebNN Worker for Task Management System
 * Handles WebNN (Web Neural Network API) computations
 */

// Worker thread context
const isWorkerContext = typeof importScripts === 'function';

if (isWorkerContext) {
    // Running in Web Worker context
    let currentJob = null;
    let shouldStop = false;
    let webnnContext = null;

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
            case 'init-webnn':
                await initializeWebNN();
                break;
            default:
                console.warn('Unknown message type:', type);
        }
    });

    async function initializeWebNN() {
        try {
            // Check for WebNN availability
            if (typeof navigator !== 'undefined' && navigator.ml) {
                webnnContext = await navigator.ml.createContext();
                
                self.postMessage({
                    type: 'webnn-initialized',
                    success: true,
                    info: {
                        contextType: 'webnn',
                        features: ['basic-ops', 'neural-networks']
                    }
                });
            } else {
                throw new Error('WebNN is not supported');
            }
        } catch (error) {
            self.postMessage({
                type: 'webnn-initialized',
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
            const job = createWebNNJob(jobType, duration, complexity);
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

    function createWebNNJob(type, duration, complexity) {
        return {
            type: type,
            duration: duration,
            complexity: complexity,
            progress: 0,
            
            async execute(progressCallback, shouldStopCallback) {
                const startTime = performance.now();
                const totalSteps = Math.max(3, complexity * 3); // Fewer steps for neural networks
                const stepDuration = duration / totalSteps;

                for (let step = 0; step < totalSteps; step++) {
                    if (shouldStopCallback && shouldStopCallback()) {
                        return null;
                    }

                    // Simulate WebNN work
                    await this.simulateWebNNWork(stepDuration, type, complexity);
                    
                    this.progress = Math.round((step + 1) / totalSteps * 100);
                    
                    if (progressCallback) {
                        progressCallback(this.progress, {
                            step: step + 1,
                            totalSteps: totalSteps,
                            type: type,
                            inferenceTime: Math.random() * 50 + 10, // 10-60ms
                            modelAccuracy: Math.random() * 0.1 + 0.9 // 90-100%
                        });
                    }
                }

                return {
                    type: type,
                    executionTime: performance.now() - startTime,
                    complexity: complexity,
                    inferenceCount: totalSteps * complexity,
                    averageInferenceTime: (performance.now() - startTime) / (totalSteps * complexity),
                    modelPrecision: 'float32',
                    success: true
                };
            },

            async simulateWebNNWork(duration, jobType, complexity) {
                const startTime = performance.now();
                
                if (webnnContext) {
                    // Use actual WebNN if available
                    await this.performWebNNInference(jobType, complexity);
                } else {
                    // Fallback to simulated neural network operations
                    switch (jobType) {
                        case 'JobA':
                            await this.simulateConvolutionalNetwork(duration, complexity);
                            break;
                        case 'JobB':
                            await this.simulateTransformerNetwork(duration, complexity);
                            break;
                        case 'JobC':
                            await this.simulateRecurrentNetwork(duration, complexity);
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

            async performWebNNInference(jobType, complexity) {
                try {
                    // Simulate WebNN model creation and inference
                    const builder = new MLGraphBuilder(webnnContext);
                    
                    // Create a simple neural network graph
                    const inputShape = [1, complexity * 10]; // Batch size 1, features
                    const hiddenSize = Math.min(128, complexity * 32);
                    const outputSize = Math.min(10, complexity);
                    
                    // Input layer
                    const input = builder.input('input', { type: 'float32', dimensions: inputShape });
                    
                    // Dense layer 1
                    const weights1Shape = [inputShape[1], hiddenSize];
                    const weights1 = builder.constant({ type: 'float32', dimensions: weights1Shape }, 
                        new Float32Array(weights1Shape[0] * weights1Shape[1]).map(() => Math.random() - 0.5));
                    
                    const bias1Shape = [hiddenSize];
                    const bias1 = builder.constant({ type: 'float32', dimensions: bias1Shape },
                        new Float32Array(hiddenSize).fill(0));
                    
                    const dense1 = builder.add(builder.matmul(input, weights1), bias1);
                    const activation1 = builder.relu(dense1);
                    
                    // Dense layer 2 (output)
                    const weights2Shape = [hiddenSize, outputSize];
                    const weights2 = builder.constant({ type: 'float32', dimensions: weights2Shape },
                        new Float32Array(weights2Shape[0] * weights2Shape[1]).map(() => Math.random() - 0.5));
                    
                    const bias2Shape = [outputSize];
                    const bias2 = builder.constant({ type: 'float32', dimensions: bias2Shape },
                        new Float32Array(outputSize).fill(0));
                    
                    const dense2 = builder.add(builder.matmul(activation1, weights2), bias2);
                    const output = builder.softmax(dense2);
                    
                    // Build the graph
                    const graph = await builder.build({ 'output': output });
                    
                    // Prepare input data
                    const inputData = new Float32Array(inputShape[0] * inputShape[1]);
                    for (let i = 0; i < inputData.length; i++) {
                        inputData[i] = Math.random() * 2 - 1;
                    }
                    
                    // Run inference
                    const results = await webnnContext.compute(graph, { 'input': inputData });
                    
                    return results['output'];
                    
                } catch (error) {
                    console.warn('WebNN inference failed, falling back to simulation:', error);
                    await this.simulateConvolutionalNetwork(100, complexity);
                }
            },

            async simulateConvolutionalNetwork(duration, complexity) {
                const startTime = performance.now();
                
                // Simulate CNN layers
                const inputSize = 32; // 32x32 input
                const channels = 3;
                const numFilters = Math.min(64, complexity * 16);
                const filterSize = 3;
                
                // Convolution operation simulation
                const input = new Float32Array(inputSize * inputSize * channels);
                for (let i = 0; i < input.length; i++) {
                    input[i] = Math.random() * 2 - 1;
                }
                
                // Simulate multiple conv layers
                const layers = Math.min(5, complexity);
                let currentFeatures = input;
                let currentSize = inputSize;
                
                for (let layer = 0; layer < layers; layer++) {
                    const outputSize = currentSize - filterSize + 1;
                    const outputFeatures = new Float32Array(outputSize * outputSize * numFilters);
                    
                    // Convolution (simplified)
                    for (let f = 0; f < numFilters; f++) {
                        for (let y = 0; y < outputSize; y++) {
                            for (let x = 0; x < outputSize; x++) {
                                let sum = 0;
                                for (let fy = 0; fy < filterSize; fy++) {
                                    for (let fx = 0; fx < filterSize; fx++) {
                                        const inputIdx = ((y + fy) * currentSize + (x + fx)) * channels;
                                        if (inputIdx < currentFeatures.length) {
                                            sum += currentFeatures[inputIdx] * (Math.random() - 0.5);
                                        }
                                    }
                                }
                                outputFeatures[(y * outputSize + x) * numFilters + f] = Math.max(0, sum); // ReLU
                            }
                        }
                    }
                    
                    currentFeatures = outputFeatures;
                    currentSize = outputSize;
                    
                    // Pooling
                    if (currentSize > 4) {
                        currentSize = Math.floor(currentSize / 2);
                    }
                }
                
                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async simulateTransformerNetwork(duration, complexity) {
                const startTime = performance.now();
                
                // Simulate transformer architecture
                const sequenceLength = Math.min(512, complexity * 100);
                const modelDim = Math.min(512, complexity * 128);
                const numHeads = Math.min(8, complexity);
                const numLayers = Math.min(6, complexity);
                
                // Input embeddings
                let input = new Float32Array(sequenceLength * modelDim);
                for (let i = 0; i < input.length; i++) {
                    input[i] = Math.random() * 2 - 1;
                }
                
                // Simulate transformer layers
                for (let layer = 0; layer < numLayers; layer++) {
                    // Multi-head attention simulation
                    const headDim = modelDim / numHeads;
                    const attentionOutput = new Float32Array(sequenceLength * modelDim);
                    
                    for (let head = 0; head < numHeads; head++) {
                        // Simplified attention computation
                        for (let pos = 0; pos < sequenceLength; pos++) {
                            for (let dim = 0; dim < headDim; dim++) {
                                let sum = 0;
                                for (let pos2 = 0; pos2 < sequenceLength; pos2++) {
                                    const weight = Math.exp(-Math.abs(pos - pos2) * 0.1);
                                    const inputIdx = pos2 * modelDim + head * headDim + dim;
                                    if (inputIdx < input.length) {
                                        sum += input[inputIdx] * weight;
                                    }
                                }
                                attentionOutput[pos * modelDim + head * headDim + dim] = sum;
                            }
                        }
                    }
                    
                    // Feed-forward network
                    const ffnOutput = new Float32Array(sequenceLength * modelDim);
                    for (let pos = 0; pos < sequenceLength; pos++) {
                        for (let dim = 0; dim < modelDim; dim++) {
                            const inputIdx = pos * modelDim + dim;
                            // Simplified FFN: linear -> ReLU -> linear
                            const intermediate = Math.max(0, attentionOutput[inputIdx] * (Math.random() - 0.5));
                            ffnOutput[inputIdx] = intermediate * (Math.random() - 0.5);
                        }
                    }
                    
                    input = ffnOutput;
                }
                
                const elapsed = performance.now() - startTime;
                if (elapsed < duration) {
                    await this.sleep(duration - elapsed);
                }
            },

            async simulateRecurrentNetwork(duration, complexity) {
                const startTime = performance.now();
                
                // Simulate LSTM/GRU network
                const sequenceLength = Math.min(1000, complexity * 200);
                const hiddenSize = Math.min(256, complexity * 64);
                const numLayers = Math.min(3, complexity);
                
                let hiddenState = new Float32Array(hiddenSize);
                let cellState = new Float32Array(hiddenSize);
                
                // Initialize states
                for (let i = 0; i < hiddenSize; i++) {
                    hiddenState[i] = 0;
                    cellState[i] = 0;
                }
                
                // Process sequence
                for (let step = 0; step < sequenceLength; step++) {
                    const input = Math.sin(step * 0.01) + Math.random() * 0.1;
                    
                    // Simplified LSTM cell
                    for (let layer = 0; layer < numLayers; layer++) {
                        const newHiddenState = new Float32Array(hiddenSize);
                        const newCellState = new Float32Array(hiddenSize);
                        
                        for (let i = 0; i < hiddenSize; i++) {
                            // Forget gate
                            const forgetGate = 1 / (1 + Math.exp(-(input * 0.5 + hiddenState[i] * 0.5)));
                            
                            // Input gate
                            const inputGate = 1 / (1 + Math.exp(-(input * 0.3 + hiddenState[i] * 0.3)));
                            
                            // Candidate values
                            const candidate = Math.tanh(input * 0.4 + hiddenState[i] * 0.4);
                            
                            // Output gate
                            const outputGate = 1 / (1 + Math.exp(-(input * 0.6 + hiddenState[i] * 0.6)));
                            
                            // Update cell state
                            newCellState[i] = forgetGate * cellState[i] + inputGate * candidate;
                            
                            // Update hidden state
                            newHiddenState[i] = outputGate * Math.tanh(newCellState[i]);
                        }
                        
                        hiddenState = newHiddenState;
                        cellState = newCellState;
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
                this.interrupted = true;
            }
        };
    }

    // Initialize WebNN on worker startup
    initializeWebNN();

} else {
    // Running in main thread context - provide worker wrapper
    class WebNNWorker {
        constructor(workerId = 'webnn-worker') {
            this.workerId = workerId;
            this.worker = null;
            this.busy = false;
            this.currentTask = null;
            this.callbacks = {};
            this.webnnInitialized = false;
            this._init();
        }

        _init() {
            try {
                // Create the worker using a blob URL for self-contained execution
                const workerCode = `
                    // WebNN Worker Implementation (Neural Network simulation)
                    self.addEventListener('message', function(e) {
                        const { taskId, taskData, command } = e.data;
                        
                        try {
                            let result;
                            switch (command) {
                                case 'execute':
                                    result = executeWebNNTask(taskData);
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
                    
                    function executeWebNNTask(taskData) {
                        // Simulate neural network inference
                        const inputSize = taskData.inputSize || 128;
                        const hiddenSize = taskData.hiddenSize || 256;
                        const outputSize = taskData.outputSize || 64;
                        
                        // Simulate forward pass
                        const input = new Float32Array(inputSize);
                        for (let i = 0; i < inputSize; i++) {
                            input[i] = Math.random() * 2 - 1; // [-1, 1]
                        }
                        
                        // Simulate hidden layer computation
                        const hidden = new Float32Array(hiddenSize);
                        for (let i = 0; i < hiddenSize; i++) {
                            let sum = 0;
                            for (let j = 0; j < inputSize; j++) {
                                sum += input[j] * (Math.random() * 2 - 1);
                            }
                            hidden[i] = Math.tanh(sum); // Activation
                        }
                        
                        // Simulate output layer
                        const output = new Float32Array(outputSize);
                        for (let i = 0; i < outputSize; i++) {
                            let sum = 0;
                            for (let j = 0; j < hiddenSize; j++) {
                                sum += hidden[j] * (Math.random() * 2 - 1);
                            }
                            output[i] = sum;
                        }
                        
                        return { 
                            inputSize, 
                            hiddenSize, 
                            outputSize,
                            inferenceTime: Math.random() * 50 + 10, // 10-60ms
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
                    console.error('WebNN Worker error:', error);
                    if (this.callbacks.onError) {
                        this.callbacks.onError(error);
                    }
                });

                // Test worker responsiveness
                this.ping();

            } catch (error) {
                console.error('Failed to create WebNN worker:', error);
                throw error;
            }
        }

        _handleMessage(data) {
            const { type, taskId } = data;

            switch (type) {
                case 'webnn-initialized':
                    this.webnnInitialized = data.success;
                    if (data.success) {
                        console.log('WebNN initialized in worker:', data.info);
                    } else {
                        console.warn('WebNN initialization failed:', data.error);
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
                    console.log(`WebNN Worker ${this.workerId} is responsive`);
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
                webnnInitialized: this.webnnInitialized
            };
        }
    }

    // Export for browser use
    if (typeof window !== 'undefined') {
        window.WebNNWorker = WebNNWorker;
    }
}
