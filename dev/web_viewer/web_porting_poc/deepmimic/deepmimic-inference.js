/**
 * DeepMimic JavaScript Inference Engine
 * Supports WebGPU, WebNN, and WebAssembly execution providers
 */

class DeepMimicInference {
    constructor() {
        this.session = null;
        this.modelInfo = null;
        this.executionProvider = null;
        this.supportedProviders = [];
        this.isInitialized = false;
    }

    /**
     * Initialize ONNX Runtime Web and detect available execution providers
     */
    async initialize() {
        try {
            // Check if ONNX Runtime Web is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web not loaded. Please include onnxruntime-web in your HTML.');
            }

            // Detect available execution providers
            await this.detectExecutionProviders();
            
            this.isInitialized = true;
            console.log('DeepMimic Inference Engine initialized');
            console.log('Available execution providers:', this.supportedProviders);
            
            return this.supportedProviders;
        } catch (error) {
            console.error('Failed to initialize DeepMimic Inference Engine:', error);
            throw error;
        }
    }

    /**
     * Detect available execution providers
     */
    async detectExecutionProviders() {
        this.supportedProviders = [];

        // Check WebAssembly (always available)
        this.supportedProviders.push('wasm');

        // Check WebGPU
        if (navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    this.supportedProviders.push('webgpu');
                }
            } catch (e) {
                console.warn('WebGPU not available:', e.message);
            }
        }

        // Check WebNN (experimental)
        if ('ml' in navigator) {
            try {
                const context = await navigator.ml.createContext();
                if (context) {
                    this.supportedProviders.push('webnn');
                }
            } catch (e) {
                console.warn('WebNN not available:', e.message);
            }
        }
    }

    /**
     * Load a DeepMimic ONNX model
     * @param {string} modelPath - Path to the ONNX model file
     * @param {string} executionProvider - Preferred execution provider ('webgpu', 'webnn', 'wasm')
     */
    async loadModel(modelPath, executionProvider = 'wasm') {
        try {
            if (!this.isInitialized) {
                await this.initialize();
            }

            // Validate execution provider
            if (!this.supportedProviders.includes(executionProvider)) {
                console.warn(`Execution provider '${executionProvider}' not available. Falling back to WebAssembly.`);
                executionProvider = 'wasm';
            }

            // Configure session options for the execution provider
            const sessionOptions = this.getSessionOptions(executionProvider);

            // Load the model
            console.log(`Loading model: ${modelPath} with provider: ${executionProvider}`);
            this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
            this.executionProvider = executionProvider;

            // Extract model information
            this.modelInfo = {
                inputName: this.session.inputNames[0],
                outputName: this.session.outputNames[0],
                inputShape: this.session.inputInfo[this.session.inputNames[0]].dims,
                outputShape: this.session.outputInfo[this.session.outputNames[0]].dims
            };

            console.log('Model loaded successfully:', this.modelInfo);
            return this.modelInfo;

        } catch (error) {
            console.error('Failed to load model:', error);
            throw error;
        }
    }

    /**
     * Get session options for different execution providers
     */
    getSessionOptions(executionProvider) {
        const options = {
            executionProviders: [],
            graphOptimizationLevel: 'all',
            executionMode: 'sequential'
        };

        switch (executionProvider) {
            case 'webgpu':
                options.executionProviders = ['webgpu', 'wasm'];
                break;
            case 'webnn':
                options.executionProviders = ['webnn', 'wasm'];
                break;
            case 'wasm':
            default:
                options.executionProviders = ['wasm'];
                break;
        }

        return options;
    }

    /**
     * Run inference on input state
     * @param {Float32Array|Array} inputState - State vector (197-dimensional for humanoid)
     * @returns {Promise<Float32Array>} - Action vector (36-dimensional for humanoid)
     */
    async predict(inputState) {
        try {
            if (!this.session) {
                throw new Error('Model not loaded. Call loadModel() first.');
            }

            // Ensure input is Float32Array
            let inputData;
            if (inputState instanceof Float32Array) {
                inputData = inputState;
            } else if (Array.isArray(inputState)) {
                inputData = new Float32Array(inputState);
            } else {
                throw new Error('Input must be Float32Array or Array');
            }

            // Validate input dimensions
            const expectedInputSize = this.modelInfo.inputShape[1]; // [null, 197]
            if (inputData.length !== expectedInputSize) {
                throw new Error(`Input size mismatch. Expected ${expectedInputSize}, got ${inputData.length}`);
            }

            // Reshape input for batch processing
            const inputTensor = new ort.Tensor('float32', inputData, [1, expectedInputSize]);

            // Run inference
            const startTime = performance.now();
            const outputs = await this.session.run({
                [this.modelInfo.inputName]: inputTensor
            });
            const inferenceTime = performance.now() - startTime;

            // Extract output
            const outputTensor = outputs[this.modelInfo.outputName];
            const actionVector = outputTensor.data;

            // Return results with timing info
            return {
                actions: new Float32Array(actionVector),
                inferenceTime: inferenceTime,
                executionProvider: this.executionProvider
            };

        } catch (error) {
            console.error('Inference failed:', error);
            throw error;
        }
    }

    /**
     * Run batch inference
     * @param {Array<Float32Array|Array>} inputBatch - Batch of state vectors
     * @returns {Promise<Array>} - Array of action vectors with timing info
     */
    async predictBatch(inputBatch) {
        const results = [];
        for (const input of inputBatch) {
            const result = await this.predict(input);
            results.push(result);
        }
        return results;
    }

    /**
     * Benchmark inference performance
     * @param {number} iterations - Number of inference iterations
     * @param {Float32Array} testInput - Test input vector
     */
    async benchmark(iterations = 100, testInput = null) {
        try {
            if (!this.session) {
                throw new Error('Model not loaded. Call loadModel() first.');
            }

            // Generate test input if not provided
            if (!testInput) {
                const inputSize = this.modelInfo.inputShape[1];
                testInput = new Float32Array(inputSize);
                for (let i = 0; i < inputSize; i++) {
                    testInput[i] = Math.random() * 2 - 1; // Random values between -1 and 1
                }
            }

            console.log(`Starting benchmark: ${iterations} iterations with ${this.executionProvider}`);
            
            const times = [];
            let totalTime = 0;

            // Warmup
            await this.predict(testInput);

            // Benchmark
            for (let i = 0; i < iterations; i++) {
                const result = await this.predict(testInput);
                times.push(result.inferenceTime);
                totalTime += result.inferenceTime;
            }

            const avgTime = totalTime / iterations;
            const minTime = Math.min(...times);
            const maxTime = Math.max(...times);
            const medianTime = times.sort((a, b) => a - b)[Math.floor(times.length / 2)];

            const benchmarkResults = {
                executionProvider: this.executionProvider,
                iterations: iterations,
                averageTime: avgTime,
                minTime: minTime,
                maxTime: maxTime,
                medianTime: medianTime,
                totalTime: totalTime,
                fps: 1000 / avgTime
            };

            console.log('Benchmark results:', benchmarkResults);
            return benchmarkResults;

        } catch (error) {
            console.error('Benchmark failed:', error);
            throw error;
        }
    }

    /**
     * Compare outputs across different execution providers
     * @param {string} modelPath - Path to ONNX model
     * @param {Float32Array} testInput - Test input vector
     * @param {number} tolerance - Numerical tolerance for comparison
     */
    async compareExecutionProviders(modelPath, testInput, tolerance = 1e-5) {
        const results = {};
        const comparisons = [];

        // Test each available execution provider
        for (const provider of this.supportedProviders) {
            try {
                console.log(`Testing with ${provider}...`);
                await this.loadModel(modelPath, provider);
                const result = await this.predict(testInput);
                results[provider] = {
                    actions: result.actions,
                    inferenceTime: result.inferenceTime,
                    success: true
                };
            } catch (error) {
                console.error(`Failed with ${provider}:`, error);
                results[provider] = {
                    error: error.message,
                    success: false
                };
            }
        }

        // Compare outputs between providers
        const providers = Object.keys(results).filter(p => results[p].success);
        for (let i = 0; i < providers.length; i++) {
            for (let j = i + 1; j < providers.length; j++) {
                const provider1 = providers[i];
                const provider2 = providers[j];
                const actions1 = results[provider1].actions;
                const actions2 = results[provider2].actions;

                const comparison = this.compareArrays(actions1, actions2, tolerance);
                comparisons.push({
                    provider1,
                    provider2,
                    maxDifference: comparison.maxDifference,
                    averageDifference: comparison.averageDifference,
                    withinTolerance: comparison.withinTolerance,
                    tolerance
                });
            }
        }

        return {
            results,
            comparisons,
            summary: {
                testedProviders: Object.keys(results),
                successfulProviders: providers,
                allWithinTolerance: comparisons.every(c => c.withinTolerance)
            }
        };
    }

    /**
     * Compare two arrays with tolerance
     */
    compareArrays(arr1, arr2, tolerance) {
        if (arr1.length !== arr2.length) {
            throw new Error('Arrays have different lengths');
        }

        let maxDiff = 0;
        let totalDiff = 0;
        let withinTolerance = true;

        for (let i = 0; i < arr1.length; i++) {
            const diff = Math.abs(arr1[i] - arr2[i]);
            maxDiff = Math.max(maxDiff, diff);
            totalDiff += diff;
            if (diff > tolerance) {
                withinTolerance = false;
            }
        }

        return {
            maxDifference: maxDiff,
            averageDifference: totalDiff / arr1.length,
            withinTolerance
        };
    }

    /**
     * Get model information
     */
    getModelInfo() {
        return this.modelInfo;
    }

    /**
     * Get current execution provider
     */
    getExecutionProvider() {
        return this.executionProvider;
    }

    /**
     * Check if model is loaded
     */
    isModelLoaded() {
        return this.session !== null;
    }

    /**
     * Cleanup resources
     */
    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
        this.modelInfo = null;
        this.executionProvider = null;
    }
}

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeepMimicInference;
} else {
    window.DeepMimicInference = DeepMimicInference;
}
