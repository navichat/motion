/**
 * DeepMimic JavaScript Inference Engine
 * 
 * This module provides WebGPU/WebNN/WASM accelerated inference for DeepMimic policies
 * converted to ONNX format, with integration to the BVH Timeline system.
 */

class DeepMimicInferenceEngine {
    constructor(options = {}) {
        this.onnxSession = null;
        this.modelMetadata = null;
        this.characterInfo = null;
        this.executionProvider = options.executionProvider || 'webgl'; // webgl, webgpu, webnn, wasm
        this.modelCache = new Map();
        this.isInitialized = false;
        
        // Performance monitoring
        this.inferenceStats = {
            totalInferences: 0,
            totalTime: 0,
            averageTime: 0,
            lastInferenceTime: 0
        };
        
        // State management
        this.currentState = null;
        this.stateHistory = [];
        this.maxHistoryLength = 100;
    }
    
    /**
     * Initialize the inference engine
     */
    async initialize() {
        try {
            // Import ONNX Runtime Web
            if (typeof ort === 'undefined') {
                await this.loadONNXRuntime();
            }
            
            // Configure execution provider
            await this.configureExecutionProvider();
            
            this.isInitialized = true;
            console.log('DeepMimic Inference Engine initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize DeepMimic Inference Engine:', error);
            throw error;
        }
    }
    
    /**
     * Load ONNX Runtime Web library
     */
    async loadONNXRuntime() {
        return new Promise((resolve, reject) => {
            if (typeof ort !== 'undefined') {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
            script.onload = () => resolve();
            script.onerror = () => reject(new Error('Failed to load ONNX Runtime Web'));
            document.head.appendChild(script);
        });
    }
    
    /**
     * Configure the execution provider based on browser capabilities
     */
    async configureExecutionProvider() {
        const providers = [];
        
        try {
            // Try WebGPU first (best performance)
            if (this.executionProvider === 'webgpu' && navigator.gpu) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    providers.push('webgpu');
                    console.log('Using WebGPU execution provider');
                }
            }
            
            // Try WebNN (good performance, better compatibility)
            if (this.executionProvider === 'webnn' && 'ml' in navigator) {
                providers.push('webnn');
                console.log('Using WebNN execution provider');
            }
            
            // Fallback to WebGL (good compatibility)
            providers.push('webgl');
            
            // Last resort: WASM (CPU)
            providers.push('wasm');
            
        } catch (error) {
            console.warn('Error configuring execution provider:', error);
            providers.push('webgl', 'wasm');
        }
        
        ort.env.executionProviders = providers;
        console.log('Configured execution providers:', providers);
    }
    
    /**
     * Load a DeepMimic policy model
     */
    async loadModel(modelPath, modelName = null) {
        try {
            if (!this.isInitialized) {
                await this.initialize();
            }
            
            const name = modelName || this.extractModelName(modelPath);
            
            // Check cache first
            if (this.modelCache.has(name)) {
                console.log(`Using cached model: ${name}`);
                return this.modelCache.get(name);
            }
            
            console.log(`Loading DeepMimic model: ${name}`);
            
            // Load ONNX model
            const session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ort.env.executionProviders
            });
            
            // Load metadata if available
            const metadataPath = modelPath.replace('.onnx', '_metadata.json');
            let metadata = null;
            
            try {
                const response = await fetch(metadataPath);
                metadata = await response.json();
            } catch (error) {
                console.warn(`Could not load metadata for ${name}:`, error);
                metadata = this.createDefaultMetadata(session);
            }
            
            const modelInfo = {
                session,
                metadata,
                name,
                inputNames: session.inputNames,
                outputNames: session.outputNames,
                inputShapes: this.getInputShapes(session),
                outputShapes: this.getOutputShapes(session)
            };
            
            // Cache the model
            this.modelCache.set(name, modelInfo);
            
            // Set as current model if first one loaded
            if (!this.onnxSession) {
                this.onnxSession = session;
                this.modelMetadata = metadata;
                this.characterInfo = metadata.character_info;
            }
            
            console.log(`Successfully loaded model: ${name}`);
            console.log(`Input shapes:`, modelInfo.inputShapes);
            console.log(`Output shapes:`, modelInfo.outputShapes);
            
            return modelInfo;
            
        } catch (error) {
            console.error(`Failed to load model ${modelPath}:`, error);
            throw error;
        }
    }
    
    /**
     * Extract model name from file path
     */
    extractModelName(modelPath) {
        return modelPath.split('/').pop().replace('.onnx', '');
    }
    
    /**
     * Create default metadata when metadata file is not available
     */
    createDefaultMetadata(session) {
        return {
            model_name: 'unknown',
            input_shape: this.getInputShapes(session)[0] || [-1, 197],
            output_shape: this.getOutputShapes(session)[0] || [-1, 48],
            character_info: this.getDefaultCharacterInfo(),
            conversion_date: new Date().toISOString()
        };
    }
    
    /**
     * Get input shapes from ONNX session
     */
    getInputShapes(session) {
        return session.inputNames.map(name => {
            const input = session.inputMetadata[name];
            return input.dims;
        });
    }
    
    /**
     * Get output shapes from ONNX session
     */
    getOutputShapes(session) {
        return session.outputNames.map(name => {
            const output = session.outputMetadata[name];
            return output.dims;
        });
    }
    
    /**
     * Get default character information
     */
    getDefaultCharacterInfo() {
        return {
            "joints": [
                {"name": "root", "parent": -1, "offset": [0, 0, 0], "type": "revolute"},
                {"name": "chest", "parent": 0, "offset": [0, 0.15, 0], "type": "revolute"},
                {"name": "neck", "parent": 1, "offset": [0, 0.15, 0], "type": "revolute"},
                {"name": "head", "parent": 2, "offset": [0, 0.1, 0], "type": "revolute"},
                {"name": "right_hip", "parent": 0, "offset": [0.1, 0, 0], "type": "revolute"},
                {"name": "right_knee", "parent": 4, "offset": [0, -0.4, 0], "type": "revolute"},
                {"name": "right_ankle", "parent": 5, "offset": [0, -0.4, 0], "type": "revolute"},
                {"name": "left_hip", "parent": 0, "offset": [-0.1, 0, 0], "type": "revolute"},
                {"name": "left_knee", "parent": 7, "offset": [0, -0.4, 0], "type": "revolute"},
                {"name": "left_ankle", "parent": 8, "offset": [0, -0.4, 0], "type": "revolute"},
                {"name": "right_shoulder", "parent": 1, "offset": [0.15, 0.1, 0], "type": "revolute"},
                {"name": "right_elbow", "parent": 10, "offset": [0.25, 0, 0], "type": "revolute"},
                {"name": "right_wrist", "parent": 11, "offset": [0.25, 0, 0], "type": "revolute"},
                {"name": "left_shoulder", "parent": 1, "offset": [-0.15, 0.1, 0], "type": "revolute"},
                {"name": "left_elbow", "parent": 13, "offset": [-0.25, 0, 0], "type": "revolute"},
                {"name": "left_wrist", "parent": 14, "offset": [-0.25, 0, 0], "type": "revolute"}
            ],
            "joint_names": ["root", "chest", "neck", "head", "right_hip", "right_knee", "right_ankle", 
                           "left_hip", "left_knee", "left_ankle", "right_shoulder", "right_elbow", 
                           "right_wrist", "left_shoulder", "left_elbow", "left_wrist"],
            "parent_indices": [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 1, 10, 11, 1, 13, 14],
            "bone_count": 16
        };
    }
    
    /**
     * Run inference on the current model
     */
    async runInference(inputState, modelName = null) {
        try {
            if (!this.isInitialized) {
                throw new Error('Inference engine not initialized');
            }
            
            const startTime = performance.now();
            
            // Select model
            let session = this.onnxSession;
            let metadata = this.modelMetadata;
            
            if (modelName && this.modelCache.has(modelName)) {
                const modelInfo = this.modelCache.get(modelName);
                session = modelInfo.session;
                metadata = modelInfo.metadata;
            }
            
            if (!session) {
                throw new Error('No model loaded');
            }
            
            // Prepare input tensor
            const inputTensor = this.prepareInputTensor(inputState, session);
            
            // Run inference
            const feeds = {};
            feeds[session.inputNames[0]] = inputTensor;
            
            const results = await session.run(feeds);
            
            // Extract action output
            const outputTensor = results[session.outputNames[0]];
            const actions = Array.from(outputTensor.data);
            
            // Update performance stats
            const inferenceTime = performance.now() - startTime;
            this.updateInferenceStats(inferenceTime);
            
            // Store state history
            this.updateStateHistory(inputState, actions);
            
            return {
                actions,
                inferenceTime,
                modelUsed: modelName || 'default',
                outputShape: outputTensor.dims
            };
            
        } catch (error) {
            console.error('Inference failed:', error);
            throw error;
        }
    }
    
    /**
     * Prepare input tensor from state data
     */
    prepareInputTensor(inputState, session) {
        // Ensure input is in the correct format
        let stateArray;
        
        if (Array.isArray(inputState)) {
            stateArray = inputState;
        } else if (inputState instanceof Float32Array) {
            stateArray = Array.from(inputState);
        } else {
            throw new Error('Invalid input state format');
        }
        
        // Get expected input shape
        const inputShape = session.inputMetadata[session.inputNames[0]].dims;
        const expectedSize = inputShape[inputShape.length - 1]; // Last dimension
        
        // Pad or truncate to expected size
        if (stateArray.length !== expectedSize) {
            console.warn(`Input size mismatch: got ${stateArray.length}, expected ${expectedSize}`);
            if (stateArray.length < expectedSize) {
                // Pad with zeros
                stateArray = stateArray.concat(new Array(expectedSize - stateArray.length).fill(0));
            } else {
                // Truncate
                stateArray = stateArray.slice(0, expectedSize);
            }
        }
        
        // Create tensor with batch dimension
        const tensorData = new Float32Array(stateArray);
        const batchedShape = [1, ...inputShape.slice(1)];
        
        return new ort.Tensor('float32', tensorData, batchedShape);
    }
    
    /**
     * Update inference performance statistics
     */
    updateInferenceStats(inferenceTime) {
        this.inferenceStats.totalInferences++;
        this.inferenceStats.totalTime += inferenceTime;
        this.inferenceStats.averageTime = this.inferenceStats.totalTime / this.inferenceStats.totalInferences;
        this.inferenceStats.lastInferenceTime = inferenceTime;
    }
    
    /**
     * Update state history for temporal coherence
     */
    updateStateHistory(inputState, outputActions) {
        this.stateHistory.push({
            timestamp: performance.now(),
            state: inputState,
            actions: outputActions
        });
        
        // Keep history within limits
        if (this.stateHistory.length > this.maxHistoryLength) {
            this.stateHistory = this.stateHistory.slice(-this.maxHistoryLength);
        }
        
        this.currentState = inputState;
    }
    
    /**
     * Convert DeepMimic actions to BVH bone rotations
     */
    actionsToBVH(actions, timestamp = 0) {
        if (!this.characterInfo) {
            throw new Error('Character information not available');
        }
        
        const bvhFrame = {
            timestamp,
            bones: {},
            rootPosition: [0, 0, 0],
            rootRotation: [0, 0, 0]
        };
        
        // DeepMimic typically outputs actions as joint torques or target angles
        // We need to convert these to BVH rotation values
        let actionIndex = 0;
        
        for (let i = 0; i < this.characterInfo.joints.length; i++) {
            const joint = this.characterInfo.joints[i];
            
            if (joint.name === 'root') {
                // Root position (first 3 values)
                if (actionIndex + 2 < actions.length) {
                    bvhFrame.rootPosition = [
                        actions[actionIndex],
                        actions[actionIndex + 1],
                        actions[actionIndex + 2]
                    ];
                    actionIndex += 3;
                }
                
                // Root rotation (next 3 values)
                if (actionIndex + 2 < actions.length) {
                    bvhFrame.rootRotation = [
                        actions[actionIndex] * 180 / Math.PI,     // Convert to degrees
                        actions[actionIndex + 1] * 180 / Math.PI,
                        actions[actionIndex + 2] * 180 / Math.PI
                    ];
                    actionIndex += 3;
                }
            } else {
                // Joint rotations (3 DOF per joint)
                if (actionIndex + 2 < actions.length) {
                    bvhFrame.bones[joint.name] = {
                        rotation: [
                            actions[actionIndex] * 180 / Math.PI,     // Convert to degrees
                            actions[actionIndex + 1] * 180 / Math.PI,
                            actions[actionIndex + 2] * 180 / Math.PI
                        ]
                    };
                    actionIndex += 3;
                }
            }
        }
        
        return bvhFrame;
    }
    
    /**
     * Get model information
     */
    getModelInfo(modelName = null) {
        if (modelName && this.modelCache.has(modelName)) {
            const modelInfo = this.modelCache.get(modelName);
            return {
                name: modelInfo.name,
                inputShapes: modelInfo.inputShapes,
                outputShapes: modelInfo.outputShapes,
                metadata: modelInfo.metadata
            };
        }
        
        if (this.onnxSession) {
            return {
                name: 'default',
                inputShapes: this.getInputShapes(this.onnxSession),
                outputShapes: this.getOutputShapes(this.onnxSession),
                metadata: this.modelMetadata
            };
        }
        
        return null;
    }
    
    /**
     * Get inference performance statistics
     */
    getPerformanceStats() {
        return { ...this.inferenceStats };
    }
    
    /**
     * Get available models
     */
    getAvailableModels() {
        return Array.from(this.modelCache.keys());
    }
    
    /**
     * Switch to a different model
     */
    switchModel(modelName) {
        if (!this.modelCache.has(modelName)) {
            throw new Error(`Model ${modelName} not loaded`);
        }
        
        const modelInfo = this.modelCache.get(modelName);
        this.onnxSession = modelInfo.session;
        this.modelMetadata = modelInfo.metadata;
        this.characterInfo = modelInfo.metadata.character_info;
        
        console.log(`Switched to model: ${modelName}`);
    }
    
    /**
     * Cleanup resources
     */
    dispose() {
        // Dispose all ONNX sessions
        for (const [name, modelInfo] of this.modelCache) {
            try {
                modelInfo.session.release();
                console.log(`Disposed model: ${name}`);
            } catch (error) {
                console.warn(`Error disposing model ${name}:`, error);
            }
        }
        
        this.modelCache.clear();
        this.onnxSession = null;
        this.modelMetadata = null;
        this.characterInfo = null;
        this.isInitialized = false;
        
        console.log('DeepMimic Inference Engine disposed');
    }
}

// Export for use in modules or global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeepMimicInferenceEngine;
} else if (typeof window !== 'undefined') {
    window.DeepMimicInferenceEngine = DeepMimicInferenceEngine;
}
