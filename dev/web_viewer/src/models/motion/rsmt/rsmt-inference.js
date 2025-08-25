/**
 * RSMT (Real-time Stylized Motion Transition) JavaScript Inference Engine
 * 
 * This module provides a complete JavaScript implementation of the RSMT system,
 * supporting real-time stylized motion transitions using three neural networks:
 * 1. DeepPhase: Skeleton -> Phase Vector encoding
 * 2. StyleVAE: Phase Vector -> Manifold space encoding/decoding  
 * 3. TransitionNet: Manifold space transition generation
 */

class RSMTInference {
    constructor(options = {}) {
        this.options = {
            executionProvider: options.executionProvider || 'wasm',
            deviceType: options.deviceType || 'cpu',
            numThreads: options.numThreads || 4,
            ...options
        };
        
        // Model sessions
        this.deepPhaseSession = null;
        this.styleVAESession = null;
        this.transitionNetSession = null;
        
        // Model information
        this.modelInfo = {
            deepPhase: null,
            styleVAE: null,
            transitionNet: null
        };
        
        // Cached data
        this.phaseDataCache = new Map();
        this.manifoldCache = new Map();
        
        // Performance tracking
        this.performanceStats = {
            deepPhaseTime: 0,
            styleVAETime: 0,
            transitionNetTime: 0,
            totalInferences: 0,
            cacheHits: 0
        };
        
        this.isInitialized = false;
        
        console.log('RSMT Inference Engine created with options:', this.options);
    }
    
    /**
     * Initialize the RSMT system with ONNX models
     */
    async initialize(modelPaths = null) {
        try {
            console.log('Initializing RSMT Inference Engine...');
            
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web not loaded. Please include onnxruntime-web.');
            }
            
            // Set session options
            const sessionOptions = {
                executionProviders: [this.options.executionProvider],
                graphOptimizationLevel: 'all',
                executionMode: 'sequential'
            };
            
            // Default model paths
            const defaultPaths = {
                deepPhase: './deepphase.onnx',
                styleVAE: './stylevae.onnx',
                transitionNet: './transitionnet.onnx'
            };
            
            const paths = { ...defaultPaths, ...modelPaths };
            
            // Load DeepPhase model
            console.log('Loading DeepPhase model...');
            this.deepPhaseSession = await ort.InferenceSession.create(paths.deepPhase, sessionOptions);
            this.modelInfo.deepPhase = this.extractModelInfo(this.deepPhaseSession);
            console.log('DeepPhase model loaded:', this.modelInfo.deepPhase);
            
            // Load StyleVAE model  
            console.log('Loading StyleVAE model...');
            this.styleVAESession = await ort.InferenceSession.create(paths.styleVAE, sessionOptions);
            this.modelInfo.styleVAE = this.extractModelInfo(this.styleVAESession);
            console.log('StyleVAE model loaded:', this.modelInfo.styleVAE);
            
            // Load TransitionNet model
            console.log('Loading TransitionNet model...');
            this.transitionNetSession = await ort.InferenceSession.create(paths.transitionNet, sessionOptions);
            this.modelInfo.transitionNet = this.extractModelInfo(this.transitionNetSession);
            console.log('TransitionNet model loaded:', this.modelInfo.transitionNet);
            
            this.isInitialized = true;
            console.log('RSMT Inference Engine initialized successfully!');
            
            return {
                success: true,
                modelInfo: this.modelInfo
            };
            
        } catch (error) {
            console.error('Failed to initialize RSMT Inference Engine:', error);
            throw error;
        }
    }
    
    /**
     * Extract model information from ONNX session
     */
    extractModelInfo(session) {
        const inputNames = session.inputNames;
        const outputNames = session.outputNames;
        
        const inputInfo = {};
        const outputInfo = {};
        
        for (const name of inputNames) {
            inputInfo[name] = session.inputInfo[name];
        }
        
        for (const name of outputNames) {
            outputInfo[name] = session.outputInfo[name];
        }
        
        return {
            inputNames,
            outputNames,
            inputInfo,
            outputInfo
        };
    }
    
    /**
     * Encode skeleton motion data to phase vectors using DeepPhase
     * @param {Float32Array|Array} skeletonData - Joint positions/rotations [frames, joints*channels]
     * @returns {Promise<Float32Array>} Phase vectors [frames, phase_dim]
     */
    async encodeToPhase(skeletonData) {
        if (!this.isInitialized) {
            throw new Error('RSMT not initialized. Call initialize() first.');
        }
        
        const startTime = performance.now();
        
        try {
            // Prepare input tensor
            const inputData = this.prepareSkeletonInput(skeletonData);
            const inputName = this.modelInfo.deepPhase.inputNames[0];
            const outputName = this.modelInfo.deepPhase.outputNames[0];
            
            // Run inference
            const feeds = { [inputName]: inputData };
            const results = await this.deepPhaseSession.run(feeds);
            
            // Extract phase vectors
            const phaseVectors = results[outputName].data;
            
            // Update performance stats
            this.performanceStats.deepPhaseTime += performance.now() - startTime;
            this.performanceStats.totalInferences++;
            
            return new Float32Array(phaseVectors);
            
        } catch (error) {
            console.error('Error in DeepPhase encoding:', error);
            throw error;
        }
    }
    
    /**
     * Encode phase vectors to manifold space using StyleVAE encoder
     * @param {Float32Array|Array} phaseVectors - Phase vectors [frames, phase_dim]
     * @returns {Promise<{mu: Float32Array, logvar: Float32Array}>} Manifold encoding
     */
    async encodeToManifold(phaseVectors) {
        if (!this.isInitialized) {
            throw new Error('RSMT not initialized. Call initialize() first.');
        }
        
        const startTime = performance.now();
        
        try {
            // Create cache key
            const cacheKey = this.createCacheKey(phaseVectors);
            if (this.manifoldCache.has(cacheKey)) {
                this.performanceStats.cacheHits++;
                return this.manifoldCache.get(cacheKey);
            }
            
            // Prepare input tensor
            const inputData = this.preparePhaseInput(phaseVectors);
            const inputName = this.modelInfo.styleVAE.inputNames[0];
            
            // Run inference
            const feeds = { [inputName]: inputData };
            const results = await this.styleVAESession.run(feeds);
            
            // Extract mu and logvar (VAE encoding)
            const outputNames = this.modelInfo.styleVAE.outputNames;
            const mu = new Float32Array(results[outputNames[0]].data);
            const logvar = outputNames.length > 1 ? new Float32Array(results[outputNames[1]].data) : null;
            
            const result = { mu, logvar };
            
            // Cache result
            this.manifoldCache.set(cacheKey, result);
            
            // Update performance stats
            this.performanceStats.styleVAETime += performance.now() - startTime;
            this.performanceStats.totalInferences++;
            
            return result;
            
        } catch (error) {
            console.error('Error in StyleVAE encoding:', error);
            throw error;
        }
    }
    
    /**
     * Decode from manifold space back to phase vectors using StyleVAE decoder
     * @param {Float32Array|Array} manifoldPoints - Points in manifold space [frames, latent_dim]
     * @returns {Promise<Float32Array>} Decoded phase vectors
     */
    async decodeFromManifold(manifoldPoints) {
        if (!this.isInitialized) {
            throw new Error('RSMT not initialized. Call initialize() first.');
        }
        
        const startTime = performance.now();
        
        try {
            // For decoding, we need to use the decoder part of StyleVAE
            // This might require a separate decoder model or combined encode/decode
            
            // Prepare input tensor for decoder
            const inputData = this.prepareManifoldInput(manifoldPoints);
            
            // If StyleVAE has decode functionality, use it
            // Otherwise, we'll need to implement a workaround
            const inputName = this.modelInfo.styleVAE.inputNames[0]; // Might need different input for decoder
            const feeds = { [inputName]: inputData };
            const results = await this.styleVAESession.run(feeds);
            
            // Extract decoded phase vectors
            const outputName = this.modelInfo.styleVAE.outputNames[0];
            const phaseVectors = new Float32Array(results[outputName].data);
            
            // Update performance stats
            this.performanceStats.styleVAETime += performance.now() - startTime;
            this.performanceStats.totalInferences++;
            
            return phaseVectors;
            
        } catch (error) {
            console.error('Error in StyleVAE decoding:', error);
            throw error;
        }
    }
    
    /**
     * Generate transition path in manifold space using TransitionNet
     * @param {Float32Array|Array} sourcePoint - Starting point in manifold space
     * @param {Float32Array|Array} targetPoint - Target point in manifold space  
     * @param {number} transitionLength - Number of frames in transition
     * @returns {Promise<Float32Array>} Transition path [frames, latent_dim]
     */
    async generateTransition(sourcePoint, targetPoint, transitionLength = 30) {
        if (!this.isInitialized) {
            throw new Error('RSMT not initialized. Call initialize() first.');
        }
        
        const startTime = performance.now();
        
        try {
            // Prepare inputs for TransitionNet
            const sourceInput = this.prepareManifoldInput(sourcePoint);
            const targetInput = this.prepareManifoldInput(targetPoint);
            
            // Create transition length tensor
            const lengthTensor = new ort.Tensor('int64', [transitionLength], [1]);
            
            // Prepare feeds based on model inputs
            const inputNames = this.modelInfo.transitionNet.inputNames;
            const feeds = {};
            
            // Map inputs based on expected input names
            if (inputNames.includes('source')) {
                feeds['source'] = sourceInput;
            } else if (inputNames.includes('input')) {
                feeds['input'] = sourceInput;
            } else {
                feeds[inputNames[0]] = sourceInput;
            }
            
            if (inputNames.includes('target')) {
                feeds['target'] = targetInput;
            } else if (inputNames.length > 1) {
                feeds[inputNames[1]] = targetInput;
            }
            
            if (inputNames.includes('length') || inputNames.includes('num_steps')) {
                const lengthInputName = inputNames.find(name => name.includes('length') || name.includes('steps'));
                if (lengthInputName) {
                    feeds[lengthInputName] = lengthTensor;
                }
            }
            
            // Run inference
            const results = await this.transitionNetSession.run(feeds);
            
            // Extract transition path
            const outputName = this.modelInfo.transitionNet.outputNames[0];
            const transitionPath = new Float32Array(results[outputName].data);
            
            // Update performance stats
            this.performanceStats.transitionNetTime += performance.now() - startTime;
            this.performanceStats.totalInferences++;
            
            return transitionPath;
            
        } catch (error) {
            console.error('Error in TransitionNet generation:', error);
            throw error;
        }
    }
    
    /**
     * Complete RSMT pipeline: skeleton -> phase -> manifold -> transition -> phase -> skeleton
     * @param {Object} options - Pipeline options
     * @returns {Promise<Object>} Generated motion data
     */
    async generateStylizedTransition(options = {}) {
        const {
            sourceMotion,           // Source skeleton motion data
            targetMotion,           // Target skeleton motion data
            transitionLength = 30,  // Number of transition frames
            styleBlending = 0.5     // Blending factor for style interpolation
        } = options;
        
        console.log('Generating stylized transition...');
        
        try {
            // Step 1: Encode source and target motions to phase vectors
            console.log('Step 1: Encoding to phase vectors...');
            const sourcePhase = await this.encodeToPhase(sourceMotion);
            const targetPhase = await this.encodeToPhase(targetMotion);
            
            // Step 2: Encode phase vectors to manifold space
            console.log('Step 2: Encoding to manifold space...');
            const sourceManifold = await this.encodeToManifold(sourcePhase);
            const targetManifold = await this.encodeToManifold(targetPhase);
            
            // Step 3: Generate transition in manifold space
            console.log('Step 3: Generating transition...');
            const transitionPath = await this.generateTransition(
                sourceManifold.mu,
                targetManifold.mu,
                transitionLength
            );
            
            // Step 4: Decode transition back to phase vectors
            console.log('Step 4: Decoding to phase vectors...');
            const transitionPhase = await this.decodeFromManifold(transitionPath);
            
            // Step 5: Convert phase vectors back to skeleton motion
            console.log('Step 5: Converting to skeleton motion...');
            const transitionMotion = await this.phaseToSkeleton(transitionPhase);
            
            console.log('Stylized transition generation complete!');
            
            return {
                sourcePhase,
                targetPhase,
                sourceManifold: sourceManifold.mu,
                targetManifold: targetManifold.mu,
                transitionPath,
                transitionPhase,
                transitionMotion,
                metadata: {
                    transitionLength,
                    styleBlending,
                    performanceStats: this.getPerformanceStats()
                }
            };
            
        } catch (error) {
            console.error('Error in stylized transition generation:', error);
            throw error;
        }
    }
    
    /**
     * Convert phase vectors back to skeleton motion data
     * This is a simplified implementation - in practice you might want a dedicated decoder
     */
    async phaseToSkeleton(phaseVectors) {
        // For now, we'll use a simple mapping from phase to skeleton
        // In a full implementation, this would be another neural network or analytical converter
        
        const numFrames = phaseVectors.length / this.getPhaseVectorSize();
        const numJoints = 22; // Standard humanoid skeleton
        const channelsPerJoint = 6; // 3 position + 3 rotation
        
        const skeletonData = new Float32Array(numFrames * numJoints * channelsPerJoint);
        
        // Simple mapping - this would be replaced with proper inverse kinematics or learned decoder
        for (let frame = 0; frame < numFrames; frame++) {
            const phaseOffset = frame * this.getPhaseVectorSize();
            const skelOffset = frame * numJoints * channelsPerJoint;
            
            for (let joint = 0; joint < numJoints; joint++) {
                const jointOffset = skelOffset + joint * channelsPerJoint;
                const phaseIdx = phaseOffset + (joint % this.getPhaseVectorSize());
                
                // Map phase values to joint transforms
                skeletonData[jointOffset + 0] = phaseVectors[phaseIdx] * 0.1; // x position
                skeletonData[jointOffset + 1] = phaseVectors[phaseIdx + 1] * 0.1; // y position  
                skeletonData[jointOffset + 2] = phaseVectors[phaseIdx + 2] * 0.1; // z position
                skeletonData[jointOffset + 3] = phaseVectors[phaseIdx + 3] * Math.PI; // x rotation
                skeletonData[jointOffset + 4] = phaseVectors[phaseIdx + 4] * Math.PI; // y rotation
                skeletonData[jointOffset + 5] = phaseVectors[phaseIdx + 5] * Math.PI; // z rotation
            }
        }
        
        return skeletonData;
    }
    
    /**
     * Helper methods for input preparation
     */
    prepareSkeletonInput(skeletonData) {
        const data = skeletonData instanceof Float32Array ? skeletonData : new Float32Array(skeletonData);
        
        // Determine input shape based on model requirements
        const inputInfo = this.modelInfo.deepPhase.inputInfo[this.modelInfo.deepPhase.inputNames[0]];
        const inputShape = inputInfo.dims;
        
        // Create tensor with appropriate shape
        return new ort.Tensor('float32', data, inputShape);
    }
    
    preparePhaseInput(phaseVectors) {
        const data = phaseVectors instanceof Float32Array ? phaseVectors : new Float32Array(phaseVectors);
        
        // Determine input shape based on model requirements
        const inputInfo = this.modelInfo.styleVAE.inputInfo[this.modelInfo.styleVAE.inputNames[0]];
        const inputShape = inputInfo.dims;
        
        return new ort.Tensor('float32', data, inputShape);
    }
    
    prepareManifoldInput(manifoldPoints) {
        const data = manifoldPoints instanceof Float32Array ? manifoldPoints : new Float32Array(manifoldPoints);
        
        // For manifold input, we need to determine the correct shape
        // This will depend on the specific model architecture
        const batchSize = 1;
        const latentDim = data.length;
        
        return new ort.Tensor('float32', data, [batchSize, latentDim]);
    }
    
    /**
     * Utility methods
     */
    createCacheKey(data) {
        // Simple hash function for caching
        let hash = 0;
        const str = data.toString();
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString();
    }
    
    getPhaseVectorSize() {
        // This should be determined from the model or configuration
        return 32; // Default phase vector size for RSMT
    }
    
    getManifoldDimension() {
        // This should be determined from the StyleVAE model
        return 8; // Default manifold dimension for RSMT
    }
    
    /**
     * Performance and statistics
     */
    getPerformanceStats() {
        const totalTime = this.performanceStats.deepPhaseTime + 
                         this.performanceStats.styleVAETime + 
                         this.performanceStats.transitionNetTime;
        
        return {
            ...this.performanceStats,
            totalTime,
            averageTime: totalTime / Math.max(1, this.performanceStats.totalInferences),
            cacheHitRate: this.performanceStats.cacheHits / Math.max(1, this.performanceStats.totalInferences)
        };
    }
    
    /**
     * Clear caches
     */
    clearCaches() {
        this.phaseDataCache.clear();
        this.manifoldCache.clear();
        console.log('RSMT caches cleared');
    }
    
    /**
     * Get model information
     */
    getModelInfo() {
        return this.modelInfo;
    }
    
    /**
     * Check if system is initialized
     */
    isReady() {
        return this.isInitialized;
    }
    
    /**
     * Dispose and cleanup resources
     */
    dispose() {
        if (this.deepPhaseSession) {
            this.deepPhaseSession.release();
            this.deepPhaseSession = null;
        }
        
        if (this.styleVAESession) {
            this.styleVAESession.release();
            this.styleVAESession = null;
        }
        
        if (this.transitionNetSession) {
            this.transitionNetSession.release();
            this.transitionNetSession = null;
        }
        
        this.clearCaches();
        this.isInitialized = false;
        
        console.log('RSMT Inference Engine disposed');
    }
}

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RSMTInference;
} else if (typeof window !== 'undefined') {
    window.RSMTInference = RSMTInference;
}
