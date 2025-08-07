/**
 * DeepMimic to BVH Converter
 * 
 * Converts DeepMimic policy outputs to BVH format and integrates with BVHTimeline system.
 * Provides real-time motion synthesis and batch processing capabilities.
 */

class DeepMimicBVHConverter {
    constructor(options = {}) {
        this.inferenceEngine = options.inferenceEngine || new DeepMimicInferenceEngine(options);
        this.timeline = options.timeline || null;
        this.frameRate = options.frameRate || 30;
        this.isInitialized = false;
        
        // State management
        this.currentPhase = 0;
        this.targetMotion = null;
        this.transitionProgress = 0;
        
        // Motion synthesis parameters
        this.motionBlending = {
            enabled: options.motionBlending !== false,
            blendFactor: options.blendFactor || 0.1,
            smoothingWindow: options.smoothingWindow || 5
        };
        
        // Performance tracking
        this.conversionStats = {
            totalFrames: 0,
            totalTime: 0,
            averageTime: 0,
            lastConversionTime: 0
        };
        
        // Motion history for temporal coherence
        this.motionHistory = [];
        this.maxHistoryLength = 100;
        
        // Default humanoid state template
        this.defaultState = this.createDefaultState();
    }
    
    /**
     * Initialize the converter
     */
    async initialize(modelPaths = []) {
        try {
            await this.inferenceEngine.initialize();
            
            // Load default models if provided
            for (const modelPath of modelPaths) {
                await this.inferenceEngine.loadModel(modelPath);
            }
            
            this.isInitialized = true;
            console.log('DeepMimic BVH Converter initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize DeepMimic BVH Converter:', error);
            throw error;
        }
    }
    
    /**
     * Create default humanoid state vector
     */
    createDefaultState() {
        // DeepMimic typically uses a 197-dimensional state vector
        // This includes: pose, velocity, target pose, phase information, etc.
        const stateSize = 197;
        const state = new Float32Array(stateSize);
        
        // Initialize with reasonable default values
        // Root position (3 values)
        state[0] = 0;  // x
        state[1] = 1;  // y (height)
        state[2] = 0;  // z
        
        // Root rotation (4 values - quaternion)
        state[3] = 0;  // x
        state[4] = 0;  // y
        state[5] = 0;  // z
        state[6] = 1;  // w
        
        // Root velocity (3 values)
        state[7] = 0;  // vx
        state[8] = 0;  // vy
        state[9] = 0;  // vz
        
        // Root angular velocity (3 values)
        state[10] = 0;  // wx
        state[11] = 0;  // wy
        state[12] = 0;  // wz
        
        // Joint positions and velocities (remaining values)
        // Initialize with neutral pose
        for (let i = 13; i < stateSize; i++) {
            state[i] = 0;
        }
        
        return state;
    }
    
    /**
     * Update character state based on target motion and current phase
     */
    updateCharacterState(targetAction = null, deltaTime = 1/30) {
        const state = new Float32Array(this.defaultState);
        
        if (targetAction) {
            // Update state based on target motion
            this.integrateTargetAction(state, targetAction, deltaTime);
        }
        
        // Update phase information (for cyclic motions)
        this.currentPhase = (this.currentPhase + deltaTime * 2 * Math.PI) % (2 * Math.PI);
        
        // Add phase to state vector (common in DeepMimic)
        if (state.length > 190) {
            state[190] = Math.sin(this.currentPhase);
            state[191] = Math.cos(this.currentPhase);
        }
        
        return state;
    }
    
    /**
     * Integrate target action into character state
     */
    integrateTargetAction(state, targetAction, deltaTime) {
        // This is a simplified integration - in practice, DeepMimic uses
        // more sophisticated physics simulation
        
        // Extract root motion from target action
        if (targetAction.length >= 6) {
            // Update root position
            state[0] += targetAction[0] * deltaTime;  // x velocity
            state[1] += targetAction[1] * deltaTime;  // y velocity
            state[2] += targetAction[2] * deltaTime;  // z velocity
            
            // Update root rotation (simplified)
            state[3] += targetAction[3] * deltaTime;  // angular velocity x
            state[4] += targetAction[4] * deltaTime;  // angular velocity y
            state[5] += targetAction[5] * deltaTime;  // angular velocity z
        }
        
        // Update velocities
        if (targetAction.length >= 3) {
            state[7] = targetAction[0];  // vx
            state[8] = targetAction[1];  // vy
            state[9] = targetAction[2];  // vz
        }
    }
    
    /**
     * Generate motion sequence using DeepMimic policy
     */
    async generateMotionSequence(options = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        const {
            duration = 5.0,           // seconds
            modelName = null,         // which policy to use
            targetMotion = null,      // reference motion (optional)
            startState = null,        // initial character state
            realTime = false          // real-time vs batch generation
        } = options;
        
        const frameCount = Math.ceil(duration * this.frameRate);
        const deltaTime = 1.0 / this.frameRate;
        const motionSequence = [];
        
        console.log(`Generating ${frameCount} frames of motion using DeepMimic...`);
        
        let currentState = startState || this.createDefaultState();
        
        for (let frame = 0; frame < frameCount; frame++) {
            const timestamp = frame * deltaTime;
            
            try {
                // Run policy inference
                const inferenceResult = await this.inferenceEngine.runInference(currentState, modelName);
                const actions = inferenceResult.actions;
                
                // Convert actions to BVH frame
                const bvhFrame = this.inferenceEngine.actionsToBVH(actions, timestamp);
                
                // Apply motion blending if enabled
                if (this.motionBlending.enabled && this.motionHistory.length > 0) {
                    this.applyMotionBlending(bvhFrame);
                }
                
                // Add frame to sequence
                motionSequence.push(bvhFrame);
                
                // Update character state for next frame
                currentState = this.updateCharacterState(actions, deltaTime);
                
                // Store in motion history
                this.updateMotionHistory(bvhFrame, actions);
                
                // Update conversion stats
                this.updateConversionStats(inferenceResult.inferenceTime);
                
                // Progress callback for real-time generation
                if (realTime && frame % 10 === 0) {
                    const progress = (frame / frameCount) * 100;
                    console.log(`Motion generation progress: ${progress.toFixed(1)}%`);
                }
                
            } catch (error) {
                console.error(`Error generating frame ${frame}:`, error);
                // Use previous frame or default pose as fallback
                if (motionSequence.length > 0) {
                    motionSequence.push({ ...motionSequence[motionSequence.length - 1] });
                } else {
                    motionSequence.push(this.createDefaultBVHFrame(timestamp));
                }
            }
        }
        
        console.log(`Motion generation complete: ${motionSequence.length} frames`);
        return motionSequence;
    }
    
    /**
     * Apply motion blending for smoother transitions
     */
    applyMotionBlending(currentFrame) {
        if (this.motionHistory.length === 0) return;
        
        const previousFrame = this.motionHistory[this.motionHistory.length - 1].bvhFrame;
        const blendFactor = this.motionBlending.blendFactor;
        
        // Blend root position
        currentFrame.rootPosition = [
            previousFrame.rootPosition[0] * (1 - blendFactor) + currentFrame.rootPosition[0] * blendFactor,
            previousFrame.rootPosition[1] * (1 - blendFactor) + currentFrame.rootPosition[1] * blendFactor,
            previousFrame.rootPosition[2] * (1 - blendFactor) + currentFrame.rootPosition[2] * blendFactor
        ];
        
        // Blend root rotation
        currentFrame.rootRotation = [
            previousFrame.rootRotation[0] * (1 - blendFactor) + currentFrame.rootRotation[0] * blendFactor,
            previousFrame.rootRotation[1] * (1 - blendFactor) + currentFrame.rootRotation[1] * blendFactor,
            previousFrame.rootRotation[2] * (1 - blendFactor) + currentFrame.rootRotation[2] * blendFactor
        ];
        
        // Blend bone rotations
        for (const boneName in currentFrame.bones) {
            if (previousFrame.bones[boneName]) {
                const prevRot = previousFrame.bones[boneName].rotation;
                const currRot = currentFrame.bones[boneName].rotation;
                
                currentFrame.bones[boneName].rotation = [
                    prevRot[0] * (1 - blendFactor) + currRot[0] * blendFactor,
                    prevRot[1] * (1 - blendFactor) + currRot[1] * blendFactor,
                    prevRot[2] * (1 - blendFactor) + currRot[2] * blendFactor
                ];
            }
        }
    }
    
    /**
     * Update motion history for temporal coherence
     */
    updateMotionHistory(bvhFrame, actions) {
        this.motionHistory.push({
            timestamp: performance.now(),
            bvhFrame: { ...bvhFrame },
            actions: [...actions]
        });
        
        // Keep history within limits
        if (this.motionHistory.length > this.maxHistoryLength) {
            this.motionHistory = this.motionHistory.slice(-this.maxHistoryLength);
        }
    }
    
    /**
     * Create default BVH frame
     */
    createDefaultBVHFrame(timestamp = 0) {
        return {
            timestamp,
            bones: {
                'root': { rotation: [0, 0, 0] },
                'chest': { rotation: [0, 0, 0] },
                'neck': { rotation: [0, 0, 0] },
                'head': { rotation: [0, 0, 0] },
                'right_hip': { rotation: [0, 0, 0] },
                'right_knee': { rotation: [0, 0, 0] },
                'right_ankle': { rotation: [0, 0, 0] },
                'left_hip': { rotation: [0, 0, 0] },
                'left_knee': { rotation: [0, 0, 0] },
                'left_ankle': { rotation: [0, 0, 0] },
                'right_shoulder': { rotation: [0, 0, 0] },
                'right_elbow': { rotation: [0, 0, 0] },
                'right_wrist': { rotation: [0, 0, 0] },
                'left_shoulder': { rotation: [0, 0, 0] },
                'left_elbow': { rotation: [0, 0, 0] },
                'left_wrist': { rotation: [0, 0, 0] }
            },
            rootPosition: [0, 1, 0],
            rootRotation: [0, 0, 0]
        };
    }
    
    /**
     * Create timeline clip from motion sequence
     */
    createTimelineClip(motionSequence, options = {}) {
        const {
            clipId = `deepmimic_motion_${Date.now()}`,
            trackName = 'deepmimic',
            startTime = 0,
            loop = false,
            metadata = {}
        } = options;
        
        return {
            id: clipId,
            trackName,
            startTime,
            endTime: startTime + (motionSequence.length / this.frameRate),
            loop,
            frames: motionSequence,
            metadata: {
                type: 'deepmimic',
                frameRate: this.frameRate,
                frameCount: motionSequence.length,
                modelUsed: metadata.modelUsed || 'unknown',
                generationTime: metadata.generationTime || Date.now(),
                ...metadata
            }
        };
    }
    
    /**
     * Add generated motion to BVH Timeline
     */
    async addMotionToTimeline(timeline, options = {}) {
        if (!timeline) {
            throw new Error('Timeline instance required');
        }
        
        const motionSequence = await this.generateMotionSequence(options);
        const clip = this.createTimelineClip(motionSequence, options);
        
        // Add clip to timeline
        timeline.addClip(clip);
        
        console.log(`Added DeepMimic motion clip: ${clip.id}`);
        return clip;
    }
    
    /**
     * Real-time motion synthesis for interactive applications
     */
    async startRealTimeMotion(timeline, options = {}) {
        if (!timeline) {
            throw new Error('Timeline instance required');
        }
        
        const {
            trackName = 'deepmimic_realtime',
            modelName = null,
            targetFPS = 30,
            bufferSize = 60  // frames to buffer ahead
        } = options;
        
        console.log('Starting real-time DeepMimic motion synthesis...');
        
        let isRunning = true;
        let frameCounter = 0;
        let currentState = this.createDefaultState();
        
        const motionLoop = async () => {
            if (!isRunning) return;
            
            try {
                const timestamp = frameCounter / targetFPS;
                
                // Generate next frame
                const inferenceResult = await this.inferenceEngine.runInference(currentState, modelName);
                const bvhFrame = this.inferenceEngine.actionsToBVH(inferenceResult.actions, timestamp);
                
                // Apply motion blending
                if (this.motionBlending.enabled && this.motionHistory.length > 0) {
                    this.applyMotionBlending(bvhFrame);
                }
                
                // Add frame to timeline buffer
                timeline.frameBuffer.addFrame(trackName, bvhFrame);
                
                // Update state for next frame
                currentState = this.updateCharacterState(inferenceResult.actions, 1/targetFPS);
                
                // Update motion history
                this.updateMotionHistory(bvhFrame, inferenceResult.actions);
                
                frameCounter++;
                
                // Schedule next frame
                setTimeout(motionLoop, 1000 / targetFPS);
                
            } catch (error) {
                console.error('Error in real-time motion synthesis:', error);
                setTimeout(motionLoop, 1000 / targetFPS);  // Continue despite errors
            }
        };
        
        // Start the motion synthesis loop
        motionLoop();
        
        // Return control object
        return {
            stop: () => {
                isRunning = false;
                console.log('Stopped real-time DeepMimic motion synthesis');
            },
            isRunning: () => isRunning,
            getFrameCount: () => frameCounter,
            getCurrentState: () => currentState
        };
    }
    
    /**
     * Update conversion performance statistics
     */
    updateConversionStats(inferenceTime) {
        this.conversionStats.totalFrames++;
        this.conversionStats.totalTime += inferenceTime;
        this.conversionStats.averageTime = this.conversionStats.totalTime / this.conversionStats.totalFrames;
        this.conversionStats.lastConversionTime = inferenceTime;
    }
    
    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        return {
            conversion: { ...this.conversionStats },
            inference: this.inferenceEngine.getPerformanceStats(),
            motionHistory: this.motionHistory.length,
            currentPhase: this.currentPhase
        };
    }
    
    /**
     * Load and switch DeepMimic policy
     */
    async loadPolicy(modelPath, modelName = null) {
        return await this.inferenceEngine.loadModel(modelPath, modelName);
    }
    
    /**
     * Switch to a different policy
     */
    switchPolicy(modelName) {
        this.inferenceEngine.switchModel(modelName);
        console.log(`Switched to DeepMimic policy: ${modelName}`);
    }
    
    /**
     * Get available policies
     */
    getAvailablePolicies() {
        return this.inferenceEngine.getAvailableModels();
    }
    
    /**
     * Reset motion state
     */
    resetMotionState() {
        this.currentPhase = 0;
        this.motionHistory = [];
        this.conversionStats = {
            totalFrames: 0,
            totalTime: 0,
            averageTime: 0,
            lastConversionTime: 0
        };
        
        console.log('DeepMimic motion state reset');
    }
    
    /**
     * Cleanup resources
     */
    dispose() {
        this.resetMotionState();
        this.inferenceEngine.dispose();
        this.isInitialized = false;
        
        console.log('DeepMimic BVH Converter disposed');
    }
}

// Export for use in modules or global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeepMimicBVHConverter;
} else if (typeof window !== 'undefined') {
    window.DeepMimicBVHConverter = DeepMimicBVHConverter;
}
