/**
 * Enhanced DeepMimic BVH Generator
 * Integrates with BVHTimeline to generate continuous motion sequences for VRM avatars
 */

class DeepMimicBVHGenerator {
    constructor(options = {}) {
        this.inference = new DeepMimicInference();
        this.frameRate = options.frameRate || 30;
        this.isInitialized = false;
        
        // State management for continuous motion
        this.currentState = null;
        this.currentPhase = 0;
        this.motionHistory = [];
        this.maxHistoryLength = options.maxHistoryLength || 100;
        
        // Motion synthesis parameters
        this.synthParams = {
            smoothingFactor: options.smoothingFactor || 0.15,
            velocityDamping: options.velocityDamping || 0.95,
            phaseIncrement: options.phaseIncrement || 0.02,
            stabilizationStrength: options.stabilizationStrength || 0.1
        };
        
        // Available models
        this.availableModels = {};
        this.currentModelName = null;
        
        // Performance tracking
        this.perfStats = {
            framesGenerated: 0,
            totalTime: 0,
            averageTime: 0,
            lastFrameTime: 0
        };
        
        // VRM bone mapping (standard VRM bone names)
        this.vrmBoneMapping = this.createVRMBoneMapping();
        
        console.log('DeepMimic BVH Generator initialized');
    }
    
    /**
     * Create VRM bone mapping for output format
     */
    createVRMBoneMapping() {
        return {
            // Core skeleton
            'hips': { index: 0, channels: ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation'] },
            'spine': { index: 1, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'chest': { index: 2, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'upperChest': { index: 3, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'neck': { index: 4, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'head': { index: 5, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left leg
            'leftUpperLeg': { index: 6, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftLowerLeg': { index: 7, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftFoot': { index: 8, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftToes': { index: 9, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right leg
            'rightUpperLeg': { index: 10, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightLowerLeg': { index: 11, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightFoot': { index: 12, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightToes': { index: 13, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left arm
            'leftShoulder': { index: 14, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftUpperArm': { index: 15, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftLowerArm': { index: 16, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftHand': { index: 17, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right arm
            'rightShoulder': { index: 18, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightUpperArm': { index: 19, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightLowerArm': { index: 20, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightHand': { index: 21, channels: ['Xrotation', 'Yrotation', 'Zrotation'] }
        };
    }
    
    /**
     * Initialize the generator with available models
     */
    async initialize(modelPaths = null) {
        try {
            await this.inference.initialize();
            
            // Auto-discover models if not provided
            if (!modelPaths) {
                modelPaths = await this.discoverAvailableModels();
            }
            
            // Load models
            for (const [name, path] of Object.entries(modelPaths)) {
                try {
                    await this.loadModel(name, path);
                    console.log(`Loaded DeepMimic model: ${name}`);
                } catch (error) {
                    console.warn(`Failed to load model ${name}:`, error.message);
                }
            }
            
            // Set default model
            const modelNames = Object.keys(this.availableModels);
            if (modelNames.length > 0) {
                this.currentModelName = modelNames[0];
                console.log(`Set default model: ${this.currentModelName}`);
            }
            
            // Initialize default state
            this.currentState = this.createDefaultState();
            
            this.isInitialized = true;
            console.log('DeepMimic BVH Generator ready');
            
        } catch (error) {
            console.error('Failed to initialize DeepMimic BVH Generator:', error);
            throw error;
        }
    }
    
    /**
     * Discover available ONNX models in the deepmimic directory
     */
    async discoverAvailableModels() {
        const basePath = '/home/barberb/motion/dev/web_viewer/web_porting_poc/deepmimic/';
        const models = {};
        
        // Common model names - these should match your available ONNX files
        const modelNames = [
            'walk', 'run', 'jump', 'dance_a', 'dance_b', 'kick', 'punch',
            'backflip', 'cartwheel', 'crawl', 'getup_facedown', 'getup_faceup',
            'roll', 'spinkick'
        ];
        
        for (const name of modelNames) {
            // Try compatible versions first
            const compatiblePath = `${basePath}compatible_humanoid3d_humanoid3d_${name}.onnx`;
            const originalPath = `${basePath}humanoid3d_humanoid3d_${name}.onnx`;
            
            models[name] = compatiblePath; // Prefer compatible versions
        }
        
        return models;
    }
    
    /**
     * Load a specific model
     */
    async loadModel(name, modelPath) {
        try {
            await this.inference.loadModel(modelPath);
            const modelInfo = this.inference.getModelInfo();
            
            this.availableModels[name] = {
                path: modelPath,
                info: modelInfo,
                loadedAt: Date.now()
            };
            
            return modelInfo;
        } catch (error) {
            console.error(`Failed to load model ${name}:`, error);
            throw error;
        }
    }
    
    /**
     * Switch to a different model
     */
    async switchModel(modelName) {
        if (!this.availableModels[modelName]) {
            throw new Error(`Model '${modelName}' not available. Available models: ${Object.keys(this.availableModels).join(', ')}`);
        }
        
        if (modelName === this.currentModelName) {
            return; // Already using this model
        }
        
        const modelInfo = this.availableModels[modelName];
        await this.inference.loadModel(modelInfo.path);
        this.currentModelName = modelName;
        
        console.log(`Switched to model: ${modelName}`);
    }
    
    /**
     * Create default character state (197-dimensional for DeepMimic humanoid)
     */
    createDefaultState() {
        const state = new Float32Array(197);
        
        // Root position (world coordinates)
        state[0] = 0.0;   // x
        state[1] = 1.0;   // y (height)
        state[2] = 0.0;   // z
        
        // Root orientation (quaternion: x, y, z, w)
        state[3] = 0.0;   // qx
        state[4] = 0.0;   // qy
        state[5] = 0.0;   // qz
        state[6] = 1.0;   // qw
        
        // Root linear velocity
        state[7] = 0.0;   // vx
        state[8] = 0.0;   // vy
        state[9] = 0.0;   // vz
        
        // Root angular velocity
        state[10] = 0.0;  // wx
        state[11] = 0.0;  // wy
        state[12] = 0.0;  // wz
        
        // Joint positions (remaining elements are joint angles and velocities)
        // Initialize with neutral pose
        for (let i = 13; i < 197; i++) {
            state[i] = 0.0;
        }
        
        // Add some variation for natural standing pose
        if (state.length > 50) {
            // Slight hip rotation for natural stance
            state[20] = 0.05;  // Left hip
            state[25] = -0.05; // Right hip
            
            // Slight knee bend
            state[21] = 0.1;   // Left knee
            state[26] = 0.1;   // Right knee
        }
        
        return state;
    }
    
    /**
     * Generate a single BVH frame from current character state
     */
    async generateFrame(time, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Generator not initialized. Call initialize() first.');
        }
        
        if (!this.currentModelName) {
            throw new Error('No model loaded. Load a model first.');
        }
        
        const startTime = performance.now();
        
        try {
            // Update state based on time and target motion
            if (options.targetMotion) {
                this.updateStateForTargetMotion(options.targetMotion, time);
            }
            
            // Update phase for cyclic motions
            this.currentPhase = (this.currentPhase + this.synthParams.phaseIncrement) % (2 * Math.PI);
            
            // Add phase information to state
            if (this.currentState.length > 190) {
                this.currentState[190] = Math.sin(this.currentPhase);
                this.currentState[191] = Math.cos(this.currentPhase);
                this.currentState[192] = this.currentPhase / (2 * Math.PI); // Normalized phase
            }
            
            // Run inference
            const result = await this.inference.predict(this.currentState);
            
            // Convert actions to BVH frame
            const bvhFrame = this.actionsToBVHFrame(result.actions, time);
            
            // Apply smoothing with previous frames
            if (this.motionHistory.length > 0) {
                this.applySmoothingToBVHFrame(bvhFrame);
            }
            
            // Update character state based on generated actions
            this.updateCharacterState(result.actions);
            
            // Store in history
            this.addToMotionHistory(bvhFrame, result.actions);
            
            // Update performance stats
            const frameTime = performance.now() - startTime;
            this.updatePerformanceStats(frameTime);
            
            return bvhFrame;
            
        } catch (error) {
            console.error('Error generating BVH frame:', error);
            return this.createDefaultBVHFrame(time);
        }
    }
    
    /**
     * Convert DeepMimic actions to BVH frame format
     */
    actionsToBVHFrame(actions, time) {
        const frame = {
            time: time,
            motionData: {},
            metadata: {
                type: 'deepmimic',
                model: this.currentModelName,
                phase: this.currentPhase,
                timestamp: Date.now()
            }
        };
        
        // Extract root motion (first 6 components typically)
        const rootPosition = [
            actions[0] || 0,  // x
            actions[1] || 1,  // y
            actions[2] || 0   // z
        ];
        
        const rootRotation = [
            actions[3] || 0,  // rx
            actions[4] || 0,  // ry
            actions[5] || 0   // rz
        ];
        
        // Process joint rotations (remaining actions)
        let actionIndex = 6;
        for (const [boneName, boneInfo] of Object.entries(this.vrmBoneMapping)) {
            const boneData = [];
            
            // Add position data for root bone
            if (boneName === 'hips') {
                boneData.push(...rootPosition);
                boneData.push(...rootRotation);
            } else {
                // Add rotation data for other bones
                for (let i = 0; i < 3; i++) {
                    if (actionIndex < actions.length) {
                        boneData.push(actions[actionIndex] * 57.2958); // Convert to degrees
                        actionIndex++;
                    } else {
                        boneData.push(0);
                    }
                }
            }
            
            frame.motionData[boneName] = boneData;
        }
        
        return frame;
    }
    
    /**
     * Apply temporal smoothing to BVH frame
     */
    applySmoothingToBVHFrame(frame) {
        if (this.motionHistory.length === 0) return;
        
        const lastFrame = this.motionHistory[this.motionHistory.length - 1].frame;
        const smoothingFactor = this.synthParams.smoothingFactor;
        
        // Smooth each bone's motion data
        for (const boneName in frame.motionData) {
            if (lastFrame.motionData[boneName]) {
                const current = frame.motionData[boneName];
                const previous = lastFrame.motionData[boneName];
                
                for (let i = 0; i < current.length; i++) {
                    if (i < previous.length) {
                        current[i] = previous[i] * (1 - smoothingFactor) + current[i] * smoothingFactor;
                    }
                }
            }
        }
    }
    
    /**
     * Update character state based on generated actions
     */
    updateCharacterState(actions) {
        const deltaTime = 1.0 / this.frameRate;
        
        // Update root position based on velocity
        if (actions.length >= 3) {
            this.currentState[0] += actions[0] * deltaTime; // x
            this.currentState[1] += actions[1] * deltaTime; // y
            this.currentState[2] += actions[2] * deltaTime; // z
        }
        
        // Update root orientation (simplified)
        if (actions.length >= 6) {
            this.currentState[3] += actions[3] * deltaTime; // qx
            this.currentState[4] += actions[4] * deltaTime; // qy
            this.currentState[5] += actions[5] * deltaTime; // qz
            
            // Normalize quaternion
            const qx = this.currentState[3];
            const qy = this.currentState[4];
            const qz = this.currentState[5];
            const qw = this.currentState[6];
            const qMag = Math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
            if (qMag > 0) {
                this.currentState[3] /= qMag;
                this.currentState[4] /= qMag;
                this.currentState[5] /= qMag;
                this.currentState[6] /= qMag;
            }
        }
        
        // Update velocities with damping
        const damping = this.synthParams.velocityDamping;
        this.currentState[7] *= damping;  // vx
        this.currentState[8] *= damping;  // vy
        this.currentState[9] *= damping;  // vz
        this.currentState[10] *= damping; // wx
        this.currentState[11] *= damping; // wy
        this.currentState[12] *= damping; // wz
    }
    
    /**
     * Update state for target motion (if provided)
     */
    updateStateForTargetMotion(targetMotion, time) {
        // This could be enhanced to incorporate reference motion data
        // For now, we'll use basic target direction/speed
        if (targetMotion.direction) {
            const speed = targetMotion.speed || 1.0;
            this.currentState[7] = targetMotion.direction[0] * speed; // vx
            this.currentState[9] = targetMotion.direction[2] * speed; // vz
        }
        
        if (targetMotion.turnRate) {
            this.currentState[11] = targetMotion.turnRate; // wy (yaw rate)
        }
    }
    
    /**
     * Add frame to motion history
     */
    addToMotionHistory(frame, actions) {
        this.motionHistory.push({
            frame: JSON.parse(JSON.stringify(frame)), // Deep copy
            actions: [...actions],
            timestamp: performance.now()
        });
        
        // Maintain history size
        if (this.motionHistory.length > this.maxHistoryLength) {
            this.motionHistory = this.motionHistory.slice(-this.maxHistoryLength);
        }
    }
    
    /**
     * Create default BVH frame for fallback
     */
    createDefaultBVHFrame(time) {
        const frame = {
            time: time,
            motionData: {},
            metadata: {
                type: 'default',
                model: 'none',
                phase: 0,
                timestamp: Date.now()
            }
        };
        
        // Create neutral pose
        for (const boneName of Object.keys(this.vrmBoneMapping)) {
            if (boneName === 'hips') {
                frame.motionData[boneName] = [0, 1, 0, 0, 0, 0]; // position + rotation
            } else {
                frame.motionData[boneName] = [0, 0, 0]; // rotation only
            }
        }
        
        return frame;
    }
    
    /**
     * Generate continuous motion sequence for timeline integration
     */
    async generateMotionClip(options = {}) {
        const {
            duration = 5.0,           // seconds
            model = null,             // model name
            targetMotion = null,      // motion parameters
            startState = null,        // initial state
            clipId = null,            // timeline clip ID
            trackName = 'deepmimic'   // timeline track
        } = options;
        
        // Switch model if specified
        if (model && model !== this.currentModelName) {
            await this.switchModel(model);
        }
        
        // Set initial state if provided
        if (startState) {
            this.currentState = new Float32Array(startState);
        }
        
        const frameCount = Math.ceil(duration * this.frameRate);
        const frames = [];
        
        console.log(`Generating ${frameCount} frames for ${duration}s motion clip using ${this.currentModelName}`);
        
        for (let i = 0; i < frameCount; i++) {
            const time = i / this.frameRate;
            const frame = await this.generateFrame(time, { targetMotion });
            frames.push(frame);
            
            // Progress logging
            if (i % 30 === 0) {
                const progress = (i / frameCount * 100).toFixed(1);
                console.log(`Motion generation progress: ${progress}%`);
            }
        }
        
        // Create timeline clip
        const clip = {
            id: clipId || `deepmimic_${this.currentModelName}_${Date.now()}`,
            type: 'deepmimic_generated',
            startTime: 0,
            duration: duration,
            weight: 1.0,
            blendMode: 'replace',
            frames: frames,
            metadata: {
                model: this.currentModelName,
                frameRate: this.frameRate,
                frameCount: frames.length,
                generationTime: Date.now(),
                targetMotion: targetMotion
            }
        };
        
        console.log(`Generated motion clip: ${clip.id} (${frames.length} frames)`);
        return clip;
    }
    
    /**
     * Create a generator function for BVHTimeline integration
     */
    createTimelineGenerator(options = {}) {
        const {
            model = null,
            targetMotion = null,
            resetState = false
        } = options;
        
        return async (time, frameIndex) => {
            // Switch model if needed
            if (model && model !== this.currentModelName) {
                await this.switchModel(model);
            }
            
            // Reset state if requested
            if (resetState && frameIndex === 0) {
                this.currentState = this.createDefaultState();
                this.currentPhase = 0;
                this.motionHistory = [];
            }
            
            return await this.generateFrame(time, { targetMotion });
        };
    }
    
    /**
     * Update performance statistics
     */
    updatePerformanceStats(frameTime) {
        this.perfStats.framesGenerated++;
        this.perfStats.totalTime += frameTime;
        this.perfStats.averageTime = this.perfStats.totalTime / this.perfStats.framesGenerated;
        this.perfStats.lastFrameTime = frameTime;
    }
    
    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        return {
            ...this.perfStats,
            fps: this.perfStats.averageTime > 0 ? 1000 / this.perfStats.averageTime : 0,
            currentModel: this.currentModelName,
            availableModels: Object.keys(this.availableModels),
            historyLength: this.motionHistory.length,
            phase: this.currentPhase
        };
    }
    
    /**
     * Reset motion state
     */
    resetState() {
        this.currentState = this.createDefaultState();
        this.currentPhase = 0;
        this.motionHistory = [];
        this.perfStats = {
            framesGenerated: 0,
            totalTime: 0,
            averageTime: 0,
            lastFrameTime: 0
        };
        
        console.log('Motion state reset');
    }
    
    /**
     * Get available models
     */
    getAvailableModels() {
        return Object.keys(this.availableModels);
    }
    
    /**
     * Get current model information
     */
    getCurrentModelInfo() {
        if (!this.currentModelName) return null;
        return this.availableModels[this.currentModelName];
    }
    
    /**
     * Cleanup resources
     */
    dispose() {
        this.inference.dispose();
        this.resetState();
        this.availableModels = {};
        this.currentModelName = null;
        this.isInitialized = false;
        
        console.log('DeepMimic BVH Generator disposed');
    }
}

// Export for use in modules or global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeepMimicBVHGenerator;
} else if (typeof window !== 'undefined') {
    window.DeepMimicBVHGenerator = DeepMimicBVHGenerator;
}
