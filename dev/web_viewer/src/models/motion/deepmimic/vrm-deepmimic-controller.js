/**
 * Simple Example: Integrating DeepMimic with BVHTimeline for VRM Avatar Animation
 * 
 * This example shows how to set up and use the DeepMimic BVH integration
 * to drive a VRM avatar with AI-generated motion sequences.
 */

class VRMDeepMimicController {
    constructor(vrmAvatar, options = {}) {
        this.vrmAvatar = vrmAvatar;
        this.options = {
            frameRate: 30,
            smoothingFactor: 0.2,
            autoCleanup: true,
            maxClips: 10,
            ...options
        };
        
        this.timeline = null;
        this.generator = null;
        this.integration = null;
        this.isInitialized = false;
        this.activeMotions = new Map();
    }
    
    /**
     * Initialize the system
     */
    async initialize() {
        try {
            console.log('Initializing VRM DeepMimic Controller...');
            
            // Create BVH Timeline
            this.timeline = new BVHTimeline({
                framerate: this.options.frameRate,
                maxBufferSize: 300,
                lookaheadFrames: 60,
                onFrameUpdate: (frame) => this.applyFrameToVRM(frame)
            });
            
            // Create DeepMimic Generator
            this.generator = new DeepMimicBVHGenerator({
                frameRate: this.options.frameRate,
                smoothingFactor: this.options.smoothingFactor
            });
            
            // Initialize with auto-discovered models
            await this.generator.initialize();
            
            // Create timeline integration
            this.integration = new BVHTimelineDeepMimicIntegration(
                this.timeline,
                this.generator,
                {
                    defaultTrack: 'character_motion',
                    bufferAhead: 2.0,
                    autoCleanup: this.options.autoCleanup,
                    maxClips: this.options.maxClips
                }
            );
            
            this.isInitialized = true;
            console.log('VRM DeepMimic Controller initialized successfully');
            console.log('Available models:', this.generator.getAvailableModels());
            
        } catch (error) {
            console.error('Failed to initialize VRM DeepMimic Controller:', error);
            throw error;
        }
    }
    
    /**
     * Apply BVH frame data to VRM avatar
     */
    async applyFrameToVRM(frame) {
        if (!this.vrmAvatar || !frame) return;
        
        try {
            const currentFrame = await this.timeline.getCurrentFrame();
            
            if (currentFrame && currentFrame.motionData) {
                this.applyMotionDataToVRM(currentFrame.motionData);
            }
        } catch (error) {
            console.warn('Error applying frame to VRM:', error);
        }
    }
    
    /**
     * Apply motion data to VRM bones
     */
    applyMotionDataToVRM(motionData) {
        if (!this.vrmAvatar.humanoid) return;
        
        for (const [boneName, boneData] of Object.entries(motionData)) {
            const vrmBone = this.vrmAvatar.humanoid.getBoneNode(boneName);
            
            if (vrmBone && boneData && boneData.length > 0) {
                if (boneName === 'hips') {
                    // Apply position and rotation for root bone
                    if (boneData.length >= 6) {
                        vrmBone.position.set(
                            boneData[0] * 0.01,  // Scale down position
                            boneData[1] * 0.01,
                            boneData[2] * 0.01
                        );
                        
                        vrmBone.rotation.set(
                            boneData[3] * Math.PI / 180,  // Convert to radians
                            boneData[4] * Math.PI / 180,
                            boneData[5] * Math.PI / 180
                        );
                    }
                } else {
                    // Apply rotation only for other bones
                    if (boneData.length >= 3) {
                        vrmBone.rotation.set(
                            boneData[0] * Math.PI / 180,
                            boneData[1] * Math.PI / 180,
                            boneData[2] * Math.PI / 180
                        );
                    }
                }
            }
        }
    }
    
    /**
     * Start a motion sequence
     */
    async startMotion(motionType, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Controller not initialized. Call initialize() first.');
        }
        
        try {
            let clipId;
            
            if (this.integration.getAvailableTemplates().includes(motionType)) {
                // Use motion template
                clipId = await this.integration.addMotionFromTemplate(
                    motionType,
                    this.timeline.currentTime,
                    options
                );
            } else {
                // Use custom motion parameters
                clipId = await this.integration.addMotionClip({
                    model: motionType,
                    startTime: this.timeline.currentTime,
                    duration: options.duration || 5.0,
                    ...options
                });
            }
            
            this.activeMotions.set(clipId, {
                type: motionType,
                startTime: this.timeline.currentTime,
                options: options
            });
            
            console.log(`Started motion: ${motionType} (${clipId})`);
            return clipId;
            
        } catch (error) {
            console.error(`Failed to start motion ${motionType}:`, error);
            throw error;
        }
    }
    
    /**
     * Start real-time controllable motion
     */
    async startRealTimeMotion(model = 'walk', initialParams = {}) {
        if (!this.isInitialized) {
            throw new Error('Controller not initialized');
        }
        
        const defaultParams = {
            speed: 1.0,
            direction: [0, 0, 1],
            ...initialParams
        };
        
        const clipId = await this.integration.addRealTimeMotion({
            model: model,
            targetMotion: defaultParams,
            trackName: 'realtime_control'
        });
        
        this.activeMotions.set(clipId, {
            type: 'realtime',
            model: model,
            realtime: true
        });
        
        console.log(`Started real-time motion: ${model} (${clipId})`);
        return clipId;
    }
    
    /**
     * Update real-time motion parameters
     */
    updateRealTimeMotion(clipId, params) {
        if (this.activeMotions.has(clipId)) {
            const motion = this.activeMotions.get(clipId);
            if (motion.realtime) {
                this.integration.updateTargetMotion(clipId, params);
                console.log(`Updated real-time motion ${clipId}:`, params);
            }
        }
    }
    
    /**
     * Stop a specific motion
     */
    stopMotion(clipId) {
        if (this.integration.removeClip(clipId)) {
            this.activeMotions.delete(clipId);
            console.log(`Stopped motion: ${clipId}`);
            return true;
        }
        return false;
    }
    
    /**
     * Stop all motions
     */
    stopAllMotions() {
        this.integration.clearAllMotions();
        this.activeMotions.clear();
        console.log('Stopped all motions');
    }
    
    /**
     * Play/pause timeline
     */
    play() {
        if (this.timeline) {
            this.timeline.play();
            console.log('Motion playback started');
        }
    }
    
    pause() {
        if (this.timeline) {
            this.timeline.pause();
            console.log('Motion playback paused');
        }
    }
    
    /**
     * Seek to specific time
     */
    seekTo(time) {
        if (this.timeline) {
            this.timeline.seek(time);
            console.log(`Seeked to time: ${time}s`);
        }
    }
    
    /**
     * Create a motion sequence
     */
    async createMotionSequence(actions, startTime = null) {
        const baseTime = startTime || this.timeline.currentTime;
        const clipIds = [];
        
        let currentTime = baseTime;
        
        for (const action of actions) {
            const clipId = await this.startMotion(action.type, {
                ...action,
                startTime: currentTime
            });
            
            clipIds.push(clipId);
            currentTime += action.duration || 5.0;
        }
        
        console.log(`Created motion sequence: ${clipIds.length} actions`);
        return clipIds;
    }
    
    /**
     * Get performance statistics
     */
    getStats() {
        if (!this.isInitialized) return null;
        
        return {
            timeline: this.timeline.getStats(),
            generator: this.generator.getPerformanceStats(),
            motions: this.integration.getMotionStats(),
            activeMotions: this.activeMotions.size
        };
    }
    
    /**
     * Get available motions
     */
    getAvailableMotions() {
        if (!this.isInitialized) return [];
        
        return {
            templates: this.integration.getAvailableTemplates(),
            models: this.generator.getAvailableModels()
        };
    }
    
    /**
     * Cleanup and dispose
     */
    dispose() {
        this.stopAllMotions();
        
        if (this.integration) {
            this.integration.dispose();
        }
        
        if (this.generator) {
            this.generator.dispose();
        }
        
        if (this.timeline) {
            this.timeline.dispose();
        }
        
        this.activeMotions.clear();
        this.isInitialized = false;
        
        console.log('VRM DeepMimic Controller disposed');
    }
}

// Usage Example
async function exampleUsage() {
    // Assume you have a VRM avatar loaded
    const vrmAvatar = await loadVRMAvatar('./avatar.vrm');
    
    // Create controller
    const controller = new VRMDeepMimicController(vrmAvatar, {
        frameRate: 30,
        smoothingFactor: 0.15,
        autoCleanup: true
    });
    
    // Initialize
    await controller.initialize();
    
    // Start playback
    controller.play();
    
    // Example 1: Simple walking motion
    const walkClipId = await controller.startMotion('walk', {
        duration: 5.0,
        loop: true
    });
    
    // Example 2: Real-time controllable motion
    const realtimeId = await controller.startRealTimeMotion('walk', {
        speed: 1.0,
        direction: [0, 0, 1]
    });
    
    // Update real-time motion parameters
    setTimeout(() => {
        controller.updateRealTimeMotion(realtimeId, {
            speed: 2.0,
            direction: [1, 0, 0] // Turn right
        });
    }, 3000);
    
    // Example 3: Motion sequence
    const sequence = [
        { type: 'walk', duration: 3.0 },
        { type: 'jump', duration: 2.0 },
        { type: 'dance', duration: 5.0 },
        { type: 'idle', duration: 2.0 }
    ];
    
    setTimeout(() => {
        controller.createMotionSequence(sequence);
    }, 10000);
    
    // Example 4: Interactive controls
    document.addEventListener('keydown', (event) => {
        switch (event.key) {
            case 'w':
                controller.updateRealTimeMotion(realtimeId, {
                    speed: 2.0,
                    direction: [0, 0, 1] // Forward
                });
                break;
            case 's':
                controller.updateRealTimeMotion(realtimeId, {
                    speed: 1.0,
                    direction: [0, 0, -1] // Backward
                });
                break;
            case 'a':
                controller.updateRealTimeMotion(realtimeId, {
                    speed: 1.5,
                    direction: [-1, 0, 0] // Left
                });
                break;
            case 'd':
                controller.updateRealTimeMotion(realtimeId, {
                    speed: 1.5,
                    direction: [1, 0, 0] // Right
                });
                break;
            case ' ':
                controller.startMotion('jump', { duration: 2.0 });
                break;
        }
    });
    
    // Monitor performance
    setInterval(() => {
        const stats = controller.getStats();
        console.log('Performance stats:', stats);
    }, 5000);
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        controller.dispose();
    });
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VRMDeepMimicController;
} else if (typeof window !== 'undefined') {
    window.VRMDeepMimicController = VRMDeepMimicController;
    window.exampleUsage = exampleUsage;
}
