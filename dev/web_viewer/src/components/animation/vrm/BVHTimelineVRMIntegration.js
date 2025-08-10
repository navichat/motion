/**
 * BVH Timeline VRM Integration
 * 
 * This module provides integration between the BVH Timeline compositor
 * and the existing VRM BVH Adapter system for driving 3D avatars.
 */

class BVHTimelineVRMIntegration {
    constructor(vrmBVHAdapter, options = {}) {
        this.vrmBVHAdapter = vrmBVHAdapter;
        this.timeline = null;
        this.isConnected = false;
        
        // Integration options
        this.realtime = options.realtime !== false; // Default to realtime
        this.preloadFrames = options.preloadFrames || 30; // Frames to preload
        this.smoothing = options.smoothing !== false; // Frame smoothing
        this.interpolation = options.interpolation || 'linear'; // 'linear', 'cubic', 'quaternion'
        
        // Frame management
        this.frameQueue = [];
        this.lastFrameTime = 0;
        this.targetFramerate = options.framerate || 30;
        this.frameBuffer = new Map(); // For non-realtime playback
        
        // Performance monitoring
        this.stats = {
            framesProcessed: 0,
            averageProcessingTime: 0,
            droppedFrames: 0,
            lastUpdateTime: 0
        };
        
        console.log('[BVH Timeline VRM Integration] Initialized');
    }
    
    /**
     * Connect to a BVH Timeline instance
     */
    connectTimeline(timeline) {
        if (this.isConnected) {
            this.disconnectTimeline();
        }
        
        this.timeline = timeline;
        
        // Override the timeline's frame update callback
        const originalCallback = timeline.onFrameUpdate;
        timeline.onFrameUpdate = (frame, time) => {
            this.handleTimelineFrame(frame, time);
            if (originalCallback) {
                originalCallback(frame, time);
            }
        };
        
        this.isConnected = true;
        console.log('[BVH Timeline VRM] Connected to timeline');
        
        return this;
    }
    
    /**
     * Disconnect from current timeline
     */
    disconnectTimeline() {
        if (this.timeline) {
            this.timeline.onFrameUpdate = null;
            this.timeline = null;
        }
        
        this.isConnected = false;
        this.frameQueue = [];
        this.frameBuffer.clear();
        
        console.log('[BVH Timeline VRM] Disconnected from timeline');
    }
    
    /**
     * Handle frame updates from timeline
     */
    handleTimelineFrame(frame, time) {
        const startTime = performance.now();
        
        try {
            if (this.realtime) {
                this.processRealtimeFrame(frame, time);
            } else {
                this.bufferFrame(frame, time);
            }
            
            this.updateStats(startTime);
            
        } catch (error) {
            console.error('[BVH Timeline VRM] Frame processing error:', error);
            this.stats.droppedFrames++;
        }
    }
    
    /**
     * Process frame in realtime mode
     */
    processRealtimeFrame(frame, time) {
        // Convert timeline frame to VRM-compatible format
        const vrmFrame = this.convertTimelineFrameToVRM(frame, time);
        
        // Apply smoothing if enabled
        if (this.smoothing && this.frameQueue.length > 0) {
            const lastFrame = this.frameQueue[this.frameQueue.length - 1];
            vrmFrame = this.smoothFrames(lastFrame.vrmFrame, vrmFrame, 0.8);
        }
        
        // Queue frame for VRM processing
        this.frameQueue.push({
            timelineFrame: frame,
            vrmFrame: vrmFrame,
            timestamp: time,
            processed: false
        });
        
        // Keep queue size manageable
        if (this.frameQueue.length > this.preloadFrames) {
            this.frameQueue.shift();
        }
        
        // Process the frame immediately in realtime mode
        this.applyFrameToVRM(vrmFrame, time);
    }
    
    /**
     * Buffer frame for non-realtime processing
     */
    bufferFrame(frame, time) {
        const frameKey = Math.floor(time * this.targetFramerate);
        this.frameBuffer.set(frameKey, {
            timelineFrame: frame,
            vrmFrame: this.convertTimelineFrameToVRM(frame, time),
            timestamp: time
        });
    }
    
    /**
     * Convert timeline frame format to VRM-compatible format
     */
    convertTimelineFrameToVRM(timelineFrame, time) {
        if (!timelineFrame.motionData || timelineFrame.motionData.length === 0) {
            return {
                timestamp: time,
                bones: {},
                metadata: timelineFrame.metadata || { type: 'default' }
            };
        }
        
        // Map timeline bone data to VRM bone structure
        const vrmBones = {};
        
        timelineFrame.motionData.forEach((boneData, boneIndex) => {
            const boneName = this.mapTimelineBoneToVRM(boneIndex);
            if (boneName && boneData.length >= 6) {
                vrmBones[boneName] = {
                    position: {
                        x: boneData[0] || 0,
                        y: boneData[1] || 0,
                        z: boneData[2] || 0
                    },
                    rotation: {
                        x: this.degreesToRadians(boneData[3] || 0),
                        y: this.degreesToRadians(boneData[4] || 0),
                        z: this.degreesToRadians(boneData[5] || 0)
                    }
                };
            }
        });
        
        return {
            timestamp: time,
            bones: vrmBones,
            metadata: timelineFrame.metadata
        };
    }
    
    /**
     * Apply VRM frame to the avatar
     */
    applyFrameToVRM(vrmFrame, time) {
        if (!this.vrmBVHAdapter || !vrmFrame.bones) {
            return;
        }
        
        try {
            // Apply bone transformations to VRM model
            for (const [boneName, boneData] of Object.entries(vrmFrame.bones)) {
                this.vrmBVHAdapter.updateBone(boneName, boneData.position, boneData.rotation);
            }

            // Also apply facial visemes to blendshapes if available
            const md = vrmFrame.metadata || {};
            if (md.faceViseme !== undefined || md.viseme !== undefined) {
                const vis = md.faceViseme ?? md.viseme;
                const { name: exprName, weight } = this.mapVisemeToBlendshape(vis);
                if (exprName && typeof this.vrmBVHAdapter.updateBlendshape === 'function') {
                    this.vrmBVHAdapter.updateBlendshape(exprName, weight);
                }
            }
            
            // Update VRM model
            if (this.vrmBVHAdapter.update) {
                this.vrmBVHAdapter.update(time);
            }
            
            this.stats.framesProcessed++;
            this.stats.lastUpdateTime = time;
            
        } catch (error) {
            console.error('[BVH Timeline VRM] VRM update error:', error);
            this.stats.droppedFrames++;
        }
    }

    /**
     * Translate viseme label/id to VRM expression name and weight [0,1].
     * Defaults to VRM mouth shapes when available.
     */
    mapVisemeToBlendshape(vis) {
        // Accept numbers (0..N) or common labels
        const v = (vis ?? '').toString().toLowerCase();
        const map = {
            '0': 'aa', 'a': 'aa', 'aa': 'aa',
            '1': 'ih', 'i': 'ih', 'ih': 'ih', 'ee': 'ih',
            '2': 'ou', 'u': 'ou', 'ou': 'ou', 'oo': 'ou',
            '3': 'e', 'e': 'e',
            '4': 'o', 'o': 'o',
            'rest': 'neutral'
        };
        const name = map[v] || 'neutral';
        const weight = (name === 'neutral') ? 0.0 : 1.0;
        return { name, weight };
    }
    
    /**
     * Smooth between two frames
     */
    smoothFrames(fromFrame, toFrame, factor) {
        const smoothedFrame = {
            timestamp: toFrame.timestamp,
            bones: {},
            metadata: toFrame.metadata
        };
        
        // Interpolate between frame bone data
        for (const boneName in toFrame.bones) {
            const fromBone = fromFrame.bones[boneName];
            const toBone = toFrame.bones[boneName];
            
            if (fromBone && toBone) {
                smoothedFrame.bones[boneName] = {
                    position: {
                        x: this.lerp(fromBone.position.x, toBone.position.x, factor),
                        y: this.lerp(fromBone.position.y, toBone.position.y, factor),
                        z: this.lerp(fromBone.position.z, toBone.position.z, factor)
                    },
                    rotation: this.interpolation === 'quaternion' 
                        ? this.slerpRotation(fromBone.rotation, toBone.rotation, factor)
                        : {
                            x: this.lerpAngle(fromBone.rotation.x, toBone.rotation.x, factor),
                            y: this.lerpAngle(fromBone.rotation.y, toBone.rotation.y, factor),
                            z: this.lerpAngle(fromBone.rotation.z, toBone.rotation.z, factor)
                        }
                };
            } else {
                smoothedFrame.bones[boneName] = toBone;
            }
        }
        
        return smoothedFrame;
    }
    
    /**
     * Manual frame playback for non-realtime mode
     */
    playFrameAtTime(time) {
        const frameKey = Math.floor(time * this.targetFramerate);
        const bufferedFrame = this.frameBuffer.get(frameKey);
        
        if (bufferedFrame) {
            this.applyFrameToVRM(bufferedFrame.vrmFrame, time);
            return true;
        }
        
        return false;
    }
    
    /**
     * Preload frames for a time range
     */
    async preloadFrames(startTime, endTime) {
        if (!this.timeline) return;
        
        const frameCount = Math.ceil((endTime - startTime) * this.targetFramerate);
        const frames = [];
        
        for (let i = 0; i < frameCount; i++) {
            const time = startTime + (i / this.targetFramerate);
            const timelineFrame = await this.timeline.getFrameAtTime(time);
            const vrmFrame = this.convertTimelineFrameToVRM(timelineFrame, time);
            
            frames.push({ time, timelineFrame, vrmFrame });
        }
        
        return frames;
    }
    
    /**
     * Bone mapping utilities
     */
    mapTimelineBoneToVRM(boneIndex) {
        // Default bone mapping - customize based on your BVH structure
        const boneMapping = {
            0: 'hips',
            1: 'spine',
            2: 'spine1',
            3: 'spine2',
            4: 'neck',
            5: 'head',
            6: 'leftShoulder',
            7: 'leftArm',
            8: 'leftForeArm',
            9: 'leftHand',
            10: 'rightShoulder',
            11: 'rightArm',
            12: 'rightForeArm',
            13: 'rightHand',
            14: 'leftUpLeg',
            15: 'leftLeg',
            16: 'leftFoot',
            17: 'rightUpLeg',
            18: 'rightLeg',
            19: 'rightFoot'
        };
        
        return boneMapping[boneIndex];
    }
    
    /**
     * Math utilities
     */
    lerp(a, b, t) {
        return a + (b - a) * t;
    }
    
    lerpAngle(a, b, t) {
        // Handle angle wrapping
        let diff = b - a;
        if (diff > Math.PI) diff -= 2 * Math.PI;
        if (diff < -Math.PI) diff += 2 * Math.PI;
        return a + diff * t;
    }
    
    slerpRotation(from, to, t) {
        // Simplified spherical interpolation for Euler angles
        // In production, you'd want proper quaternion SLERP
        return {
            x: this.lerpAngle(from.x, to.x, t),
            y: this.lerpAngle(from.y, to.y, t),
            z: this.lerpAngle(from.z, to.z, t)
        };
    }
    
    degreesToRadians(degrees) {
        return degrees * (Math.PI / 180);
    }
    
    radiansToDegrees(radians) {
        return radians * (180 / Math.PI);
    }
    
    /**
     * Default frame for fallback
     */
    getDefaultVRMFrame() {
        return {
            timestamp: 0,
            bones: {},
            metadata: { type: 'default' }
        };
    }
    
    /**
     * Performance monitoring
     */
    updateStats(startTime) {
        const processingTime = performance.now() - startTime;
        
        // Update running average
        const count = this.stats.framesProcessed;
        this.stats.averageProcessingTime = 
            (this.stats.averageProcessingTime * count + processingTime) / (count + 1);
    }
    
    getStats() {
        return {
            ...this.stats,
            isConnected: this.isConnected,
            realtime: this.realtime,
            queueSize: this.frameQueue.length,
            bufferSize: this.frameBuffer.size
        };
    }
    
    /**
     * Configuration methods
     */
    setRealtime(enabled) {
        this.realtime = enabled;
        if (!enabled) {
            this.frameQueue = [];
        }
        console.log('[BVH Timeline VRM] Realtime mode:', enabled);
    }
    
    setSmoothing(enabled) {
        this.smoothing = enabled;
        console.log('[BVH Timeline VRM] Smoothing:', enabled);
    }
    
    setInterpolation(mode) {
        this.interpolation = mode;
        console.log('[BVH Timeline VRM] Interpolation mode:', mode);
    }
    
    /**
     * Cleanup
     */
    dispose() {
        this.disconnectTimeline();
        this.frameQueue = [];
        this.frameBuffer.clear();
        console.log('[BVH Timeline VRM] Disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BVHTimelineVRMIntegration;
} else {
    window.BVHTimelineVRMIntegration = BVHTimelineVRMIntegration;
}
