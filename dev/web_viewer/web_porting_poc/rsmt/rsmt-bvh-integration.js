/**
 * RSMT BVH Integration
 * Integrates RSMT with BVH Timeline system for real-time stylized motion transitions
 */

class RSMTBVHIntegration {
    constructor(timeline, rsmt, options = {}) {
        this.timeline = timeline;
        this.rsmt = rsmt;
        
        this.options = {
            frameRate: 30,
            defaultTransitionLength: 30,
            blendingStrength: 0.5,
            cacheTransitions: true,
            maxCacheSize: 50,
            ...options
        };
        
        // Transition cache
        this.transitionCache = new Map();
        
        // VRM bone mapping for RSMT output
        this.vrmBoneMapping = this.createVRMBoneMapping();
        
        // Performance tracking
        this.performanceStats = {
            transitionsGenerated: 0,
            totalTime: 0,
            cacheHits: 0,
            activeSessions: 0
        };
        
        // Active transition sessions
        this.activeSessions = new Map();
        
        console.log('RSMT BVH Integration initialized');
    }
    
    /**
     * Create VRM bone mapping for RSMT skeleton output
     */
    createVRMBoneMapping() {
        return {
            // Root and spine
            'hips': { index: 0, channels: ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation'] },
            'spine': { index: 1, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'chest': { index: 2, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'upperChest': { index: 3, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'neck': { index: 4, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'head': { index: 5, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left leg
            'leftUpperLeg': { index: 14, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftLowerLeg': { index: 15, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftFoot': { index: 16, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftToes': { index: 17, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right leg  
            'rightUpperLeg': { index: 18, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightLowerLeg': { index: 19, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightFoot': { index: 20, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightToes': { index: 21, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left arm
            'leftShoulder': { index: 6, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftUpperArm': { index: 7, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftLowerArm': { index: 8, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'leftHand': { index: 9, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right arm
            'rightShoulder': { index: 10, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightUpperArm': { index: 11, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightLowerArm': { index: 12, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            'rightHand': { index: 13, channels: ['Xrotation', 'Yrotation', 'Zrotation'] }
        };
    }
    
    /**
     * Add a stylized transition between two motion clips
     */
    async addStylizedTransition(options = {}) {
        const {
            sourceClipId,
            targetClipId,
            startTime,
            transitionLength = this.options.defaultTransitionLength,
            style = 'smooth',
            trackName = 'rsmt_transition',
            weight = 1.0,
            blendMode = 'replace'
        } = options;
        
        console.log(`Adding RSMT stylized transition: ${sourceClipId} -> ${targetClipId}`);
        
        try {
            // Get source and target motion data
            const sourceMotion = await this.extractMotionFromClip(sourceClipId);
            const targetMotion = await this.extractMotionFromClip(targetClipId);
            
            if (!sourceMotion || !targetMotion) {
                throw new Error('Could not extract motion data from clips');
            }
            
            // Generate transition using RSMT
            const transitionData = await this.generateRSMTTransition({
                sourceMotion,
                targetMotion,
                transitionLength,
                style
            });
            
            // Create BVH clip from transition data
            const bvhClip = this.createBVHClipFromTransition(transitionData, {
                startTime,
                transitionLength,
                weight,
                blendMode,
                style
            });
            
            // Add to timeline
            const clipId = this.timeline.addClip(trackName, bvhClip);
            
            // Store session info
            this.activeSessions.set(clipId, {
                sourceClipId,
                targetClipId,
                style,
                transitionData,
                addedAt: Date.now()
            });
            
            // Update stats
            this.performanceStats.transitionsGenerated++;
            this.performanceStats.activeSessions = this.activeSessions.size;
            
            console.log(`RSMT transition added: ${clipId}`);
            return clipId;
            
        } catch (error) {
            console.error('Error adding stylized transition:', error);
            throw error;
        }
    }
    
    /**
     * Generate RSMT transition between two motion sequences
     */
    async generateRSMTTransition(options = {}) {
        const {
            sourceMotion,
            targetMotion,
            transitionLength,
            style = 'smooth'
        } = options;
        
        const startTime = performance.now();
        
        try {
            // Check cache first
            const cacheKey = this.createTransitionCacheKey(sourceMotion, targetMotion, transitionLength, style);
            if (this.options.cacheTransitions && this.transitionCache.has(cacheKey)) {
                this.performanceStats.cacheHits++;
                return this.transitionCache.get(cacheKey);
            }
            
            // Prepare motion data for RSMT
            const sourceSkeletonData = this.prepareMotionForRSMT(sourceMotion);
            const targetSkeletonData = this.prepareMotionForRSMT(targetMotion);
            
            // Generate stylized transition using RSMT
            const transitionResult = await this.rsmt.generateStylizedTransition({
                sourceMotion: sourceSkeletonData,
                targetMotion: targetSkeletonData,
                transitionLength: transitionLength,
                styleBlending: this.getStyleBlending(style)
            });
            
            // Convert RSMT output to BVH frames
            const bvhFrames = this.convertRSMTToBVHFrames(transitionResult.transitionMotion, transitionLength);
            
            const result = {
                frames: bvhFrames,
                metadata: {
                    style,
                    transitionLength,
                    sourcePhase: transitionResult.sourcePhase,
                    targetPhase: transitionResult.targetPhase,
                    manifoldPath: transitionResult.transitionPath,
                    performanceStats: transitionResult.metadata.performanceStats
                }
            };
            
            // Cache result if enabled
            if (this.options.cacheTransitions) {
                this.transitionCache.set(cacheKey, result);
                this.cleanupCache();
            }
            
            // Update performance stats
            this.performanceStats.totalTime += performance.now() - startTime;
            
            return result;
            
        } catch (error) {
            console.error('Error generating RSMT transition:', error);
            throw error;
        }
    }
    
    /**
     * Prepare motion data for RSMT input format
     */
    prepareMotionForRSMT(motionData) {
        // Extract key frame for transition (typically last frame of source, first frame of target)
        if (motionData.frames && motionData.frames.length > 0) {
            const frame = motionData.frames[motionData.frames.length - 1]; // Use last frame
            return this.bvhFrameToSkeletonData(frame);
        } else if (motionData.motionData) {
            // Direct motion data
            return this.motionDataToSkeletonData(motionData.motionData);
        } else {
            throw new Error('Invalid motion data format');
        }
    }
    
    /**
     * Convert BVH frame to skeleton data format expected by RSMT
     */
    bvhFrameToSkeletonData(bvhFrame) {
        const numJoints = Object.keys(this.vrmBoneMapping).length;
        const channelsPerJoint = 6; // 3 position + 3 rotation
        const skeletonData = new Float32Array(numJoints * channelsPerJoint);
        
        let dataIndex = 0;
        
        for (const [boneName, boneInfo] of Object.entries(this.vrmBoneMapping)) {
            if (bvhFrame.motionData && bvhFrame.motionData[boneName]) {
                const boneData = bvhFrame.motionData[boneName];
                
                // Copy bone data (ensuring we have 6 values per joint)
                for (let i = 0; i < channelsPerJoint; i++) {
                    skeletonData[dataIndex + i] = boneData[i] || 0;
                }
            } else {
                // Default values if bone data not available
                for (let i = 0; i < channelsPerJoint; i++) {
                    skeletonData[dataIndex + i] = 0;
                }
            }
            
            dataIndex += channelsPerJoint;
        }
        
        return skeletonData;
    }
    
    /**
     * Convert motion data to skeleton data format
     */
    motionDataToSkeletonData(motionData) {
        const numJoints = Object.keys(this.vrmBoneMapping).length;
        const channelsPerJoint = 6;
        const skeletonData = new Float32Array(numJoints * channelsPerJoint);
        
        let dataIndex = 0;
        
        for (const [boneName, boneInfo] of Object.entries(this.vrmBoneMapping)) {
            if (motionData[boneName]) {
                const boneData = motionData[boneName];
                
                for (let i = 0; i < Math.min(channelsPerJoint, boneData.length); i++) {
                    skeletonData[dataIndex + i] = boneData[i];
                }
            }
            
            dataIndex += channelsPerJoint;
        }
        
        return skeletonData;
    }
    
    /**
     * Convert RSMT skeleton output to BVH frames
     */
    convertRSMTToBVHFrames(skeletonData, numFrames) {
        const frames = [];
        const numJoints = Object.keys(this.vrmBoneMapping).length;
        const channelsPerJoint = 6;
        const frameSize = numJoints * channelsPerJoint;
        
        for (let frame = 0; frame < numFrames; frame++) {
            const frameOffset = frame * frameSize;
            const bvhFrame = {
                time: frame / this.options.frameRate,
                motionData: {},
                metadata: {
                    type: 'rsmt_transition',
                    frame: frame,
                    timestamp: Date.now()
                }
            };
            
            let dataIndex = 0;
            
            for (const [boneName, boneInfo] of Object.entries(this.vrmBoneMapping)) {
                const boneData = [];
                
                for (let i = 0; i < channelsPerJoint; i++) {
                    const valueIndex = frameOffset + dataIndex + i;
                    if (valueIndex < skeletonData.length) {
                        boneData.push(skeletonData[valueIndex]);
                    } else {
                        boneData.push(0);
                    }
                }
                
                bvhFrame.motionData[boneName] = boneData;
                dataIndex += channelsPerJoint;
            }
            
            frames.push(bvhFrame);
        }
        
        return frames;
    }
    
    /**
     * Create BVH clip from transition data
     */
    createBVHClipFromTransition(transitionData, options = {}) {
        const {
            startTime,
            transitionLength,
            weight,
            blendMode,
            style
        } = options;
        
        const generator = async (time, frameIndex) => {
            const localFrame = Math.floor((time - startTime) * this.options.frameRate);
            
            if (localFrame >= 0 && localFrame < transitionData.frames.length) {
                const frame = transitionData.frames[localFrame];
                return {
                    ...frame,
                    time: time
                };
            } else {
                // Return default frame if out of range
                return this.createDefaultBVHFrame(time);
            }
        };
        
        return new BVHClip({
            id: `rsmt_transition_${Date.now()}`,
            type: 'rsmt_transition',
            startTime: startTime,
            duration: transitionLength / this.options.frameRate,
            weight: weight,
            blendMode: blendMode,
            generator: generator,
            metadata: {
                style: style,
                transitionLength: transitionLength,
                generatedBy: 'RSMT',
                ...transitionData.metadata
            }
        });
    }
    
    /**
     * Extract motion data from existing timeline clip
     */
    async extractMotionFromClip(clipId) {
        // This would need to interface with your timeline system
        // For now, we'll return a placeholder
        const track = this.findTrackContainingClip(clipId);
        if (track) {
            const clip = track.clips.find(c => c.id === clipId);
            if (clip && clip.generator) {
                // Generate a frame from the clip to use as motion data
                const frame = await clip.generator(clip.startTime, 0);
                return { frames: [frame] };
            }
        }
        
        return null;
    }
    
    /**
     * Find track containing specific clip
     */
    findTrackContainingClip(clipId) {
        for (const track of Object.values(this.timeline.tracks)) {
            if (track.clips.some(clip => clip.id === clipId)) {
                return track;
            }
        }
        return null;
    }
    
    /**
     * Get style blending factor based on style name
     */
    getStyleBlending(style) {
        const styleMap = {
            'smooth': 0.3,
            'sharp': 0.8,
            'fluid': 0.2,
            'energetic': 0.9,
            'gentle': 0.1,
            'dramatic': 0.7,
            'natural': 0.5
        };
        
        return styleMap[style] || 0.5;
    }
    
    /**
     * Create cache key for transitions
     */
    createTransitionCacheKey(sourceMotion, targetMotion, length, style) {
        // Simple hash based on motion data characteristics
        const sourceHash = this.hashMotionData(sourceMotion);
        const targetHash = this.hashMotionData(targetMotion);
        return `${sourceHash}_${targetHash}_${length}_${style}`;
    }
    
    /**
     * Hash motion data for caching
     */
    hashMotionData(motionData) {
        // Simple hash of motion data
        let hash = 0;
        const str = JSON.stringify(motionData).substring(0, 100); // First 100 chars
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString();
    }
    
    /**
     * Cleanup cache when it gets too large
     */
    cleanupCache() {
        if (this.transitionCache.size > this.options.maxCacheSize) {
            // Remove oldest entries
            const entries = Array.from(this.transitionCache.entries());
            const toRemove = entries.slice(0, entries.length - this.options.maxCacheSize);
            for (const [key] of toRemove) {
                this.transitionCache.delete(key);
            }
        }
    }
    
    /**
     * Create default BVH frame
     */
    createDefaultBVHFrame(time) {
        const frame = {
            time: time,
            motionData: {},
            metadata: {
                type: 'default',
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
     * Performance and statistics
     */
    getPerformanceStats() {
        return {
            ...this.performanceStats,
            averageTransitionTime: this.performanceStats.totalTime / Math.max(1, this.performanceStats.transitionsGenerated),
            cacheHitRate: this.performanceStats.cacheHits / Math.max(1, this.performanceStats.transitionsGenerated),
            cacheSize: this.transitionCache.size
        };
    }
    
    /**
     * Remove transition session
     */
    removeTransition(clipId) {
        if (this.activeSessions.has(clipId)) {
            this.timeline.removeClip(this.findTrackContainingClip(clipId)?.name || 'rsmt_transition', clipId);
            this.activeSessions.delete(clipId);
            this.performanceStats.activeSessions = this.activeSessions.size;
            return true;
        }
        return false;
    }
    
    /**
     * Clear all transitions
     */
    clearAllTransitions() {
        for (const clipId of this.activeSessions.keys()) {
            this.removeTransition(clipId);
        }
        this.transitionCache.clear();
        console.log('All RSMT transitions cleared');
    }
    
    /**
     * Dispose and cleanup
     */
    dispose() {
        this.clearAllTransitions();
        this.activeSessions.clear();
        this.transitionCache.clear();
        console.log('RSMT BVH Integration disposed');
    }
}

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RSMTBVHIntegration;
} else if (typeof window !== 'undefined') {
    window.RSMTBVHIntegration = RSMTBVHIntegration;
}
