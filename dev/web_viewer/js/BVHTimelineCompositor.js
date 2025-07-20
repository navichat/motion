/**
 * BVH Timeline Compositor
 * Manages multiple timeline tracks with different BVH animation sources
 * and composites them into a single BVH output at any given timestamp
 */

class BVHTimelineCompositor {
    constructor(options = {}) {
        this.tracks = new Map();
        this.currentTime = 0;
        this.frameRate = options.frameRate || 30; // FPS
        this.frameDuration = 1000 / this.frameRate; // ms per frame
        this.blendMode = options.blendMode || 'hierarchical'; // 'hierarchical', 'additive', 'override'
        
        // BVH skeleton structure - will be populated when first BVH is loaded
        this.skeleton = null;
        this.jointHierarchy = null;
        this.jointNames = [];
        this.jointChannels = new Map();
        
        // Cache for performance
        this.frameCache = new Map();
        this.cacheSize = options.cacheSize || 1000;
        
        // Event system
        this.eventListeners = new Map();
        
        this.log('BVH Timeline Compositor initialized', { frameRate: this.frameRate, blendMode: this.blendMode });
    }
    
    /**
     * Add a timeline track
     */
    addTrack(trackId, config = {}) {
        const track = {
            id: trackId,
            type: config.type || 'static', // 'static', 'rsmt', 'neural', 'audio2gesture', 'faceformer'
            priority: config.priority || 0, // Higher numbers have higher priority
            enabled: config.enabled !== false,
            muted: config.muted || false,
            weight: config.weight || 1.0, // Blend weight (0.0 - 1.0)
            clips: [], // Array of timeline clips
            generator: config.generator || null, // Function for procedural generation
            channels: config.channels || 'all', // Which joints/channels this track affects
            blendMode: config.blendMode || 'replace', // 'replace', 'additive', 'multiply'
            
            // Track-specific settings
            ...config
        };
        
        this.tracks.set(trackId, track);
        this.log('Track added', { trackId, type: track.type, priority: track.priority });
        
        this.emit('trackAdded', { trackId, track });
        return track;
    }
    
    /**
     * Remove a timeline track
     */
    removeTrack(trackId) {
        if (this.tracks.has(trackId)) {
            this.tracks.delete(trackId);
            this.clearCache();
            this.emit('trackRemoved', { trackId });
            this.log('Track removed', { trackId });
        }
    }
    
    /**
     * Add a clip to a track
     */
    addClip(trackId, clip) {
        const track = this.tracks.get(trackId);
        if (!track) {
            throw new Error(`Track ${trackId} not found`);
        }
        
        const clipData = {
            id: clip.id || `clip_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            startTime: clip.startTime || 0,
            duration: clip.duration || 1000,
            bvhData: clip.bvhData || null,
            bvhFile: clip.bvhFile || null,
            
            // Clip properties
            enabled: clip.enabled !== false,
            weight: clip.weight || 1.0,
            fadeIn: clip.fadeIn || 0,
            fadeOut: clip.fadeOut || 0,
            loop: clip.loop || false,
            speed: clip.speed || 1.0,
            offset: clip.offset || 0,
            
            // Audio-specific (for audio2gesture tracks)
            audioFile: clip.audioFile || null,
            audioBuffer: clip.audioBuffer || null,
            
            // Neural network specific
            prompt: clip.prompt || null,
            parameters: clip.parameters || {},
            
            // RSMT specific
            sourceClip: clip.sourceClip || null,
            targetClip: clip.targetClip || null,
            transitionDuration: clip.transitionDuration || 500,
            
            ...clip
        };
        
        // Insert clip in chronological order
        const insertIndex = track.clips.findIndex(c => c.startTime > clipData.startTime);
        if (insertIndex === -1) {
            track.clips.push(clipData);
        } else {
            track.clips.splice(insertIndex, 0, clipData);
        }
        
        this.clearCache();
        this.emit('clipAdded', { trackId, clip: clipData });
        this.log('Clip added', { trackId, clipId: clipData.id, startTime: clipData.startTime, duration: clipData.duration });
        
        return clipData;
    }
    
    /**
     * Remove a clip from a track
     */
    removeClip(trackId, clipId) {
        const track = this.tracks.get(trackId);
        if (!track) return false;
        
        const clipIndex = track.clips.findIndex(c => c.id === clipId);
        if (clipIndex !== -1) {
            const removedClip = track.clips.splice(clipIndex, 1)[0];
            this.clearCache();
            this.emit('clipRemoved', { trackId, clip: removedClip });
            this.log('Clip removed', { trackId, clipId });
            return true;
        }
        return false;
    }
    
    /**
     * Get the composited BVH frame at a given timestamp
     */
    async getFrameAtTime(timestamp) {
        // Check cache first
        const cacheKey = `frame_${timestamp}`;
        if (this.frameCache.has(cacheKey)) {
            return this.frameCache.get(cacheKey);
        }
        
        // Get active clips at this timestamp from all tracks
        const activeClips = this.getActiveClipsAtTime(timestamp);
        
        if (activeClips.length === 0) {
            // Return default T-pose or rest pose
            return this.getDefaultPose();
        }
        
        // Generate frames for each active clip
        const clipFrames = await Promise.all(
            activeClips.map(({ track, clip, localTime, weight }) => 
                this.generateClipFrame(track, clip, localTime, weight)
            )
        );
        
        // Composite all frames together
        const compositedFrame = this.compositeFrames(clipFrames, timestamp);
        
        // Cache the result
        this.cacheFrame(cacheKey, compositedFrame);
        
        return compositedFrame;
    }
    
    /**
     * Get all active clips at a given timestamp
     */
    getActiveClipsAtTime(timestamp) {
        const activeClips = [];
        
        // Sort tracks by priority (highest first)
        const sortedTracks = Array.from(this.tracks.values())
            .filter(track => track.enabled && !track.muted)
            .sort((a, b) => b.priority - a.priority);
        
        for (const track of sortedTracks) {
            for (const clip of track.clips) {
                if (!clip.enabled) continue;
                
                const clipEndTime = clip.startTime + clip.duration;
                if (timestamp >= clip.startTime && timestamp < clipEndTime) {
                    const localTime = timestamp - clip.startTime;
                    const weight = this.calculateClipWeight(clip, localTime);
                    
                    activeClips.push({
                        track,
                        clip,
                        localTime,
                        weight
                    });
                }
            }
        }
        
        return activeClips;
    }
    
    /**
     * Calculate clip weight considering fade in/out
     */
    calculateClipWeight(clip, localTime) {
        let weight = clip.weight;
        
        // Apply fade in
        if (clip.fadeIn > 0 && localTime < clip.fadeIn) {
            weight *= localTime / clip.fadeIn;
        }
        
        // Apply fade out
        if (clip.fadeOut > 0) {
            const fadeOutStart = clip.duration - clip.fadeOut;
            if (localTime > fadeOutStart) {
                const fadeProgress = (localTime - fadeOutStart) / clip.fadeOut;
                weight *= (1.0 - fadeProgress);
            }
        }
        
        return Math.max(0, Math.min(1, weight));
    }
    
    /**
     * Generate a frame for a specific clip
     */
    async generateClipFrame(track, clip, localTime, weight) {
        let frame = null;
        
        switch (track.type) {
            case 'static':
                frame = await this.generateStaticFrame(clip, localTime);
                break;
                
            case 'rsmt':
                frame = await this.generateRSMTFrame(clip, localTime);
                break;
                
            case 'neural':
                frame = await this.generateNeuralFrame(clip, localTime);
                break;
                
            case 'audio2gesture':
                frame = await this.generateAudio2GestureFrame(clip, localTime);
                break;
                
            case 'faceformer':
                frame = await this.generateFaceformerFrame(clip, localTime);
                break;
                
            default:
                if (track.generator && typeof track.generator === 'function') {
                    frame = await track.generator(clip, localTime, weight);
                } else {
                    frame = this.getDefaultPose();
                }
        }
        
        return {
            frame,
            track,
            clip,
            weight,
            channels: track.channels
        };
    }
    
    /**
     * Generate frame from static BVH data
     */
    async generateStaticFrame(clip, localTime) {
        if (!clip.bvhData && !clip.bvhFile) {
            return this.getDefaultPose();
        }
        
        let bvhData = clip.bvhData;
        if (!bvhData && clip.bvhFile) {
            bvhData = await this.loadBVHFile(clip.bvhFile);
        }
        
        if (!bvhData || !bvhData.frames) {
            return this.getDefaultPose();
        }
        
        // Calculate frame index considering speed and looping
        const adjustedTime = localTime * clip.speed + clip.offset;
        let frameIndex = Math.floor(adjustedTime / this.frameDuration);
        
        if (clip.loop && bvhData.frames.length > 0) {
            frameIndex = frameIndex % bvhData.frames.length;
        } else {
            frameIndex = Math.min(frameIndex, bvhData.frames.length - 1);
        }
        
        return bvhData.frames[frameIndex] || this.getDefaultPose();
    }
    
    /**
     * Generate frame using RSMT transition
     */
    async generateRSMTFrame(clip, localTime) {
        // This would integrate with your RSMT system
        if (typeof window !== 'undefined' && window.RSMTGenerator) {
            return await window.RSMTGenerator.generateTransitionFrame(
                clip.sourceClip,
                clip.targetClip,
                localTime,
                clip.transitionDuration,
                clip.parameters
            );
        }
        
        // Fallback: simple interpolation between source and target
        const progress = Math.min(1, localTime / clip.transitionDuration);
        return this.interpolateFrames(clip.sourceClip, clip.targetClip, progress);
    }
    
    /**
     * Generate frame using neural network
     */
    async generateNeuralFrame(clip, localTime) {
        // This would integrate with your DeepPhase or other neural networks
        if (typeof window !== 'undefined' && window.NeuralGenerator) {
            return await window.NeuralGenerator.generateFrame(
                clip.prompt,
                localTime,
                clip.parameters
            );
        }
        
        // Fallback
        return this.getDefaultPose();
    }
    
    /**
     * Generate frame using Audio2Gesture
     */
    async generateAudio2GestureFrame(clip, localTime) {
        // This would integrate with your Audio2Gesture system
        if (typeof window !== 'undefined' && window.Audio2GestureGenerator) {
            return await window.Audio2GestureGenerator.generateFrame(
                clip.audioBuffer || clip.audioFile,
                localTime,
                clip.parameters
            );
        }
        
        // Fallback: generate procedural gesture based on audio analysis
        return this.generateProceduralGesture(localTime);
    }
    
    /**
     * Generate frame using Faceformer
     */
    async generateFaceformerFrame(clip, localTime) {
        // This would integrate with your Faceformer system
        if (typeof window !== 'undefined' && window.FaceformerGenerator) {
            return await window.FaceformerGenerator.generateFrame(
                clip.audioBuffer || clip.audioFile,
                localTime,
                clip.parameters
            );
        }
        
        // Fallback: generate procedural facial animation
        return this.generateProceduralFacialAnimation(localTime);
    }
    
    /**
     * Composite multiple frames into a single frame
     */
    compositeFrames(clipFrames, timestamp) {
        if (clipFrames.length === 0) {
            return this.getDefaultPose();
        }
        
        if (clipFrames.length === 1) {
            return clipFrames[0].frame;
        }
        
        // Initialize result frame
        let resultFrame = this.getDefaultPose();
        
        // Group frames by channel type for proper blending
        const channelGroups = this.groupFramesByChannels(clipFrames);
        
        // Blend each channel group
        for (const [channelType, frames] of channelGroups.entries()) {
            resultFrame = this.blendChannelFrames(resultFrame, frames, channelType);
        }
        
        return resultFrame;
    }
    
    /**
     * Group frames by their affected channels
     */
    groupFramesByChannels(clipFrames) {
        const groups = new Map();
        
        for (const clipFrame of clipFrames) {
            const channels = clipFrame.channels || 'all';
            
            if (!groups.has(channels)) {
                groups.set(channels, []);
            }
            groups.get(channels).push(clipFrame);
        }
        
        return groups;
    }
    
    /**
     * Blend frames for specific channels
     */
    blendChannelFrames(baseFrame, clipFrames, channelType) {
        let resultFrame = { ...baseFrame };
        
        // Sort by track priority
        clipFrames.sort((a, b) => b.track.priority - a.track.priority);
        
        for (const clipFrame of clipFrames) {
            resultFrame = this.blendTwoFrames(
                resultFrame,
                clipFrame.frame,
                clipFrame.weight,
                clipFrame.track.blendMode,
                channelType
            );
        }
        
        return resultFrame;
    }
    
    /**
     * Blend two frames together
     */
    blendTwoFrames(baseFrame, overlayFrame, weight, blendMode, channelType) {
        if (!overlayFrame || weight <= 0) return baseFrame;
        if (weight >= 1 && blendMode === 'replace') return overlayFrame;
        
        const result = { ...baseFrame };
        
        // Get joint indices to blend based on channel type
        const jointsToBlend = this.getJointsForChannelType(channelType);
        
        for (const jointIndex of jointsToBlend) {
            if (overlayFrame[jointIndex] !== undefined) {
                switch (blendMode) {
                    case 'replace':
                        result[jointIndex] = this.interpolateJointData(
                            baseFrame[jointIndex],
                            overlayFrame[jointIndex],
                            weight
                        );
                        break;
                        
                    case 'additive':
                        result[jointIndex] = this.addJointData(
                            baseFrame[jointIndex],
                            overlayFrame[jointIndex],
                            weight
                        );
                        break;
                        
                    case 'multiply':
                        result[jointIndex] = this.multiplyJointData(
                            baseFrame[jointIndex],
                            overlayFrame[jointIndex],
                            weight
                        );
                        break;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Get joint indices for a specific channel type
     */
    getJointsForChannelType(channelType) {
        if (channelType === 'all') {
            return Array.from({ length: this.jointNames.length }, (_, i) => i);
        }
        
        // Define channel mappings
        const channelMappings = {
            'body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], // Spine, arms, legs
            'face': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], // Facial joints
            'hands': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], // Hand joints
            'spine': [0, 1, 2, 3], // Spine joints only
            'arms': [4, 5, 6, 7, 8, 9], // Arm joints
            'legs': [10, 11, 12, 13, 14], // Leg joints
        };
        
        return channelMappings[channelType] || [];
    }
    
    /**
     * Interpolate between joint data
     */
    interpolateJointData(joint1, joint2, t) {
        if (!joint1 || !joint2) return joint1 || joint2;
        
        const result = {};
        
        // Interpolate position
        if (joint1.position && joint2.position) {
            result.position = {
                x: joint1.position.x + (joint2.position.x - joint1.position.x) * t,
                y: joint1.position.y + (joint2.position.y - joint1.position.y) * t,
                z: joint1.position.z + (joint2.position.z - joint1.position.z) * t
            };
        }
        
        // Interpolate rotation (quaternion slerp would be better)
        if (joint1.rotation && joint2.rotation) {
            result.rotation = {
                x: joint1.rotation.x + (joint2.rotation.x - joint1.rotation.x) * t,
                y: joint1.rotation.y + (joint2.rotation.y - joint1.rotation.y) * t,
                z: joint1.rotation.z + (joint2.rotation.z - joint1.rotation.z) * t
            };
        }
        
        return result;
    }
    
    /**
     * Add joint data (for additive blending)
     */
    addJointData(joint1, joint2, weight) {
        if (!joint1 || !joint2) return joint1 || joint2;
        
        const result = { ...joint1 };
        
        if (joint1.position && joint2.position) {
            result.position = {
                x: joint1.position.x + joint2.position.x * weight,
                y: joint1.position.y + joint2.position.y * weight,
                z: joint1.position.z + joint2.position.z * weight
            };
        }
        
        if (joint1.rotation && joint2.rotation) {
            result.rotation = {
                x: joint1.rotation.x + joint2.rotation.x * weight,
                y: joint1.rotation.y + joint2.rotation.y * weight,
                z: joint1.rotation.z + joint2.rotation.z * weight
            };
        }
        
        return result;
    }
    
    /**
     * Multiply joint data
     */
    multiplyJointData(joint1, joint2, weight) {
        if (!joint1 || !joint2) return joint1 || joint2;
        
        const result = { ...joint1 };
        
        if (joint1.rotation && joint2.rotation) {
            // For rotations, multiplication means combining rotations
            const factor = 1 + (weight - 1) * weight;
            result.rotation = {
                x: joint1.rotation.x * (1 + joint2.rotation.x * weight * factor),
                y: joint1.rotation.y * (1 + joint2.rotation.y * weight * factor),
                z: joint1.rotation.z * (1 + joint2.rotation.z * weight * factor)
            };
        }
        
        return result;
    }
    
    /**
     * Get default T-pose
     */
    getDefaultPose() {
        // Return a basic T-pose structure
        const defaultFrame = {};
        
        for (let i = 0; i < this.jointNames.length; i++) {
            defaultFrame[i] = {
                position: { x: 0, y: 0, z: 0 },
                rotation: { x: 0, y: 0, z: 0 }
            };
        }
        
        return defaultFrame;
    }
    
    /**
     * Generate procedural gesture
     */
    generateProceduralGesture(time) {
        const frame = this.getDefaultPose();
        
        // Add some simple procedural arm movement
        const armSwing = Math.sin(time * 0.002) * 0.3;
        if (frame[4]) frame[4].rotation.z = armSwing; // Left arm
        if (frame[7]) frame[7].rotation.z = -armSwing; // Right arm
        
        return frame;
    }
    
    /**
     * Generate procedural facial animation
     */
    generateProceduralFacialAnimation(time) {
        const frame = this.getDefaultPose();
        
        // Add simple blinking and mouth movement
        const blink = Math.sin(time * 0.01) > 0.95 ? 0.8 : 0;
        const mouthMove = Math.sin(time * 0.005) * 0.2;
        
        // These would be actual facial joint indices
        if (frame[15]) frame[15].rotation.x = blink; // Eyes
        if (frame[20]) frame[20].rotation.y = mouthMove; // Mouth
        
        return frame;
    }
    
    /**
     * Simple frame interpolation
     */
    interpolateFrames(frame1, frame2, t) {
        if (!frame1 || !frame2) return frame1 || frame2 || this.getDefaultPose();
        
        const result = {};
        const allJoints = new Set([...Object.keys(frame1), ...Object.keys(frame2)]);
        
        for (const jointIndex of allJoints) {
            const joint1 = frame1[jointIndex] || { position: {x:0,y:0,z:0}, rotation: {x:0,y:0,z:0} };
            const joint2 = frame2[jointIndex] || { position: {x:0,y:0,z:0}, rotation: {x:0,y:0,z:0} };
            
            result[jointIndex] = this.interpolateJointData(joint1, joint2, t);
        }
        
        return result;
    }
    
    /**
     * Load BVH file
     */
    async loadBVHFile(filePath) {
        try {
            const response = await fetch(filePath);
            const bvhContent = await response.text();
            return this.parseBVH(bvhContent);
        } catch (error) {
            this.log('Error loading BVH file', { filePath, error: error.message });
            return null;
        }
    }
    
    /**
     * Parse BVH content (simplified parser)
     */
    parseBVH(bvhContent) {
        // This is a simplified BVH parser
        // You would need a more robust parser for production use
        const lines = bvhContent.split('\n');
        const data = { joints: [], frames: [] };
        
        let isMotionSection = false;
        let frameCount = 0;
        let frameTime = 0;
        
        for (const line of lines) {
            const trimmed = line.trim();
            
            if (trimmed.startsWith('MOTION')) {
                isMotionSection = true;
                continue;
            }
            
            if (isMotionSection) {
                if (trimmed.startsWith('Frames:')) {
                    frameCount = parseInt(trimmed.split(':')[1]);
                } else if (trimmed.startsWith('Frame Time:')) {
                    frameTime = parseFloat(trimmed.split(':')[1]);
                } else if (trimmed && !trimmed.startsWith('Frames') && !trimmed.startsWith('Frame Time')) {
                    // Parse frame data
                    const values = trimmed.split(/\s+/).map(parseFloat);
                    const frame = this.parseFrameData(values);
                    data.frames.push(frame);
                }
            }
        }
        
        return data;
    }
    
    /**
     * Parse frame data from BVH values
     */
    parseFrameData(values) {
        const frame = {};
        let valueIndex = 0;
        
        // Root position (first 3 values typically)
        frame[0] = {
            position: {
                x: values[valueIndex++] || 0,
                y: values[valueIndex++] || 0,
                z: values[valueIndex++] || 0
            },
            rotation: {
                x: values[valueIndex++] || 0,
                y: values[valueIndex++] || 0,
                z: values[valueIndex++] || 0
            }
        };
        
        // Parse remaining joints (3 rotation values each)
        let jointIndex = 1;
        while (valueIndex < values.length && jointIndex < 50) { // Limit to reasonable number
            frame[jointIndex] = {
                position: { x: 0, y: 0, z: 0 },
                rotation: {
                    x: values[valueIndex++] || 0,
                    y: values[valueIndex++] || 0,
                    z: values[valueIndex++] || 0
                }
            };
            jointIndex++;
        }
        
        return frame;
    }
    
    /**
     * Cache management
     */
    cacheFrame(key, frame) {
        if (this.frameCache.size >= this.cacheSize) {
            // Remove oldest entries
            const keysToRemove = Array.from(this.frameCache.keys()).slice(0, this.cacheSize * 0.2);
            keysToRemove.forEach(k => this.frameCache.delete(k));
        }
        
        this.frameCache.set(key, frame);
    }
    
    clearCache() {
        this.frameCache.clear();
    }
    
    /**
     * Set current time and get frame
     */
    async setTime(timestamp) {
        this.currentTime = timestamp;
        const frame = await this.getFrameAtTime(timestamp);
        this.emit('timeChanged', { timestamp, frame });
        return frame;
    }
    
    /**
     * Play timeline
     */
    play(startTime = null) {
        if (startTime !== null) {
            this.currentTime = startTime;
        }
        
        this.isPlaying = true;
        this.playStartTime = Date.now();
        this.playOffset = this.currentTime;
        
        this.playLoop();
        this.emit('playStarted', { currentTime: this.currentTime });
    }
    
    /**
     * Pause timeline
     */
    pause() {
        this.isPlaying = false;
        this.emit('playPaused', { currentTime: this.currentTime });
    }
    
    /**
     * Stop timeline
     */
    stop() {
        this.isPlaying = false;
        this.currentTime = 0;
        this.emit('playStopped');
    }
    
    /**
     * Play loop
     */
    async playLoop() {
        if (!this.isPlaying) return;
        
        const now = Date.now();
        this.currentTime = this.playOffset + (now - this.playStartTime);
        
        const frame = await this.getFrameAtTime(this.currentTime);
        this.emit('frameUpdate', { timestamp: this.currentTime, frame });
        
        requestAnimationFrame(() => this.playLoop());
    }
    
    /**
     * Event system
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.eventListeners.has(event)) {
            const listeners = this.eventListeners.get(event);
            const index = listeners.indexOf(callback);
            if (index !== -1) {
                listeners.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    this.log('Event callback error', { event, error: error.message });
                }
            });
        }
    }
    
    /**
     * Logging
     */
    log(message, data = null) {
        if (console && console.log) {
            console.log(`[BVHTimelineCompositor] ${message}`, data || '');
        }
    }
    
    /**
     * Get timeline status
     */
    getStatus() {
        return {
            currentTime: this.currentTime,
            isPlaying: this.isPlaying,
            trackCount: this.tracks.size,
            totalClips: Array.from(this.tracks.values()).reduce((sum, track) => sum + track.clips.length, 0),
            cacheSize: this.frameCache.size,
            frameRate: this.frameRate
        };
    }
    
    /**
     * Export timeline to JSON
     */
    exportTimeline() {
        const tracks = {};
        this.tracks.forEach((track, id) => {
            tracks[id] = {
                ...track,
                generator: null // Can't serialize functions
            };
        });
        
        return {
            frameRate: this.frameRate,
            blendMode: this.blendMode,
            tracks: tracks,
            currentTime: this.currentTime
        };
    }
    
    /**
     * Import timeline from JSON
     */
    importTimeline(data) {
        this.frameRate = data.frameRate || 30;
        this.blendMode = data.blendMode || 'hierarchical';
        this.currentTime = data.currentTime || 0;
        
        this.tracks.clear();
        this.clearCache();
        
        if (data.tracks) {
            Object.entries(data.tracks).forEach(([id, track]) => {
                this.tracks.set(id, track);
            });
        }
        
        this.emit('timelineImported', data);
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BVHTimelineCompositor;
} else if (typeof window !== 'undefined') {
    window.BVHTimelineCompositor = BVHTimelineCompositor;
}
