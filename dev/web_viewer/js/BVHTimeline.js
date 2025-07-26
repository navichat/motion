/**
 * BVH Timeline Compositor
 * 
 * A comprehensive timeline system for compositing BVH animations from multiple sources:
 * - Static BVH animations (premade clips)
 * - RSMT transition generation
 * - Deep phase neural network generation
 * - Audio2gesture/Faceformer face animation
 * - Real-time composition and blending
 */

// 1) Pull in our three helpers as ES modules
import SimplePoseSearchEngine from './SimplePoseSearchEngine.js';
//import getCurrentPose            from './getCurrentPose.js';
import smartTransition           from './smartTransition.js';
import { requestAnimationFrame, cancelAnimationFrame } from './animation_frame.js';

// now BVHTimeline can bind to them

class BVHTimeline {
    // Now accept an array of animations plus options
    constructor(animations = [], options = {}) {
        this.framerate = options.framerate || 30; // FPS
        this.frameTime = 1.0 / this.framerate; // seconds per frame

        // —— NEW: build a Map of your animations, index them for searching, and bind your helpers ——
        this.animations    = new Map( animations.map(a=>[a.animationId, a]) );
        this.searchEngine  = new SimplePoseSearchEngine();
        this.searchEngine.indexAnimations(animations);
        //this.getCurrentPose  = getCurrentPose.bind(this);
        //this.smartTransition = smartTransition.bind(this);

        // transition‐state defaults
        this.isTransitioning   = false;
        this.transitionTarget  = { pose: null };
        this.transitionProgress= 0;
        this.similarityThreshold = options.similarityThreshold ?? 0.5;

        // Timeline tracks for different animation sources
        this.tracks = {
            base: new BVHTrack('base', { priority: 0 }), // Base body animations
            transitions: new BVHTrack('transitions', { priority: 1 }), // RSMT transitions
            deepphase: new BVHTrack('deepphase', { priority: 2 }), // Deep phase generations
            face: new BVHTrack('face', { priority: 3 }), // Face/head animations
            audio: new BVHTrack('audio', { priority: 4 }), // Audio-driven gestures
            override: new BVHTrack('override', { priority: 5 }) // High priority overrides
        };
        
        // Current playback state
        this.currentTime = 0;
        this.isPlaying = false;
        this.startTime = null;
        
        // Animation frame request
        this.animationFrameId = null;
        
        // Frame buffer system
        this.frameBuffer = new BVHFrameBuffer({
            maxBufferSize: options.maxBufferSize || 300, // 10 seconds at 30fps
            lookaheadFrames: options.lookaheadFrames || 60, // 2 seconds at 30fps
            framerate: this.framerate,
            cleanupInterval: options.cleanupInterval || 1000, // 1 second
            staleThreshold: options.staleThreshold || 5000 // 5 seconds
        });
        
        // Buffer management
        this.bufferUpdateId = null;
        this.lastBufferUpdate = 0;
        this.bufferUpdateInterval = options.bufferUpdateInterval || 100; // 100ms
        
        // Callbacks
        this.onFrameUpdate = options.onFrameUpdate || null;
        this.onTrackChange = options.onTrackChange || null;
        this.onBufferUpdate = options.onBufferUpdate || null;
        
        // BVH skeleton structure cache
        this.skeletonTemplate = null;
        this.boneMapping = new Map();
        
        // Composition settings
        this.blendingMode = options.blendingMode || 'additive'; // 'replace', 'additive', 'weighted'
        this.globalWeight = 1.0;
        
        // Performance monitoring
        this.performanceStats = {
            frameGenerationTime: 0,
            bufferHitRate: 0,
            totalFramesRequested: 0,
            bufferHits: 0,
            bufferMisses: 0
        };
        
        console.log('[BVHTimeline] Initialized with framerate:', this.framerate);
        console.log('[BVHTimeline] Frame buffer initialized with lookahead:', options.lookaheadFrames || 60);
    }
    

    // *** ADD THIS NEW ASYNC INIT METHOD ***
    async init() {
        // Load the THREE.js library from the CDN once
        //this.THREE = await import('https://cdn.jsdelivr.net/npm/three@0.177.0/build/three.module.js');
        // In Node.js, import the locally installed 'three' package from `npm install three`
        this.THREE = await import('three');

        // Now that `this.THREE` exists, we can safely import and bind the helpers
        const { default: getCurrentPose } = await import('./getCurrentPose.js');
        const { default: smartTransition } = await import('./smartTransition.js');

        this.getCurrentPose = getCurrentPose.bind(this);
        this.smartTransition = smartTransition.bind(this);

        console.log('[BVHTimeline] THREE.js loaded and helpers initialized.');
    }

    /**
     * Add a BVH animation clip to a specific track
     */
    addClip(trackName, clip) {
        if (!this.tracks[trackName]) {
            throw new Error(`Track '${trackName}' does not exist`);
        }
        
        return this.tracks[trackName].addClip(clip);
    }
    
    /**
     * Add a static BVH animation
     */
    addBVHAnimation(trackName, bvhData, startTime, options = {}) {
        const clip = new BVHClip({
            type: 'static',
            bvhData: bvhData,
            startTime: startTime,
            duration: options.duration || this.calculateBVHDuration(bvhData),
            weight: options.weight || 1.0,
            blendMode: options.blendMode || 'replace',
            loop: options.loop || false,
            ...options
        });
        
        return this.addClip(trackName, clip);
    }
    
    /**
     * Add an RSMT transition
     */
    addRSMTTransition(startTime, fromPose, toPose, duration, options = {}) {
        const clip = new BVHClip({
            type: 'rsmt_transition',
            startTime: startTime,
            duration: duration,
            weight: options.weight || 1.0,
            blendMode: options.blendMode || 'replace',
            generator: async (time, frameIndex) => {
                return await this.generateRSMTFrame(fromPose, toPose, time / duration);
            },
            metadata: { fromPose, toPose, ...options }
        });
        
        return this.addClip('transitions', clip);
    }
    
    /**
     * Add Deep Phase generated animation
     */
    addDeepPhaseGeneration(startTime, prompt, duration, options = {}) {
        const clip = new BVHClip({
            type: 'deepphase_generation',
            startTime: startTime,
            duration: duration,
            weight: options.weight || 1.0,
            blendMode: options.blendMode || 'replace',
            generator: async (time, frameIndex) => {
                return await this.generateDeepPhaseFrame(prompt, time, frameIndex);
            },
            metadata: { prompt, ...options }
        });
        
        return this.addClip('deepphase', clip);
    }
    
    /**
     * Add audio-driven animation (Faceformer/Audio2Gesture)
     */
    addAudioAnimation(audioBuffer, startTime, options = {}) {
        const duration = audioBuffer.duration;
        const clip = new BVHClip({
            type: 'audio_driven',
            startTime: startTime,
            duration: duration,
            weight: options.weight || 1.0,
            blendMode: options.blendMode || 'additive',
            generator: async (time, frameIndex) => {
                return await this.generateAudioFrame(audioBuffer, time, options.animationType || 'face');
            },
            metadata: { audioBuffer, animationType: options.animationType, ...options }
        });
        
        const trackName = options.animationType === 'gesture' ? 'audio' : 'face';
        return this.addClip(trackName, clip);
    }
    
    /**
     * Get current composed BVH frame at current time
     */
    async getCurrentFrame() {
        return await this.getFrameAtTime(this.currentTime);
    }
    
    /**
     * Get composed BVH frame at specific time with buffering
     */
    async getFrameAtTime(time) {
        const startTime = performance.now();
        this.performanceStats.totalFramesRequested++;
        
        // Try to get from buffer first
        const bufferedFrame = this.frameBuffer.getFrame(time);
        if (bufferedFrame) {
            this.performanceStats.bufferHits++;
            this.updateBufferHitRate();
            return bufferedFrame;
        }
        
        this.performanceStats.bufferMisses++;
        
        // Generate frame and add to buffer
        const frame = await this.generateFrameAtTime(time);
        this.frameBuffer.addFrame(time, frame);
        
        // Update performance stats
        this.performanceStats.frameGenerationTime = performance.now() - startTime;
        this.updateBufferHitRate();
        
        return frame;
    }
    
    /**
     * Get multiple frames with lookahead (for renderer optimization)
     */
    async getFramesWithLookahead(currentTime, lookaheadCount = null) {
        const maxLookahead = lookaheadCount || this.frameBuffer.lookaheadFrames;
        const frames = [];
        
        // Get current frame
        const currentFrame = await this.getFrameAtTime(currentTime);
        frames.push({ time: currentTime, frame: currentFrame });
        
        // Get lookahead frames
        for (let i = 1; i <= maxLookahead; i++) {
            const futureTime = currentTime + (i * this.frameTime);
            
            // Check if we have clips that extend to this time
            if (!this.hasActiveClipsAtTime(futureTime)) {
                break; // No point in generating frames beyond active content
            }
            
            try {
                const futureFrame = await this.getFrameAtTime(futureTime);
                frames.push({ time: futureTime, frame: futureFrame });
            } catch (error) {
                console.warn(`[BVHTimeline] Failed to generate lookahead frame at ${futureTime}:`, error);
                break; // Stop lookahead on error
            }
        }
        
        return frames;
    }
    
    /**
     * Generate frame at specific time (internal method)
     */
    async generateFrameAtTime(time) {
        const activeClips = this.getActiveClipsAtTime(time);
        
        if (activeClips.length === 0) {
            return this.getDefaultFrame(time);
        }
        
        // Sort clips by track priority
        activeClips.sort((a, b) => {
            const priorityA = this.tracks[a.trackName]?.priority || 0;
            const priorityB = this.tracks[b.trackName]?.priority || 0;
            return priorityA - priorityB;
        });
        
        // Generate/get frames from all active clips
        const clipFrames = await Promise.all(
            activeClips.map(async (clipInfo) => {
                const localTime = time - clipInfo.clip.startTime;
                const frame = await clipInfo.clip.getFrameAtTime(localTime);
                return {
                    frame: frame,
                    weight: clipInfo.clip.weight,
                    blendMode: clipInfo.clip.blendMode,
                    trackName: clipInfo.trackName,
                    priority: this.tracks[clipInfo.trackName].priority
                };
            })
        );
        
        // Compose the final frame
        return this.composeFrames(clipFrames, time);
    }
    
    /**
     * Check if there are active clips at a specific time
     */
    hasActiveClipsAtTime(time) {
        for (const track of Object.values(this.tracks)) {
            if (track.getActiveClipsAtTime(time).length > 0) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Get all active clips at a specific time
     */
    getActiveClipsAtTime(time) {
        const activeClips = [];
        
        for (const [trackName, track] of Object.entries(this.tracks)) {
            const clips = track.getActiveClipsAtTime(time);
            clips.forEach(clip => {
                activeClips.push({ clip, trackName });
            });
        }
        
        return activeClips;
    }
    
    /**
     * Compose multiple BVH frames into a single frame
     */
    composeFrames(clipFrames, time) {
        if (clipFrames.length === 0) {
            return this.getDefaultFrame(time);
        }
        
        if (clipFrames.length === 1) {
            return clipFrames[0].frame;
        }
        
        // Start with the lowest priority (base) frame
        let composedFrame = this.cloneFrame(clipFrames[0].frame);
        composedFrame.time = time; // Update timestamp
        
        // Apply higher priority frames based on blend mode
        for (let i = 1; i < clipFrames.length; i++) {
            const clipFrame = clipFrames[i];
            composedFrame = this.blendFrames(
                composedFrame,
                clipFrame.frame,
                clipFrame.weight,
                clipFrame.blendMode,
                clipFrame.trackName
            );
        }
        
        // Add composition metadata
        composedFrame.metadata = composedFrame.metadata || {};
        composedFrame.metadata.composedFrom = clipFrames.map(cf => ({
            track: cf.trackName,
            weight: cf.weight,
            blendMode: cf.blendMode
        }));
        
        return composedFrame;
    }
    
    /**
     * Blend two BVH frames together
     */
    blendFrames(baseFrame, overlayFrame, weight, blendMode, trackName) {
        if (!overlayFrame || weight <= 0) {
            return baseFrame;
        }
        
        const result = this.cloneFrame(baseFrame);
        
        switch (blendMode) {
            case 'replace':
                return this.blendReplace(result, overlayFrame, weight, trackName);
            
            case 'additive':
                return this.blendAdditive(result, overlayFrame, weight, trackName);
            
            case 'weighted':
                return this.blendWeighted(result, overlayFrame, weight, trackName);
            
            case 'mask':
                return this.blendMask(result, overlayFrame, weight, trackName);
                
            default:
                console.warn(`Unknown blend mode: ${blendMode}`);
                return result;
        }
    }
    
    /**
     * Replace blending - overlay completely replaces base in affected bones
     */
    blendReplace(baseFrame, overlayFrame, weight, trackName) {
        const affectedBones = this.getTrackBoneInfluence(trackName);
        
        if (!overlayFrame.motionData) return baseFrame;
        
        overlayFrame.motionData.forEach((boneData, boneIndex) => {
            const boneName = this.getBoneName(boneIndex);
            
            if (affectedBones.has(boneName)) {
                if (baseFrame.motionData[boneIndex]) {
                    // Interpolate based on weight
                    for (let i = 0; i < boneData.length; i++) {
                        baseFrame.motionData[boneIndex][i] = 
                            baseFrame.motionData[boneIndex][i] * (1 - weight) + 
                            boneData[i] * weight;
                    }
                } else {
                    baseFrame.motionData[boneIndex] = [...boneData];
                }
            }
        });
        
        return baseFrame;
    }
    
    /**
     * Additive blending - overlay adds to base
     */
    blendAdditive(baseFrame, overlayFrame, weight, trackName) {
        const affectedBones = this.getTrackBoneInfluence(trackName);
        
        if (!overlayFrame.motionData) return baseFrame;
        
        overlayFrame.motionData.forEach((boneData, boneIndex) => {
            const boneName = this.getBoneName(boneIndex);
            
            if (affectedBones.has(boneName) && baseFrame.motionData[boneIndex]) {
                for (let i = 0; i < boneData.length; i++) {
                    baseFrame.motionData[boneIndex][i] += boneData[i] * weight;
                }
            }
        });
        
        return baseFrame;
    }
    
    /**
     * Weighted blending - linear interpolation
     */
    blendWeighted(baseFrame, overlayFrame, weight, trackName) {
        const affectedBones = this.getTrackBoneInfluence(trackName);
        
        if (!overlayFrame.motionData) return baseFrame;
        
        overlayFrame.motionData.forEach((boneData, boneIndex) => {
            const boneName = this.getBoneName(boneIndex);
            
            if (affectedBones.has(boneName) && baseFrame.motionData[boneIndex]) {
                for (let i = 0; i < boneData.length; i++) {
                    baseFrame.motionData[boneIndex][i] = 
                        baseFrame.motionData[boneIndex][i] * (1 - weight) + 
                        boneData[i] * weight;
                }
            }
        });
        
        return baseFrame;
    }
    
    /**
     * Mask blending - overlay affects only specific bone regions
     */
    blendMask(baseFrame, overlayFrame, weight, trackName) {
        // Similar to replace but with strict bone masking
        return this.blendReplace(baseFrame, overlayFrame, weight, trackName);
    }
    
    /**
     * Get bone influence for different tracks
     */
    getTrackBoneInfluence(trackName) {
        const influences = {
            base: new Set(['hips', 'spine', 'spine1', 'spine2', 'leftLeg', 'rightLeg', 'leftArm', 'rightArm']),
            transitions: new Set(['hips', 'spine', 'spine1', 'spine2', 'leftLeg', 'rightLeg', 'leftArm', 'rightArm']),
            deepphase: new Set(['hips', 'spine', 'spine1', 'spine2', 'leftLeg', 'rightLeg', 'leftArm', 'rightArm']),
            face: new Set(['head', 'neck', 'jaw', 'leftEye', 'rightEye']),
            audio: new Set(['leftArm', 'rightArm', 'leftHand', 'rightHand', 'head', 'neck']),
            override: new Set() // Empty = affects all bones
        };
        
        return influences[trackName] || new Set();
    }
    
    /**
     * Example: kick off a smart transition into a new animation
     */
    async transitionTo(targetAnimationId) {
      // the smartTransition helper returns
      // [ newAnimId, bestFrameIndex, distance, fullResults ]
      const [ newId, frameIndex, distance, results ] =
        await smartTransition(this, targetAnimationId);

      // flip our state over to transitioning
      this.isTransitioning    = true;
      this.transitionProgress = 0;

      this.transitionTarget   = {
        animationId: newId, // Store the ID for later
        frameIndex: frameIndex, // Store the index for later
        pose: this.animations.get(newId).poses[frameIndex]
      };

      // We DO NOT change the currentAnimationId or currentFrame here.
      // The animate loop will continue using the OLD animation as the source.
      console.log(`↳ Starting blend to ${newId} frame ${frameIndex} (dist ${distance.toFixed(3)})`);
      return results;
    }

    /**
     * Start timeline playback
     */
    play() {
        if (this.isPlaying) return;
        
        this.isPlaying = true;
        this.startTime = performance.now() - (this.currentTime * 1000);
        
        // Start buffer management
        this.startBufferManagement();
        
        this.scheduleNextFrame();
        
        console.log('[BVHTimeline] Playback started at time:', this.currentTime);
    }
    
    /**
     * Pause timeline playback
     */
    pause() {
        this.isPlaying = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        
        // Stop buffer management
        this.stopBufferManagement();
        
        console.log('[BVHTimeline] Playback paused at time:', this.currentTime);
    }
    
    /**
     * Seek to specific time
     */
    seek(time) {
        const oldTime = this.currentTime;
        this.currentTime = Math.max(0, time);
        this.startTime = performance.now() - (this.currentTime * 1000);
        
        // Clear buffer if seeking far from current position
        if (Math.abs(time - oldTime) > 2.0) { // 2 second threshold
            this.frameBuffer.clearStaleFrames(time);
        }
        
        // Pre-buffer frames around seek position
        this.prebufferAroundTime(time);
        
        if (this.onFrameUpdate) {
            this.getCurrentFrame().then(frame => {
                this.onFrameUpdate(frame, this.currentTime);
            });
        }
        
        console.log('[BVHTimeline] Seeked to time:', this.currentTime);
    }
    
    /**
     * Start buffer management system
     */
    startBufferManagement() {
        if (this.bufferUpdateId) return;
        
        const updateBuffer = () => {
            if (!this.isPlaying) return;
            
            const now = performance.now();
            if (now - this.lastBufferUpdate >= this.bufferUpdateInterval) {
                this.updateBuffer();
                this.lastBufferUpdate = now;
            }
            
            this.bufferUpdateId = requestAnimationFrame(updateBuffer);
        };
        
        updateBuffer();
    }
    
    /**
     * Stop buffer management system
     */
    stopBufferManagement() {
        if (this.bufferUpdateId) {
            cancelAnimationFrame(this.bufferUpdateId);
            this.bufferUpdateId = null;
        }
    }
    
    /**
     * Update frame buffer with lookahead frames
     */
    async updateBuffer() {
        try {
            // Clean up stale frames
            this.frameBuffer.cleanup(this.currentTime);
            
            // Prebuffer upcoming frames
            await this.prebufferFrames();
            
            // Notify listeners of buffer update
            if (this.onBufferUpdate) {
                this.onBufferUpdate(this.frameBuffer.getStats());
            }
            
        } catch (error) {
            console.warn('[BVHTimeline] Buffer update failed:', error);
        }
    }
    
    /**
     * Prebuffer frames ahead of current time
     */
    async prebufferFrames() {
        const startTime = this.currentTime;
        const endTime = startTime + (this.frameBuffer.lookaheadFrames * this.frameTime);
        
        // Only prebuffer if we have active clips in the range
        if (!this.hasActiveClipsInRange(startTime, endTime)) {
            return;
        }
        
        const promises = [];
        for (let time = startTime; time <= endTime; time += this.frameTime) {
            // Skip if already buffered
            if (this.frameBuffer.hasFrame(time)) {
                continue;
            }
            
            // Limit concurrent generations to avoid overwhelming the system
            if (promises.length >= 5) {
                await Promise.all(promises);
                promises.length = 0;
            }
            
            promises.push(this.prebufferSingleFrame(time));
        }
        
        if (promises.length > 0) {
            await Promise.all(promises);
        }
    }
    
    /**
     * Prebuffer a single frame
     */
    async prebufferSingleFrame(time) {
        try {
            const frame = await this.generateFrameAtTime(time);
            this.frameBuffer.addFrame(time, frame);
        } catch (error) {
            console.warn(`[BVHTimeline] Failed to prebuffer frame at ${time}:`, error);
        }
    }
    
    /**
     * Prebuffer frames around a specific time (used for seeking)
     */
    async prebufferAroundTime(centerTime) {
        const radius = 1.0; // 1 second radius
        const startTime = centerTime - radius;
        const endTime = centerTime + radius;
        
        const promises = [];
        for (let time = startTime; time <= endTime; time += this.frameTime) {
            if (time >= 0 && !this.frameBuffer.hasFrame(time)) {
                promises.push(this.prebufferSingleFrame(time));
            }
        }
        
        await Promise.all(promises);
    }
    
    /**
     * Check if there are active clips in a time range
     */
    hasActiveClipsInRange(startTime, endTime) {
        for (const track of Object.values(this.tracks)) {
            for (const clip of track.clips) {
                const clipStart = clip.startTime;
                const clipEnd = clip.startTime + clip.duration;
                
                // Check for overlap
                if (clipStart < endTime && clipEnd > startTime) {
                    return true;
                }
            }
        }
        return false;
    }
    
    /**
     * Schedule next animation frame
     */
    scheduleNextFrame() {
        if (!this.isPlaying) return;
        
        this.animationFrameId = requestAnimationFrame(() => {
            this.updateTimeline();
            this.scheduleNextFrame();
        });
    }
    
    /**
     * Update timeline for current frame
     */
    async updateTimeline() {
        if (!this.isPlaying) return;
        
        const now = performance.now();
        this.currentTime = (now - this.startTime) / 1000;

        const deltaTime = this.frameTime; // Approximate time since last frame

        // --- STATE UPDATE LOGIC ---
        // Advance the current animation frame to keep the source pose dynamic
        const anim = this.animations.get(this.currentAnimationId);
        if (anim) {
            this.currentFrame = (this.currentFrame + (deltaTime * anim.fps));
            if (this.currentFrame >= anim.poses.length) {
                this.currentFrame = 0; // Simple loop
            }
        }

        // If a transition is active, advance its progress
        if (this.isTransitioning) {
            this.transitionProgress += deltaTime / 3.0; // Assuming a 3s transition duration for now

            if (this.transitionProgress >= 1.0) {
                // --- COMPLETE THE TRANSITION ---
                this.isTransitioning = false;
                this.currentAnimationId = this.transitionTarget.animationId;
                this.currentFrame = this.transitionTarget.frameIndex;
            }
        }

        // --- POSE CALCULATION & CALLBACK ---
        const currentPose = this.getCurrentPose(this);

        if (this.onFrameUpdate) {
            this.onFrameUpdate(currentPose, this.currentTime);
        }
    }


    /**
     * Generate RSMT transition frame
     */
    async generateRSMTFrame(fromPose, toPose, progress) {
        // TODO: Integrate with RSMT server
        // This is a placeholder for RSMT transition generation
        try {
            const response = await fetch('/api/rsmt/transition', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fromPose: fromPose,
                    toPose: toPose,
                    progress: progress
                })
            });
            
            if (response.ok) {
                const bvhData = await response.json();
                return this.parseBVHFrame(bvhData);
            }
        } catch (error) {
            console.warn('[BVHTimeline] RSMT generation failed:', error);
        }
        
        // Fallback: linear interpolation
        return this.interpolatePoses(fromPose, toPose, progress);
    }
    
    /**
     * Generate Deep Phase animation frame
     */
    async generateDeepPhaseFrame(prompt, time, frameIndex) {
        // TODO: Integrate with Deep Phase neural network
        try {
            const response = await fetch('/api/deepphase/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    time: time,
                    frameIndex: frameIndex
                })
            });
            
            if (response.ok) {
                const bvhData = await response.json();
                return this.parseBVHFrame(bvhData);
            }
        } catch (error) {
            console.warn('[BVHTimeline] Deep Phase generation failed:', error);
        }
        
        return this.getDefaultFrame();
    }
    
    /**
     * Generate audio-driven animation frame
     */
    async generateAudioFrame(audioBuffer, time, animationType) {
        // TODO: Integrate with Faceformer/Audio2Gesture
        try {
            const audioSample = this.getAudioSampleAtTime(audioBuffer, time);
            
            const response = await fetch('/api/audio2gesture/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audioSample: audioSample,
                    time: time,
                    animationType: animationType
                })
            });
            
            if (response.ok) {
                const bvhData = await response.json();
                return this.parseBVHFrame(bvhData);
            }
        } catch (error) {
            console.warn('[BVHTimeline] Audio animation generation failed:', error);
        }
        
        return this.getDefaultFrame();
    }
    
    /**
     * Utility methods
     */
    calculateBVHDuration(bvhData) {
        // Parse BVH to get frame count and frame time
        const lines = bvhData.split('\n');
        let frameCount = 0;
        let frameTime = this.frameTime;
        
        for (const line of lines) {
            if (line.startsWith('Frames:')) {
                frameCount = parseInt(line.split(':')[1].trim());
            } else if (line.startsWith('Frame Time:')) {
                frameTime = parseFloat(line.split(':')[1].trim());
            }
        }
        
        return frameCount * frameTime;
    }
    
    cloneFrame(frame) {
        return {
            ...frame,
            motionData: frame.motionData ? frame.motionData.map(bone => [...bone]) : []
        };
    }
    
    getDefaultFrame(time = null) {
        return {
            time: time !== null ? time : this.currentTime,
            motionData: [],
            metadata: { type: 'default' }
        };
    }
    
    /**
     * Update buffer hit rate statistics
     */
    updateBufferHitRate() {
        if (this.performanceStats.totalFramesRequested > 0) {
            this.performanceStats.bufferHitRate = 
                this.performanceStats.bufferHits / this.performanceStats.totalFramesRequested;
        }
    }
    
    /**
     * Add clip to track with buffer invalidation
     */
    addClip(trackName, clip) {
        if (!this.tracks[trackName]) {
            throw new Error(`Track '${trackName}' does not exist`);
        }
        
        const clipId = this.tracks[trackName].addClip(clip);
        
        // Invalidate buffer for affected time range
        this.frameBuffer.invalidateRange(clip.startTime, clip.startTime + clip.duration);
        
        if (this.onTrackChange) {
            this.onTrackChange(trackName, 'clip_added', { clipId, clip });
        }
        
        return clipId;
    }
    
    /**
     * Remove clip from track with buffer invalidation
     */
    removeClip(trackName, clipId) {
        if (this.tracks[trackName]) {
            const clip = this.tracks[trackName].clips.find(c => c.id === clipId);
            const removed = this.tracks[trackName].removeClip(clipId);
            
            if (removed && clip) {
                // Invalidate buffer for affected time range
                this.frameBuffer.invalidateRange(clip.startTime, clip.startTime + clip.duration);
                
                if (this.onTrackChange) {
                    this.onTrackChange(trackName, 'clip_removed', { clipId, clip });
                }
            }
            
            return removed;
        }
        return false;
    }
    
    /**
     * Clear all clips from a track with buffer invalidation
     */
    clearTrack(trackName) {
        if (this.tracks[trackName]) {
            this.tracks[trackName].clear();
            
            // Clear entire buffer since we don't know what was affected
            this.frameBuffer.clear();
            
            if (this.onTrackChange) {
                this.onTrackChange(trackName, 'track_cleared', {});
            }
        }
    }
    
    /**
     * Clear all tracks with buffer invalidation
     */
    clearAll() {
        Object.values(this.tracks).forEach(track => track.clear());
        this.frameBuffer.clear();
        
        if (this.onTrackChange) {
            this.onTrackChange('all', 'all_cleared', {});
        }
    }
    
    /**
     * Get enhanced timeline statistics including buffer info
     */
    getStats() {
        const stats = {
            currentTime: this.currentTime,
            isPlaying: this.isPlaying,
            totalTracks: Object.keys(this.tracks).length,
            tracks: {},
            buffer: this.frameBuffer.getStats(),
            performance: { ...this.performanceStats }
        };
        
        for (const [name, track] of Object.entries(this.tracks)) {
            stats.tracks[name] = {
                clipCount: track.clips.length,
                totalDuration: track.getTotalDuration(),
                priority: track.priority
            };
        }
        
        return stats;
    }
    
    /**
     * Get frame buffer statistics
     */
    getBufferStats() {
        return this.frameBuffer.getStats();
    }
    
    /**
     * Force buffer cleanup
     */
    cleanupBuffer() {
        this.frameBuffer.cleanup(this.currentTime);
    }
    
    /**
     * Clear frame buffer
     */
    clearBuffer() {
        this.frameBuffer.clear();
    }
    
    /**
     * Dispose timeline and cleanup resources
     */
    dispose() {
        this.pause();
        this.stopBufferManagement();
        this.frameBuffer.dispose();
        this.clearAll();
        
        console.log('[BVHTimeline] Timeline disposed');
    }
    
    interpolatePoses(fromPose, toPose, progress) {
        // Simple linear interpolation between poses
        const result = this.cloneFrame(fromPose);
        
        if (toPose.motionData && fromPose.motionData) {
            fromPose.motionData.forEach((fromBone, boneIndex) => {
                const toBone = toPose.motionData[boneIndex];
                if (toBone) {
                    for (let i = 0; i < fromBone.length; i++) {
                        result.motionData[boneIndex][i] = 
                            fromBone[i] * (1 - progress) + toBone[i] * progress;
                    }
                }
            });
        }
        
        return result;
    }
    
    parseBVHFrame(bvhData) {
        // Parse BVH frame data into internal format
        // This is a simplified parser - you may need more sophisticated parsing
        return {
            time: this.currentTime,
            motionData: bvhData.motionData || [],
            metadata: bvhData.metadata || {}
        };
    }
    
    getBoneName(boneIndex) {
        // Map bone index to name based on skeleton
        return this.boneMapping.get(boneIndex) || `bone_${boneIndex}`;
    }
    
    getAudioSampleAtTime(audioBuffer, time) {
        // Extract audio sample at specific time
        const sampleRate = audioBuffer.sampleRate;
        const sampleIndex = Math.floor(time * sampleRate);
        const channelData = audioBuffer.getChannelData(0);
        
        // Return a small window around the sample
        const windowSize = 1024;
        const start = Math.max(0, sampleIndex - windowSize / 2);
        const end = Math.min(channelData.length, start + windowSize);
        
        return Array.from(channelData.slice(start, end));
    }
    
    /**
     * Remove clip from track
     */
    removeClip(trackName, clipId) {
        if (this.tracks[trackName]) {
            return this.tracks[trackName].removeClip(clipId);
        }
        return false;
    }
    
    /**
     * Clear all clips from a track
     */
    clearTrack(trackName) {
        if (this.tracks[trackName]) {
            this.tracks[trackName].clear();
        }
    }
    
    /**
     * Clear all tracks
     */
    clearAll() {
        Object.values(this.tracks).forEach(track => track.clear());
    }
    
    /**
     * Get timeline statistics
     */
    getStats() {
        const stats = {
            currentTime: this.currentTime,
            isPlaying: this.isPlaying,
            totalTracks: Object.keys(this.tracks).length,
            tracks: {}
        };
        
        for (const [name, track] of Object.entries(this.tracks)) {
            stats.tracks[name] = {
                clipCount: track.clips.length,
                totalDuration: track.getTotalDuration(),
                priority: track.priority
            };
        }
        
        return stats;
    }
}

/**
 * Timeline Track - represents a single track containing multiple clips
 */
class BVHTrack {
    constructor(name, options = {}) {
        this.name = name;
        this.priority = options.priority || 0;
        this.clips = [];
        this.muted = false;
        this.solo = false;
        this.volume = 1.0;
    }
    
    addClip(clip) {
        clip.id = clip.id || this.generateClipId();
        this.clips.push(clip);
        this.sortClips();
        return clip.id;
    }
    
    removeClip(clipId) {
        const index = this.clips.findIndex(clip => clip.id === clipId);
        if (index !== -1) {
            this.clips.splice(index, 1);
            return true;
        }
        return false;
    }
    
    getActiveClipsAtTime(time) {
        return this.clips.filter(clip => 
            time >= clip.startTime && 
            time < (clip.startTime + clip.duration)
        );
    }
    
    sortClips() {
        this.clips.sort((a, b) => a.startTime - b.startTime);
    }
    
    clear() {
        this.clips = [];
    }
    
    getTotalDuration() {
        if (this.clips.length === 0) return 0;
        return Math.max(...this.clips.map(clip => clip.startTime + clip.duration));
    }
    
    generateClipId() {
        return `${this.name}_clip_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}

/**
 * BVH Clip - represents a single animation clip on a track
 */
class BVHClip {
    constructor(options) {
        this.id = options.id || null;
        this.type = options.type || 'static'; // 'static', 'generated', 'rsmt_transition', etc.
        this.startTime = options.startTime || 0;
        this.duration = options.duration || 1;
        this.weight = options.weight || 1.0;
        this.blendMode = options.blendMode || 'replace';
        this.loop = options.loop || false;
        
        // Data sources
        this.bvhData = options.bvhData || null; // For static clips
        this.generator = options.generator || null; // For generated clips
        
        // Metadata
        this.metadata = options.metadata || {};
        
        // Caching for generated content
        this.frameCache = new Map();
        this.cacheEnabled = options.cacheEnabled !== false;
    }
    
    async getFrameAtTime(localTime) {
        // Handle looping
        if (this.loop && localTime > this.duration) {
            localTime = localTime % this.duration;
        }
        
        // Check cache first
        const cacheKey = Math.floor(localTime * 30); // 30 FPS cache resolution
        if (this.cacheEnabled && this.frameCache.has(cacheKey)) {
            return this.frameCache.get(cacheKey);
        }
        
        let frame;
        
        if (this.type === 'static' && this.bvhData) {
            frame = this.getStaticFrame(localTime);
        } else if (this.generator) {
            const frameIndex = Math.floor(localTime * 30); // Assuming 30 FPS
            frame = await this.generator(localTime, frameIndex);
        } else {
            frame = this.getDefaultFrame();
        }
        
        // Cache the result
        if (this.cacheEnabled && frame) {
            this.frameCache.set(cacheKey, frame);
        }
        
        return frame;
    }
    
    getStaticFrame(localTime) {
      const data = this.bvhData;
      if (!data || !data.poses || !data.poses.length) {
        return this.getDefaultFrame();
      }
      // wrap or clamp the time to the duration of the clip
      const frameIndex = Math.floor(localTime / data.frameTime) % data.poses.length;
      return {
        time: localTime,
        motionData: data.poses[frameIndex],
        metadata: { type: 'static', source: this.id }
      };
    }
    
    getDefaultFrame() {
        return {
            time: 0,
            motionData: [],
            metadata: { type: 'default' }
        };
    }
    
    clearCache() {
        this.frameCache.clear();
    }
}

/**
 * BVH Frame Buffer - manages frame buffering, cleanup, and lookahead
 */
class BVHFrameBuffer {
    constructor(options = {}) {
        this.maxBufferSize = options.maxBufferSize || 300; // Max frames to keep
        this.lookaheadFrames = options.lookaheadFrames || 60; // Frames to buffer ahead
        this.framerate = options.framerate || 30;
        this.frameTime = 1.0 / this.framerate;
        this.cleanupInterval = options.cleanupInterval || 1000; // ms
        this.staleThreshold = options.staleThreshold || 5000; // ms
        
        // Frame storage - Map<timeKey, {frame, timestamp, accessCount}>
        this.frames = new Map();
        
        // Cleanup management
        this.lastCleanup = 0;
        this.cleanupCounter = 0;
        
        // Statistics
        this.stats = {
            totalFrames: 0,
            bufferSize: 0,
            oldestFrame: null,
            newestFrame: null,
            hitRate: 0,
            cleanupCount: 0,
            memoryUsage: 0
        };
        
        console.log('[BVHFrameBuffer] Initialized with max size:', this.maxBufferSize);
    }
    
    /**
     * Get time key for frame storage (quantized to frame boundaries)
     */
    getTimeKey(time) {
        return Math.floor(time / this.frameTime) * this.frameTime;
    }
    
    /**
     * Add frame to buffer
     */
    addFrame(time, frame) {
        const timeKey = this.getTimeKey(time);
        const now = performance.now();
        
        const frameData = {
            frame: frame,
            timestamp: now,
            accessCount: 1,
            size: this.estimateFrameSize(frame)
        };
        
        this.frames.set(timeKey, frameData);
        this.updateStats();
        
        // Cleanup if buffer is too large
        if (this.frames.size > this.maxBufferSize) {
            this.cleanup();
        }
    }
    
    /**
     * Get frame from buffer
     */
    getFrame(time) {
        const timeKey = this.getTimeKey(time);
        const frameData = this.frames.get(timeKey);
        
        if (frameData) {
            frameData.accessCount++;
            frameData.timestamp = performance.now(); // Update access time
            return frameData.frame;
        }
        
        return null;
    }
    
    /**
     * Check if frame exists in buffer
     */
    hasFrame(time) {
        const timeKey = this.getTimeKey(time);
        return this.frames.has(timeKey);
    }
    
    /**
     * Invalidate frames in a specific time range
     */
    invalidateRange(startTime, endTime) {
        const startKey = this.getTimeKey(startTime);
        const endKey = this.getTimeKey(endTime);
        
        let removedCount = 0;
        for (const [timeKey] of this.frames) {
            if (timeKey >= startKey && timeKey <= endKey) {
                this.frames.delete(timeKey);
                removedCount++;
            }
        }
        
        if (removedCount > 0) {
            this.updateStats();
            console.log(`[BVHFrameBuffer] Invalidated ${removedCount} frames in range [${startTime}, ${endTime}]`);
        }
    }
    
    /**
     * Clean up stale frames
     */
    cleanup(currentTime = null) {
        const now = performance.now();
        
        // Skip if cleanup was done recently
        if (now - this.lastCleanup < this.cleanupInterval) {
            return;
        }
        
        let removedCount = 0;
        const cutoffTime = now - this.staleThreshold;
        
        for (const [timeKey, frameData] of this.frames) {
            let shouldRemove = false;
            
            // Remove if too old (not accessed recently)
            if (frameData.timestamp < cutoffTime) {
                shouldRemove = true;
            }
            
            // Remove if too far from current time (if provided)
            if (currentTime !== null) {
                const timeDiff = Math.abs(timeKey - currentTime);
                const maxDistance = this.lookaheadFrames * this.frameTime * 2; // 2x lookahead distance
                
                if (timeDiff > maxDistance) {
                    shouldRemove = true;
                }
            }
            
            if (shouldRemove) {
                this.frames.delete(timeKey);
                removedCount++;
            }
        }
        
        // If still too large, remove oldest frames
        if (this.frames.size > this.maxBufferSize) {
            const sortedFrames = Array.from(this.frames.entries())
                .sort((a, b) => a[1].timestamp - b[1].timestamp);
            
            const toRemove = this.frames.size - this.maxBufferSize;
            for (let i = 0; i < toRemove && i < sortedFrames.length; i++) {
                this.frames.delete(sortedFrames[i][0]);
                removedCount++;
            }
        }
        
        this.lastCleanup = now;
        this.cleanupCounter++;
        this.updateStats();
        
        if (removedCount > 0) {
            console.log(`[BVHFrameBuffer] Cleanup #${this.cleanupCounter}: removed ${removedCount} stale frames`);
        }
    }
    
    /**
     * Clear frames around a specific time (used for seeking)
     */
    clearStaleFrames(centerTime) {
        const radius = this.lookaheadFrames * this.frameTime;
        const startTime = centerTime - radius;
        const endTime = centerTime + radius;
        
        let removedCount = 0;
        for (const [timeKey] of this.frames) {
            if (timeKey < startTime || timeKey > endTime) {
                this.frames.delete(timeKey);
                removedCount++;
            }
        }
        
        if (removedCount > 0) {
            this.updateStats();
            console.log(`[BVHFrameBuffer] Cleared ${removedCount} frames outside range [${startTime}, ${endTime}]`);
        }
    }
    
    /**
     * Clear all frames
     */
    clear() {
        const count = this.frames.size;
        this.frames.clear();
        this.updateStats();
        
        if (count > 0) {
            console.log(`[BVHFrameBuffer] Cleared all ${count} frames`);
        }
    }
    
    /**
     * Estimate frame size for memory usage tracking
     */
    estimateFrameSize(frame) {
        let size = 0;
        
        // Estimate motion data size
        if (frame.motionData && Array.isArray(frame.motionData)) {
            size += frame.motionData.length * 6 * 8; // 6 floats per bone * 8 bytes per float
        }
        
        // Estimate metadata size
        if (frame.metadata) {
            size += JSON.stringify(frame.metadata).length * 2; // Rough estimate
        }
        
        size += 64; // Base frame object overhead
        
        return size;
    }
    
    /**
     * Update buffer statistics
     */
    updateStats() {
        this.stats.bufferSize = this.frames.size;
        this.stats.totalFrames = this.frames.size;
        this.stats.cleanupCount = this.cleanupCounter;
        
        if (this.frames.size > 0) {
            const timeKeys = Array.from(this.frames.keys()).sort((a, b) => a - b);
            this.stats.oldestFrame = timeKeys[0];
            this.stats.newestFrame = timeKeys[timeKeys.length - 1];
            
            // Calculate memory usage
            this.stats.memoryUsage = Array.from(this.frames.values())
                .reduce((total, frameData) => total + frameData.size, 0);
        } else {
            this.stats.oldestFrame = null;
            this.stats.newestFrame = null;
            this.stats.memoryUsage = 0;
        }
    }
    
    /**
     * Get buffer statistics
     */
    getStats() {
        this.updateStats();
        return {
            ...this.stats,
            utilizationPercent: (this.stats.bufferSize / this.maxBufferSize) * 100,
            timeSpan: this.stats.newestFrame && this.stats.oldestFrame ? 
                this.stats.newestFrame - this.stats.oldestFrame : 0,
            averageFrameSize: this.stats.bufferSize > 0 ? 
                this.stats.memoryUsage / this.stats.bufferSize : 0
        };
    }
    
    /**
     * Get frames in a time range
     */
    getFramesInRange(startTime, endTime) {
        const frames = [];
        
        for (const [timeKey, frameData] of this.frames) {
            if (timeKey >= startTime && timeKey <= endTime) {
                frames.push({
                    time: timeKey,
                    frame: frameData.frame,
                    accessCount: frameData.accessCount
                });
            }
        }
        
        return frames.sort((a, b) => a.time - b.time);
    }
    
    /**
     * Get buffer health status
     */
    getHealthStatus() {
        const stats = this.getStats();
        
        return {
            healthy: stats.utilizationPercent < 90,
            utilizationPercent: stats.utilizationPercent,
            memoryUsageMB: stats.memoryUsage / (1024 * 1024),
            recommendedAction: stats.utilizationPercent > 90 ? 'cleanup' : 'normal'
        };
    }
    
    /**
     * Dispose buffer and cleanup resources
     */
    dispose() {
        this.clear();
        console.log('[BVHFrameBuffer] Buffer disposed');
    }
}



// =================================================================================
//  3. IMPLEMENTATION OF MISSING LOADER FUNCTIONS
// =================================================================================

/**
 * A basic BVH parser. In a real application, this would be more robust.
 * @param {string} bvhText - The text content of a .bvh file.
 * @returns {object} Parsed animation data.
 */
function yourBVHParser(bvhText) {
    const lines = bvhText.split('\n');
    let motionIndex = lines.findIndex(line => line.trim().startsWith('MOTION'));
    if (motionIndex === -1) throw new Error('No MOTION section in BVH file');

    let frameCountLine = lines.find(line => line.trim().startsWith('Frames:'));
    let frameTimeLine = lines.find(line => line.trim().startsWith('Frame Time:'));

    const frameCount = frameCountLine ? parseInt(frameCountLine.split(':')[1]) : 0;
    const frameTime = frameTimeLine ? parseFloat(frameTimeLine.split(':')[1]) : 0.0333;

    const frames = [];
    const motionDataStartIndex = lines.indexOf(frameTimeLine) + 1;

    for (let i = motionDataStartIndex; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line) {
            const values = line.split(/\s+/).filter(Boolean).map(v => parseFloat(v));

            // *** THE FIX: Check if EVERY value in the array is a valid number. ***
            const hasNaN = values.some(v => isNaN(v));

            if (values.length > 1 && !hasNaN) {
                frames.push(new Float32Array(values));
            } else if (hasNaN) {
                // Optional but recommended: log the corrupted line for debugging.
                console.warn(`[BVHParser] Discarded corrupted data line: "${line}"`);
            }
        }
    }

    // Generate a timestamp for each frame.
    const timestamps = frames.map((_, index) => index * frameTime);

    return {
        poses: frames,
        timestamps: timestamps,
        frameTime: frameTime,
        fps: 1 / frameTime
    };
}

let fs, path;
fs = await import('fs/promises');
path = await import('path');

/**
 * Loads a list of BVH files.
 * @param {Array<object>} fileList - Array of objects with { url, id }.
 * @param {Function} parser - The parser function to process the BVH text.
 * @returns {Promise<Array<object>>} A promise that resolves to an array of animation objects.
 */
async function loadAnimations(fileList, parser) {
    let text;
    const animationPromises = fileList.map(async (fileInfo) => {
        try {
            const scriptDir = path.dirname(import.meta.url.replace('file://', ''));
            const filePath = path.resolve(scriptDir, '..', fileInfo.url); // Go up one dir from /js
            console.log("filePath = ", filePath);
            text = await fs.readFile(filePath, 'utf-8');

            const parsedData = parser(text);
            return {
                animationId: fileInfo.id,
                ...parsedData
            };
        } catch (error) {
            console.error(`Error loading animation ${fileInfo.id}:`, error);
            return null; // Return null for failed loads
        }
    });

    const animations = await Promise.all(animationPromises);
    return animations.filter(anim => anim !== null); // Filter out any that failed to load
}


// =================================================================================
//  4. EXAMPLE USAGE (MODIFIED TO WORK)
// =================================================================================

(async () => {
  // Load motions
  const list = [
    { url: './angry_reference.bvh', id: 'angry' },
    { url: './robot_reference.bvh',  id: 'robot'  }
    // Add other animations here
  ];

  console.log("Loading animations...");
  const animations = await loadAnimations(list, yourBVHParser);
  console.log("Animations loaded:", animations);

  // Instantiate and start
  // 1. Create the timeline instance (it's not ready yet)
  const timeline = new BVHTimeline(animations, { framerate: 30 });

  // 2. *** CALL THE NEW ASYNC INIT METHOD AND WAIT FOR IT ***
  await timeline.init();

  // 3. Now it's safe to use the timeline
  timeline.play();

  // Example onFrameUpdate callback
  timeline.onFrameUpdate = (frame, time) => {
    // In a real app, you would send this frame to your renderer
    // console.log(`Time: ${time.toFixed(2)}s, Pose Root:`, frame.slice(0, 3));
  };

  // Optionally trigger a transition later
  setTimeout(() => {
      console.log("\n>>> TRIGGERING SMART TRANSITION TO 'robot' <<<\n");
      timeline.transitionTo('robot');
  }, 3000); // Transition after 3 seconds
})();


// Export for use in other modules
// — ESM exports ——
export default BVHTimeline;
export { BVHTrack, BVHClip, BVHFrameBuffer };
