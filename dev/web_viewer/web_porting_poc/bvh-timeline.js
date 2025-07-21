/**
 * Simplified BVH Timeline Implementation
 * For unified animation systems testing
 */

class BVHClip {
    constructor(options = {}) {
        this.id = options.id || `clip_${Date.now()}`;
        this.type = options.type || 'motion';
        this.startTime = options.startTime || 0;
        this.duration = options.duration || 1.0;
        this.weight = options.weight || 1.0;
        this.blendMode = options.blendMode || 'replace'; // replace, add, blend
        this.data = options.data || null;
        this.generator = options.generator || null;
        this.metadata = options.metadata || {};
        
        // Animation state
        this.isActive = false;
        this.localTime = 0;
    }
    
    /**
     * Get frame data at specific time
     */
    async getFrameAt(time) {
        this.localTime = time;
        
        if (this.generator && typeof this.generator === 'function') {
            return await this.generator(time, Math.floor(time * 30)); // Assume 30 FPS
        }
        
        if (this.data && Array.isArray(this.data)) {
            const frameIndex = Math.floor(time * 30) % this.data.length;
            return this.data[frameIndex];
        }
        
        // Return empty frame if no data
        return this.createEmptyFrame();
    }
    
    /**
     * Create empty BVH frame
     */
    createEmptyFrame() {
        return {
            Hips: [0, 0, 0, 0, 0, 0],
            Chest: [0, 0, 0],
            Neck: [0, 0, 0],
            Head: [0, 0, 0],
            LeftShoulder: [0, 0, 0],
            LeftElbow: [0, 0, 0],
            LeftWrist: [0, 0, 0],
            RightShoulder: [0, 0, 0],
            RightElbow: [0, 0, 0],
            RightWrist: [0, 0, 0],
            LeftHip: [0, 0, 0],
            LeftKnee: [0, 0, 0],
            LeftAnkle: [0, 0, 0],
            RightHip: [0, 0, 0],
            RightKnee: [0, 0, 0],
            RightAnkle: [0, 0, 0]
        };
    }
    
    /**
     * Check if clip is active at given time
     */
    isActiveAt(time) {
        return time >= this.startTime && time <= this.startTime + this.duration;
    }
}

class BVHTrack {
    constructor(name, options = {}) {
        this.name = name;
        this.clips = [];
        this.muted = false;
        this.solo = false;
        this.volume = options.volume || 1.0;
        this.blendMode = options.blendMode || 'replace';
    }
    
    /**
     * Add clip to track
     */
    addClip(clip) {
        if (!(clip instanceof BVHClip)) {
            clip = new BVHClip(clip);
        }
        
        this.clips.push(clip);
        this.clips.sort((a, b) => a.startTime - b.startTime);
        
        return clip;
    }
    
    /**
     * Remove clip from track
     */
    removeClip(clipId) {
        const index = this.clips.findIndex(clip => clip.id === clipId);
        if (index !== -1) {
            return this.clips.splice(index, 1)[0];
        }
        return null;
    }
    
    /**
     * Get all active clips at given time
     */
    getActiveClips(time) {
        return this.clips.filter(clip => clip.isActiveAt(time));
    }
    
    /**
     * Generate composite frame for track at given time
     */
    async generateFrame(time) {
        if (this.muted) return null;
        
        const activeClips = this.getActiveClips(time);
        if (activeClips.length === 0) return null;
        
        let compositeFrame = null;
        let totalWeight = 0;
        
        for (const clip of activeClips) {
            const localTime = time - clip.startTime;
            const clipFrame = await clip.getFrameAt(localTime);
            const weight = clip.weight * this.volume;
            
            if (!compositeFrame) {
                compositeFrame = this.cloneFrame(clipFrame);
                totalWeight = weight;
            } else {
                this.blendFrames(compositeFrame, clipFrame, weight / (totalWeight + weight));
                totalWeight += weight;
            }
        }
        
        return compositeFrame;
    }
    
    /**
     * Clone a BVH frame
     */
    cloneFrame(frame) {
        const clone = {};
        for (const [joint, data] of Object.entries(frame)) {
            if (Array.isArray(data)) {
                clone[joint] = [...data];
            } else {
                clone[joint] = data;
            }
        }
        return clone;
    }
    
    /**
     * Blend two BVH frames
     */
    blendFrames(baseFrame, overlayFrame, weight) {
        for (const joint in overlayFrame) {
            if (baseFrame[joint] && Array.isArray(baseFrame[joint]) && Array.isArray(overlayFrame[joint])) {
                for (let i = 0; i < Math.min(baseFrame[joint].length, overlayFrame[joint].length); i++) {
                    baseFrame[joint][i] = baseFrame[joint][i] * (1 - weight) + overlayFrame[joint][i] * weight;
                }
            } else if (overlayFrame[joint] && !baseFrame[joint]) {
                baseFrame[joint] = Array.isArray(overlayFrame[joint]) ? [...overlayFrame[joint]] : overlayFrame[joint];
            }
        }
    }
    
    /**
     * Clear all clips
     */
    clear() {
        this.clips = [];
    }
}

class BVHTimeline {
    constructor(options = {}) {
        this.tracks = new Map();
        this.currentTime = 0;
        this.isPlaying = false;
        this.frameRate = options.frameRate || 30;
        this.duration = options.duration || 10.0;
        
        // Playback state
        this.playbackSpeed = 1.0;
        this.loop = options.loop || false;
        
        // Frame buffer
        this.currentFrame = null;
        this.frameHistory = [];
        this.maxHistorySize = options.maxHistorySize || 100;
        
        // Event callbacks
        this.onFrameGenerated = options.onFrameGenerated || null;
        this.onTimeChanged = options.onTimeChanged || null;
        
        console.log('🎬 BVH Timeline initialized');
    }
    
    /**
     * Add or get track
     */
    getTrack(name, create = true) {
        if (!this.tracks.has(name) && create) {
            this.tracks.set(name, new BVHTrack(name));
        }
        return this.tracks.get(name);
    }
    
    /**
     * Add clip to specific track
     */
    addClip(trackName, clipData) {
        const track = this.getTrack(trackName);
        const clip = track.addClip(clipData);
        
        console.log(`📎 Added clip "${clip.id}" to track "${trackName}"`);
        return clip;
    }
    
    /**
     * Remove clip from track
     */
    removeClip(trackName, clipId) {
        const track = this.getTrack(trackName, false);
        if (track) {
            const clip = track.removeClip(clipId);
            if (clip) {
                console.log(`🗑️ Removed clip "${clipId}" from track "${trackName}"`);
                return clip;
            }
        }
        return null;
    }
    
    /**
     * Update timeline to specific time
     */
    async update(time) {
        this.currentTime = Math.max(0, time);
        
        // Handle looping
        if (this.loop && this.currentTime > this.duration) {
            this.currentTime = this.currentTime % this.duration;
        }
        
        // Generate current frame
        await this.generateCurrentFrame();
        
        // Call time change callback
        if (this.onTimeChanged) {
            this.onTimeChanged(this.currentTime);
        }
    }
    
    /**
     * Generate composite frame from all tracks
     */
    async generateCurrentFrame() {
        let compositeFrame = null;
        const activeFrames = [];
        
        // Collect frames from all tracks
        for (const [trackName, track] of this.tracks) {
            const trackFrame = await track.generateFrame(this.currentTime);
            if (trackFrame) {
                activeFrames.push({
                    trackName: trackName,
                    frame: trackFrame,
                    blendMode: track.blendMode
                });
            }
        }
        
        // Composite frames
        if (activeFrames.length > 0) {
            compositeFrame = this.compositeFrames(activeFrames);
            
            // Add metadata
            compositeFrame._metadata = {
                time: this.currentTime,
                sources: activeFrames.map(f => f.trackName),
                frameCount: this.frameHistory.length + 1,
                timestamp: Date.now()
            };
        }
        
        // Update current frame
        this.currentFrame = compositeFrame;
        
        // Add to history
        if (compositeFrame) {
            this.frameHistory.push({
                time: this.currentTime,
                frame: this.cloneFrame(compositeFrame)
            });
            
            // Limit history size
            if (this.frameHistory.length > this.maxHistorySize) {
                this.frameHistory.shift();
            }
        }
        
        // Call frame generated callback
        if (this.onFrameGenerated && compositeFrame) {
            this.onFrameGenerated(compositeFrame);
        }
        
        return compositeFrame;
    }
    
    /**
     * Composite multiple frames with different blend modes
     */
    compositeFrames(frameData) {
        if (frameData.length === 0) return null;
        if (frameData.length === 1) return this.cloneFrame(frameData[0].frame);
        
        // Start with first frame
        let composite = this.cloneFrame(frameData[0].frame);
        
        // Blend additional frames
        for (let i = 1; i < frameData.length; i++) {
            const { frame, blendMode } = frameData[i];
            
            switch (blendMode) {
                case 'add':
                    this.addFrames(composite, frame);
                    break;
                case 'multiply':
                    this.multiplyFrames(composite, frame);
                    break;
                case 'overlay':
                    this.overlayFrames(composite, frame);
                    break;
                case 'replace':
                default:
                    this.replaceFrames(composite, frame);
                    break;
            }
        }
        
        return composite;
    }
    
    /**
     * Frame blending operations
     */
    replaceFrames(base, overlay) {
        for (const joint in overlay) {
            if (Array.isArray(overlay[joint])) {
                base[joint] = [...overlay[joint]];
            }
        }
    }
    
    addFrames(base, overlay) {
        for (const joint in overlay) {
            if (base[joint] && Array.isArray(base[joint]) && Array.isArray(overlay[joint])) {
                for (let i = 0; i < Math.min(base[joint].length, overlay[joint].length); i++) {
                    base[joint][i] += overlay[joint][i];
                }
            }
        }
    }
    
    multiplyFrames(base, overlay) {
        for (const joint in overlay) {
            if (base[joint] && Array.isArray(base[joint]) && Array.isArray(overlay[joint])) {
                for (let i = 0; i < Math.min(base[joint].length, overlay[joint].length); i++) {
                    base[joint][i] *= overlay[joint][i];
                }
            }
        }
    }
    
    overlayFrames(base, overlay) {
        // Smart overlay: replace non-zero values, keep zero values from base
        for (const joint in overlay) {
            if (base[joint] && Array.isArray(base[joint]) && Array.isArray(overlay[joint])) {
                for (let i = 0; i < Math.min(base[joint].length, overlay[joint].length); i++) {
                    if (Math.abs(overlay[joint][i]) > 0.001) { // Non-zero threshold
                        base[joint][i] = overlay[joint][i];
                    }
                }
            }
        }
    }
    
    /**
     * Clone frame data
     */
    cloneFrame(frame) {
        const clone = {};
        for (const [joint, data] of Object.entries(frame)) {
            if (Array.isArray(data)) {
                clone[joint] = [...data];
            } else {
                clone[joint] = data;
            }
        }
        return clone;
    }
    
    /**
     * Playback controls
     */
    play() {
        this.isPlaying = true;
        console.log('▶️ Timeline playback started');
    }
    
    pause() {
        this.isPlaying = false;
        console.log('⏸️ Timeline playback paused');
    }
    
    stop() {
        this.isPlaying = false;
        this.currentTime = 0;
        console.log('⏹️ Timeline playback stopped');
    }
    
    seek(time) {
        this.currentTime = Math.max(0, Math.min(time, this.duration));
        this.update(this.currentTime);
        console.log(`⏭️ Timeline seeked to ${this.currentTime.toFixed(2)}s`);
    }
    
    /**
     * Get timeline info
     */
    getInfo() {
        return {
            currentTime: this.currentTime,
            duration: this.duration,
            isPlaying: this.isPlaying,
            frameRate: this.frameRate,
            trackCount: this.tracks.size,
            totalClips: Array.from(this.tracks.values()).reduce((sum, track) => sum + track.clips.length, 0)
        };
    }
    
    /**
     * Clear all tracks and clips
     */
    clearAllTracks() {
        this.tracks.clear();
        this.currentFrame = null;
        this.frameHistory = [];
        console.log('🗑️ All tracks cleared');
    }
    
    /**
     * Export timeline data
     */
    exportData() {
        const data = {
            duration: this.duration,
            frameRate: this.frameRate,
            tracks: {}
        };
        
        for (const [name, track] of this.tracks) {
            data.tracks[name] = {
                clips: track.clips.map(clip => ({
                    id: clip.id,
                    type: clip.type,
                    startTime: clip.startTime,
                    duration: clip.duration,
                    weight: clip.weight,
                    metadata: clip.metadata
                }))
            };
        }
        
        return data;
    }
    
    /**
     * Get current frame as text representation
     */
    getCurrentFrameAsText() {
        if (!this.currentFrame) return 'No frame data';
        
        let text = `Timeline Frame (t=${this.currentTime.toFixed(3)}s)\n`;
        text += `Sources: ${this.currentFrame._metadata?.sources?.join(', ') || 'unknown'}\n\n`;
        
        const joints = Object.entries(this.currentFrame).filter(([key]) => !key.startsWith('_'));
        for (const [joint, data] of joints) {
            if (Array.isArray(data)) {
                const values = data.map(v => v.toFixed(2)).join(', ');
                text += `${joint}: [${values}]\n`;
            }
        }
        
        return text;
    }
    
    /**
     * Dispose timeline
     */
    dispose() {
        this.clearAllTracks();
        this.isPlaying = false;
        console.log('🧹 BVH Timeline disposed');
    }
}

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BVHTimeline, BVHTrack, BVHClip };
} else if (typeof window !== 'undefined') {
    window.BVHTimeline = BVHTimeline;
    window.BVHTrack = BVHTrack;
    window.BVHClip = BVHClip;
}
