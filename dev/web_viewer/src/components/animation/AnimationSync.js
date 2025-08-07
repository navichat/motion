/**
 * AnimationSync - Handles synchronization between audio and VRM animations
 * Provides timing coordination for gesture and facial animations
 */

class AnimationSync {
    constructor() {
        this.isInitialized = false;
        this.syncData = null;
        this.animationStartTime = null;
        this.audioStartTime = null;
        this.syncOffset = 0; // Offset in milliseconds to sync audio and animation
        this.callbacks = {
            onSyncStart: [],
            onSyncUpdate: [],
            onSyncEnd: []
        };
    }

    initialize() {
        this.isInitialized = true;
        console.log('✅ AnimationSync initialized');
        return true;
    }

    /**
     * Create synchronization data from audio and animation
     * @param {AudioBuffer} audioBuffer - The audio buffer
     * @param {Array} gestureFrames - Array of gesture animation frames
     * @param {Array} facialFrames - Array of facial animation frames (optional)
     * @param {number} fps - Animation frame rate (default: 30)
     * @returns {Object} - Synchronization data object
     */
    createSyncData(audioBuffer, gestureFrames, facialFrames = null, fps = 30) {
        if (!audioBuffer || !gestureFrames) {
            throw new Error('Audio buffer and gesture frames are required for sync');
        }

        const audioDuration = audioBuffer.duration;
        const animationDuration = gestureFrames.length / fps;
        const frameTime = 1 / fps; // Time per frame in seconds

        console.log(`🎬 Creating sync data:`);
        console.log(`  Audio duration: ${audioDuration.toFixed(3)}s`);
        console.log(`  Animation duration: ${animationDuration.toFixed(3)}s`);
        console.log(`  Frame rate: ${fps} FPS`);

        // Create timing map for each animation frame
        const timingMap = [];
        for (let i = 0; i < gestureFrames.length; i++) {
            const frameTime_ms = (i * frameTime) * 1000; // Convert to milliseconds
            timingMap.push({
                frameIndex: i,
                timeMs: frameTime_ms,
                gestureData: gestureFrames[i],
                facialData: facialFrames ? facialFrames[i] : null
            });
        }

        // Calculate sync metrics
        const durationDiff = Math.abs(audioDuration - animationDuration);
        const syncQuality = durationDiff < 0.1 ? 'excellent' : durationDiff < 0.5 ? 'good' : 'fair';

        this.syncData = {
            audioDuration,
            animationDuration,
            fps,
            frameTime,
            timingMap,
            syncQuality,
            durationDiff,
            totalFrames: gestureFrames.length,
            hasGestures: true,
            hasFacial: facialFrames !== null
        };

        console.log(`✅ Sync data created with ${syncQuality} quality`);
        console.log(`  Duration difference: ${durationDiff.toFixed(3)}s`);

        return this.syncData;
    }

    /**
     * Start synchronized playback
     * @param {AudioContext} audioContext - Web Audio API context
     * @param {AudioBuffer} audioBuffer - Audio to play
     * @param {Function} animationCallback - Callback to update animation frames
     * @param {Object} options - Sync options
     */
    async startSyncedPlayback(audioContext, audioBuffer, animationCallback, options = {}) {
        if (!this.syncData) {
            throw new Error('No sync data available. Call createSyncData first.');
        }

        const {
            loop = false,
            fadeInDuration = 0,
            fadeOutDuration = 0,
            playbackRate = 1.0
        } = options;

        console.log('🎬 Starting synchronized playback...');

        // Create audio source
        const audioSource = audioContext.createBufferSource();
        audioSource.buffer = audioBuffer;
        audioSource.playbackRate.value = playbackRate;

        // Add fade effects if requested
        let gainNode = null;
        if (fadeInDuration > 0 || fadeOutDuration > 0) {
            gainNode = audioContext.createGain();
            audioSource.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            if (fadeInDuration > 0) {
                gainNode.gain.setValueAtTime(0, audioContext.currentTime);
                gainNode.gain.linearRampToValueAtTime(1, audioContext.currentTime + fadeInDuration);
            }
            
            if (fadeOutDuration > 0) {
                const fadeStartTime = audioContext.currentTime + audioBuffer.duration - fadeOutDuration;
                gainNode.gain.setValueAtTime(1, fadeStartTime);
                gainNode.gain.linearRampToValueAtTime(0, fadeStartTime + fadeOutDuration);
            }
        } else {
            audioSource.connect(audioContext.destination);
        }

        // Set up timing
        this.audioStartTime = audioContext.currentTime;
        this.animationStartTime = performance.now();

        // Start audio
        audioSource.start(0);

        // Trigger sync start callbacks
        this.triggerCallbacks('onSyncStart', {
            audioStartTime: this.audioStartTime,
            animationStartTime: this.animationStartTime,
            syncData: this.syncData
        });

        // Start animation sync loop
        this.startAnimationLoop(animationCallback, playbackRate);

        // Handle audio end
        audioSource.onended = () => {
            console.log('🏁 Synchronized playback ended');
            this.stopAnimationLoop();
            
            if (loop && this.isPlaying) {
                // Restart if looping
                setTimeout(() => {
                    this.startSyncedPlayback(audioContext, audioBuffer, animationCallback, options);
                }, 100);
            } else {
                this.triggerCallbacks('onSyncEnd', {
                    totalDuration: performance.now() - this.animationStartTime,
                    syncData: this.syncData
                });
            }
        };

        return {
            audioSource,
            gainNode,
            stop: () => {
                audioSource.stop();
                this.stopAnimationLoop();
            }
        };
    }

    /**
     * Start the animation synchronization loop
     */
    startAnimationLoop(animationCallback, playbackRate = 1.0) {
        this.isPlaying = true;
        let lastFrameIndex = -1;

        const animationLoop = () => {
            if (!this.isPlaying) return;

            const currentTime = performance.now();
            const elapsedTime = (currentTime - this.animationStartTime) * playbackRate;
            
            // Find the current frame based on elapsed time
            const currentFrameIndex = Math.floor(elapsedTime / (this.syncData.frameTime * 1000));
            
            if (currentFrameIndex < this.syncData.totalFrames && currentFrameIndex !== lastFrameIndex) {
                const frameData = this.syncData.timingMap[currentFrameIndex];
                
                if (frameData) {
                    // Call the animation callback with frame data
                    animationCallback(frameData, currentFrameIndex, elapsedTime);
                    
                    // Trigger sync update callbacks
                    this.triggerCallbacks('onSyncUpdate', {
                        frameIndex: currentFrameIndex,
                        frameData,
                        elapsedTime,
                        progress: currentFrameIndex / this.syncData.totalFrames
                    });
                    
                    lastFrameIndex = currentFrameIndex;
                }
            }

            // Continue the loop
            requestAnimationFrame(animationLoop);
        };

        // Start the animation loop
        requestAnimationFrame(animationLoop);
    }

    /**
     * Stop the animation synchronization loop
     */
    stopAnimationLoop() {
        this.isPlaying = false;
        console.log('⏹️ Animation sync loop stopped');
    }

    /**
     * Add event listeners for sync events
     */
    addEventListener(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        } else {
            console.warn(`Unknown event: ${event}`);
        }
    }

    /**
     * Remove event listeners
     */
    removeEventListener(event, callback) {
        if (this.callbacks[event]) {
            const index = this.callbacks[event].indexOf(callback);
            if (index > -1) {
                this.callbacks[event].splice(index, 1);
            }
        }
    }

    /**
     * Trigger callbacks for specific events
     */
    triggerCallbacks(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${event} callback:`, error);
                }
            });
        }
    }

    /**
     * Calculate timing adjustments to improve sync
     */
    calculateSyncAdjustment(actualFrameTime, expectedFrameTime) {
        const timeDiff = actualFrameTime - expectedFrameTime;
        
        // If we're behind by more than half a frame, skip ahead
        if (timeDiff > this.syncData.frameTime * 500) { // 500ms for half frame at 30fps
            return 1; // Skip next frame
        }
        
        // If we're ahead by more than half a frame, delay
        if (timeDiff < -this.syncData.frameTime * 500) {
            return -1; // Repeat current frame
        }
        
        return 0; // No adjustment needed
    }

    /**
     * Get sync statistics
     */
    getSyncStats() {
        if (!this.syncData) return null;

        return {
            audioDuration: this.syncData.audioDuration,
            animationDuration: this.syncData.animationDuration,
            syncQuality: this.syncData.syncQuality,
            durationDiff: this.syncData.durationDiff,
            totalFrames: this.syncData.totalFrames,
            fps: this.syncData.fps,
            hasGestures: this.syncData.hasGestures,
            hasFacial: this.syncData.hasFacial,
            isPlaying: this.isPlaying,
            syncOffset: this.syncOffset
        };
    }

    /**
     * Adjust sync offset to compensate for timing drift
     */
    adjustSyncOffset(offsetMs) {
        this.syncOffset += offsetMs;
        console.log(`🔧 Sync offset adjusted by ${offsetMs}ms (total: ${this.syncOffset}ms)`);
    }

    /**
     * Reset sync offset
     */
    resetSyncOffset() {
        this.syncOffset = 0;
        console.log('🔄 Sync offset reset');
    }

    /**
     * Create a simple beat detection for rhythm-based gestures
     */
    static detectBeats(audioBuffer, options = {}) {
        const {
            sensitivity = 0.5,
            minBeatInterval = 0.3, // Minimum time between beats in seconds
            windowSize = 1024
        } = options;

        const audioData = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;
        const beats = [];
        
        // Simple energy-based beat detection
        let lastBeatTime = -minBeatInterval;
        
        for (let i = 0; i < audioData.length - windowSize; i += windowSize / 2) {
            const window = audioData.slice(i, i + windowSize);
            
            // Calculate energy
            let energy = 0;
            for (let j = 0; j < window.length; j++) {
                energy += window[j] * window[j];
            }
            energy = Math.sqrt(energy / window.length);
            
            // Simple threshold-based beat detection
            const currentTime = i / sampleRate;
            if (energy > sensitivity && (currentTime - lastBeatTime) > minBeatInterval) {
                beats.push({
                    time: currentTime,
                    energy: energy,
                    sample: i
                });
                lastBeatTime = currentTime;
            }
        }
        
        console.log(`🥁 Detected ${beats.length} beats in ${audioBuffer.duration.toFixed(2)}s audio`);
        return beats;
    }

    /**
     * Create gesture emphasis points based on audio analysis
     */
    static createGestureEmphasis(audioBuffer, gestureFrames, fps = 30) {
        const beats = AnimationSync.detectBeats(audioBuffer);
        const emphasisFrames = [];
        
        beats.forEach(beat => {
            const frameIndex = Math.floor(beat.time * fps);
            if (frameIndex < gestureFrames.length) {
                emphasisFrames.push({
                    frameIndex,
                    time: beat.time,
                    energy: beat.energy,
                    emphasisType: beat.energy > 0.7 ? 'strong' : 'medium'
                });
            }
        });
        
        console.log(`✨ Created ${emphasisFrames.length} gesture emphasis points`);
        return emphasisFrames;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AnimationSync };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.AnimationSync = AnimationSync;
}
