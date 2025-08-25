/**
 * FaceFormer BVH Timeline Integration Example
 * 
 * This example demonstrates how to integrate FaceFormer neural network output
 * with the BVH Timeline compositor system for real-time facial animation.
 */

class FaceFormerTimelineIntegration {
    constructor(timeline, options = {}) {
        this.timeline = timeline;
        this.converter = new FaceFormerBVHConverter(options.converter || {});
        
        // Integration options
        this.realtime = options.realtime !== false;
        this.trackName = options.trackName || 'faceformer_facial';
        this.priority = options.priority || 100; // High priority for facial
        this.channels = options.channels || ['face', 'head']; // Target face and head bones
        
        // FaceFormer model integration
        this.faceFormerModel = null;
        this.isModelLoaded = false;
        this.processingQueue = [];
        this.isProcessing = false;
        
        // Performance monitoring
        this.stats = {
            framesProcessed: 0,
            averageLatency: 0,
            modelInferenceTime: 0,
            conversionTime: 0,
            totalTime: 0
        };
        
        console.log('[FaceFormer Timeline Integration] Initialized');
    }
    
    /**
     * Initialize and load FaceFormer model
     */
    async initializeFaceFormer(modelPath, options = {}) {
        try {
            console.log('[FaceFormer Timeline] Loading FaceFormer model...');
            
            // Check if FaceFormer is available in global scope
            if (typeof window !== 'undefined' && window.FaceFormer) {
                this.faceFormerModel = new window.FaceFormer(modelPath, options);
            } else if (typeof global !== 'undefined' && global.FaceFormer) {
                this.faceFormerModel = new global.FaceFormer(modelPath, options);
            } else {
                // Try to load via dynamic import or create mock
                try {
                    const FaceFormerModule = await import(modelPath);
                    this.faceFormerModel = new FaceFormerModule.default(options);
                } catch (importError) {
                    console.warn('[FaceFormer Timeline] FaceFormer not found, creating mock model');
                    this.faceFormerModel = this.createMockFaceFormer();
                }
            }
            
            // Initialize the model
            if (this.faceFormerModel.initialize) {
                await this.faceFormerModel.initialize();
            }
            
            this.isModelLoaded = true;
            console.log('[FaceFormer Timeline] FaceFormer model loaded successfully');
            
            return true;
            
        } catch (error) {
            console.error('[FaceFormer Timeline] Failed to load FaceFormer model:', error);
            
            // Create mock model for development/testing
            this.faceFormerModel = this.createMockFaceFormer();
            this.isModelLoaded = true;
            
            return false;
        }
    }
    
    /**
     * Create a mock FaceFormer model for testing
     */
    createMockFaceFormer() {
        return {
            predict: async (audioInput, options = {}) => {
                // Simulate model inference time
                await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 20));
                
                // Generate mock facial animation data
                const frameCount = Math.floor(audioInput.length / 1000); // Approximate frames from audio
                const frames = [];
                
                for (let i = 0; i < frameCount; i++) {
                    const t = i / frameCount;
                    
                    // Generate realistic-looking blend shape values
                    frames.push({
                        blendshapes: {
                            // Eye movements
                            eyeBlinkLeft: Math.max(0, Math.sin(t * 12) * 0.3 + 0.1),
                            eyeBlinkRight: Math.max(0, Math.sin(t * 12 + 0.1) * 0.3 + 0.1),
                            eyeLookUpLeft: Math.sin(t * 8) * 0.2,
                            eyeLookUpRight: Math.sin(t * 8 + 0.2) * 0.2,
                            
                            // Jaw movement (correlated with audio)
                            jawOpen: Math.max(0, Math.sin(t * 20) * 0.4 + Math.random() * 0.2),
                            jawLeft: Math.sin(t * 15) * 0.1,
                            jawRight: Math.cos(t * 15) * 0.1,
                            
                            // Mouth movements
                            mouthSmileLeft: Math.max(0, Math.sin(t * 6) * 0.3),
                            mouthSmileRight: Math.max(0, Math.sin(t * 6 + 0.1) * 0.3),
                            mouthFrownLeft: Math.max(0, -Math.sin(t * 6) * 0.2),
                            mouthFrownRight: Math.max(0, -Math.sin(t * 6 + 0.1) * 0.2),
                            
                            // Eyebrow movements
                            browInnerUp: Math.sin(t * 4) * 0.2,
                            browOuterUpLeft: Math.sin(t * 5) * 0.15,
                            browOuterUpRight: Math.sin(t * 5 + 0.3) * 0.15,
                            
                            // Cheek movements
                            cheekPuffLeft: Math.max(0, Math.sin(t * 10) * 0.1),
                            cheekPuffRight: Math.max(0, Math.sin(t * 10 + 0.2) * 0.1)
                        },
                        confidence: 0.8 + Math.random() * 0.2,
                        timestamp: i * (1000 / 30) // 30 FPS
                    });
                }
                
                return {
                    frames: frames,
                    metadata: {
                        model: 'MockFaceFormer',
                        version: '1.0.0',
                        audioLength: audioInput.length,
                        frameCount: frameCount,
                        frameRate: 30
                    }
                };
            },
            
            isMock: true
        };
    }
    
    /**
     * Process audio input and generate facial animation
     */
    async processAudio(audioInput, options = {}) {
        if (!this.isModelLoaded) {
            throw new Error('FaceFormer model not loaded');
        }
        
        const startTime = performance.now();
        
        try {
            // Run FaceFormer inference
            const inferenceStart = performance.now();
            const faceFormerOutput = await this.faceFormerModel.predict(audioInput, options);
            const inferenceTime = performance.now() - inferenceStart;
            
            // Convert to BVH timeline clips
            const conversionStart = performance.now();
            const timelineClips = await this.convertToTimelineClips(faceFormerOutput, options);
            const conversionTime = performance.now() - conversionStart;
            
            // Add clips to timeline
            for (const clip of timelineClips) {
                await this.addClipToTimeline(clip, options);
            }
            
            // Update statistics
            const totalTime = performance.now() - startTime;
            this.updateStats(inferenceTime, conversionTime, totalTime);
            
            console.log('[FaceFormer Timeline] Audio processed successfully:', {
                audioLength: audioInput.length,
                clipCount: timelineClips.length,
                totalTime: `${totalTime.toFixed(2)}ms`,
                inferenceTime: `${inferenceTime.toFixed(2)}ms`,
                conversionTime: `${conversionTime.toFixed(2)}ms`
            });
            
            return {
                success: true,
                clips: timelineClips,
                timing: {
                    inference: inferenceTime,
                    conversion: conversionTime,
                    total: totalTime
                }
            };
            
        } catch (error) {
            console.error('[FaceFormer Timeline] Audio processing failed:', error);
            throw error;
        }
    }
    
    /**
     * Convert FaceFormer output to timeline clips
     */
    async convertToTimelineClips(faceFormerOutput, options = {}) {
        const clips = [];
        const frames = faceFormerOutput.frames || [faceFormerOutput];
        const startTime = options.startTime || 0;
        
        // Create main facial animation clip
        const facialClip = this.converter.createTimelineClip(frames, startTime);
        facialClip.trackName = this.trackName;
        facialClip.priority = this.priority;
        facialClip.channels = this.channels;
        facialClip.blending = 'override'; // Facial animation typically overrides
        
        clips.push(facialClip);
        
        // If the output contains separate eye tracking, create additional clips
        if (options.separateEyeTracking && frames.some(f => f.eyeTracking)) {
            const eyeClip = this.createEyeTrackingClip(frames, startTime);
            clips.push(eyeClip);
        }
        
        // If the output contains separate mouth shapes, create additional clips
        if (options.separateMouthShapes && frames.some(f => f.mouthShapes)) {
            const mouthClip = this.createMouthShapeClip(frames, startTime);
            clips.push(mouthClip);
        }
        
        return clips;
    }
    
    /**
     * Add clip to timeline with proper configuration
     */
    async addClipToTimeline(clip, options = {}) {
        // Ensure the track exists
        if (!this.timeline.hasTrack(clip.trackName)) {
            this.timeline.addTrack(clip.trackName, {
                type: 'faceformer',
                priority: clip.priority,
                channels: clip.channels,
                blending: clip.blending || 'additive'
            });
        }
        
        // Add the clip to the track
        this.timeline.addClip(clip.trackName, clip, {
            startTime: clip.startTime,
            duration: clip.duration,
            loop: options.loop || false,
            fadeIn: options.fadeIn || 0.1,
            fadeOut: options.fadeOut || 0.1
        });
        
        console.log('[FaceFormer Timeline] Added clip to track:', {
            trackName: clip.trackName,
            startTime: clip.startTime,
            duration: clip.duration,
            frameCount: clip.frames.length
        });
    }
    
    /**
     * Real-time processing for live audio input
     */
    async processLiveAudio(audioChunk, timestamp) {
        if (this.isProcessing && this.realtime) {
            // Queue for later processing if we're in realtime mode
            this.processingQueue.push({ audioChunk, timestamp });
            return;
        }
        
        this.isProcessing = true;
        
        try {
            const result = await this.processAudio(audioChunk, {
                startTime: timestamp,
                realtime: true
            });
            
            // Process any queued audio chunks
            if (this.processingQueue.length > 0) {
                const next = this.processingQueue.shift();
                setTimeout(() => this.processLiveAudio(next.audioChunk, next.timestamp), 0);
            }
            
            return result;
            
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Create specialized clips for different facial features
     */
    createEyeTrackingClip(frames, startTime) {
        const eyeFrames = frames.map(frame => ({
            blendshapes: {
                eyeLookUpLeft: frame.eyeTracking?.leftEye?.up || 0,
                eyeLookDownLeft: frame.eyeTracking?.leftEye?.down || 0,
                eyeLookInLeft: frame.eyeTracking?.leftEye?.in || 0,
                eyeLookOutLeft: frame.eyeTracking?.leftEye?.out || 0,
                eyeLookUpRight: frame.eyeTracking?.rightEye?.up || 0,
                eyeLookDownRight: frame.eyeTracking?.rightEye?.down || 0,
                eyeLookInRight: frame.eyeTracking?.rightEye?.in || 0,
                eyeLookOutRight: frame.eyeTracking?.rightEye?.out || 0
            },
            confidence: frame.confidence || 1.0,
            timestamp: frame.timestamp
        }));
        
        const clip = this.converter.createTimelineClip(eyeFrames, startTime);
        clip.trackName = 'faceformer_eyes';
        clip.priority = this.priority + 10; // Higher priority for eyes
        clip.channels = ['eyes'];
        clip.blending = 'override';
        
        return clip;
    }
    
    createMouthShapeClip(frames, startTime) {
        const mouthFrames = frames.map(frame => ({
            blendshapes: frame.mouthShapes || {},
            confidence: frame.confidence || 1.0,
            timestamp: frame.timestamp
        }));
        
        const clip = this.converter.createTimelineClip(mouthFrames, startTime);
        clip.trackName = 'faceformer_mouth';
        clip.priority = this.priority + 5; // High priority for mouth
        clip.channels = ['mouth', 'jaw'];
        clip.blending = 'override';
        
        return clip;
    }
    
    /**
     * Batch processing for pre-recorded audio
     */
    async processBatchAudio(audioSegments, options = {}) {
        const results = [];
        const batchStartTime = options.startTime || 0;
        let currentTime = batchStartTime;
        
        for (let i = 0; i < audioSegments.length; i++) {
            const segment = audioSegments[i];
            const segmentDuration = segment.duration || (segment.audio.length / segment.sampleRate);
            
            try {
                const result = await this.processAudio(segment.audio, {
                    startTime: currentTime,
                    ...options
                });
                
                results.push({
                    index: i,
                    startTime: currentTime,
                    duration: segmentDuration,
                    result: result
                });
                
                currentTime += segmentDuration;
                
            } catch (error) {
                console.error(`[FaceFormer Timeline] Failed to process segment ${i}:`, error);
                results.push({
                    index: i,
                    startTime: currentTime,
                    duration: segmentDuration,
                    error: error.message
                });
            }
        }
        
        return {
            totalSegments: audioSegments.length,
            successfulSegments: results.filter(r => !r.error).length,
            totalDuration: currentTime - batchStartTime,
            results: results
        };
    }
    
    /**
     * Timeline synchronization utilities
     */
    synchronizeWithTimeline(options = {}) {
        // Set up timeline callbacks for FaceFormer integration
        const originalOnFrameUpdate = this.timeline.onFrameUpdate;
        
        this.timeline.onFrameUpdate = (frame, time) => {
            // Process FaceFormer frames
            if (frame.metadata?.source === 'faceformer') {
                this.handleFaceFormerFrame(frame, time);
            }
            
            // Call original callback
            if (originalOnFrameUpdate) {
                originalOnFrameUpdate(frame, time);
            }
        };
        
        console.log('[FaceFormer Timeline] Synchronized with timeline');
    }
    
    handleFaceFormerFrame(frame, time) {
        // Custom processing for FaceFormer frames
        if (this.onFaceFormerFrame) {
            this.onFaceFormerFrame(frame, time);
        }
    }
    
    /**
     * Statistics and monitoring
     */
    updateStats(inferenceTime, conversionTime, totalTime) {
        this.stats.framesProcessed++;
        
        const count = this.stats.framesProcessed;
        this.stats.averageLatency = 
            (this.stats.averageLatency * (count - 1) + totalTime) / count;
        this.stats.modelInferenceTime = 
            (this.stats.modelInferenceTime * (count - 1) + inferenceTime) / count;
        this.stats.conversionTime = 
            (this.stats.conversionTime * (count - 1) + conversionTime) / count;
        this.stats.totalTime = 
            (this.stats.totalTime * (count - 1) + totalTime) / count;
    }
    
    getStats() {
        return {
            ...this.stats,
            isModelLoaded: this.isModelLoaded,
            isMockModel: this.faceFormerModel?.isMock || false,
            queueSize: this.processingQueue.length,
            isProcessing: this.isProcessing,
            converterStats: this.converter.getStats()
        };
    }
    
    /**
     * Configuration methods
     */
    setRealtime(enabled) {
        this.realtime = enabled;
        console.log('[FaceFormer Timeline] Realtime mode:', enabled);
    }
    
    setTrackPriority(priority) {
        this.priority = priority;
        
        // Update existing tracks
        if (this.timeline.hasTrack(this.trackName)) {
            this.timeline.updateTrack(this.trackName, { priority });
        }
        
        console.log('[FaceFormer Timeline] Track priority updated:', priority);
    }
    
    setChannels(channels) {
        this.channels = channels;
        
        // Update existing tracks
        if (this.timeline.hasTrack(this.trackName)) {
            this.timeline.updateTrack(this.trackName, { channels });
        }
        
        console.log('[FaceFormer Timeline] Channels updated:', channels);
    }
    
    /**
     * Cleanup and disposal
     */
    dispose() {
        // Clean up FaceFormer model
        if (this.faceFormerModel && this.faceFormerModel.dispose) {
            this.faceFormerModel.dispose();
        }
        
        // Clean up converter
        this.converter.dispose();
        
        // Clear processing queue
        this.processingQueue = [];
        
        // Reset stats
        this.stats = {
            framesProcessed: 0,
            averageLatency: 0,
            modelInferenceTime: 0,
            conversionTime: 0,
            totalTime: 0
        };
        
        console.log('[FaceFormer Timeline Integration] Disposed');
    }
}

// Usage example
async function createFaceFormerTimelineExample() {
    // Create timeline instance
    const timeline = new BVHTimelineCompositor({
        frameRate: 30,
        maxTracks: 10
    });
    
    // Create FaceFormer integration
    const faceFormerIntegration = new FaceFormerTimelineIntegration(timeline, {
        realtime: true,
        trackName: 'facial_animation',
        priority: 100,
        channels: ['face', 'head'],
        converter: {
            scaleFactor: 1.0,
            smoothing: true,
            smoothingFactor: 0.8
        }
    });
    
    // Initialize FaceFormer model
    await faceFormerIntegration.initializeFaceFormer('/path/to/faceformer/model');
    
    // Set up timeline synchronization
    faceFormerIntegration.synchronizeWithTimeline();
    
    // Example: Process audio for facial animation
    const audioData = new Float32Array(16000); // 1 second of audio at 16kHz
    const result = await faceFormerIntegration.processAudio(audioData, {
        startTime: 0,
        realtime: false
    });
    
    console.log('FaceFormer processing result:', result);
    
    // Start timeline playback
    timeline.play();
    
    return { timeline, faceFormerIntegration };
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceFormerTimelineIntegration;
} else {
    window.FaceFormerTimelineIntegration = FaceFormerTimelineIntegration;
}
