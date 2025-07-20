/**
 * Audio2Gesture BVH Timeline Integration
 * 
 * This module integrates Audio2Gesture neural network body gesture generation
 * with the BVH Timeline compositor system for full-body animation from audio.
 */

class Audio2GestureTimelineIntegration {
    constructor(timeline, options = {}) {
        this.timeline = timeline;
        this.converter = new Audio2GestureBVHConverter(options.converter || {});
        
        // Integration options
        this.realtime = options.realtime !== false;
        this.trackName = options.trackName || 'audio2gesture_body';
        this.priority = options.priority || 80; // Lower priority than facial
        this.channels = options.channels || ['body', 'arms', 'hands']; // Target body channels
        
        // Audio2Gesture model integration
        this.audio2GestureModel = null;
        this.isModelLoaded = false;
        this.processingQueue = [];
        this.isProcessing = false;
        
        // Audio processing
        this.audioProcessor = new Audio2GestureAudioProcessor({
            sampleRate: options.sampleRate || 16000,
            frameSize: options.frameSize || 1024,
            hopLength: options.hopLength || 512
        });
        
        // Performance monitoring
        this.stats = {
            framesProcessed: 0,
            averageLatency: 0,
            modelInferenceTime: 0,
            conversionTime: 0,
            audioProcessingTime: 0,
            totalTime: 0
        };
        
        console.log('[Audio2Gesture Timeline Integration] Initialized');
    }
    
    /**
     * Initialize and load Audio2Gesture model
     */
    async initializeAudio2Gesture(modelPath, options = {}) {
        try {
            console.log('[Audio2Gesture Timeline] Loading Audio2Gesture model...');
            
            // Check if Audio2Gesture is available in global scope
            if (typeof window !== 'undefined' && window.Audio2Gesture) {
                this.audio2GestureModel = new window.Audio2Gesture(modelPath, options);
            } else if (typeof global !== 'undefined' && global.Audio2Gesture) {
                this.audio2GestureModel = new global.Audio2Gesture(modelPath, options);
            } else {
                // Try to load via dynamic import or create mock
                try {
                    const Audio2GestureModule = await import(modelPath);
                    this.audio2GestureModel = new Audio2GestureModule.default(options);
                } catch (importError) {
                    console.warn('[Audio2Gesture Timeline] Audio2Gesture not found, creating mock model');
                    this.audio2GestureModel = this.createMockAudio2Gesture();
                }
            }
            
            // Initialize the model
            if (this.audio2GestureModel.initialize) {
                await this.audio2GestureModel.initialize();
            }
            
            this.isModelLoaded = true;
            console.log('[Audio2Gesture Timeline] Audio2Gesture model loaded successfully');
            
            return true;
            
        } catch (error) {
            console.error('[Audio2Gesture Timeline] Failed to load Audio2Gesture model:', error);
            
            // Create mock model for development/testing
            this.audio2GestureModel = this.createMockAudio2Gesture();
            this.isModelLoaded = true;
            
            return false;
        }
    }
    
    /**
     * Create a mock Audio2Gesture model for testing
     */
    createMockAudio2Gesture() {
        return {
            predict: async (audioFeatures, options = {}) => {
                // Simulate model inference time
                await new Promise(resolve => setTimeout(resolve, 20 + Math.random() * 40));
                
                const frameCount = audioFeatures.length || Math.floor(audioFeatures.duration * 30);
                const poses = [];
                
                for (let i = 0; i < frameCount; i++) {
                    const t = i / frameCount;
                    const audioIntensity = audioFeatures.amplitude || Math.random();
                    
                    // Generate realistic pose parameters (72 dimensions for SMPL)
                    const pose = new Array(72).fill(0);
                    
                    // Root rotation (global orientation)
                    pose[0] = Math.sin(t * 4) * 0.1 * audioIntensity;  // Slight sway
                    pose[1] = 0;
                    pose[2] = Math.cos(t * 3) * 0.05 * audioIntensity;
                    
                    // Spine rotations (indices 3-11)
                    pose[3] = Math.sin(t * 2) * 0.15 * audioIntensity;   // Spine X
                    pose[4] = Math.cos(t * 1.5) * 0.1 * audioIntensity; // Spine Y
                    pose[6] = Math.sin(t * 1.8) * 0.1 * audioIntensity;  // Spine1 X
                    pose[9] = Math.sin(t * 2.2) * 0.08 * audioIntensity; // Spine2 X
                    
                    // Neck and head (indices 12-17)
                    pose[12] = Math.sin(t * 5) * 0.2 * audioIntensity;   // Neck X
                    pose[13] = Math.cos(t * 4) * 0.15 * audioIntensity;  // Neck Y
                    pose[15] = Math.sin(t * 6) * 0.1 * audioIntensity;   // Head X
                    pose[16] = Math.cos(t * 5.5) * 0.08 * audioIntensity; // Head Y
                    
                    // Left arm (indices 18-26)
                    pose[18] = Math.sin(t * 3 + 0.5) * 0.6 * audioIntensity; // Left shoulder X
                    pose[19] = Math.cos(t * 2.5) * 0.3 * audioIntensity;     // Left shoulder Y
                    pose[20] = Math.sin(t * 2.8) * 0.2 * audioIntensity;     // Left shoulder Z
                    pose[21] = Math.sin(t * 4 + 1) * 0.8 * audioIntensity;   // Left arm X
                    pose[22] = Math.cos(t * 3.2) * 0.4 * audioIntensity;     // Left arm Y
                    pose[24] = Math.max(0, Math.sin(t * 5) * 0.9 * audioIntensity); // Left forearm (elbow bend)
                    
                    // Right arm (indices 27-35)
                    pose[27] = Math.sin(t * 3 + 2) * 0.6 * audioIntensity;   // Right shoulder X
                    pose[28] = Math.cos(t * 2.5 + 1) * 0.3 * audioIntensity; // Right shoulder Y
                    pose[29] = Math.sin(t * 2.8 + 1.5) * 0.2 * audioIntensity; // Right shoulder Z
                    pose[30] = Math.sin(t * 4 + 3) * 0.8 * audioIntensity;   // Right arm X
                    pose[31] = Math.cos(t * 3.2 + 2) * 0.4 * audioIntensity; // Right arm Y
                    pose[33] = Math.max(0, Math.sin(t * 5 + 1) * 0.9 * audioIntensity); // Right forearm
                    
                    // Hands (basic finger curling)
                    for (let j = 36; j < 48; j++) { // Left hand
                        pose[j] = Math.sin(t * 8 + j) * 0.3 * audioIntensity;
                    }
                    for (let j = 48; j < 60; j++) { // Right hand
                        pose[j] = Math.sin(t * 8 + j + 3) * 0.3 * audioIntensity;
                    }
                    
                    // Legs (subtle movement)
                    if (options.includeLowerBody) {
                        pose[60] = Math.sin(t * 1.5) * 0.1 * audioIntensity;     // Left hip
                        pose[63] = Math.sin(t * 1.5 + Math.PI) * 0.1 * audioIntensity; // Right hip
                        pose[66] = Math.max(0, Math.sin(t * 2) * 0.2 * audioIntensity); // Left knee
                        pose[69] = Math.max(0, Math.sin(t * 2 + 0.5) * 0.2 * audioIntensity); // Right knee
                    }
                    
                    poses.push({
                        body_pose: pose,
                        left_hand: new Array(15).fill(0).map(() => Math.random() * 0.2 - 0.1),
                        right_hand: new Array(15).fill(0).map(() => Math.random() * 0.2 - 0.1),
                        confidence: 0.8 + Math.random() * 0.2,
                        timestamp: i * (1000 / 30) // 30 FPS
                    });
                }
                
                return {
                    poses: poses,
                    metadata: {
                        model: 'MockAudio2Gesture',
                        version: '1.0.0',
                        inputFeatures: audioFeatures.length || 'unknown',
                        outputFrames: poses.length,
                        frameRate: 30
                    }
                };
            },
            
            isMock: true
        };
    }
    
    /**
     * Process audio input and generate body gestures
     */
    async processAudio(audioInput, options = {}) {
        if (!this.isModelLoaded) {
            throw new Error('Audio2Gesture model not loaded');
        }
        
        const startTime = performance.now();
        
        try {
            // Extract audio features
            const audioProcessingStart = performance.now();
            const audioFeatures = await this.audioProcessor.extractFeatures(audioInput, options);
            const audioProcessingTime = performance.now() - audioProcessingStart;
            
            // Run Audio2Gesture inference
            const inferenceStart = performance.now();
            const audio2GestureOutput = await this.audio2GestureModel.predict(audioFeatures, {
                includeLowerBody: this.converter.options.enableLowerBody,
                includeFingers: this.converter.options.enableFingers,
                emotionalContext: options.emotionalContext || 'neutral',
                ...options
            });
            const inferenceTime = performance.now() - inferenceStart;
            
            // Convert to BVH timeline clips
            const conversionStart = performance.now();
            const timelineClips = await this.convertToTimelineClips(audio2GestureOutput, audioFeatures, options);
            const conversionTime = performance.now() - conversionStart;
            
            // Add clips to timeline
            for (const clip of timelineClips) {
                await this.addClipToTimeline(clip, options);
            }
            
            // Update statistics
            const totalTime = performance.now() - startTime;
            this.updateStats(inferenceTime, conversionTime, totalTime, audioProcessingTime);
            
            console.log('[Audio2Gesture Timeline] Audio processed successfully:', {
                audioLength: audioInput.length,
                clipCount: timelineClips.length,
                totalTime: `${totalTime.toFixed(2)}ms`,
                inferenceTime: `${inferenceTime.toFixed(2)}ms`,
                conversionTime: `${conversionTime.toFixed(2)}ms`,
                audioProcessingTime: `${audioProcessingTime.toFixed(2)}ms`
            });
            
            return {
                success: true,
                clips: timelineClips,
                timing: {
                    audioProcessing: audioProcessingTime,
                    inference: inferenceTime,
                    conversion: conversionTime,
                    total: totalTime
                },
                features: audioFeatures
            };
            
        } catch (error) {
            console.error('[Audio2Gesture Timeline] Audio processing failed:', error);
            throw error;
        }
    }
    
    /**
     * Convert Audio2Gesture output to timeline clips
     */
    async convertToTimelineClips(audio2GestureOutput, audioFeatures, options = {}) {
        const clips = [];
        const poses = audio2GestureOutput.poses || [audio2GestureOutput];
        const startTime = options.startTime || 0;
        
        // Create main body animation clip
        const bodyClip = this.converter.createTimelineClip(poses, audioFeatures, startTime);
        bodyClip.trackName = this.trackName;
        bodyClip.priority = this.priority;
        bodyClip.channels = this.channels;
        bodyClip.blending = 'additive'; // Body gestures typically blend with other animations
        
        clips.push(bodyClip);
        
        // If the output contains separate upper/lower body, create additional clips
        if (options.separateUpperLower) {
            const upperBodyClip = this.createUpperBodyClip(poses, audioFeatures, startTime);
            const lowerBodyClip = this.createLowerBodyClip(poses, audioFeatures, startTime);
            clips.push(upperBodyClip, lowerBodyClip);
        }
        
        // If the output contains detailed hand gestures, create hand clips
        if (options.separateHands && poses.some(p => p.left_hand || p.right_hand)) {
            const handClips = this.createHandGestureClips(poses, audioFeatures, startTime);
            clips.push(...handClips);
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
                type: 'audio2gesture',
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
            fadeIn: options.fadeIn || 0.2,
            fadeOut: options.fadeOut || 0.2
        });
        
        console.log('[Audio2Gesture Timeline] Added clip to track:', {
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
                realtime: true,
                emotionalContext: this.detectEmotionalContext(audioChunk)
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
     * Detect emotional context from audio for gesture style adaptation
     */
    detectEmotionalContext(audioChunk) {
        // Simple emotion detection based on audio characteristics
        const amplitude = this.audioProcessor.calculateAmplitude(audioChunk);
        const energy = this.audioProcessor.calculateRMS(audioChunk);
        
        if (amplitude > 0.7 && energy > 0.6) {
            return 'excited';
        } else if (amplitude < 0.2 && energy < 0.3) {
            return 'calm';
        } else if (energy > 0.8) {
            return 'emphatic';
        } else {
            return 'neutral';
        }
    }
    
    /**
     * Create specialized clips for different body parts
     */
    createUpperBodyClip(poses, audioFeatures, startTime) {
        // Filter poses to only include upper body
        const upperBodyPoses = poses.map(pose => ({
            body_pose: pose.body_pose ? pose.body_pose.slice(0, 60) : [], // Upper body only
            left_hand: pose.left_hand,
            right_hand: pose.right_hand,
            confidence: pose.confidence,
            timestamp: pose.timestamp
        }));
        
        const clip = this.converter.createTimelineClip(upperBodyPoses, audioFeatures, startTime);
        clip.trackName = 'audio2gesture_upper';
        clip.priority = this.priority + 5; // Higher priority for upper body
        clip.channels = ['arms', 'hands', 'spine', 'head'];
        clip.blending = 'override';
        
        return clip;
    }
    
    createLowerBodyClip(poses, audioFeatures, startTime) {
        // Filter poses to only include lower body
        const lowerBodyPoses = poses.map(pose => ({
            body_pose: pose.body_pose ? pose.body_pose.slice(60) : [], // Lower body only
            confidence: pose.confidence,
            timestamp: pose.timestamp
        }));
        
        const clip = this.converter.createTimelineClip(lowerBodyPoses, audioFeatures, startTime);
        clip.trackName = 'audio2gesture_lower';
        clip.priority = this.priority - 5; // Lower priority for lower body
        clip.channels = ['legs', 'feet'];
        clip.blending = 'additive';
        
        return clip;
    }
    
    createHandGestureClips(poses, audioFeatures, startTime) {
        const clips = [];
        
        // Left hand clip
        const leftHandPoses = poses.map(pose => ({
            hand_pose: pose.left_hand,
            confidence: pose.confidence,
            timestamp: pose.timestamp
        })).filter(pose => pose.hand_pose);
        
        if (leftHandPoses.length > 0) {
            const leftClip = this.converter.createTimelineClip(leftHandPoses, audioFeatures, startTime);
            leftClip.trackName = 'audio2gesture_left_hand';
            leftClip.priority = this.priority + 10;
            leftClip.channels = ['left_hand', 'left_fingers'];
            leftClip.blending = 'override';
            clips.push(leftClip);
        }
        
        // Right hand clip
        const rightHandPoses = poses.map(pose => ({
            hand_pose: pose.right_hand,
            confidence: pose.confidence,
            timestamp: pose.timestamp
        })).filter(pose => pose.hand_pose);
        
        if (rightHandPoses.length > 0) {
            const rightClip = this.converter.createTimelineClip(rightHandPoses, audioFeatures, startTime);
            rightClip.trackName = 'audio2gesture_right_hand';
            rightClip.priority = this.priority + 10;
            rightClip.channels = ['right_hand', 'right_fingers'];
            rightClip.blending = 'override';
            clips.push(rightClip);
        }
        
        return clips;
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
                    emotionalContext: segment.emotionalContext || 'neutral',
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
                console.error(`[Audio2Gesture Timeline] Failed to process segment ${i}:`, error);
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
        // Set up timeline callbacks for Audio2Gesture integration
        const originalOnFrameUpdate = this.timeline.onFrameUpdate;
        
        this.timeline.onFrameUpdate = (frame, time) => {
            // Process Audio2Gesture frames
            if (frame.metadata?.source === 'audio2gesture') {
                this.handleAudio2GestureFrame(frame, time);
            }
            
            // Call original callback
            if (originalOnFrameUpdate) {
                originalOnFrameUpdate(frame, time);
            }
        };
        
        console.log('[Audio2Gesture Timeline] Synchronized with timeline');
    }
    
    handleAudio2GestureFrame(frame, time) {
        // Custom processing for Audio2Gesture frames
        if (this.onAudio2GestureFrame) {
            this.onAudio2GestureFrame(frame, time);
        }
    }
    
    /**
     * Statistics and monitoring
     */
    updateStats(inferenceTime, conversionTime, totalTime, audioProcessingTime) {
        this.stats.framesProcessed++;
        
        const count = this.stats.framesProcessed;
        this.stats.averageLatency = 
            (this.stats.averageLatency * (count - 1) + totalTime) / count;
        this.stats.modelInferenceTime = 
            (this.stats.modelInferenceTime * (count - 1) + inferenceTime) / count;
        this.stats.conversionTime = 
            (this.stats.conversionTime * (count - 1) + conversionTime) / count;
        this.stats.audioProcessingTime = 
            (this.stats.audioProcessingTime * (count - 1) + audioProcessingTime) / count;
        this.stats.totalTime = 
            (this.stats.totalTime * (count - 1) + totalTime) / count;
    }
    
    getStats() {
        return {
            ...this.stats,
            isModelLoaded: this.isModelLoaded,
            isMockModel: this.audio2GestureModel?.isMock || false,
            queueSize: this.processingQueue.length,
            isProcessing: this.isProcessing,
            converterStats: this.converter.getStats(),
            audioProcessorStats: this.audioProcessor.getStats()
        };
    }
    
    /**
     * Configuration methods
     */
    setRealtime(enabled) {
        this.realtime = enabled;
        console.log('[Audio2Gesture Timeline] Realtime mode:', enabled);
    }
    
    setTrackPriority(priority) {
        this.priority = priority;
        
        // Update existing tracks
        if (this.timeline.hasTrack(this.trackName)) {
            this.timeline.updateTrack(this.trackName, { priority });
        }
        
        console.log('[Audio2Gesture Timeline] Track priority updated:', priority);
    }
    
    setChannels(channels) {
        this.channels = channels;
        
        // Update existing tracks
        if (this.timeline.hasTrack(this.trackName)) {
            this.timeline.updateTrack(this.trackName, { channels });
        }
        
        console.log('[Audio2Gesture Timeline] Channels updated:', channels);
    }
    
    setGestureIntensity(intensity) {
        this.converter.options.gestureIntensity = intensity;
        console.log('[Audio2Gesture Timeline] Gesture intensity updated:', intensity);
    }
    
    setEmotionalModulation(enabled) {
        this.converter.options.emotionalModulation = enabled;
        console.log('[Audio2Gesture Timeline] Emotional modulation:', enabled);
    }
    
    /**
     * Cleanup and disposal
     */
    dispose() {
        // Clean up Audio2Gesture model
        if (this.audio2GestureModel && this.audio2GestureModel.dispose) {
            this.audio2GestureModel.dispose();
        }
        
        // Clean up converter
        this.converter.dispose();
        
        // Clean up audio processor
        this.audioProcessor.dispose();
        
        // Clear processing queue
        this.processingQueue = [];
        
        // Reset stats
        this.stats = {
            framesProcessed: 0,
            averageLatency: 0,
            modelInferenceTime: 0,
            conversionTime: 0,
            audioProcessingTime: 0,
            totalTime: 0
        };
        
        console.log('[Audio2Gesture Timeline Integration] Disposed');
    }
}

/**
 * Audio2Gesture Audio Processor
 * 
 * Handles audio feature extraction for Audio2Gesture models
 */
class Audio2GestureAudioProcessor {
    constructor(options = {}) {
        this.sampleRate = options.sampleRate || 16000;
        this.frameSize = options.frameSize || 1024;
        this.hopLength = options.hopLength || 512;
        this.melBins = options.melBins || 80;
        
        this.stats = {
            featuresExtracted: 0,
            totalProcessingTime: 0
        };
    }
    
    /**
     * Extract comprehensive audio features for gesture generation
     */
    async extractFeatures(audioData, options = {}) {
        const startTime = performance.now();
        
        try {
            const features = {
                // Basic properties
                length: audioData.length,
                duration: audioData.length / this.sampleRate,
                sampleRate: this.sampleRate,
                
                // Time-domain features
                amplitude: this.calculateAmplitude(audioData),
                rms: this.calculateRMS(audioData),
                zeroCrossingRate: this.calculateZeroCrossingRate(audioData),
                
                // Spectral features (simplified)
                spectralCentroid: this.calculateSpectralCentroid(audioData),
                spectralRolloff: this.calculateSpectralRolloff(audioData),
                spectralFlux: this.calculateSpectralFlux(audioData),
                
                // Rhythm features
                tempo: this.estimateTempo(audioData),
                beatStrength: this.calculateBeatStrength(audioData),
                rhythmRegularity: this.calculateRhythmRegularity(audioData),
                
                // Pitch features
                fundamentalFreq: this.estimateFundamentalFrequency(audioData),
                pitch: this.estimatePitch(audioData),
                pitchVariability: this.calculatePitchVariability(audioData),
                
                // Energy features
                shortTimeEnergy: this.calculateShortTimeEnergy(audioData),
                energyVariability: this.calculateEnergyVariability(audioData),
                
                // Mel-scale features (simplified)
                melSpectrogram: this.calculateMelSpectrogram(audioData),
                
                // Prosodic features
                loudness: this.calculateLoudness(audioData),
                prosody: this.extractProsody(audioData),
                
                timestamp: Date.now()
            };
            
            this.stats.featuresExtracted++;
            this.stats.totalProcessingTime += performance.now() - startTime;
            
            return features;
            
        } catch (error) {
            console.error('[Audio2Gesture Audio Processor] Feature extraction failed:', error);
            return this.getDefaultFeatures(audioData);
        }
    }
    
    // Audio analysis methods (simplified implementations)
    calculateAmplitude(audioData) {
        return audioData.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0);
    }
    
    calculateRMS(audioData) {
        const sum = audioData.reduce((acc, sample) => acc + sample * sample, 0);
        return Math.sqrt(sum / audioData.length);
    }
    
    calculateZeroCrossingRate(audioData) {
        let crossings = 0;
        for (let i = 1; i < audioData.length; i++) {
            if ((audioData[i] >= 0) !== (audioData[i - 1] >= 0)) {
                crossings++;
            }
        }
        return crossings / audioData.length;
    }
    
    calculateSpectralCentroid(audioData) {
        // Simplified spectral centroid
        let weightedSum = 0;
        let magnitudeSum = 0;
        
        for (let i = 0; i < audioData.length; i++) {
            const magnitude = Math.abs(audioData[i]);
            weightedSum += i * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    }
    
    calculateSpectralRolloff(audioData) {
        // Simplified spectral rolloff
        let totalEnergy = 0;
        const energies = audioData.map(sample => sample * sample);
        totalEnergy = energies.reduce((sum, energy) => sum + energy, 0);
        
        const threshold = totalEnergy * 0.85;
        let cumulativeEnergy = 0;
        
        for (let i = 0; i < energies.length; i++) {
            cumulativeEnergy += energies[i];
            if (cumulativeEnergy >= threshold) {
                return i / energies.length;
            }
        }
        
        return 1.0;
    }
    
    calculateSpectralFlux(audioData) {
        // Simplified spectral flux (change in spectrum)
        const windowSize = Math.floor(audioData.length / 10);
        let totalFlux = 0;
        
        for (let i = windowSize; i < audioData.length - windowSize; i += windowSize) {
            const window1 = audioData.slice(i - windowSize, i);
            const window2 = audioData.slice(i, i + windowSize);
            
            const energy1 = window1.reduce((sum, sample) => sum + sample * sample, 0);
            const energy2 = window2.reduce((sum, sample) => sum + sample * sample, 0);
            
            totalFlux += Math.abs(energy2 - energy1);
        }
        
        return totalFlux;
    }
    
    estimateTempo(audioData) {
        // Simplified tempo estimation using energy variations
        const beatIntervals = this.findBeatIntervals(audioData);
        if (beatIntervals.length < 2) return 120; // Default tempo
        
        const avgInterval = beatIntervals.reduce((sum, interval) => sum + interval, 0) / beatIntervals.length;
        return Math.round(60 / (avgInterval / this.sampleRate));
    }
    
    findBeatIntervals(audioData) {
        // Simple beat detection
        const windowSize = Math.floor(this.sampleRate * 0.1); // 100ms windows
        const energies = [];
        
        for (let i = 0; i < audioData.length - windowSize; i += windowSize) {
            const window = audioData.slice(i, i + windowSize);
            const energy = window.reduce((sum, sample) => sum + sample * sample, 0) / windowSize;
            energies.push(energy);
        }
        
        // Find peaks (simplified)
        const beats = [];
        const threshold = energies.reduce((sum, energy) => sum + energy, 0) / energies.length * 1.5;
        
        for (let i = 1; i < energies.length - 1; i++) {
            if (energies[i] > threshold && energies[i] > energies[i - 1] && energies[i] > energies[i + 1]) {
                beats.push(i * windowSize);
            }
        }
        
        // Calculate intervals
        const intervals = [];
        for (let i = 1; i < beats.length; i++) {
            intervals.push(beats[i] - beats[i - 1]);
        }
        
        return intervals;
    }
    
    calculateBeatStrength(audioData) {
        const beatIntervals = this.findBeatIntervals(audioData);
        if (beatIntervals.length === 0) return 0;
        
        // Regularity indicates strong beat
        const avgInterval = beatIntervals.reduce((sum, interval) => sum + interval, 0) / beatIntervals.length;
        const variance = beatIntervals.reduce((sum, interval) => sum + Math.pow(interval - avgInterval, 2), 0) / beatIntervals.length;
        
        return Math.max(0, 1 - Math.sqrt(variance) / avgInterval);
    }
    
    calculateRhythmRegularity(audioData) {
        const beatIntervals = this.findBeatIntervals(audioData);
        if (beatIntervals.length < 3) return 0;
        
        const avgInterval = beatIntervals.reduce((sum, interval) => sum + interval, 0) / beatIntervals.length;
        const deviations = beatIntervals.map(interval => Math.abs(interval - avgInterval));
        const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / deviations.length;
        
        return Math.max(0, 1 - (avgDeviation / avgInterval));
    }
    
    estimateFundamentalFrequency(audioData) {
        // Simplified autocorrelation-based pitch detection
        const minPeriod = Math.floor(this.sampleRate / 500); // 500 Hz max
        const maxPeriod = Math.floor(this.sampleRate / 50);  // 50 Hz min
        
        let maxCorrelation = 0;
        let bestPeriod = minPeriod;
        
        for (let period = minPeriod; period <= maxPeriod && period < audioData.length / 2; period++) {
            let correlation = 0;
            const samples = audioData.length - period;
            
            for (let i = 0; i < samples; i++) {
                correlation += audioData[i] * audioData[i + period];
            }
            
            correlation /= samples;
            
            if (correlation > maxCorrelation) {
                maxCorrelation = correlation;
                bestPeriod = period;
            }
        }
        
        return this.sampleRate / bestPeriod;
    }
    
    estimatePitch(audioData) {
        const fundamentalFreq = this.estimateFundamentalFrequency(audioData);
        
        // Convert frequency to MIDI note number
        const midiNote = 12 * Math.log2(fundamentalFreq / 440) + 69;
        return Math.max(0, Math.min(127, Math.round(midiNote)));
    }
    
    calculatePitchVariability(audioData) {
        // Analyze pitch changes over time
        const windowSize = Math.floor(audioData.length / 10);
        const pitches = [];
        
        for (let i = 0; i < audioData.length - windowSize; i += windowSize) {
            const window = audioData.slice(i, i + windowSize);
            const pitch = this.estimateFundamentalFrequency(window);
            pitches.push(pitch);
        }
        
        if (pitches.length < 2) return 0;
        
        const avgPitch = pitches.reduce((sum, pitch) => sum + pitch, 0) / pitches.length;
        const variance = pitches.reduce((sum, pitch) => sum + Math.pow(pitch - avgPitch, 2), 0) / pitches.length;
        
        return Math.sqrt(variance) / (avgPitch + 1e-6);
    }
    
    calculateShortTimeEnergy(audioData) {
        const windowSize = Math.floor(this.frameSize / 2);
        const energies = [];
        
        for (let i = 0; i < audioData.length - windowSize; i += this.hopLength) {
            const window = audioData.slice(i, i + windowSize);
            const energy = window.reduce((sum, sample) => sum + sample * sample, 0) / windowSize;
            energies.push(energy);
        }
        
        return energies;
    }
    
    calculateEnergyVariability(audioData) {
        const energies = this.calculateShortTimeEnergy(audioData);
        if (energies.length < 2) return 0;
        
        const avgEnergy = energies.reduce((sum, energy) => sum + energy, 0) / energies.length;
        const variance = energies.reduce((sum, energy) => sum + Math.pow(energy - avgEnergy, 2), 0) / energies.length;
        
        return Math.sqrt(variance) / (avgEnergy + 1e-6);
    }
    
    calculateMelSpectrogram(audioData) {
        // Simplified mel-scale spectrogram
        // In a real implementation, you'd use proper FFT and mel filter banks
        const windowSize = this.frameSize;
        const numFrames = Math.floor((audioData.length - windowSize) / this.hopLength) + 1;
        const melSpec = [];
        
        for (let frame = 0; frame < numFrames; frame++) {
            const start = frame * this.hopLength;
            const end = Math.min(start + windowSize, audioData.length);
            const window = audioData.slice(start, end);
            
            // Simplified mel bins (just energy in different frequency ranges)
            const melFrame = new Array(this.melBins).fill(0);
            const binSize = window.length / this.melBins;
            
            for (let bin = 0; bin < this.melBins; bin++) {
                const binStart = Math.floor(bin * binSize);
                const binEnd = Math.floor((bin + 1) * binSize);
                
                for (let i = binStart; i < binEnd && i < window.length; i++) {
                    melFrame[bin] += window[i] * window[i];
                }
                
                melFrame[bin] /= (binEnd - binStart);
            }
            
            melSpec.push(melFrame);
        }
        
        return melSpec;
    }
    
    calculateLoudness(audioData) {
        // Simplified loudness calculation (A-weighted would be more accurate)
        const rms = this.calculateRMS(audioData);
        return 20 * Math.log10(rms + 1e-6); // Convert to dB
    }
    
    extractProsody(audioData) {
        // Extract prosodic features
        const pitch = this.estimateFundamentalFrequency(audioData);
        const energy = this.calculateRMS(audioData);
        const tempo = this.estimateTempo(audioData);
        
        return {
            pitch: pitch,
            energy: energy,
            tempo: tempo,
            stress: energy > 0.5 ? 1 : 0, // Simplified stress detection
            accent: this.detectAccent(audioData)
        };
    }
    
    detectAccent(audioData) {
        // Simplified accent detection based on energy peaks
        const energies = this.calculateShortTimeEnergy(audioData);
        const threshold = energies.reduce((sum, energy) => sum + energy, 0) / energies.length * 1.5;
        
        return energies.filter(energy => energy > threshold).length / energies.length;
    }
    
    getDefaultFeatures(audioData) {
        return {
            length: audioData.length,
            duration: audioData.length / this.sampleRate,
            amplitude: 0.5,
            rms: 0.3,
            fundamentalFreq: 150,
            tempo: 120,
            beatStrength: 0.5,
            timestamp: Date.now()
        };
    }
    
    getStats() {
        return { ...this.stats };
    }
    
    dispose() {
        this.stats = {
            featuresExtracted: 0,
            totalProcessingTime: 0
        };
    }
}

// Usage example
async function createAudio2GestureTimelineExample() {
    // Create timeline instance
    const timeline = new BVHTimelineCompositor({
        frameRate: 30,
        maxTracks: 15
    });
    
    // Create Audio2Gesture integration
    const audio2GestureIntegration = new Audio2GestureTimelineIntegration(timeline, {
        realtime: true,
        trackName: 'body_gestures',
        priority: 80,
        channels: ['body', 'arms', 'hands'],
        converter: {
            scaleFactor: 1.0,
            smoothing: true,
            smoothingFactor: 0.7,
            gestureIntensity: 1.0,
            emotionalModulation: true
        }
    });
    
    // Initialize Audio2Gesture model
    await audio2GestureIntegration.initializeAudio2Gesture('/path/to/audio2gesture/model');
    
    // Set up timeline synchronization
    audio2GestureIntegration.synchronizeWithTimeline();
    
    // Example: Process audio for body gesture animation
    const audioData = new Float32Array(48000); // 3 seconds of audio at 16kHz
    const result = await audio2GestureIntegration.processAudio(audioData, {
        startTime: 0,
        emotionalContext: 'neutral',
        realtime: false
    });
    
    console.log('Audio2Gesture processing result:', result);
    
    // Start timeline playback
    timeline.play();
    
    return { timeline, audio2GestureIntegration };
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Audio2GestureTimelineIntegration, Audio2GestureAudioProcessor };
} else {
    window.Audio2GestureTimelineIntegration = Audio2GestureTimelineIntegration;
    window.Audio2GestureAudioProcessor = Audio2GestureAudioProcessor;
}
