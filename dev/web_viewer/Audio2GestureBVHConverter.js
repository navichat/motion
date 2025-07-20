/**
 * Audio2Gesture to BVH Converter
 * Converts Audio2Gesture neural network outputs to BVH bone transformations
 * Compatible with BVH Timeline buffering system
 * 
 * Follows patterns established in FaceformerBVHConverter for consistency
 */
class Audio2GestureBVHConverter {
    constructor(options = {}) {
        this.options = {
            modelPath: options.modelPath || './models/audio2gesture_step_fixed.onnx',
            sampleRate: options.sampleRate || 44100,
            frameRate: options.frameRate || 30, // BVH frame rate
            bufferSize: options.bufferSize || 60, // Pre-buffer 2 seconds
            lookaheadFrames: options.lookaheadFrames || 10,
            cleanupThreshold: options.cleanupThreshold || 180, // Cleanup after 6 seconds
            ...options
        };

        // Neural network components
        this.generator = null;
        this.initialized = false;

        // Audio processing
        this.audioContext = null;
        this.audioBuffer = [];
        this.processingQueue = [];

        // Animation state
        this.currentHiddenState = null;
        this.previousMotion = null;
        this.isGenerating = false;

        // Performance tracking
        this.performanceStats = {
            audioProcessingTime: 0,
            neuralInferenceTime: 0,
            bvhConversionTime: 0,
            totalProcessingTime: 0,
            framesGenerated: 0,
            averageFrameTime: 0
        };

        // Gesture bone mapping (48 values from audio2gesture)
        this.gestureBoneMap = this.createGestureBoneMapping();
        
        console.log('🎭 Audio2GestureBVHConverter initialized with options:', this.options);
    }

    createGestureBoneMapping() {
        // Map 48 audio2gesture output values to BVH bone transformations
        // Audio2Gesture typically outputs full-body gesture parameters
        return {
            // Upper body focus for gesture animation
            spine: { indices: [0, 1, 2], type: 'rotation' }, // Spine rotation
            leftShoulder: { indices: [3, 4, 5], type: 'rotation' },
            rightShoulder: { indices: [6, 7, 8], type: 'rotation' },
            leftElbow: { indices: [9, 10, 11], type: 'rotation' },
            rightElbow: { indices: [12, 13, 14], type: 'rotation' },
            leftWrist: { indices: [15, 16, 17], type: 'rotation' },
            rightWrist: { indices: [18, 19, 20], type: 'rotation' },
            
            // Torso and posture
            chest: { indices: [21, 22, 23], type: 'rotation' },
            neck: { indices: [24, 25, 26], type: 'rotation' },
            head: { indices: [27, 28, 29], type: 'rotation' },
            
            // Lower body (subtle movements)
            hips: { indices: [30, 31, 32], type: 'rotation' },
            leftHip: { indices: [33, 34, 35], type: 'rotation' },
            rightHip: { indices: [36, 37, 38], type: 'rotation' },
            
            // Additional gesture parameters
            leftHand: { indices: [39, 40, 41], type: 'rotation' },
            rightHand: { indices: [42, 43, 44], type: 'rotation' },
            
            // Root motion (optional)
            rootPosition: { indices: [45, 46, 47], type: 'position' }
        };
    }

    async initialize() {
        try {
            console.log('🎬 Initializing Audio2Gesture to BVH Converter...');

            // Check dependencies
            if (typeof Audio2GestureWebGenerator === 'undefined') {
                throw new Error('Audio2GestureWebGenerator not available');
            }

            // Initialize neural network generator
            this.generator = new Audio2GestureWebGenerator(this.options.modelPath);
            const initSuccess = await this.generator.initialize();
            
            if (!initSuccess) {
                throw new Error('Failed to initialize Audio2Gesture generator');
            }

            // Initialize audio context
            if (typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined') {
                const AudioContextClass = AudioContext || webkitAudioContext;
                this.audioContext = new AudioContextClass({ sampleRate: this.options.sampleRate });
                console.log('🎵 Audio context initialized');
            }

            // Initialize animation state
            this.currentHiddenState = this.generator.createInitialHiddenState();
            this.previousMotion = new Float32Array(48).fill(0); // Neutral pose

            this.initialized = true;
            console.log('✅ Audio2GestureBVHConverter initialization complete');
            
            return true;
        } catch (error) {
            console.error('❌ Audio2GestureBVHConverter initialization failed:', error);
            this.initialized = false;
            return false;
        }
    }

    async processAudioToGestures(audioData, options = {}) {
        if (!this.initialized) {
            throw new Error('Converter not initialized. Call initialize() first.');
        }

        const startTime = performance.now();
        console.log('🎵 Processing audio to gestures...');

        try {
            // Extract audio features (simplified - real implementation would use proper feature extraction)
            const audioFeatures = this.extractAudioFeatures(audioData);
            
            // Generate gesture parameters using neural network
            const lexemeType = options.expressiveness || 'neutral';
            const numFrames = options.duration ? Math.ceil(options.duration * this.options.frameRate) : 30;
            
            const generationResult = await this.generator.generateGestureSequence(
                audioFeatures,
                lexemeType,
                numFrames
            );

            // Convert gesture frames to BVH bone transformations
            const bvhFrames = this.convertGesturesToBVH(generationResult.frames);

            const processingTime = performance.now() - startTime;
            this.updatePerformanceStats(processingTime, generationResult.frames.length);

            console.log(`✅ Generated ${bvhFrames.length} BVH frames in ${processingTime.toFixed(1)}ms`);

            return {
                bvhFrames,
                metadata: {
                    duration: numFrames / this.options.frameRate,
                    frameRate: this.options.frameRate,
                    audioLength: audioData ? audioData.length : 0,
                    expressiveness: lexemeType,
                    processingTime,
                    neuralMetrics: generationResult.metrics
                }
            };

        } catch (error) {
            console.error('❌ Audio to gesture processing failed:', error);
            throw error;
        }
    }

    extractAudioFeatures(audioData) {
        // Simplified audio feature extraction
        // Real implementation would use proper MFCC/mel-spectrogram extraction
        if (!audioData || audioData.length === 0) {
            return null; // Use synthetic features in generator
        }

        // Convert audio to feature format expected by Audio2Gesture
        const windowSize = 30;
        const featureSize = 80;
        const features = new Float32Array(featureSize * windowSize);

        // Simple feature extraction (placeholder)
        for (let i = 0; i < features.length; i++) {
            const audioIndex = Math.floor((i / features.length) * audioData.length);
            features[i] = audioData[audioIndex] || 0;
        }

        return features;
    }

    convertGesturesToBVH(gestureFrames) {
        const bvhFrames = [];

        for (let frameIndex = 0; frameIndex < gestureFrames.length; frameIndex++) {
            const gestureData = gestureFrames[frameIndex];
            const bvhFrame = this.convertSingleGestureToBVH(gestureData, frameIndex);
            bvhFrames.push(bvhFrame);
        }

        return bvhFrames;
    }

    convertSingleGestureToBVH(gestureData, frameIndex = 0) {
        const bvhTransforms = {};
        const timestamp = frameIndex / this.options.frameRate;

        // Convert each bone group
        for (const [boneName, mapping] of Object.entries(this.gestureBoneMap)) {
            const values = mapping.indices.map(i => gestureData[i] || 0);
            
            if (mapping.type === 'rotation') {
                // Convert to rotation (assuming values are in radians or normalized)
                bvhTransforms[boneName] = {
                    rotation: {
                        x: this.normalizeRotation(values[0]),
                        y: this.normalizeRotation(values[1]),
                        z: this.normalizeRotation(values[2])
                    }
                };
            } else if (mapping.type === 'position') {
                // Convert to position
                bvhTransforms[boneName] = {
                    position: {
                        x: values[0] * 10, // Scale appropriately
                        y: values[1] * 10,
                        z: values[2] * 10
                    }
                };
            }
        }

        return {
            frameNumber: frameIndex,
            timestamp: timestamp,
            transforms: bvhTransforms,
            metadata: {
                source: 'audio2gesture',
                confidence: this.calculateGestureConfidence(gestureData),
                energy: this.calculateGestureEnergy(gestureData)
            }
        };
    }

    normalizeRotation(value) {
        // Normalize neural network output to reasonable rotation range
        // Assuming network outputs values roughly in [-1, 1]
        return Math.max(-180, Math.min(180, value * 45)); // Scale to ±45 degrees
    }

    calculateGestureConfidence(gestureData) {
        // Simple confidence metric based on gesture magnitude
        const magnitude = Math.sqrt(gestureData.reduce((sum, val) => sum + val * val, 0));
        return Math.min(1.0, magnitude / 10); // Normalize to [0, 1]
    }

    calculateGestureEnergy(gestureData) {
        // Calculate gesture energy for animation blending
        const upperBodyIndices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; // Arms primarily
        const upperBodyEnergy = upperBodyIndices.reduce((sum, i) => {
            return sum + Math.abs(gestureData[i] || 0);
        }, 0);
        return upperBodyEnergy / upperBodyIndices.length;
    }

    async createTimelineClip(audioData, options = {}) {
        console.log('🎬 Creating Audio2Gesture timeline clip...');

        const result = await this.processAudioToGestures(audioData, options);
        
        // Create timeline clip compatible with BVHTimeline
        const clip = {
            id: `audio2gesture_${Date.now()}`,
            type: 'audio2gesture',
            startTime: options.startTime || 0,
            duration: result.metadata.duration,
            priority: options.priority || 3, // Medium priority (after face, before base)
            blendMode: options.blendMode || 'additive',
            frames: result.bvhFrames,
            metadata: {
                ...result.metadata,
                audioLength: audioData ? audioData.length : 0,
                expressiveness: options.expressiveness || 'neutral',
                source: 'audio2gesture-neural'
            }
        };

        // Pre-buffer frames for timeline if buffer is available
        if (options.timeline && options.timeline.buffer) {
            console.log(`📦 Pre-buffering ${result.bvhFrames.length} gesture frames...`);
            
            for (let i = 0; i < result.bvhFrames.length; i++) {
                const frameTime = clip.startTime + (i / this.options.frameRate);
                options.timeline.buffer.setFrame(clip.id, frameTime, result.bvhFrames[i]);
            }
            
            console.log('✅ Gesture frames pre-buffered to timeline');
        }

        return clip;
    }

    async generateRealtimeGesture(audioChunk, previousState = null) {
        if (!this.initialized) {
            throw new Error('Converter not initialized');
        }

        const startTime = performance.now();

        try {
            // Use previous state if available
            if (previousState) {
                this.currentHiddenState = previousState.hiddenState;
                this.previousMotion = previousState.motion;
            }

            // Process audio chunk
            const audioFeatures = this.extractAudioFeatures(audioChunk);
            const audioWindow = this.generator.prepareAudioWindow(audioFeatures);
            const lexemeFeatures = this.generator.createLexemeFeatures('neutral');
            
            // Convert previous motion to tensor
            const prevMotionTensor = new ort.Tensor('float32', this.previousMotion, [1, 48]);

            // Generate single gesture frame
            const result = await this.generator.generateSingleStep(
                audioWindow,
                prevMotionTensor,
                lexemeFeatures,
                this.currentHiddenState
            );

            // Update state
            this.currentHiddenState = result.newHiddenState;
            this.previousMotion = Array.from(result.newMotion.data);

            // Convert to BVH frame
            const bvhFrame = this.convertSingleGestureToBVH(this.previousMotion);

            const processingTime = performance.now() - startTime;
            this.updatePerformanceStats(processingTime, 1);

            return {
                bvhFrame,
                state: {
                    hiddenState: this.currentHiddenState,
                    motion: this.previousMotion
                },
                metrics: {
                    processingTime,
                    timestamp: performance.now()
                }
            };

        } catch (error) {
            console.error('❌ Realtime gesture generation failed:', error);
            throw error;
        }
    }

    updatePerformanceStats(processingTime, frameCount) {
        this.performanceStats.totalProcessingTime += processingTime;
        this.performanceStats.framesGenerated += frameCount;
        this.performanceStats.averageFrameTime = this.performanceStats.totalProcessingTime / this.performanceStats.framesGenerated;
    }

    getPerformanceStats() {
        return {
            ...this.performanceStats,
            currentFPS: this.performanceStats.averageFrameTime > 0 ? 1000 / this.performanceStats.averageFrameTime : 0
        };
    }

    // Utility methods for integration
    static async createFromAudio(audioData, options = {}) {
        const converter = new Audio2GestureBVHConverter(options);
        await converter.initialize();
        return converter.createTimelineClip(audioData, options);
    }

    static isAvailable() {
        return typeof Audio2GestureWebGenerator !== 'undefined' && typeof ort !== 'undefined';
    }

    dispose() {
        if (this.audioContext) {
            this.audioContext.close();
        }
        this.initialized = false;
        console.log('🧹 Audio2GestureBVHConverter disposed');
    }
}

// Demo function
async function demoAudio2GestureBVHConverter() {
    console.log('🎭 Audio2Gesture to BVH Converter Demo');
    console.log('=====================================\n');

    try {
        const converter = new Audio2GestureBVHConverter({
            frameRate: 30,
            bufferSize: 30
        });

        if (!await converter.initialize()) {
            console.log('❌ Failed to initialize converter');
            return;
        }

        // Test 1: Generate gesture animation from synthetic audio
        console.log('🧪 Test 1: Synthetic audio gesture generation');
        const syntheticAudio = new Float32Array(1000).map(() => Math.sin(Math.random() * Math.PI) * 0.1);
        
        const result1 = await converter.processAudioToGestures(syntheticAudio, {
            duration: 2.0,
            expressiveness: 'expressive'
        });

        console.log(`✅ Generated ${result1.bvhFrames.length} frames`);
        console.log(`   Processing time: ${result1.metadata.processingTime.toFixed(1)}ms`);
        console.log(`   Frame rate: ${result1.metadata.frameRate} FPS`);

        // Test 2: Create timeline clip
        console.log('\n🧪 Test 2: Timeline clip creation');
        const clip = await converter.createTimelineClip(syntheticAudio, {
            startTime: 0,
            duration: 2.0,
            priority: 3,
            expressiveness: 'subtle'
        });

        console.log(`✅ Created timeline clip: ${clip.id}`);
        console.log(`   Duration: ${clip.duration}s`);
        console.log(`   Frames: ${clip.frames.length}`);
        console.log(`   Priority: ${clip.priority}`);

        // Test 3: Realtime generation test
        console.log('\n🧪 Test 3: Realtime gesture generation');
        let state = null;
        const realtimeFrames = [];

        for (let i = 0; i < 5; i++) {
            const audioChunk = new Float32Array(100).map(() => Math.sin(i * 0.5) * 0.1);
            const result = await converter.generateRealtimeGesture(audioChunk, state);
            
            state = result.state;
            realtimeFrames.push(result.bvhFrame);
            
            console.log(`   Frame ${i + 1}: ${result.metrics.processingTime.toFixed(1)}ms`);
        }

        // Performance summary
        const stats = converter.getPerformanceStats();
        console.log('\n📊 Performance Summary:');
        console.log(`   Frames generated: ${stats.framesGenerated}`);
        console.log(`   Average frame time: ${stats.averageFrameTime.toFixed(1)}ms`);
        console.log(`   Average FPS: ${stats.currentFPS.toFixed(1)}`);

        console.log('\n🎉 Audio2Gesture to BVH Converter Demo Complete!');
        
        return {
            gestureResult: result1,
            timelineClip: clip,
            realtimeFrames,
            performanceStats: stats
        };

    } catch (error) {
        console.error('❌ Demo failed:', error);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Audio2GestureBVHConverter };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.Audio2GestureBVHConverter = Audio2GestureBVHConverter;
    window.demoAudio2GestureBVHConverter = demoAudio2GestureBVHConverter;
}
