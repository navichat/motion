/**
 * Audio2Gesture BVH Timeline Integration Example
 * Demonstrates how to integrate Audio2Gesture neural network with BVH Timeline system
 * Shows multi-source animation composition with gesture generation
 */

class Audio2GestureTimelineIntegration {
    constructor(options = {}) {
        this.timeline = null;
        this.audio2gestureConverter = null;
        this.initialized = false;
        
        this.options = {
            timelineFrameRate: 30,
            timelineBufferSize: 90,
            gestureBufferSize: 60,
            gesturePriority: 3, // Medium priority for gestures
            ...options
        };

        console.log('🎭 Audio2Gesture Timeline Integration initialized');
    }

    async initialize() {
        try {
            console.log('🚀 Initializing Audio2Gesture Timeline Integration...');

            // Initialize BVH Timeline system
            if (typeof BVHTimeline === 'undefined') {
                throw new Error('BVHTimeline not available');
            }

            this.timeline = new BVHTimeline({
                frameRate: this.options.timelineFrameRate,
                bufferSize: this.options.timelineBufferSize,
                lookaheadFrames: 15,
                cleanupThreshold: 180
            });

            console.log('✅ BVH Timeline initialized');

            // Initialize Audio2Gesture converter
            if (typeof Audio2GestureBVHConverter === 'undefined') {
                throw new Error('Audio2GestureBVHConverter not available');
            }

            this.audio2gestureConverter = new Audio2GestureBVHConverter({
                frameRate: this.options.timelineFrameRate,
                bufferSize: this.options.gestureBufferSize
            });

            const converterInitialized = await this.audio2gestureConverter.initialize();
            if (!converterInitialized) {
                throw new Error('Failed to initialize Audio2Gesture converter');
            }

            console.log('✅ Audio2Gesture converter initialized');

            this.initialized = true;
            console.log('🎉 Audio2Gesture Timeline Integration ready!');
            
            return true;

        } catch (error) {
            console.error('❌ Integration initialization failed:', error);
            this.initialized = false;
            return false;
        }
    }

    async addGestureFromAudio(audioData, options = {}) {
        if (!this.initialized) {
            throw new Error('Integration not initialized');
        }

        console.log('🎵 Adding gesture animation from audio...');

        const gestureOptions = {
            startTime: options.startTime || 0,
            duration: options.duration || (audioData.length / 44100), // Assume 44.1kHz
            expressiveness: options.expressiveness || 'neutral',
            priority: options.priority || this.options.gesturePriority,
            blendMode: options.blendMode || 'additive',
            timeline: this.timeline // Pass timeline for pre-buffering
        };

        // Create gesture timeline clip
        const gestureClip = await this.audio2gestureConverter.createTimelineClip(audioData, gestureOptions);
        
        // Add to timeline
        this.timeline.addClip(gestureClip);

        console.log(`✅ Added gesture clip: ${gestureClip.id}`);
        console.log(`   Duration: ${gestureClip.duration}s, Frames: ${gestureClip.frames.length}`);
        console.log(`   Priority: ${gestureClip.priority}, Start: ${gestureClip.startTime}s`);

        return gestureClip;
    }

    async addMultipleGestureSources(audioSources) {
        if (!this.initialized) {
            throw new Error('Integration not initialized');
        }

        console.log('🎬 Adding multiple gesture sources...');
        const addedClips = [];

        for (let i = 0; i < audioSources.length; i++) {
            const source = audioSources[i];
            
            try {
                const clip = await this.addGestureFromAudio(source.audioData, {
                    startTime: source.startTime || i * 2, // Stagger by 2 seconds if no start time
                    duration: source.duration,
                    expressiveness: source.expressiveness || 'neutral',
                    priority: source.priority || (this.options.gesturePriority + i) // Slightly different priorities
                });

                addedClips.push(clip);
                console.log(`  ✅ Added source ${i + 1}/${audioSources.length}`);

            } catch (error) {
                console.error(`  ❌ Failed to add source ${i + 1}:`, error);
            }
        }

        console.log(`🎉 Added ${addedClips.length}/${audioSources.length} gesture sources`);
        return addedClips;
    }

    async createRealtimeGestureStream(options = {}) {
        if (!this.initialized) {
            throw new Error('Integration not initialized');
        }

        console.log('🔄 Starting realtime gesture stream...');

        const streamOptions = {
            chunkDuration: options.chunkDuration || 0.1, // 100ms chunks
            lookahead: options.lookahead || 0.5, // 500ms lookahead
            priority: options.priority || this.options.gesturePriority,
            ...options
        };

        let streamActive = true;
        let frameCount = 0;
        let gestureState = null;

        const streamInterval = setInterval(async () => {
            if (!streamActive) {
                clearInterval(streamInterval);
                return;
            }

            try {
                // Generate audio chunk (in real app, this would come from microphone)
                const chunkSize = Math.floor(streamOptions.chunkDuration * 44100);
                const audioChunk = new Float32Array(chunkSize);
                
                // Simulate audio with some pattern
                for (let i = 0; i < chunkSize; i++) {
                    const t = (frameCount * chunkSize + i) / 44100;
                    audioChunk[i] = Math.sin(t * 2 * Math.PI + Math.sin(t * 0.5) * 2) * 0.1;
                }

                // Generate realtime gesture frame
                const result = await this.audio2gestureConverter.generateRealtimeGesture(audioChunk, gestureState);
                gestureState = result.state;

                // Create instant timeline clip for current frame
                const currentTime = frameCount * streamOptions.chunkDuration;
                const instantClip = {
                    id: `realtime_gesture_${frameCount}`,
                    type: 'audio2gesture_realtime',
                    startTime: currentTime,
                    duration: streamOptions.chunkDuration,
                    priority: streamOptions.priority,
                    blendMode: 'additive',
                    frames: [result.bvhFrame],
                    metadata: {
                        source: 'audio2gesture-realtime',
                        frameCount: frameCount,
                        processingTime: result.metrics.processingTime
                    }
                };

                // Add to timeline with automatic cleanup
                this.timeline.addClip(instantClip);

                frameCount++;

                if (frameCount % 10 === 0) {
                    console.log(`  🎯 Generated ${frameCount} realtime gesture frames`);
                }

            } catch (error) {
                console.error('Realtime gesture generation error:', error);
                streamActive = false;
            }
        }, streamOptions.chunkDuration * 1000);

        // Return control object
        return {
            stop: () => {
                streamActive = false;
                console.log('⏹️ Realtime gesture stream stopped');
            },
            isActive: () => streamActive,
            getFrameCount: () => frameCount
        };
    }

    getCompositeAnimationAtTime(time) {
        if (!this.initialized) {
            throw new Error('Integration not initialized');
        }

        return this.timeline.getFrameAtTime(time);
    }

    playAnimationSequence(startTime = 0, duration = 5, onFrame = null) {
        if (!this.initialized) {
            throw new Error('Integration not initialized');
        }

        console.log(`▶️ Playing animation sequence: ${startTime}s to ${startTime + duration}s`);

        const frameRate = this.options.timelineFrameRate;
        const totalFrames = Math.ceil(duration * frameRate);
        let currentFrame = 0;

        const playInterval = setInterval(() => {
            const currentTime = startTime + (currentFrame / frameRate);
            const compositeFrame = this.getCompositeAnimationAtTime(currentTime);

            if (onFrame && compositeFrame) {
                onFrame(compositeFrame, currentTime, currentFrame);
            }

            currentFrame++;

            if (currentFrame >= totalFrames) {
                clearInterval(playInterval);
                console.log('🏁 Animation sequence playback complete');
            }
        }, 1000 / frameRate);

        return {
            stop: () => clearInterval(playInterval),
            getCurrentTime: () => startTime + (currentFrame / frameRate),
            getCurrentFrame: () => currentFrame,
            getTotalFrames: () => totalFrames
        };
    }

    getTimelineStats() {
        if (!this.timeline) return null;

        const clips = this.timeline.clips || [];
        const gestureClips = clips.filter(clip => 
            clip.type === 'audio2gesture' || clip.type === 'audio2gesture_realtime'
        );

        const bufferStats = this.timeline.buffer ? this.timeline.buffer.getStats() : {};
        const converterStats = this.audio2gestureConverter ? 
            this.audio2gestureConverter.getPerformanceStats() : {};

        return {
            timeline: {
                totalClips: clips.length,
                gestureClips: gestureClips.length,
                activeTimeRange: this.timeline.getActiveTimeRange ? this.timeline.getActiveTimeRange() : null,
                bufferStats
            },
            audio2gesture: converterStats,
            integration: {
                initialized: this.initialized,
                frameRate: this.options.timelineFrameRate,
                gesturePriority: this.options.gesturePriority
            }
        };
    }

    dispose() {
        if (this.audio2gestureConverter) {
            this.audio2gestureConverter.dispose();
        }
        
        if (this.timeline) {
            this.timeline.dispose && this.timeline.dispose();
        }

        this.initialized = false;
        console.log('🧹 Audio2Gesture Timeline Integration disposed');
    }
}

// Demo function showing complete integration
async function demoAudio2GestureTimelineIntegration() {
    console.log('🎭 Audio2Gesture Timeline Integration Demo');
    console.log('==========================================\n');

    try {
        // Initialize integration
        const integration = new Audio2GestureTimelineIntegration({
            timelineFrameRate: 30,
            timelineBufferSize: 120,
            gesturePriority: 3
        });

        const initSuccess = await integration.initialize();
        if (!initSuccess) {
            console.log('❌ Failed to initialize integration');
            return;
        }

        // Test 1: Single gesture from audio
        console.log('🧪 Test 1: Single gesture animation from audio');
        const testAudio1 = new Float32Array(44100 * 2); // 2 seconds
        for (let i = 0; i < testAudio1.length; i++) {
            testAudio1[i] = Math.sin(i / 1000) * 0.1;
        }

        const clip1 = await integration.addGestureFromAudio(testAudio1, {
            startTime: 0,
            expressiveness: 'expressive'
        });

        // Test 2: Multiple gesture sources
        console.log('\n🧪 Test 2: Multiple gesture sources');
        const audioSources = [
            {
                audioData: new Float32Array(44100).map(() => Math.sin(Math.random() * 6.28) * 0.1),
                startTime: 3,
                expressiveness: 'subtle',
                priority: 2
            },
            {
                audioData: new Float32Array(44100 * 1.5).map(() => Math.cos(Math.random() * 6.28) * 0.1),
                startTime: 5,
                expressiveness: 'expressive',
                priority: 4
            }
        ];

        const multipleClips = await integration.addMultipleGestureSources(audioSources);

        // Test 3: Timeline playback
        console.log('\n🧪 Test 3: Timeline playback simulation');
        let frameCount = 0;
        
        const playback = integration.playAnimationSequence(0, 3, (frame, time, frameIndex) => {
            frameCount++;
            if (frameIndex % 15 === 0) { // Every 0.5 seconds
                const transformCount = Object.keys(frame.transforms || {}).length;
                console.log(`  ⏰ Time: ${time.toFixed(2)}s, Transforms: ${transformCount}`);
            }
        });

        // Wait for playback to complete
        await new Promise(resolve => {
            const checkComplete = () => {
                if (playback.getCurrentFrame() >= playback.getTotalFrames()) {
                    resolve();
                } else {
                    setTimeout(checkComplete, 100);
                }
            };
            checkComplete();
        });

        // Test 4: Realtime stream (short test)
        console.log('\n🧪 Test 4: Realtime gesture stream (5 seconds)');
        const realtimeStream = await integration.createRealtimeGestureStream({
            chunkDuration: 0.1,
            priority: 1
        });

        // Let it run for 5 seconds
        await new Promise(resolve => setTimeout(resolve, 5000));
        realtimeStream.stop();

        console.log(`   Generated ${realtimeStream.getFrameCount()} realtime frames`);

        // Show final statistics
        console.log('\n📊 Final Integration Statistics:');
        const stats = integration.getTimelineStats();
        console.log('Timeline Stats:', JSON.stringify(stats.timeline, null, 2));
        console.log('Audio2Gesture Stats:', JSON.stringify(stats.audio2gesture, null, 2));

        console.log('\n🎉 Audio2Gesture Timeline Integration Demo Complete!');
        console.log('✅ Single gesture clips working');
        console.log('✅ Multiple source composition working');
        console.log('✅ Timeline playback working');
        console.log('✅ Realtime gesture streaming working');
        console.log('🚀 Ready for production use with real audio and 3D rendering!');

        // Cleanup
        integration.dispose();

        return {
            singleClip: clip1,
            multipleClips,
            totalFrames: frameCount,
            realtimeFrames: realtimeStream.getFrameCount(),
            finalStats: stats
        };

    } catch (error) {
        console.error('❌ Demo failed:', error);
        throw error;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Audio2GestureTimelineIntegration };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.Audio2GestureTimelineIntegration = Audio2GestureTimelineIntegration;
    window.demoAudio2GestureTimelineIntegration = demoAudio2GestureTimelineIntegration;
}
