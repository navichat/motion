const ort = require('onnxruntime-web');
const fs = require('fs');

class Audio2GestureWebGenerator {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
        this.initialized = false;
    }

    async initialize() {
        try {
            console.log('🎭 Initializing Audio2Gesture Web Generator...');
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['cpu']
            });
            this.initialized = true;
            console.log('✅ Audio2Gesture generator initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize Audio2Gesture generator:', error);
            return false;
        }
    }

    createInitialHiddenState() {
        // Initialize hidden state with zeros [4, 1, 1024]
        const hiddenStateSize = 4 * 1 * 1024;
        const hiddenStateData = new Float32Array(hiddenStateSize);
        return new ort.Tensor('float32', hiddenStateData, [4, 1, 1024]);
    }

    prepareAudioWindow(audioFeatures, windowSize = 30) {
        // Prepare fixed-size audio window [1, 80, 30]
        const audioData = new Float32Array(1 * 80 * windowSize);
        
        // If audioFeatures is provided, copy it (truncate or pad as needed)
        if (audioFeatures && audioFeatures.length > 0) {
            const flatAudio = audioFeatures.flat ? audioFeatures.flat(2) : audioFeatures;
            const copyLength = Math.min(flatAudio.length, audioData.length);
            for (let i = 0; i < copyLength; i++) {
                audioData[i] = flatAudio[i];
            }
        } else {
            // Use random audio for testing
            for (let i = 0; i < audioData.length; i++) {
                audioData[i] = (Math.random() - 0.5) * 0.1; // Small random values
            }
        }
        
        return new ort.Tensor('float32', audioData, [1, 80, windowSize]);
    }

    createLexemeFeatures(lexemeType = 'neutral') {
        // Create lexeme features [1, 96]
        const lexemeData = new Float32Array(96);
        
        // Simple lexeme encoding (in real app, this would be more sophisticated)
        switch (lexemeType) {
            case 'expressive':
                lexemeData.fill(0.5);
                break;
            case 'subtle':
                lexemeData.fill(0.2);
                break;
            case 'neutral':
            default:
                lexemeData.fill(0.1);
                break;
        }
        
        return new ort.Tensor('float32', lexemeData, [1, 96]);
    }

    async generateSingleStep(audioWindow, prevMotion, currentLexeme, hiddenState) {
        if (!this.initialized) {
            throw new Error('Generator not initialized. Call initialize() first.');
        }

        const feeds = {
            audio_window: audioWindow,
            prev_motion: prevMotion,
            current_lexeme: currentLexeme,
            hidden_state: hiddenState
        };

        const results = await this.session.run(feeds);
        return {
            newMotion: results.new_motion,
            newHiddenState: results.new_hidden_state
        };
    }

    async generateGestureSequence(audioFeatures = null, lexemeType = 'neutral', numFrames = 10) {
        if (!this.initialized) {
            throw new Error('Generator not initialized. Call initialize() first.');
        }

        console.log(`🎬 Generating ${numFrames} gesture frames...`);
        
        // Prepare inputs
        const audioWindow = this.prepareAudioWindow(audioFeatures);
        const currentLexeme = this.createLexemeFeatures(lexemeType);
        let hiddenState = this.createInitialHiddenState();
        
        // Initialize first motion frame with zeros
        let currentMotion = new ort.Tensor('float32', new Float32Array(48), [1, 48]);
        
        const generatedFrames = [];
        const performanceMetrics = [];
        
        for (let step = 0; step < numFrames; step++) {
            const startTime = performance.now();
            
            console.log(`  🔄 Generating frame ${step + 1}/${numFrames}...`);
            
            // Generate next motion frame
            const result = await this.generateSingleStep(
                audioWindow,
                currentMotion,
                currentLexeme,
                hiddenState
            );
            
            // Update state for next iteration
            currentMotion = result.newMotion;
            hiddenState = result.newHiddenState;
            
            // Store generated frame
            const motionData = Array.from(result.newMotion.data);
            generatedFrames.push(motionData);
            
            const endTime = performance.now();
            const frameTime = endTime - startTime;
            performanceMetrics.push(frameTime);
            
            // Show progress
            if (step < 5 || step % 5 === 4) {
                console.log(`    ✨ Frame ${step + 1}: [${motionData.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...] (${frameTime.toFixed(1)}ms)`);
            }
        }
        
        const totalTime = performanceMetrics.reduce((a, b) => a + b, 0);
        const avgTime = totalTime / numFrames;
        
        console.log(`🎉 Generation complete!`);
        console.log(`   📊 Total time: ${totalTime.toFixed(1)}ms`);
        console.log(`   ⚡ Average per frame: ${avgTime.toFixed(1)}ms`);
        console.log(`   🚀 Generation rate: ${(1000 / avgTime).toFixed(1)} FPS`);
        
        return {
            frames: generatedFrames,
            metrics: {
                totalTime,
                avgTime,
                fps: 1000 / avgTime,
                numFrames
            }
        };
    }

    async generateWithRealAudio(audioPath, lexemeType = 'neutral', numFrames = 20) {
        // Placeholder for real audio processing
        // In a full implementation, this would:
        // 1. Load and preprocess audio file
        // 2. Extract audio features (e.g., MFCC, mel-spectrogram)
        // 3. Generate gestures synchronized to audio
        
        console.log(`🎵 Loading audio from ${audioPath}...`);
        console.log('⚠️  Real audio processing not implemented yet');
        console.log('🔄 Using synthetic audio features instead...');
        
        return await this.generateGestureSequence(null, lexemeType, numFrames);
    }
}

// Demo usage
async function demoAudio2GestureGeneration() {
    console.log('🎭 Audio2Gesture Web Generator Demo');
    console.log('=====================================\n');
    
    try {
        // Initialize generator
        const generator = new Audio2GestureWebGenerator('./audio2gesture_step_fixed.onnx');
        const initSuccess = await generator.initialize();
        
        if (!initSuccess) {
            console.log('❌ Failed to initialize generator');
            return;
        }
        
        // Test 1: Short sequence with neutral gestures
        console.log('🧪 Test 1: Neutral gesture sequence (5 frames)');
        const result1 = await generator.generateGestureSequence(null, 'neutral', 5);
        
        // Test 2: Longer sequence with expressive gestures
        console.log('\n🧪 Test 2: Expressive gesture sequence (10 frames)');
        const result2 = await generator.generateGestureSequence(null, 'expressive', 10);
        
        // Test 3: Performance test
        console.log('\n🧪 Test 3: Performance test (20 frames)');
        const result3 = await generator.generateGestureSequence(null, 'subtle', 20);
        
        // Show final results
        console.log('\n📊 Demo Results Summary:');
        console.log(`   Test 1 (5 frames): ${result1.metrics.avgTime.toFixed(1)}ms/frame, ${result1.metrics.fps.toFixed(1)} FPS`);
        console.log(`   Test 2 (10 frames): ${result2.metrics.avgTime.toFixed(1)}ms/frame, ${result2.metrics.fps.toFixed(1)} FPS`);
        console.log(`   Test 3 (20 frames): ${result3.metrics.avgTime.toFixed(1)}ms/frame, ${result3.metrics.fps.toFixed(1)} FPS`);
        
        // Save results for inspection
        const demoResults = {
            test1: result1,
            test2: result2,
            test3: result3,
            timestamp: new Date().toISOString()
        };
        
        fs.writeFileSync('audio2gesture_demo_results.json', JSON.stringify(demoResults, null, 2));
        console.log('\n💾 Demo results saved to audio2gesture_demo_results.json');
        
        console.log('\n🎉 Audio2Gesture Web Generator Demo Complete!');
        console.log('✅ Multi-step autoregressive generation working');
        console.log('🚀 Ready for integration with real audio and 3D rendering');
        
    } catch (error) {
        console.error('❌ Demo failed:', error);
    }
}

// Run demo if called directly
if (require.main === module) {
    demoAudio2GestureGeneration();
}

module.exports = { Audio2GestureWebGenerator };
