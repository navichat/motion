const ort = require('onnxruntime-web');
const { ChunkedModelLoader } = require('./chunked_model_loader');

class Audio2GestureChunkedGenerator {
    constructor() {
        this.session = null;
        this.loader = new ChunkedModelLoader();
        this.isInitialized = false;
    }
    
    /**
     * Initialize the generator with chunked model support
     * @param {string} modelPath - Path to model file or chunks metadata
     */
    async initialize(modelPath) {
        console.log('🚀 Initializing Audio2Gesture Generator with chunked model support...');
        
        try {
            // Load model (automatically handles chunks)
            const modelData = await this.loader.loadModel(modelPath);
            
            // Create ONNX session
            console.log('🔧 Creating ONNX Runtime session...');
            this.session = await ort.InferenceSession.create(modelData, {
                executionProviders: ['cpu']
            });
            
            console.log('✅ Model loaded successfully!');
            console.log('📊 Model inputs:', Object.keys(this.session.inputMetadata));
            console.log('📊 Model outputs:', Object.keys(this.session.outputMetadata));
            
            this.isInitialized = true;
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize model:', error.message);
            throw error;
        }
    }
    
    /**
     * Generate motion sequence from audio features
     * @param {Object} inputs - Audio and motion inputs
     * @param {number} numFrames - Number of frames to generate
     * @returns {Array} Generated motion sequence
     */
    async generateSequence(inputs, numFrames = 20) {
        if (!this.isInitialized) {
            throw new Error('Generator not initialized. Call initialize() first.');
        }
        
        console.log(`🎭 Generating ${numFrames} frames of motion...`);
        
        // For single-step model (audio2gesture_step_fixed.onnx)
        if (this.session.inputMetadata.audio_window) {
            return await this.generateWithStepModel(inputs, numFrames);
        } 
        // For full model (motion_generator.onnx)
        else {
            return await this.generateWithFullModel(inputs, numFrames);
        }
    }
    
    /**
     * Generate with single-step model
     */
    async generateWithStepModel(inputs, numFrames) {
        const { audioWindow, prevMotion, currentLexeme, hiddenState } = inputs;
        
        let currentMotion = new ort.Tensor('float32', new Float32Array(prevMotion), [1, 48]);
        let currentHiddenState = new ort.Tensor('float32', new Float32Array(hiddenState), [4, 1, 1024]);
        
        const audioWindowTensor = new ort.Tensor('float32', new Float32Array(audioWindow), [1, 80, 30]);
        const currentLexemeTensor = new ort.Tensor('float32', new Float32Array(currentLexeme), [1, 96]);
        
        const generatedFrames = [];
        const performanceMetrics = [];
        
        for (let step = 0; step < numFrames; step++) {
            const startTime = performance.now();
            
            console.log(`  🔄 Generating frame ${step + 1}/${numFrames}...`);
            
            const feeds = {
                audio_window: audioWindowTensor,
                prev_motion: currentMotion,
                current_lexeme: currentLexemeTensor,
                hidden_state: currentHiddenState
            };
            
            const results = await this.session.run(feeds);
            
            // Update for next iteration
            currentMotion = results.new_motion;
            currentHiddenState = results.new_hidden_state;
            
            // Store frame
            const motionData = Array.from(results.new_motion.data);
            generatedFrames.push(motionData);
            
            const endTime = performance.now();
            const frameTime = endTime - startTime;
            performanceMetrics.push(frameTime);
            
            if (step < 5 || step % 5 === 4) {
                console.log(`    ✨ Frame ${step + 1}: [${motionData.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...] (${frameTime.toFixed(1)}ms)`);
            }
        }
        
        return this.summarizeResults(generatedFrames, performanceMetrics);
    }
    
    /**
     * Generate with full model
     */
    async generateWithFullModel(inputs, numFrames) {
        const { audInput, moInput, lxmInput } = inputs;
        
        const audInputTensor = new ort.Tensor('float32', new Float32Array(audInput), [1, 80, 100]);
        const moInputTensor = new ort.Tensor('float32', new Float32Array(moInput), [1, 48, 100]);
        const lxmInputTensor = new ort.Tensor('float32', new Float32Array(lxmInput), [1, 96, 10]);
        
        console.log('🚀 Running full model inference...');
        const startTime = performance.now();
        
        const feeds = {
            aud_input: audInputTensor,
            mo_input: moInputTensor,
            lxm_input: lxmInputTensor
        };
        
        const results = await this.session.run(feeds);
        const endTime = performance.now();
        
        const outputData = Array.from(results.output.data);
        const outputShape = results.output.dims; // [1, 48, 80]
        
        // Reshape output to frames
        const framesPerOutput = outputShape[2]; // 80 frames
        const motionDim = outputShape[1]; // 48 dimensions
        
        const generatedFrames = [];
        for (let frame = 0; frame < Math.min(framesPerOutput, numFrames); frame++) {
            const frameData = [];
            for (let dim = 0; dim < motionDim; dim++) {
                const index = dim * framesPerOutput + frame;
                frameData.push(outputData[index]);
            }
            generatedFrames.push(frameData);
        }
        
        const totalTime = endTime - startTime;
        console.log(`🎉 Generated ${generatedFrames.length} frames in ${totalTime.toFixed(1)}ms`);
        
        return {
            frames: generatedFrames,
            metrics: {
                totalTime,
                avgTime: totalTime / generatedFrames.length,
                fps: (generatedFrames.length * 1000) / totalTime,
                numFrames: generatedFrames.length
            }
        };
    }
    
    /**
     * Create sample inputs for testing
     */
    createSampleInputs(modelType = 'step') {
        if (modelType === 'step') {
            return {
                audioWindow: Array.from({length: 2400}, () => Math.random() * 0.1), // [80 * 30]
                prevMotion: Array.from({length: 48}, () => Math.random() * 0.05),
                currentLexeme: Array.from({length: 96}, () => Math.random() * 0.02),
                hiddenState: Array.from({length: 4096}, () => 0.0) // [4 * 1 * 1024]
            };
        } else {
            return {
                audInput: Array.from({length: 8000}, () => Math.random() * 0.1), // [80 * 100]
                moInput: Array.from({length: 4800}, () => Math.random() * 0.05), // [48 * 100]
                lxmInput: Array.from({length: 960}, () => Math.random() * 0.02)   // [96 * 10]
            };
        }
    }
    
    /**
     * Summarize generation results
     */
    summarizeResults(generatedFrames, performanceMetrics) {
        const totalTime = performanceMetrics.reduce((a, b) => a + b, 0);
        const avgTime = totalTime / performanceMetrics.length;
        
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
                numFrames: generatedFrames.length
            }
        };
    }
    
    /**
     * Get cache information
     */
    getCacheInfo() {
        return this.loader.getCacheInfo();
    }
    
    /**
     * Clear model cache
     */
    clearCache() {
        this.loader.clearCache();
    }
}

module.exports = { Audio2GestureChunkedGenerator };
