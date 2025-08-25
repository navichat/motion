// JavaScript implementation for autoregressive generation with the core step model
// Browser-compatible version - ONNX Runtime loaded globally via CDN

class FaceFormerWebGenerator {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
    }

    async initialize() {
        console.log('Loading FaceFormer core step model...');
        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['cpu']  // Force CPU backend
        });
        console.log('Model loaded successfully!');
        
        // Print model info
        console.log('\nModel inputs:');
        for (const input of this.session.inputMetadata) {
            console.log(`  ${input.name}: ${JSON.stringify(input.shape)} (${input.type})`);
        }
    }

    async generateSequence(audioFeatures, template, oneHot, maxFrames = 100) {
        if (!this.session) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        console.log('Starting autoregressive generation...');
        
        // Initialize with style embedding
        const batchSize = 1;
        const featureDim = 64;
        const numSubjects = 3;
        const audioSeqLen = Math.floor(audioFeatures.length / 768);
        const vertexDim = template.length;
        
        console.log('Tensor shapes:');
        console.log(`Audio: [${batchSize}, ${audioSeqLen}, 768]`);
        console.log(`Template: [${batchSize}, 1, ${vertexDim}]`);
        console.log(`OneHot: [${batchSize}, ${numSubjects}]`);
        
        // Convert inputs to ONNX tensors
        const audioTensor = new ort.Tensor('float32', new Float32Array(audioFeatures), [batchSize, audioSeqLen, 768]);
        const templateTensor = new ort.Tensor('float32', new Float32Array(template), [batchSize, 1, vertexDim]);
        const oneHotTensor = new ort.Tensor('float32', new Float32Array(oneHot), [batchSize, numSubjects]);
        
        // Initialize vertex embeddings with style embedding (using first embedding from sample data)
        // For the first step, we need a [1, 1, 64] embedding
        let currentVerticeEmb = new Array(batchSize * 1 * featureDim).fill(0.1);
        
        const generatedVertices = [];
        
        for (let i = 0; i < maxFrames; i++) {
            console.log(`Generation step ${i + 1}/${maxFrames}`);
            
            const currentSeqLen = i + 1;
            const verticeEmbTensor = new ort.Tensor('float32', new Float32Array(currentVerticeEmb), [batchSize, currentSeqLen, featureDim]);
            
            console.log(`Current embedding shape: [${batchSize}, ${currentSeqLen}, ${featureDim}]`);
            
            // Run one step of generation
            const feeds = {
                audio_features: audioTensor,
                vertice_emb: verticeEmbTensor,
                one_hot: oneHotTensor,
                template: templateTensor
            };
            
            try {
                const results = await this.session.run(feeds);
                
                console.log('Available outputs:', Object.keys(results));
                
                // Check if outputs exist
                if (!results.new_vertice_out || !results.updated_vertice_emb) {
                    console.error('Missing expected outputs from model');
                    console.log('Available outputs:', Object.keys(results));
                    break;
                }
                
                // Extract results
                const newVerticeOut = Array.from(results.new_vertice_out.data);
                const updatedVerticeEmb = Array.from(results.updated_vertice_emb.data);
                
                console.log(`Output shapes: new_out=${results.new_vertice_out.dims}, updated_emb=${results.updated_vertice_emb.dims}`);
                
                // Store the generated vertex
                generatedVertices.push(newVerticeOut);
                
                // Update embeddings for next iteration
                currentVerticeEmb = updatedVerticeEmb;
                
                // Optional: Add stopping condition based on some criteria
                // if (shouldStop(newVerticeOut)) break;
                
            } catch (error) {
                console.error(`Error in generation step ${i + 1}:`, error.message);
                console.error('Full error:', error);
                break;
            }
        }
        
        console.log('Generation complete!');
        return generatedVertices;
    }
}

// Alternative approach: Pre-process audio with a separate model
class FaceFormerPreprocessor {
    constructor(audioModelPath) {
        this.audioModelPath = audioModelPath;
        this.audioSession = null;
    }

    async initialize() {
        // You would export the audio processing part separately
        // this.audioSession = await ort.InferenceSession.create(this.audioModelPath);
    }

    async processAudio(rawAudio, sampleRate = 16000) {
        // Pre-process audio to features
        // This would replace the Wav2Vec2 processing
        // For now, return dummy features
        const audioLength = rawAudio.length;
        const featuresLength = Math.floor(audioLength / 320); // Rough downsampling
        const features = new Array(featuresLength * 768).fill(0).map(() => Math.random() * 0.1);
        return features;
    }
}

// Usage example
async function testFaceFormerGeneration() {
    try {
        const generator = new FaceFormerWebGenerator('./faceformer_core_step.onnx');
        await generator.initialize();
        
        // Load sample data
        // Note: In browser environment, sample data would need to be loaded via fetch()
        // const response = await fetch('./faceformer_sample_data.json');
        // const sampleData = await response.json();
        
        // For now, using placeholder data
        const sampleData = {
            core_step: {
                audio_features: [[[0.1, 0.2]]],
                template: [[[0.3, 0.4]]],
                one_hot: [[1, 0, 0]]
            }
        };
        const coreStepData = sampleData.core_step;
        
        // Extract flat arrays for the first step
        const audioFeatures = coreStepData.audio_features.flat(2);
        const template = coreStepData.template.flat(2);
        const oneHot = coreStepData.one_hot.flat();
        
        console.log('Input shapes:');
        console.log('Audio features length:', audioFeatures.length);
        console.log('Template length:', template.length);
        console.log('One hot length:', oneHot.length);
        
        // Generate sequence
        const generated = await generator.generateSequence(audioFeatures, template, oneHot, 3); // Only 3 steps for testing
        
        console.log('Generated sequence length:', generated.length);
        if (generated.length > 0) {
            console.log('First frame sample:', generated[0].slice(0, 10));
            console.log('Generation successful!');
        } else {
            console.log('No frames generated');
        }
        
    } catch (error) {
        console.error('Generation failed:', error);
        console.error('Stack trace:', error.stack);
    }
}

// Export for browser global usage
if (typeof window !== 'undefined') {
    window.FaceFormerWebGenerator = FaceFormerWebGenerator;
    window.FaceFormerPreprocessor = FaceFormerPreprocessor;
    window.testFaceFormerGeneration = testFaceFormerGeneration;
}

// Also export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FaceFormerWebGenerator, FaceFormerPreprocessor, testFaceFormerGeneration };
}

// For direct execution in Node.js
if (typeof require !== 'undefined' && require.main === module) {
    testFaceFormerGeneration();
}
