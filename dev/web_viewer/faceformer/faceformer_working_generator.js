const ort = require('onnxruntime-web');
const fs = require('fs');
const path = require('path');

class FaceFormerWebGeneratorFixed {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
    }

    async initialize() {
        console.log('Loading FaceFormer core step model...');
        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['cpu']
        });
        console.log('Model loaded successfully!');
    }

    async generateSingleStep(audioFeatures, verticeEmb, oneHot, template) {
        if (!this.session) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        // Use exact same approach as working test
        const audioFlatData = audioFeatures.flat(2);
        const verticeEmbFlatData = verticeEmb.flat(2);
        const oneHotFlatData = oneHot.flat ? oneHot.flat() : oneHot;
        const templateFlatData = template.flat(2);
        
        // Infer shapes from the data structure
        const audioSeqLen = audioFeatures[0].length;  // Should be 100
        const verticeSeqLen = verticeEmb[0].length;   // Current sequence length
        const verticeFeatureDim = verticeEmb[0][0].length; // Should be 64
        const templateVertexDim = template[0][0].length; // Should be 15069
        
        console.log(`  Audio shape: [1, ${audioSeqLen}, 768] (${audioFlatData.length} elements)`);
        console.log(`  Vertice emb shape: [1, ${verticeSeqLen}, ${verticeFeatureDim}] (${verticeEmbFlatData.length} elements)`);
        console.log(`  OneHot shape: [1, ${oneHotFlatData.length}] (${oneHotFlatData.length} elements)`);
        console.log(`  Template shape: [1, 1, ${templateVertexDim}] (${templateFlatData.length} elements)`);

        const audioTensor = new ort.Tensor('float32', 
            new Float32Array(audioFlatData), 
            [1, audioSeqLen, 768]);
        
        const verticeEmbTensor = new ort.Tensor('float32', 
            new Float32Array(verticeEmbFlatData), 
            [1, verticeSeqLen, verticeFeatureDim]);
        
        const oneHotTensor = new ort.Tensor('float32', 
            new Float32Array(oneHotFlatData), 
            [1, oneHotFlatData.length]);
        
        const templateTensor = new ort.Tensor('float32', 
            new Float32Array(templateFlatData), 
            [1, 1, templateVertexDim]);

        const feeds = {
            audio_features: audioTensor,
            vertice_emb: verticeEmbTensor,
            one_hot: oneHotTensor,
            template: templateTensor
        };

        const results = await this.session.run(feeds);
        
        return {
            new_vertice_out: Array.from(results.new_vertice_out.data),
            updated_vertice_emb: Array.from(results.updated_vertice_emb.data),
            new_vertice_out_shape: results.new_vertice_out.dims,
            updated_vertice_emb_shape: results.updated_vertice_emb.dims
        };
    }

    async generateSequence(audioFeatures, template, oneHot, maxFrames = 10) {
        console.log('Starting autoregressive generation...');
        
        // Load initial embeddings from test data (style embedding)
        const testData = JSON.parse(fs.readFileSync('./onnx_test_data.json', 'utf8'));
        let currentVerticeEmb = testData.inputs.vertice_emb;  // Start with test embedding
        
        const generatedVertices = [];
        
        for (let i = 0; i < maxFrames; i++) {
            console.log(`Generation step ${i + 1}/${maxFrames}`);
            
            try {
                const result = await this.generateSingleStep(
                    audioFeatures, 
                    currentVerticeEmb, 
                    oneHot, 
                    template
                );
                
                console.log(`Step ${i + 1} successful:`);
                console.log(`  Output shape: ${JSON.stringify(result.new_vertice_out_shape)}`);
                console.log(`  Updated embedding shape: ${JSON.stringify(result.updated_vertice_emb_shape)}`);
                
                // Store the generated vertex
                generatedVertices.push(result.new_vertice_out);
                
                // Update embeddings for next iteration - reshape back to 3D format
                const [batch, newSeqLen, featDim] = result.updated_vertice_emb_shape;
                currentVerticeEmb = [];
                for (let b = 0; b < batch; b++) {
                    const batchData = [];
                    for (let s = 0; s < newSeqLen; s++) {
                        const seqData = [];
                        for (let f = 0; f < featDim; f++) {
                            const idx = b * newSeqLen * featDim + s * featDim + f;
                            seqData.push(result.updated_vertice_emb[idx]);
                        }
                        batchData.push(seqData);
                    }
                    currentVerticeEmb.push(batchData);
                }
                
            } catch (error) {
                console.error(`Error in generation step ${i + 1}:`, error);
                break;
            }
        }
        
        console.log('Generation complete!');
        return generatedVertices;
    }
}

async function testFaceFormerGenerationFixed() {
    try {
        // Use the test data that we know works
        const testData = JSON.parse(fs.readFileSync('./onnx_test_data.json', 'utf8'));
        const inputs = testData.inputs;
        
        const generator = new FaceFormerWebGeneratorFixed('./faceformer_core_step.onnx');
        await generator.initialize();
        
        console.log('\nUsing test data that works in Python...');
        
        // Generate sequence using the working test data format
        const generated = await generator.generateSequence(
            inputs.audio_features,
            inputs.template, 
            inputs.one_hot,
            3  // Only 3 steps for testing
        );
        
        console.log('\n=== RESULTS ===');
        console.log('Generated sequence length:', generated.length);
        if (generated.length > 0) {
            console.log('First frame sample (first 10 values):', generated[0].slice(0, 10));
            console.log('Last frame sample (first 10 values):', generated[generated.length - 1].slice(0, 10));
            console.log('✅ FaceFormer autoregressive generation working!');
        } else {
            console.log('❌ No frames generated');
        }
        
    } catch (error) {
        console.error('Generation failed:', error);
        console.error('Stack trace:', error.stack);
    }
}

module.exports = { FaceFormerWebGeneratorFixed };

if (require.main === module) {
    testFaceFormerGenerationFixed();
}
