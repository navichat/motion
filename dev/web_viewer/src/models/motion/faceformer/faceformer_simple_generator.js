const ort = require('onnxruntime-web');
const fs = require('fs');

class FaceFormerSimpleGenerator {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
        this.maxSeqLen = 20;
    }

    async initialize() {
        console.log('Loading FaceFormer simple step model...');
        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['cpu']
        });
        console.log('Model loaded successfully!');
        
        // Load sample data to get max_seq_len
        const sampleData = JSON.parse(fs.readFileSync('./faceformer_simple_sample_data.json', 'utf8'));
        this.maxSeqLen = sampleData.max_seq_len;
        console.log(`Max sequence length: ${this.maxSeqLen}`);
    }

    async generateSequence(audioFeatures, template, oneHot, maxFrames = 10) {
        if (!this.session) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        console.log('Starting simplified autoregressive generation...');
        
        const batchSize = 1;
        const featureDim = 64;
        
        // Initialize fixed-size sequence buffer with zeros
        let sequenceBuffer = new Array(batchSize * this.maxSeqLen * featureDim).fill(0);
        
        // Initialize with style embedding (first position)
        // For simplicity, we'll start with the first embedding from sample data
        const sampleData = JSON.parse(fs.readFileSync('./faceformer_simple_sample_data.json', 'utf8'));
        const initialSeq = sampleData.vertice_sequence[0]; // Get the first batch
        
        // Copy initial sequence to buffer
        for (let i = 0; i < Math.min(initialSeq.length, this.maxSeqLen); i++) {
            for (let j = 0; j < featureDim; j++) {
                const bufferIdx = i * featureDim + j;
                sequenceBuffer[bufferIdx] = initialSeq[i][j];
            }
        }
        
        let currentLength = 5; // Start with initial length from sample
        const generatedVertices = [];
        
        // Prepare static inputs using correct data structure understanding
        const audioSeqLen = audioFeatures[0].length;        // 100
        const audioFeatureDim = audioFeatures[0][0].length;  // 768
        const templateSeqLen = template[0].length;           // 1
        const templateVertexDim = template[0][0].length;     // 15069
        const oneHotDim = oneHot[0].length;                  // 3
        
        console.log(`Input tensor shapes:`);
        console.log(`  Audio: [${batchSize}, ${audioSeqLen}, ${audioFeatureDim}]`);
        console.log(`  Template: [${batchSize}, ${templateSeqLen}, ${templateVertexDim}]`);
        console.log(`  OneHot: [${batchSize}, ${oneHotDim}]`);
        
        const audioTensor = new ort.Tensor('float32', 
            new Float32Array(audioFeatures.flat(2)), 
            [batchSize, audioSeqLen, audioFeatureDim]);
        
        const oneHotTensor = new ort.Tensor('float32', 
            new Float32Array(oneHot.flat()), 
            [batchSize, oneHotDim]);
        
        const templateTensor = new ort.Tensor('float32', 
            new Float32Array(template.flat(2)), 
            [batchSize, templateSeqLen, templateVertexDim]);
        
        for (let step = 0; step < maxFrames && currentLength < this.maxSeqLen; step++) {
            console.log(`Generation step ${step + 1}/${maxFrames} (current length: ${currentLength})`);
            
            try {
                // Create tensors for this step
                const sequenceTensor = new ort.Tensor('float32', 
                    new Float32Array(sequenceBuffer), 
                    [batchSize, this.maxSeqLen, featureDim]);
                
                const lengthTensor = new ort.Tensor('int64', 
                    new BigInt64Array([BigInt(currentLength)]), 
                    [batchSize, 1]);
                
                // Run inference
                const feeds = {
                    audio_features: audioTensor,
                    vertice_sequence: sequenceTensor,
                    current_length: lengthTensor,
                    one_hot: oneHotTensor,
                    template: templateTensor
                };
                
                console.log(`  Input shapes: seq=[${sequenceTensor.dims}], len=[${lengthTensor.dims}]`);
                
                const results = await this.session.run(feeds);
                
                // Extract results
                const newVerticeOut = Array.from(results.new_vertice_out.data);
                const updatedSequence = Array.from(results.updated_sequence.data);
                const newLength = Number(results.new_length.data[0]);
                
                console.log(`  Output shapes: new_out=${results.new_vertice_out.dims}, new_len=${newLength}`);
                console.log(`  Generated vertex sample:`, newVerticeOut.slice(0, 5));
                
                // Store the generated vertex
                generatedVertices.push(newVerticeOut);
                
                // Update sequence buffer and length
                sequenceBuffer = updatedSequence;
                currentLength = newLength;
                
                // Check if we've reached max length
                if (currentLength >= this.maxSeqLen) {
                    console.log('Reached maximum sequence length');
                    break;
                }
                
            } catch (error) {
                console.error(`Error in generation step ${step + 1}:`, error);
                console.error('Error details:', error.message);
                break;
            }
        }
        
        console.log('Generation complete!');
        return generatedVertices;
    }
}

async function testSimpleGeneration() {
    try {
        // Load sample data
        const sampleData = JSON.parse(fs.readFileSync('./faceformer_simple_sample_data.json', 'utf8'));
        
        const generator = new FaceFormerSimpleGenerator('./faceformer_simple_step.onnx');
        await generator.initialize();
        
        console.log('\nUsing simplified approach...');
        
        // Generate sequence
        const generated = await generator.generateSequence(
            sampleData.audio_features,
            sampleData.template, 
            sampleData.one_hot,
            5  // Generate 5 steps
        );
        
        console.log('\n=== RESULTS ===');
        console.log('Generated sequence length:', generated.length);
        if (generated.length > 0) {
            console.log('First frame sample:', generated[0].slice(0, 10));
            if (generated.length > 1) {
                console.log('Last frame sample:', generated[generated.length - 1].slice(0, 10));
            }
            console.log('✅ Multi-step FaceFormer generation working!');
            
            // Save results
            const results = {
                generated_frames: generated,
                num_frames: generated.length,
                frame_size: generated[0].length
            };
            
            fs.writeFileSync('generation_results.json', JSON.stringify(results, null, 2));
            console.log('Results saved to generation_results.json');
            
        } else {
            console.log('❌ No frames generated');
        }
        
    } catch (error) {
        console.error('Generation failed:', error);
        console.error('Stack trace:', error.stack);
    }
}

module.exports = { FaceFormerSimpleGenerator };

if (require.main === module) {
    testSimpleGeneration();
}
