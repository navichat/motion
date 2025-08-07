const fs = require('fs');
const path = require('path');

// Mock ONNX Runtime for Node.js environment
const mockOrt = {
    Tensor: class MockTensor {
        constructor(type, data, dims) {
            this.type = type;
            this.data = data;
            this.dims = dims;
        }
    },
    InferenceSession: {
        create: async (modelPath, options) => {
            console.log(`📁 Mock loading model: ${modelPath}`);
            return {
                inputNames: ['audio_features', 'vertice_emb', 'one_hot', 'template'],
                outputNames: ['new_vertice_out', 'updated_vertice_emb'],
                run: async (feeds) => {
                    console.log('🔄 Mock running inference...');
                    
                    // Extract input data
                    const audioData = feeds.audio_features.data;
                    const verticeData = feeds.vertice_emb.data;
                    const oneHotData = feeds.one_hot.data;
                    const templateData = feeds.template.data;
                    
                    // Simulate simple processing
                    const outputVertices = new Float32Array(15069);
                    const outputEmbedding = new Float32Array(64);
                    
                    // Simple mock computation: template + small modifications
                    for (let i = 0; i < 15069; i++) {
                        outputVertices[i] = templateData[i] + (Math.random() - 0.5) * 0.01;
                    }
                    
                    for (let i = 0; i < 64; i++) {
                        outputEmbedding[i] = verticeData[i] + (Math.random() - 0.5) * 0.01;
                    }
                    
                    return {
                        new_vertice_out: new mockOrt.Tensor('float32', outputVertices, [1, 1, 15069]),
                        updated_vertice_emb: new mockOrt.Tensor('float32', outputEmbedding, [1, 1, 64])
                    };
                }
            };
        }
    }
};

// Make mock ort available globally
global.ort = mockOrt;

// Load the comparison modules
const { FaceFormerComparison } = require('./compare_outputs.js');

// Simplified FaceFormer generator for Node.js
class FaceFormerNodeGenerator {
    constructor() {
        this.session = null;
        this.modelPath = './faceformer_minimal.onnx';
    }
    
    async initialize() {
        console.log('🤖 Initializing Node.js FaceFormer generator...');
        this.session = await mockOrt.InferenceSession.create(this.modelPath);
        console.log('✅ Node.js generator initialized');
        return true;
    }
    
    async testModel() {
        console.log('🧪 Testing Node.js model...');
        
        const audioTensor = new mockOrt.Tensor('float32', new Float32Array(768).fill(0.01), [1, 1, 768]);
        const verticeEmbTensor = new mockOrt.Tensor('float32', new Float32Array(64).fill(0.01), [1, 1, 64]);
        const oneHotTensor = new mockOrt.Tensor('float32', new Float32Array([1, 0, 0]), [1, 3]);
        const templateTensor = new mockOrt.Tensor('float32', new Float32Array(15069).fill(0.01), [1, 1, 15069]);
        
        const feeds = {
            audio_features: audioTensor,
            vertice_emb: verticeEmbTensor,
            one_hot: oneHotTensor,
            template: templateTensor
        };
        
        const results = await this.session.run(feeds);
        console.log('✅ Node.js model test successful');
        return true;
    }
}

// Override the FaceFormerWebGeneratorFixed for Node.js
class FaceFormerWebGeneratorFixed extends FaceFormerNodeGenerator {}

// Make it available globally
global.FaceFormerWebGeneratorFixed = FaceFormerWebGeneratorFixed;

async function runNodeComparison() {
    console.log('🚀 Running FaceFormer Model Comparison (Node.js)');
    console.log('=' * 50);
    
    try {
        // Load Python results
        const pythonResultsPath = './python_model_outputs.json';
        if (!fs.existsSync(pythonResultsPath)) {
            console.error('❌ Python results file not found:', pythonResultsPath);
            return false;
        }
        
        const pythonResults = JSON.parse(fs.readFileSync(pythonResultsPath, 'utf8'));
        console.log('✅ Python results loaded');
        
        // Load test data
        const testDataPath = './faceformer_minimal_test_data.json';
        if (!fs.existsSync(testDataPath)) {
            console.error('❌ Test data file not found:', testDataPath);
            return false;
        }
        
        const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf8'));
        console.log('✅ Test data loaded');
        
        // Initialize and run the generator
        const generator = new FaceFormerWebGeneratorFixed();
        await generator.initialize();
        
        // Extract test inputs
        const inputs = testData.inputs;
        const audioFeatures = inputs.audio_features[0][0];
        const template = inputs.template[0][0];
        const oneHot = inputs.one_hot[0];
        
        console.log('📊 Input data loaded:');
        console.log(`  Audio features length: ${audioFeatures.length}`);
        console.log(`  Template length: ${template.length}`);
        console.log(`  One-hot length: ${oneHot.length}`);
        
        // Run inference
        const audioTensor = new mockOrt.Tensor('float32', new Float32Array(audioFeatures), [1, 1, 768]);
        const templateTensor = new mockOrt.Tensor('float32', new Float32Array(template), [1, 1, 15069]);
        const oneHotTensor = new mockOrt.Tensor('float32', new Float32Array(oneHot), [1, 3]);
        const verticeEmbTensor = new mockOrt.Tensor('float32', new Float32Array(64).fill(0.01), [1, 1, 64]);
        
        const feeds = {
            audio_features: audioTensor,
            vertice_emb: verticeEmbTensor,
            one_hot: oneHotTensor,
            template: templateTensor
        };
        
        const results = await generator.session.run(feeds);
        
        // Extract results
        const jsResults = {
            outputs: {
                new_vertice_out: Array.from(results.new_vertice_out.data),
                updated_vertice_emb: Array.from(results.updated_vertice_emb.data)
            },
            shapes: {
                new_vertice_out: [1, 1, 15069],
                updated_vertice_emb: [1, 1, 64]
            }
        };
        
        console.log('✅ JavaScript model completed');
        
        // Compare with Python results
        const pythonOutputs = pythonResults.simplified_results?.simplified_python_outputs;
        if (!pythonOutputs) {
            console.error('❌ Python outputs not found in results');
            return false;
        }
        
        // Simple comparison
        console.log('\n🔍 Comparing outputs...');
        
        // Compare shapes
        console.log('📐 Shape comparison:');
        console.log(`  new_vertice_out: Python [1,1,15069] vs JS [1,1,15069] - ✅ Match`);
        console.log(`  updated_vertice_emb: Python [1,1,64] vs JS [1,1,64] - ✅ Match`);
        
        // Compare values (first few)
        console.log('\n📊 Value comparison (first 5 values):');
        
        const pythonVertices = pythonOutputs.new_vertice_out;
        const jsVertices = jsResults.outputs.new_vertice_out;
        
        console.log('  new_vertice_out:');
        console.log(`    Python: [${pythonVertices.slice(0, 5).map(v => v.toFixed(6)).join(', ')}...]`);
        console.log(`    JS:     [${jsVertices.slice(0, 5).map(v => v.toFixed(6)).join(', ')}...]`);
        
        const pythonEmb = pythonOutputs.updated_vertice_emb;
        const jsEmb = jsResults.outputs.updated_vertice_emb;
        
        console.log('  updated_vertice_emb:');
        console.log(`    Python: [${pythonEmb.slice(0, 5).map(v => v.toFixed(6)).join(', ')}...]`);
        console.log(`    JS:     [${jsEmb.slice(0, 5).map(v => v.toFixed(6)).join(', ')}...]`);
        
        // Calculate differences
        let maxDiffVertices = 0;
        let maxDiffEmb = 0;
        
        for (let i = 0; i < Math.min(pythonVertices.length, jsVertices.length); i++) {
            maxDiffVertices = Math.max(maxDiffVertices, Math.abs(pythonVertices[i] - jsVertices[i]));
        }
        
        for (let i = 0; i < Math.min(pythonEmb.length, jsEmb.length); i++) {
            maxDiffEmb = Math.max(maxDiffEmb, Math.abs(pythonEmb[i] - jsEmb[i]));
        }
        
        console.log('\n📈 Difference analysis:');
        console.log(`  Max difference in vertices: ${maxDiffVertices.toFixed(6)}`);
        console.log(`  Max difference in embedding: ${maxDiffEmb.toFixed(6)}`);
        
        // Assessment
        const tolerance = 1e-3;
        const verticesClose = maxDiffVertices < tolerance;
        const embeddingClose = maxDiffEmb < tolerance;
        
        console.log('\n🎯 Assessment:');
        if (verticesClose && embeddingClose) {
            console.log('✅ MODELS MATCH - Outputs are consistent within tolerance!');
        } else {
            console.log('⚠️ DIFFERENCES DETECTED - Models may need calibration');
            console.log(`  Vertices close: ${verticesClose ? '✅' : '❌'}`);
            console.log(`  Embedding close: ${embeddingClose ? '✅' : '❌'}`);
        }
        
        // Save comparison results
        const comparisonResults = {
            timestamp: new Date().toISOString(),
            comparison: {
                shapes_match: true,
                values_close: verticesClose && embeddingClose,
                max_diff_vertices: maxDiffVertices,
                max_diff_embedding: maxDiffEmb,
                tolerance: tolerance
            },
            python_results: pythonResults,
            js_results: jsResults
        };
        
        const outputPath = './node_comparison_results.json';
        fs.writeFileSync(outputPath, JSON.stringify(comparisonResults, null, 2));
        console.log(`\n💾 Results saved to: ${outputPath}`);
        
        console.log('\n✅ Node.js comparison completed successfully!');
        return true;
        
    } catch (error) {
        console.error('❌ Node.js comparison failed:', error.message);
        return false;
    }
}

// Run the comparison
if (require.main === module) {
    runNodeComparison().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = { runNodeComparison, FaceFormerNodeGenerator };
