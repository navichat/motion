const fs = require('fs');
const { FaceFormerWeightLoader, SimpleFaceFormerJS } = require('./weight_loader.cjs');

// Mock ONNX Runtime for Node.js environment with exact weight loading capability
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
            console.log(`📁 Loading model: ${modelPath}`);
            
            // Check if this is the exact weights model
            const isExactWeights = modelPath.includes('python_weights.onnx');
            
            if (isExactWeights && fs.existsSync(modelPath)) {
                console.log('✅ Using ONNX model with exact Python weights');
                return {
                    inputNames: ['audio_features', 'vertice_emb', 'one_hot', 'template'],
                    outputNames: ['new_vertice_out', 'updated_vertice_emb'],
                    useExactWeights: true,
                    run: async (feeds) => {
                        console.log('🔄 Running inference with exact weights...');
                        
                        // Load the Python weights and run exact computation
                        const weightLoader = new FaceFormerWeightLoader();
                        await weightLoader.loadPythonWeights('./faceformer_python_weights.json');
                        
                        // Extract input data
                        const audioData = Array.from(feeds.audio_features.data);
                        const verticeData = Array.from(feeds.vertice_emb.data);
                        const oneHotData = Array.from(feeds.one_hot.data);
                        const templateData = Array.from(feeds.template.data);
                        
                        // Run exact computation using loaded weights
                        const model = new SimpleFaceFormerJS();
                        weightLoader.applyToCustomModel(model);
                        
                        const inputs = {
                            audio_features: audioData,
                            vertice_emb: verticeData,
                            one_hot: oneHotData,
                            template: templateData
                        };
                        
                        const outputs = model.forward(inputs);
                        
                        return {
                            new_vertice_out: new mockOrt.Tensor('float32', new Float32Array(outputs.vertices), [1, 1, 15069]),
                            updated_vertice_emb: new mockOrt.Tensor('float32', new Float32Array(outputs.embedding), [1, 1, 64])
                        };
                    }
                };
            } else {
                // Fallback to original mock behavior
                console.log('⚠️ Using mock model (weights not available)');
                return {
                    inputNames: ['audio_features', 'vertice_emb', 'one_hot', 'template'],
                    outputNames: ['new_vertice_out', 'updated_vertice_emb'],
                    useExactWeights: false,
                    run: async (feeds) => {
                        console.log('🔄 Mock running inference...');
                        
                        const templateData = feeds.template.data;
                        
                        // Simple mock computation
                        const outputVertices = new Float32Array(15069);
                        const outputEmbedding = new Float32Array(64);
                        
                        for (let i = 0; i < 15069; i++) {
                            outputVertices[i] = templateData[i] + (Math.random() - 0.5) * 0.01;
                        }
                        
                        for (let i = 0; i < 64; i++) {
                            outputEmbedding[i] = (Math.random() - 0.5) * 0.01;
                        }
                        
                        return {
                            new_vertice_out: new mockOrt.Tensor('float32', outputVertices, [1, 1, 15069]),
                            updated_vertice_emb: new mockOrt.Tensor('float32', outputEmbedding, [1, 1, 64])
                        };
                    }
                };
            }
        }
    }
};

// Make mock ort available globally
global.ort = mockOrt;

// Updated FaceFormer generator for Node.js with weight loading
class FaceFormerNodeGeneratorWithWeights {
    constructor() {
        this.session = null;
        this.modelPath = './faceformer_python_weights.onnx';
        this.fallbackPath = './faceformer_minimal.onnx';
        this.useExactWeights = false;
    }
    
    async initialize() {
        console.log('🤖 Initializing FaceFormer with exact Python weights...');
        
        // Try to load model with exact weights first
        if (fs.existsSync(this.modelPath)) {
            this.session = await mockOrt.InferenceSession.create(this.modelPath);
            this.useExactWeights = this.session.useExactWeights;
            console.log('✅ Initialized with exact Python weights');
        } else {
            // Fallback to original model
            this.session = await mockOrt.InferenceSession.create(this.fallbackPath);
            this.useExactWeights = false;
            console.log('⚠️ Initialized with fallback model (no exact weights)');
        }
        
        return true;
    }
    
    async testModel() {
        console.log('🧪 Testing model with weight loading...');
        
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
        console.log(`✅ Model test successful (exact weights: ${this.useExactWeights})`);
        return true;
    }
}

// Make it available globally
global.FaceFormerWebGeneratorFixed = FaceFormerNodeGeneratorWithWeights;

async function runExactWeightComparison() {
    console.log('🚀 Running FaceFormer Exact Weight Comparison');
    console.log('=' * 50);
    
    try {
        // Load Python results for reference
        const pythonResultsPath = './python_model_outputs.json';
        if (!fs.existsSync(pythonResultsPath)) {
            console.error('❌ Python results file not found:', pythonResultsPath);
            return false;
        }
        
        const pythonResults = JSON.parse(fs.readFileSync(pythonResultsPath, 'utf8'));
        const pythonOutputs = pythonResults.simplified_results?.simplified_python_outputs;
        
        if (!pythonOutputs) {
            console.error('❌ Python outputs not found in results');
            return false;
        }
        
        console.log('✅ Python reference outputs loaded');
        
        // Load test data
        const testDataPath = './faceformer_minimal_test_data.json';
        const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf8'));
        
        // Initialize generator with exact weights
        const generator = new FaceFormerNodeGeneratorWithWeights();
        await generator.initialize();
        
        // Prepare the exact same inputs as Python
        const inputs = testData.inputs;
        const audioFeatures = inputs.audio_features[0][0];
        const template = inputs.template[0][0];
        const oneHot = inputs.one_hot[0];
        const verticeEmb = inputs.vertice_emb[0][0]; // Use the exact embedding from test data
        
        console.log('📊 Input data prepared (exact match with Python):');
        console.log(`  Audio features length: ${audioFeatures.length}`);
        console.log(`  Template length: ${template.length}`);
        console.log(`  One-hot length: ${oneHot.length}`);
        console.log(`  Vertice embedding length: ${verticeEmb.length}`);
        
        // Run inference with exact weights
        const audioTensor = new mockOrt.Tensor('float32', new Float32Array(audioFeatures), [1, 1, 768]);
        const templateTensor = new mockOrt.Tensor('float32', new Float32Array(template), [1, 1, 15069]);
        const oneHotTensor = new mockOrt.Tensor('float32', new Float32Array(oneHot), [1, 3]);
        const verticeEmbTensor = new mockOrt.Tensor('float32', new Float32Array(verticeEmb), [1, 1, 64]);
        
        const feeds = {
            audio_features: audioTensor,
            vertice_emb: verticeEmbTensor,
            one_hot: oneHotTensor,
            template: templateTensor
        };
        
        console.log('🔄 Running inference with exact Python weights...');
        const results = await generator.session.run(feeds);
        
        // Extract results
        const jsResults = {
            new_vertice_out: Array.from(results.new_vertice_out.data),
            updated_vertice_emb: Array.from(results.updated_vertice_emb.data)
        };
        
        console.log('✅ JavaScript inference completed with exact weights');
        
        // Compare with Python outputs (should be EXACTLY the same)
        console.log('\n🔍 EXACT WEIGHT COMPARISON:');
        
        // Compare shapes
        console.log('📐 Shape comparison:');
        console.log(`  new_vertice_out: Python ${pythonOutputs.new_vertice_out.length} vs JS ${jsResults.new_vertice_out.length} - ${pythonOutputs.new_vertice_out.length === jsResults.new_vertice_out.length ? '✅' : '❌'}`);
        console.log(`  updated_vertice_emb: Python ${pythonOutputs.updated_vertice_emb.length} vs JS ${jsResults.updated_vertice_emb.length} - ${pythonOutputs.updated_vertice_emb.length === jsResults.updated_vertice_emb.length ? '✅' : '❌'}`);
        
        // Compare exact values
        let maxVertexDiff = 0;
        let maxEmbDiff = 0;
        let vertexMatches = 0;
        let embMatches = 0;
        
        // Vertex comparison
        for (let i = 0; i < Math.min(pythonOutputs.new_vertice_out.length, jsResults.new_vertice_out.length); i++) {
            const diff = Math.abs(pythonOutputs.new_vertice_out[i] - jsResults.new_vertice_out[i]);
            maxVertexDiff = Math.max(maxVertexDiff, diff);
            if (diff < 1e-10) vertexMatches++;
        }
        
        // Embedding comparison
        for (let i = 0; i < Math.min(pythonOutputs.updated_vertice_emb.length, jsResults.updated_vertice_emb.length); i++) {
            const diff = Math.abs(pythonOutputs.updated_vertice_emb[i] - jsResults.updated_vertice_emb[i]);
            maxEmbDiff = Math.max(maxEmbDiff, diff);
            if (diff < 1e-10) embMatches++;
        }
        
        console.log('\n📊 EXACT VALUE COMPARISON:');
        console.log(`  Max vertex difference: ${maxVertexDiff.toExponential(3)}`);
        console.log(`  Max embedding difference: ${maxEmbDiff.toExponential(3)}`);
        console.log(`  Exact vertex matches: ${vertexMatches}/${pythonOutputs.new_vertice_out.length} (${(vertexMatches/pythonOutputs.new_vertice_out.length*100).toFixed(1)}%)`);
        console.log(`  Exact embedding matches: ${embMatches}/${pythonOutputs.updated_vertice_emb.length} (${(embMatches/pythonOutputs.updated_vertice_emb.length*100).toFixed(1)}%)`);
        
        // Sample value comparison
        console.log('\n📊 Sample value comparison (first 5):');
        console.log('  Vertices:');
        console.log(`    Python: [${pythonOutputs.new_vertice_out.slice(0, 5).map(v => v.toFixed(8)).join(', ')}...]`);
        console.log(`    JS:     [${jsResults.new_vertice_out.slice(0, 5).map(v => v.toFixed(8)).join(', ')}...]`);
        console.log('  Embedding:');
        console.log(`    Python: [${pythonOutputs.updated_vertice_emb.slice(0, 5).map(v => v.toFixed(8)).join(', ')}...]`);
        console.log(`    JS:     [${jsResults.updated_vertice_emb.slice(0, 5).map(v => v.toFixed(8)).join(', ')}...]`);
        
        // Final assessment
        const perfectMatch = maxVertexDiff < 1e-10 && maxEmbDiff < 1e-10;
        const nearPerfectMatch = maxVertexDiff < 1e-6 && maxEmbDiff < 1e-6;
        
        console.log('\n🎯 FINAL ASSESSMENT:');
        if (perfectMatch) {
            console.log('🎉 PERFECT MATCH! - Weights transferred successfully');
            console.log('✅ Zero variance achieved - models are identical');
        } else if (nearPerfectMatch) {
            console.log('✅ NEAR PERFECT MATCH - Weights transferred with minimal numerical error');
            console.log('✅ Variance essentially eliminated');
        } else {
            console.log('⚠️ DIFFERENCES DETECTED - Weight transfer may have issues');
            console.log('🔧 Check weight loading implementation');
        }
        
        // Save exact comparison results
        const exactResults = {
            timestamp: new Date().toISOString(),
            exact_weight_comparison: true,
            python_results: pythonResults,
            js_results: jsResults,
            comparison: {
                max_vertex_diff: maxVertexDiff,
                max_embedding_diff: maxEmbDiff,
                vertex_matches: vertexMatches,
                embedding_matches: embMatches,
                perfect_match: perfectMatch,
                near_perfect_match: nearPerfectMatch,
                using_exact_weights: generator.useExactWeights
            }
        };
        
        fs.writeFileSync('./exact_weight_comparison.json', JSON.stringify(exactResults, null, 2));
        console.log('\n💾 Exact comparison results saved to: exact_weight_comparison.json');
        
        return perfectMatch || nearPerfectMatch;
        
    } catch (error) {
        console.error('❌ Exact weight comparison failed:', error.message);
        return false;
    }
}

// Run the exact weight comparison
if (require.main === module) {
    runExactWeightComparison().then(success => {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🏁 EXACT WEIGHT COMPARISON ${success ? 'SUCCESSFUL' : 'FAILED'}`);
        console.log(`${'='.repeat(60)}`);
        process.exit(success ? 0 : 1);
    });
}

module.exports = { runExactWeightComparison, FaceFormerNodeGeneratorWithWeights };
