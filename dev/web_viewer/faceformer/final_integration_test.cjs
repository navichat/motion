// Final Integration Test - Demonstrating Zero Variance Achievement
const fs = require('fs');
const { createEnhancedFaceFormer } = require('./faceformer_enhanced_generator.cjs');

// Mock ONNX Runtime for Node.js testing
const mockOrt = {
    Tensor: class MockTensor {
        constructor(type, data, dims) {
            this.type = type;
            this.data = data;
            this.dims = dims;
        }
    },
    InferenceSession: {
        create: async () => ({
            run: async () => {
                throw new Error('Mock ONNX - exact weights preferred');
            }
        })
    }
};

global.ort = mockOrt;
global.performance = {
    now: () => Date.now()
};

async function runFinalIntegrationTest() {
    console.log('🚀 Final Integration Test: Zero Variance Achievement');
    console.log('=' * 60);
    
    try {
        // Test 1: Initialize with exact weights
        console.log('\n📌 Test 1: Enhanced Generator Initialization');
        const generator = await createEnhancedFaceFormer({
            useExactWeights: true,
            pythonWeightsPath: './faceformer_python_weights.json'
        });
        
        const status = generator.getStatus();
        console.log('✅ Generator Status:', status);
        
        // Test 2: Load reference data
        console.log('\n📌 Test 2: Loading Reference Data');
        const pythonResults = JSON.parse(fs.readFileSync('./python_model_outputs.json', 'utf8'));
        const testData = JSON.parse(fs.readFileSync('./faceformer_minimal_test_data.json', 'utf8'));
        
        const pythonOutputs = pythonResults.simplified_results?.simplified_python_outputs;
        if (!pythonOutputs) {
            throw new Error('Python reference outputs not found');
        }
        
        console.log('✅ Reference data loaded');
        
        // Test 3: Generate with exact weights
        console.log('\n📌 Test 3: Generation with Exact Weights');
        const inputs = testData.inputs;
        const audioFeatures = inputs.audio_features[0][0];
        const verticeEmb = inputs.vertice_emb[0][0];
        const oneHot = inputs.one_hot[0];
        const template = inputs.template[0][0];
        
        const result = await generator.generateFrame(audioFeatures, verticeEmb, oneHot, template);
        console.log('✅ Frame generated with exact weights');
        console.log(`📊 Using exact weights: ${result.usingExactWeights}`);
        console.log(`🎯 Consistency level: ${result.consistency}`);
        
        // Test 4: Verify zero variance
        console.log('\n📌 Test 4: Zero Variance Verification');
        
        // Compare with Python reference
        let maxVertexDiff = 0;
        let maxEmbDiff = 0;
        let exactMatches = 0;
        
        for (let i = 0; i < result.vertices.length; i++) {
            const diff = Math.abs(result.vertices[i] - pythonOutputs.new_vertice_out[i]);
            maxVertexDiff = Math.max(maxVertexDiff, diff);
            if (diff < 1e-10) exactMatches++;
        }
        
        for (let i = 0; i < result.embedding.length; i++) {
            const diff = Math.abs(result.embedding[i] - pythonOutputs.updated_vertice_emb[i]);
            maxEmbDiff = Math.max(maxEmbDiff, diff);
        }
        
        console.log(`📊 Max vertex difference: ${maxVertexDiff.toExponential(3)}`);
        console.log(`📊 Max embedding difference: ${maxEmbDiff.toExponential(3)}`);
        console.log(`📊 Exact matches: ${exactMatches}/${result.vertices.length} (${(exactMatches/result.vertices.length*100).toFixed(1)}%)`);
        
        // Test 5: Performance benchmark
        console.log('\n📌 Test 5: Performance Benchmark');
        const benchmarkResults = await generator.benchmark(5);
        console.log(`⚡ Average generation time: ${benchmarkResults.averageMs.toFixed(2)}ms`);
        
        // Test 6: Consistency check
        console.log('\n📌 Test 6: Multi-run Consistency Check');
        const run1 = await generator.generateFrame(audioFeatures, verticeEmb, oneHot, template);
        const run2 = await generator.generateFrame(audioFeatures, verticeEmb, oneHot, template);
        
        let consistencyDiff = 0;
        for (let i = 0; i < run1.vertices.length; i++) {
            consistencyDiff = Math.max(consistencyDiff, Math.abs(run1.vertices[i] - run2.vertices[i]));
        }
        
        console.log(`🔄 Multi-run consistency: ${consistencyDiff === 0 ? 'Perfect' : consistencyDiff.toExponential(3)}`);
        
        // Final Assessment
        console.log('\n🎯 FINAL ASSESSMENT:');
        const zeroVariance = maxVertexDiff < 1e-6 && maxEmbDiff < 1e-6;
        const perfectConsistency = consistencyDiff === 0;
        
        if (zeroVariance && perfectConsistency) {
            console.log('🎉 SUCCESS: Zero variance achieved!');
            console.log('✅ JavaScript model produces identical outputs to Python');
            console.log('✅ Perfect consistency across multiple runs');
            console.log('✅ Enhanced generator fully functional');
        } else {
            console.log('⚠️ PARTIAL SUCCESS: Near-zero variance achieved');
            console.log(`  Variance level: ${Math.max(maxVertexDiff, maxEmbDiff).toExponential(3)}`);
        }
        
        // Save final results
        const finalResults = {
            timestamp: new Date().toISOString(),
            test: 'final_integration',
            generator_status: status,
            benchmark: benchmarkResults,
            variance_analysis: {
                max_vertex_diff: maxVertexDiff,
                max_embedding_diff: maxEmbDiff,
                exact_matches: exactMatches,
                total_elements: result.vertices.length,
                consistency_diff: consistencyDiff
            },
            success: zeroVariance && perfectConsistency,
            python_reference: pythonResults.timestamp,
            notes: 'Final integration test demonstrating zero variance achievement'
        };
        
        fs.writeFileSync('./final_integration_results.json', JSON.stringify(finalResults, null, 2));
        console.log('\n💾 Final results saved to: final_integration_results.json');
        
        return finalResults.success;
        
    } catch (error) {
        console.error('❌ Final integration test failed:', error.message);
        console.error(error.stack);
        return false;
    }
}

// Run the final integration test
if (require.main === module) {
    runFinalIntegrationTest().then(success => {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🏁 FINAL INTEGRATION TEST ${success ? 'PASSED' : 'FAILED'}`);
        console.log(`🎯 ZERO VARIANCE ${success ? 'ACHIEVED' : 'NOT ACHIEVED'}`);
        console.log(`${'='.repeat(60)}`);
        process.exit(success ? 0 : 1);
    });
}

module.exports = { runFinalIntegrationTest };
