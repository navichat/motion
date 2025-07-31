/**
 * Direct test of quantization improvements in CPU worker
 */

console.log('🧠 Testing Enhanced Quantization System...');

// Create a mock worker environment
const mockSelf = {
    postMessage: (data) => {
        if (data.type === 'completed' && data.result.modelOutput) {
            const output = data.result.modelOutput;
            console.log(`\n✅ ${data.taskId} completed!`);
            console.log(`🔧 Precision Mode: ${output.precision_mode}`);
            console.log(`⚡ Quantization: ${output.quantization_enabled ? 'ENABLED' : 'DISABLED'}`);
            if (output.quantization_enabled) {
                console.log(`🚀 Speed Improvement: ${output.speed_improvement_percent}%`);
                console.log(`💾 Memory Reduction: ${output.memory_reduction_factor}x`);
                console.log(`📊 Quality Retention: ${output.quality_retention_percent}%`);
                console.log(`⏱️  Execution Time: ${output.inference_time_ms}ms`);
                console.log(`🎯 Quantization Type: ${output.quantization_type}`);
                console.log(`📝 Description: ${output.quantization_description}`);
            }
        }
    },
    performance: {
        memory: {
            jsHeapSizeLimit: 4294967296, // 4GB
            usedJSHeapSize: 1073741824   // 1GB used
        }
    }
};

global.self = mockSelf;

// Load the CPU worker code
const fs = require('fs');
const path = require('path');

const workerCode = fs.readFileSync(
    path.join(__dirname, 'dev/web_viewer/js/workers/cpu-worker-simple.js'), 
    'utf8'
);

// Extract and test the quantization functions
eval(workerCode);

// Test quantization selection
async function testQuantizationSystem() {
    console.log('\n🔬 Testing quantization selection...');
    
    // Test with different memory scenarios
    const scenarios = [
        { memory: 0.4, expected: 'fp32' },
        { memory: 0.7, expected: 'int8' },
        { memory: 1.2, expected: 'fp16' },
        { memory: 2.5, expected: 'int4' }
    ];
    
    for (const scenario of scenarios) {
        const quantization = await selectOptimalQuantization('1.1B', scenario.memory, {});
        console.log(`💾 ${scenario.memory}GB → ${quantization.type} (${quantization.speedMultiplier}x faster)`);
    }
    
    // Test model execution
    console.log('\n🤖 Testing TinyLlama with quantization...');
    await executeLanguageModel('test_tinyllama', 2000, 1, 'TinyLlama');
    
    console.log('\n🤖 Testing DiabloGPT with quantization...');
    await executeLanguageModel('test_diablo', 2000, 1, 'DiabloGPT');
    
    console.log('\n🎉 Quantization system test complete!');
}

testQuantizationSystem().catch(console.error);
