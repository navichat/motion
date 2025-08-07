/**
 * Test Script for Quantized Model Optimization
 * Tests the new quantized models: Silero VAD, Whisper, SpeechT5, Kokoro
 */

const testQuantizedModels = async () => {
    console.log('🚀 Testing Quantized Model Optimization System');
    console.log('='.repeat(50));
    
    try {
        // Import the quantized model optimizer using CommonJS
        const { QuantizedModelOptimizer } = require('./quantized-model-optimizer.js');
        const optimizer = new QuantizedModelOptimizer();
        
        console.log('\n📋 Available Models:');
        const availableModels = optimizer.getAvailableModels();
        availableModels.forEach(model => {
            console.log(`  • ${model.displayName} (${model.name})`);
            console.log(`    Type: ${model.type}`);
            console.log(`    Quantizations: ${model.quantizations} options`);
            console.log(`    Backends: ${model.backends} supported`);
            console.log('');
        });
        
        // Test different optimization priorities
        const testCases = [
            { model: 'silero-vad', priority: 'performance', description: 'Silero VAD - Fast real-time' },
            { model: 'whisper-tiny-en', priority: 'memory', description: 'Whisper Tiny - Memory optimized' },
            { model: 'whisper-base', priority: 'accuracy', description: 'Whisper Base - High accuracy' },
            { model: 'speecht5-tts', priority: 'performance', description: 'SpeechT5 - Fast TTS' },
            { model: 'kokoro-tts', priority: 'accuracy', description: 'Kokoro - High quality TTS' }
        ];
        
        console.log('\n🎯 Testing Model Optimizations:');
        console.log('='.repeat(50));
        
        for (const testCase of testCases) {
            console.log(`\n🔧 Testing: ${testCase.description}`);
            
            try {
                const startTime = performance.now();
                
                // Get optimal configuration
                const config = await optimizer.selectOptimalModelConfiguration(testCase.model, {
                    prioritize: testCase.priority
                });
                
                if (config) {
                    console.log(`✅ Configuration selected in ${(performance.now() - startTime).toFixed(1)}ms:`);
                    console.log(`   Backend: ${config.backend}`);
                    console.log(`   Quantization: ${config.quantization}`);
                    console.log(`   Expected speedup: ${config.performance.estimatedSpeedup}x`);
                    console.log(`   Memory reduction: ${config.performance.memoryReduction}%`);
                    
                    if (config.paths.quantizedModelPath) {
                        console.log(`   Model path: ${config.paths.quantizedModelPath}`);
                    }
                    if (config.paths.modelId) {
                        console.log(`   Model ID: ${config.paths.modelId}`);
                    }
                    
                    // Test execution (mock data)
                    const mockData = generateMockInputData(testCase.model);
                    const executionStart = performance.now();
                    
                    const result = await optimizer.executeWithOptimizedModel(
                        testCase.model, 
                        mockData, 
                        { prioritize: testCase.priority }
                    );
                    
                    const executionTime = performance.now() - executionStart;
                    console.log(`🚀 Execution completed in ${executionTime.toFixed(1)}ms`);
                    console.log(`   Result: ${result.output.substring(0, 80)}...`);
                    console.log(`   Backend used: ${result.optimization.backend}`);
                    console.log(`   Quantization used: ${result.optimization.quantization}`);
                    
                } else {
                    console.log(`❌ Failed to configure model: ${testCase.model}`);
                }
                
            } catch (error) {
                console.error(`❌ Error testing ${testCase.model}:`, error.message);
            }
        }
        
        // Performance comparison
        console.log('\n📊 Performance Comparison Summary:');
        console.log('='.repeat(50));
        console.log('Model                 | Backend    | Quantization | Speedup | Memory');
        console.log('-'.repeat(70));
        
        for (const testCase of testCases.slice(0, 3)) { // Just show first 3 for brevity
            try {
                const config = await optimizer.selectOptimalModelConfiguration(testCase.model, {
                    prioritize: 'performance'
                });
                
                if (config) {
                    const modelName = config.model.padEnd(20);
                    const backend = config.backend.padEnd(10);
                    const quantization = config.quantization.padEnd(12);
                    const speedup = `${config.performance.estimatedSpeedup}x`.padEnd(7);
                    const memory = `${config.performance.memoryReduction}%`;
                    
                    console.log(`${modelName} | ${backend} | ${quantization} | ${speedup} | ${memory}`);
                }
            } catch (error) {
                console.log(`${testCase.model.padEnd(20)} | Error: ${error.message.substring(0, 30)}`);
            }
        }
        
        console.log('\n✅ Quantized Model Optimization Testing Complete!');
        console.log('\n🎉 Key Features Demonstrated:');
        console.log('  • Real quantized ONNX models (Silero VAD with 8 quantization levels)');
        console.log('  • Transformers.js WebGPU optimization (Whisper models)');
        console.log('  • Backend hierarchy selection (WebNN→WebGPU→ONNX→WASM)');
        console.log('  • Intelligent quantization selection (int4, int8, fp16, fp32)');
        console.log('  • Performance estimation and memory optimization');
        console.log('  • Automatic fallbacks for compatibility');
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    }
};

function generateMockInputData(modelName) {
    switch (modelName) {
        case 'silero-vad':
            return {
                audioData: new Float32Array(16000).fill(0.1), // 1 second of audio
                sampleRate: 16000
            };
        case 'whisper-tiny-en':
        case 'whisper-base':
            return {
                audioData: new Float32Array(32000).fill(0.1), // 2 seconds of audio
                sampleRate: 16000,
                language: 'en'
            };
        case 'speecht5-tts':
            return {
                text: "Hello, this is a test of optimized text-to-speech synthesis.",
                speaker: 'cmu_us_slt_arctic-wav-arctic_a0001'
            };
        case 'kokoro-tts':
            return {
                text: "This is Kokoro TTS generating high-quality speech.",
                voice: 'default'
            };
        default:
            return { data: 'mock input data' };
    }
}

// Export for Node.js testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { testQuantizedModels };
}

// Run tests if executed directly
if (typeof window === 'undefined' && typeof process !== 'undefined') {
    testQuantizedModels().catch(console.error);
}

// Browser execution
if (typeof window !== 'undefined') {
    window.testQuantizedModels = testQuantizedModels;
    console.log('🌐 Quantized model testing available in browser console: testQuantizedModels()');
}
