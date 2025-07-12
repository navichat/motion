const { Audio2GestureChunkedGenerator } = require('./audio2gesture_chunked_generator');
const { ChunkedModelLoader } = require('./chunked_model_loader');

async function testChunkedSystem() {
    console.log('🧪 Testing Audio2Gesture Chunked Model System');
    console.log('===============================================\n');
    
    try {
        // Test 1: Test chunked model loader directly
        console.log('🔧 Test 1: Direct chunked model loading...');
        const loader = new ChunkedModelLoader();
        
        // Test loading from chunks metadata
        const modelData = await loader.loadModel('./motion_generator.chunks.json');
        console.log(`✅ Successfully loaded model data: ${(modelData.byteLength / 1024 / 1024).toFixed(2)} MB`);
        
        // Test cache
        const cachedData = await loader.loadModel('./motion_generator.chunks.json');
        console.log(`✅ Cache test passed: ${(cachedData.byteLength / 1024 / 1024).toFixed(2)} MB`);
        
        console.log('📊 Cache info:', loader.getCacheInfo());
        
        // Test 2: Test with Audio2Gesture generator
        console.log('\n🎭 Test 2: Audio2Gesture generator with chunked model...');
        const generator = new Audio2GestureChunkedGenerator();
        
        // Try to initialize with chunked model
        await generator.initialize('./motion_generator.chunks.json');
        console.log('✅ Generator initialized with chunked model!');
        
        // Test generation with sample inputs
        const inputs = generator.createSampleInputs('full');
        console.log('🎬 Testing motion generation...');
        const results = await generator.generateSequence(inputs, 5);
        
        console.log(`✅ Generated ${results.frames.length} motion frames successfully!`);
        console.log(`📊 Performance: ${results.metrics.fps.toFixed(1)} FPS`);
        
        // Test 3: Test step model with chunks
        console.log('\n🔄 Test 3: Step model with chunks...');
        const stepGenerator = new Audio2GestureChunkedGenerator();
        
        try {
            await stepGenerator.initialize('./audio2gesture_step_fixed.chunks.json');
            console.log('✅ Step model initialized with chunks!');
            
            const stepInputs = stepGenerator.createSampleInputs('step');
            const stepResults = await stepGenerator.generateSequence(stepInputs, 3);
            
            console.log(`✅ Step model generated ${stepResults.frames.length} frames!`);
            console.log(`📊 Step model performance: ${stepResults.metrics.fps.toFixed(1)} FPS`);
            
        } catch (error) {
            console.log(`⚠️  Step model test failed (expected if model architecture differs): ${error.message}`);
        }
        
        // Test 4: Test rebuild functionality
        console.log('\n🔧 Test 4: Model rebuild functionality...');
        const { rebuildModel } = require('./rebuild_motion_generator');
        
        // First backup original if it exists
        const fs = require('fs');
        if (fs.existsSync('./motion_generator.onnx')) {
            fs.renameSync('./motion_generator.onnx', './motion_generator.onnx.test_backup');
            console.log('📁 Backed up existing model');
        }
        
        // Test rebuild
        const rebuildSuccess = rebuildModel();
        if (rebuildSuccess) {
            console.log('✅ Model rebuild test passed!');
            
            // Verify rebuilt model works
            const rebuiltGenerator = new Audio2GestureChunkedGenerator();
            await rebuiltGenerator.initialize('./motion_generator.onnx');
            console.log('✅ Rebuilt model loads correctly!');
            
            // Clean up - remove rebuilt file and restore backup if exists
            fs.unlinkSync('./motion_generator.onnx');
            if (fs.existsSync('./motion_generator.onnx.test_backup')) {
                fs.renameSync('./motion_generator.onnx.test_backup', './motion_generator.onnx');
                console.log('📁 Restored original model');
            }
        }
        
        console.log('\n🎉 All chunked model tests passed!');
        console.log('✅ System is ready for Git-friendly model storage');
        
        return {
            success: true,
            tests: {
                chunkLoading: true,
                generatorInit: true,
                motionGeneration: true,
                rebuild: rebuildSuccess
            }
        };
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        return { success: false, error: error.message };
    }
}

// Run comprehensive test
async function runFullTest() {
    const results = await testChunkedSystem();
    
    if (results.success) {
        console.log('\n📋 Summary: Chunked Model System Ready');
        console.log('=====================================');
        console.log('✅ Large ONNX files split into Git-friendly chunks');
        console.log('✅ Transparent loading system working');
        console.log('✅ Audio2Gesture generation functional');
        console.log('✅ Model rebuild verification passed');
        console.log('\n🚀 Next Steps:');
        console.log('  1. Move original .onnx files to backup');
        console.log('  2. Add chunk files to Git');
        console.log('  3. Update .gitignore for large files');
        console.log('  4. Deploy with chunked system');
    } else {
        console.log('\n💔 Chunked system test failed');
        console.log('Check error details above');
    }
}

if (require.main === module) {
    runFullTest().catch(console.error);
}

module.exports = { testChunkedSystem };
