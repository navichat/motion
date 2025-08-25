/**
 * Quick test script for unified animation system with DeepMimic
 */

console.log('🚀 Testing Unified Animation System with DeepMimic...');

// Import the unified animation model test
import('./unified_animation_model_test.js').then(async (module) => {
    try {
        console.log('📦 Module loaded successfully');
        
        // Create test instance
        const testSuite = new module.UnifiedAnimationModelTest();
        
        console.log('🔍 Starting basic connectivity test...');
        
        // Test ONNX Runtime availability
        if (typeof ort !== 'undefined') {
            console.log('✅ ONNX Runtime available');
            console.log(`📊 Version: ${ort.version || 'unknown'}`);
        } else {
            console.log('❌ ONNX Runtime not available');
            return;
        }
        
        // Test model paths
        console.log('🗂️ Testing model paths...');
        const modelCounts = {};
        
        for (const [systemName, models] of Object.entries(testSuite.models)) {
            modelCounts[systemName] = Object.keys(models).length;
            console.log(`   ${systemName}: ${modelCounts[systemName]} models`);
            
            for (const [modelName, config] of Object.entries(models)) {
                console.log(`     - ${modelName}: ${config.path}`);
            }
        }
        
        console.log(`\n📊 Total systems: ${Object.keys(testSuite.models).length}`);
        console.log(`📊 Total models: ${Object.values(modelCounts).reduce((a, b) => a + b, 0)}`);
        
        // Test DeepMimic specifically
        console.log('\n🤖 Testing DeepMimic integration...');
        
        if (testSuite.models.deepmimic) {
            console.log('✅ DeepMimic models configured:');
            for (const [modelName, config] of Object.entries(testSuite.models.deepmimic)) {
                console.log(`   - ${modelName}: ${config.path}`);
            }
            
            // Test DeepMimic BVH frame generation
            console.log('\n🎬 Testing DeepMimic BVH frame generation...');
            try {
                const bvhFrame = await testSuite.generateDeepMimicBVHFrame();
                console.log('✅ DeepMimic BVH frame generated successfully');
                console.log(`📊 Frame joints: ${Object.keys(bvhFrame).filter(k => !k.startsWith('_')).length}`);
                console.log(`🔢 Sample joint data:`, Object.entries(bvhFrame).slice(0, 3));
                
                if (bvhFrame._metadata) {
                    console.log(`📝 Metadata:`, bvhFrame._metadata);
                }
                
            } catch (error) {
                console.log('❌ DeepMimic BVH generation failed:', error.message);
            }
            
            // Test data generation functions
            console.log('\n🧪 Testing DeepMimic data generation...');
            try {
                const stateData = testSuite.generateDeepMimicStateData([1, 197]);
                console.log(`✅ State data generated: ${stateData.length} values`);
                console.log(`📊 State sample: [${Array.from(stateData.slice(0, 5)).map(v => v.toFixed(3)).join(', ')}...]`);
                
                const actionData = testSuite.generateDeepMimicTestData([1, 43]);
                console.log(`✅ Action data generated: ${actionData.length} values`);
                console.log(`📊 Action sample: [${Array.from(actionData.slice(0, 5)).map(v => v.toFixed(3)).join(', ')}...]`);
                
            } catch (error) {
                console.log('❌ Data generation failed:', error.message);
            }
            
        } else {
            console.log('❌ DeepMimic models not found in configuration');
        }
        
        // Test frame compositing
        console.log('\n🎨 Testing frame compositing capabilities...');
        try {
            const testFrames = {
                rsmt: { Hips: [0, 1, 0, 0, 0, 0], Chest: [0, 0, 0] },
                faceformer: { Head: [5, 0, 0], Neck: [2, 0, 0] },
                audiogesture: { LeftShoulder: [10, 0, 0], RightShoulder: [-10, 0, 0] },
                deepmimic: { LeftHip: [0, 0, 15], RightHip: [0, 0, -15] }
            };
            
            const composite = testSuite.compositeFrames(testFrames);
            console.log('✅ Frame compositing successful');
            console.log(`📊 Composite joints: ${Object.keys(composite).length}`);
            console.log(`🎭 Sample composite data:`, Object.entries(composite).slice(0, 3));
            
        } catch (error) {
            console.log('❌ Frame compositing failed:', error.message);
        }
        
        console.log('\n🎯 Basic connectivity test completed!');
        console.log('💡 To run full test suite, use: testSuite.runCompleteTest()');
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    }
    
}).catch(error => {
    console.error('❌ Failed to load module:', error);
});

// Export for global access
window.testUnifiedAnimation = () => {
    console.log('🔄 Reloading unified animation test...');
    location.reload();
};

console.log('📋 Quick test commands:');
console.log('  - testUnifiedAnimation() - Reload and retest');
console.log('  - Check console for test results');
