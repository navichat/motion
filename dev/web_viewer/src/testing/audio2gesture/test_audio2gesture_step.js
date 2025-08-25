const ort = require('onnxruntime-web');
const fs = require('fs');

async function testAudio2GestureStepModel() {
    try {
        console.log('🎭 Testing Audio2Gesture Single-Step Model');
        console.log('ONNX Runtime Web version:', ort.version);
        
        // Load test data from Python export
        const testData = JSON.parse(fs.readFileSync('./audio2gesture_step_test_data.json', 'utf8'));
        
        console.log('\n📦 Loading Audio2Gesture step model...');
        const session = await ort.InferenceSession.create('./audio2gesture_step_fixed.onnx', {
            executionProviders: ['cpu']
        });
        
        console.log('✅ Audio2Gesture step model loaded successfully!');
        
        // Print model metadata
        console.log('\n🔍 Model Inputs:');
        for (const input of session.inputMetadata) {
            console.log(`  ${input.name}: ${JSON.stringify(input.shape)} (${input.type})`);
        }
        
        console.log('\n🔍 Model Outputs:');
        for (const output of session.outputMetadata) {
            console.log(`  ${output.name}: ${JSON.stringify(output.shape)} (${output.type})`);
        }
        
        // Prepare input tensors
        console.log('\n⚙️ Preparing input tensors...');
        
        const audioWindowData = new Float32Array(testData.audio_window.flat(2));
        const prevMotionData = new Float32Array(testData.prev_motion.flat());
        const currentLexemeData = new Float32Array(testData.current_lexeme.flat());
        const hiddenStateData = new Float32Array(testData.hidden_state.flat(2));
        
        const audioWindowTensor = new ort.Tensor('float32', audioWindowData, [1, 80, 30]);
        const prevMotionTensor = new ort.Tensor('float32', prevMotionData, [1, 48]);
        const currentLexemeTensor = new ort.Tensor('float32', currentLexemeData, [1, 96]);
        const hiddenStateTensor = new ort.Tensor('float32', hiddenStateData, [4, 1, 1024]);
        
        console.log('✅ Input tensors created:');
        console.log(`  audio_window: ${JSON.stringify(audioWindowTensor.dims)}`);
        console.log(`  prev_motion: ${JSON.stringify(prevMotionTensor.dims)}`);
        console.log(`  current_lexeme: ${JSON.stringify(currentLexemeTensor.dims)}`);
        console.log(`  hidden_state: ${JSON.stringify(hiddenStateTensor.dims)}`);
        
        // Run single-step inference
        console.log('\n🚀 Running single-step inference...');
        const feeds = {
            audio_window: audioWindowTensor,
            prev_motion: prevMotionTensor,
            current_lexeme: currentLexemeTensor,
            hidden_state: hiddenStateTensor
        };
        
        const results = await session.run(feeds);
        
        console.log('✅ Single-step inference successful!');
        console.log('📊 Output shapes:');
        Object.keys(results).forEach(name => {
            console.log(`  ${name}: ${JSON.stringify(results[name].dims)}`);
        });
        
        // Validate against Python outputs
        console.log('\n🔬 Validating against Python results...');
        
        const jsNewMotion = Array.from(results.new_motion.data);
        const jsNewHiddenState = Array.from(results.new_hidden_state.data);
        
        const pyNewMotion = testData.new_motion.flat();
        const pyNewHiddenState = testData.new_hidden_state.flat(2);
        
        // Compare motion outputs
        const motionDiff = jsNewMotion.slice(0, 10).map((v, i) => Math.abs(v - pyNewMotion[i]));
        const maxMotionDiff = Math.max(...motionDiff);
        
        // Compare hidden state outputs
        const hiddenDiff = jsNewHiddenState.slice(0, 10).map((v, i) => Math.abs(v - pyNewHiddenState[i]));
        const maxHiddenDiff = Math.max(...hiddenDiff);
        
        console.log('📈 Validation Results:');
        console.log(`  Motion output max difference: ${maxMotionDiff.toExponential(3)}`);
        console.log(`  Hidden state max difference: ${maxHiddenDiff.toExponential(3)}`);
        
        if (maxMotionDiff < 1e-5 && maxHiddenDiff < 1e-5) {
            console.log('✅ JavaScript and Python outputs match perfectly!');
        } else if (maxMotionDiff < 1e-3 && maxHiddenDiff < 1e-3) {
            console.log('✅ JavaScript and Python outputs match (within acceptable tolerance)!');
        } else {
            console.log('⚠️ JavaScript and Python outputs differ significantly');
        }
        
        console.log('\n🎉 Audio2Gesture single-step model test complete!');
        console.log('Ready for multi-step autoregressive generation.');
        
        return {
            success: true,
            newMotion: jsNewMotion,
            newHiddenState: jsNewHiddenState,
            maxMotionDiff,
            maxHiddenDiff
        };
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        return { success: false, error: error.message };
    }
}

// Run the test
testAudio2GestureStepModel().then(result => {
    if (result.success) {
        console.log('\n🎯 Next Step: Implement multi-step autoregressive generator');
        console.log('   Similar to FaceFormer\'s successful approach');
    } else {
        console.log('\n💔 Test failed. Check model export and dependencies.');
    }
});
