const ort = require('onnxruntime-web');
const fs = require('fs');
const path = require('path');

async function testModelBasic() {
    try {
        console.log('ONNX Runtime Web version:', ort.version);
        
        // Load test data from Python
        const testData = JSON.parse(fs.readFileSync('./sample_data.json', 'utf8'));
        
        console.log('Loading Audio2Gesture model...');
        const session = await ort.InferenceSession.create('./motion_generator.onnx', {
            executionProviders: ['cpu']  // Force CPU backend
        });
        
        console.log('Audio2Gesture model loaded successfully!');
        
        // Print model info - different API for ONNX Runtime Web
        console.log('\nInputs:');
        try {
            for (const input of session.inputMetadata) {
                console.log(`  ${input.name}: ${JSON.stringify(input.shape)} (${input.type})`);
            }
        } catch (e) {
            console.log('Using alternative input inspection...');
            console.log('Session object keys:', Object.keys(session));
        }
        
        console.log('\nOutputs:');
        try {
            for (const output of session.outputMetadata) {
                console.log(`  ${output.name}: ${JSON.stringify(output.shape)} (${output.type})`);
            }
        } catch (e) {
            console.log('Using alternative output inspection...');
        }
        
        // Prepare inputs - Audio2Gesture uses aud_input, mo_input, lxm_input
        const inputs = testData;
        
        console.log('\nInput shapes:');
        const audInputFlatData = inputs.aud_input.flat(2);
        const moInputFlatData = inputs.mo_input.flat(2);
        const lxmInputFlatData = inputs.lxm_input.flat(2);
        
        console.log(`  aud_input: ${audInputFlatData.length} elements (expected: ${1*80*100})`);
        console.log(`  mo_input: ${moInputFlatData.length} elements (expected: ${1*48*100})`);
        console.log(`  lxm_input: ${lxmInputFlatData.length} elements (expected: ${1*96*10})`);
        
        // Create ONNX tensors
        const audInputTensor = new ort.Tensor('float32', 
            new Float32Array(audInputFlatData), 
            [1, 80, 100]);
        
        const moInputTensor = new ort.Tensor('float32', 
            new Float32Array(moInputFlatData), 
            [1, 48, 100]);
        
        const lxmInputTensor = new ort.Tensor('float32', 
            new Float32Array(lxmInputFlatData), 
            [1, 96, 10]);
        
        console.log('\nTensor shapes created:');
        console.log(`  aud_input: ${JSON.stringify(audInputTensor.dims)}`);
        console.log(`  mo_input: ${JSON.stringify(moInputTensor.dims)}`);
        console.log(`  lxm_input: ${JSON.stringify(lxmInputTensor.dims)}`);
        
        // Run inference
        console.log('\nRunning inference...');
        const feeds = {
            aud_input: audInputTensor,
            mo_input: moInputTensor,
            lxm_input: lxmInputTensor
        };
        
        const results = await session.run(feeds);
        
        console.log('Inference successful!');
        console.log('Output names:', Object.keys(results));
        
        Object.keys(results).forEach(name => {
            console.log(`  ${name}: ${JSON.stringify(results[name].dims)}`);
        });
        
        // Compare with Python results
        const pyOutputs = testData.output;
        const jsOutput1 = Array.from(results[Object.keys(results)[0]].data);
        
        // Handle nested Python output structure
        const pyOutput1 = Array.isArray(pyOutputs[0]) ? pyOutputs.flat(2) : pyOutputs.flat();
        
        console.log('\nOutput analysis:');
        console.log('Python output structure:', Array.isArray(pyOutputs[0]) ? 'nested arrays' : 'flat array');
        console.log('Python output length:', pyOutput1.length);
        console.log('JavaScript output length:', jsOutput1.length);
        
        console.log('\nPython output first 10 values:', pyOutput1.slice(0, 10));
        console.log('JavaScript output first 10 values:', jsOutput1.slice(0, 10));
        
        // Check for NaN values
        const hasNaNPy = pyOutput1.slice(0, 10).some(v => isNaN(v));
        const hasNaNJs = jsOutput1.slice(0, 10).some(v => isNaN(v));
        
        console.log('Python output has NaN:', hasNaNPy);
        console.log('JavaScript output has NaN:', hasNaNJs);
        
        if (!hasNaNPy && !hasNaNJs && pyOutput1.length === jsOutput1.length) {
            const diff = jsOutput1.slice(0, 10).map((v, i) => Math.abs(v - pyOutput1[i]));
            console.log('\nFirst 10 output differences:', diff);
            console.log('Max difference in first 10:', Math.max(...diff));
            
            // Test more values
            const diff100 = jsOutput1.slice(0, Math.min(100, pyOutput1.length)).map((v, i) => Math.abs(v - pyOutput1[i]));
            console.log('Max difference in first 100:', Math.max(...diff100));
            
            if (Math.max(...diff100) < 1e-4) {
                console.log('✅ JavaScript and Python outputs match (within 1e-4 tolerance)!');
            } else {
                console.log('⚠️ JavaScript and Python outputs differ');
            }
        } else {
            console.log('⚠️ Cannot compare outputs - length mismatch or NaN values');
            console.log(`   Python length: ${pyOutput1.length}, JavaScript length: ${jsOutput1.length}`);
        }
        
    } catch (error) {
        console.error('Test failed:', error);
        if (error.code) {
            console.error('Error code:', error.code);
        }
        if (error.message) {
            console.error('Error message:', error.message);
        }
        console.error('Full error object:', error);
    }
}

testModelBasic();
