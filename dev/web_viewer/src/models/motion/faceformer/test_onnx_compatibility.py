#!/usr/bin/env python3
"""
Test ONNX model inputs and outputs to verify compatibility
"""

import onnxruntime as ort
import numpy as np
import json
import os

def test_onnx_model_inputs():
    """Test ONNX model to verify input/output shapes and types"""
    
    print("🔍 Testing ONNX model inputs and outputs...")
    
    models_dir = "./models"
    datasets = ['vocaset', 'biwi']
    
    for dataset in datasets:
        print(f"\n📋 Testing {dataset.upper()} model...")
        
        # Load config
        config_path = f"{models_dir}/faceformer_{dataset}_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        model_path = f"{models_dir}/faceformer_{dataset}_full.onnx"
        session = ort.InferenceSession(model_path)
        
        print(f"  Model: {model_path}")
        print(f"  Providers: {session.get_providers()}")
        
        # Print input details
        print("  📥 Model Inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.type} {inp.shape}")
        
        # Print output details
        print("  📤 Model Outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.type} {out.shape}")
        
        # Test with correct inputs
        batch_size = 1
        seq_len = 25
        
        print(f"  🧪 Testing inference with seq_len={seq_len}...")
        
        # Create test inputs
        audio_features = np.random.randn(batch_size, seq_len, config['audio_input_dim']).astype(np.float32)
        template = np.random.randn(batch_size, config['vertice_dim']).astype(np.float32)
        subject_id = np.array([0], dtype=np.int64)
        
        inputs = {
            'audio_features': audio_features,
            'template': template,
            'subject_id': subject_id
        }
        
        print(f"    Input shapes:")
        for name, tensor in inputs.items():
            print(f"      {name}: {tensor.shape} ({tensor.dtype})")
        
        try:
            outputs = session.run(None, inputs)
            
            print(f"    ✅ Inference successful!")
            print(f"    Output shapes:")
            for i, out in enumerate(outputs):
                print(f"      {session.get_outputs()[i].name}: {out.shape} ({out.dtype})")
            
            # Verify output shape
            expected_shape = (batch_size, seq_len, config['vertice_dim'])
            actual_shape = outputs[0].shape
            
            if actual_shape == expected_shape:
                print(f"    ✅ Output shape correct: {actual_shape}")
            else:
                print(f"    ❌ Output shape mismatch: expected {expected_shape}, got {actual_shape}")
                
        except Exception as e:
            print(f"    ❌ Inference failed: {e}")
        
        print()

def create_web_compatible_test():
    """Create a simple test that mimics web input format"""
    
    print("🌐 Creating web-compatible test...")
    
    # Create a simple JavaScript test file
    js_test = """
// Test ONNX model in browser format
async function testONNXModel() {
    try {
        console.log('🔍 Loading ONNX model...');
        
        const session = await ort.InferenceSession.create('./models/faceformer_vocaset_full.onnx');
        
        console.log('📥 Model inputs:', session.inputNames);
        console.log('📤 Model outputs:', session.outputNames);
        
        // Create test data
        const batchSize = 1;
        const seqLen = 25;
        const audioInputDim = 768;
        const verticeDim = 15069;
        
        const audioFeatures = new Float32Array(batchSize * seqLen * audioInputDim);
        for (let i = 0; i < audioFeatures.length; i++) {
            audioFeatures[i] = Math.random() * 0.1 - 0.05;
        }
        
        const template = new Float32Array(batchSize * verticeDim);
        for (let i = 0; i < template.length; i++) {
            template[i] = Math.random() * 0.01;
        }
        
        const subjectId = new BigInt64Array([0n]);
        
        // Create tensors
        const feeds = {
            'audio_features': new ort.Tensor('float32', audioFeatures, [batchSize, seqLen, audioInputDim]),
            'template': new ort.Tensor('float32', template, [batchSize, verticeDim]),
            'subject_id': new ort.Tensor('int64', subjectId, [batchSize])
        };
        
        console.log('🧪 Running inference...');
        console.log('Input tensor shapes:', {
            audio_features: feeds.audio_features.dims,
            template: feeds.template.dims,
            subject_id: feeds.subject_id.dims
        });
        
        const outputs = await session.run(feeds);
        
        console.log('✅ Inference successful!');
        console.log('Output shape:', outputs.vertices.dims);
        console.log('Output type:', outputs.vertices.type);
        
        return true;
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        return false;
    }
}

// Run test when page loads
window.addEventListener('load', testONNXModel);
""";
    
    with open('test_onnx_web.js', 'w') as f:
        f.write(js_test)
    
    # Create simple HTML test page
    html_test = """
<!DOCTYPE html>
<html>
<head>
    <title>ONNX Model Test</title>
</head>
<body>
    <h1>ONNX Model Compatibility Test</h1>
    <p>Check browser console for results</p>
    
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js"></script>
    <script src="./test_onnx_web.js"></script>
</body>
</html>
""";
    
    with open('test_onnx_model.html', 'w') as f:
        f.write(html_test)
    
    print("📄 Created test_onnx_model.html for browser testing")

def main():
    os.chdir("/home/barberb/motion/dev/web_viewer/faceformer")
    
    test_onnx_model_inputs()
    create_web_compatible_test()
    
    print("✅ ONNX model testing complete!")
    print()
    print("🌐 To test in browser:")
    print("  1. Open test_onnx_model.html in a browser")
    print("  2. Check console for detailed results")
    print("  3. Verify input/output compatibility")

if __name__ == "__main__":
    main()
