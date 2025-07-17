
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
