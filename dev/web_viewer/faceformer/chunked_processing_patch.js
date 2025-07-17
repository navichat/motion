// Chunked Processing Extension for Full FaceFormer
// Add this to full_faceformer_web.js

// Replace the generateWithONNXTransformer method with this version:

async generateWithONNXTransformer(audioFeatures, template, subjectId) {
    console.log('📦 Generating with ONNX Runtime transformer...');
    
    const seqLen = audioFeatures.length;
    const batchSize = 1;
    
    // Check for sequence length limits to avoid memory issues
    const maxSeqLen = 200;
    
    if (seqLen > maxSeqLen) {
        console.log(`⚠️ Sequence too long (${seqLen}), processing in chunks of ${maxSeqLen}`);
        return await this.processLongSequenceChunked(audioFeatures, template, subjectId, maxSeqLen);
    }
    
    try {
        // Prepare tensors for transformer
        const audioTensorData = new Float32Array(audioFeatures.flat());
        const templateTensorData = new Float32Array(template);
        const subjectTensorData = new BigInt64Array([BigInt(subjectId)]);
        
        console.log(`🔍 Preparing tensors: seq_len=${seqLen}, audio_dim=${this.config.audio_input_dim}, vertex_dim=${this.config.vertice_dim}`);
        
        const feeds = {
            audio_features: new ort.Tensor('float32', audioTensorData, [batchSize, seqLen, this.config.audio_input_dim]),
            template: new ort.Tensor('float32', templateTensorData, [batchSize, this.config.vertice_dim]),
            subject_id: new ort.Tensor('int64', subjectTensorData, [1])
        };
        
        console.log(`🎯 Input shapes: audio[${feeds.audio_features.dims}], template[${feeds.template.dims}], subject[${feeds.subject_id.dims}]`);
        
        // Run full transformer inference
        const startTime = performance.now();
        const results = await this.session.run(feeds);
        const inferenceTime = performance.now() - startTime;
        
        this.performanceStats.transformerTime = inferenceTime;
        
        console.log(`✅ ONNX transformer inference: ${inferenceTime.toFixed(2)}ms`);
        
        // Extract and reshape results
        const vertices = Array.from(results.vertices.data);
        const reshapedVertices = this.reshapeVertices(vertices, seqLen);
        
        return {
            vertices: reshapedVertices,
            backend: 'onnxruntime-web',
            inferenceTime: inferenceTime,
            audioProcessingTime: this.performanceStats.audioProcessingTime,
            totalTime: this.performanceStats.audioProcessingTime + inferenceTime
        };
        
    } catch (error) {
        console.error('❌ ONNX transformer failed:', error.message);
        console.error('❌ Error details:', error);
        
        // Check for memory-related errors
        if (error.toString().includes('9523168') || error.toString().includes('memory') || error.toString().includes('allocation')) {
            console.log('💡 Memory error detected - sequence was too long');
        }
        
        throw error;
    }
}

// Add these new methods:

async processLongSequenceChunked(audioFeatures, template, subjectId, maxChunkSize) {
    console.log(`🔧 Processing long sequence in chunks of ${maxChunkSize} frames...`);
    
    const totalFrames = audioFeatures.length;
    const chunks = Math.ceil(totalFrames / maxChunkSize);
    const allVertices = [];
    let totalInferenceTime = 0;
    
    for (let i = 0; i < chunks; i++) {
        const startIdx = i * maxChunkSize;
        const endIdx = Math.min(startIdx + maxChunkSize, totalFrames);
        const chunkFeatures = audioFeatures.slice(startIdx, endIdx);
        
        console.log(`📊 Processing chunk ${i + 1}/${chunks}: frames ${startIdx}-${endIdx}`);
        
        try {
            const chunkResult = await this.processChunkDirect(chunkFeatures, template, subjectId);
            allVertices.push(...chunkResult.vertices);
            totalInferenceTime += chunkResult.inferenceTime;
            
            // Small delay to prevent system overwhelm
            if (i < chunks - 1) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
        } catch (error) {
            console.error(`❌ Failed to process chunk ${i + 1}:`, error);
            throw error;
        }
    }
    
    const avgTimePerChunk = totalInferenceTime / chunks;
    console.log(`✅ Long sequence processed: ${chunks} chunks, ${totalInferenceTime.toFixed(2)}ms total, ${avgTimePerChunk.toFixed(2)}ms avg per chunk`);
    
    return {
        vertices: allVertices,
        backend: 'onnxruntime-web-chunked',
        inferenceTime: totalInferenceTime,
        audioProcessingTime: this.performanceStats.audioProcessingTime,
        totalTime: this.performanceStats.audioProcessingTime + totalInferenceTime,
        chunksProcessed: chunks,
        avgTimePerChunk: avgTimePerChunk
    };
}

async processChunkDirect(audioFeatures, template, subjectId) {
    const seqLen = audioFeatures.length;
    const batchSize = 1;
    
    const audioTensorData = new Float32Array(audioFeatures.flat());
    const templateTensorData = new Float32Array(template);
    const subjectTensorData = new BigInt64Array([BigInt(subjectId)]);
    
    const feeds = {
        audio_features: new ort.Tensor('float32', audioTensorData, [batchSize, seqLen, this.config.audio_input_dim]),
        template: new ort.Tensor('float32', templateTensorData, [batchSize, this.config.vertice_dim]),
        subject_id: new ort.Tensor('int64', subjectTensorData, [1])
    };
    
    const startTime = performance.now();
    const results = await this.session.run(feeds);
    const inferenceTime = performance.now() - startTime;
    
    const vertices = Array.from(results.vertices.data);
    const reshapedVertices = this.reshapeVertices(vertices, seqLen);
    
    return {
        vertices: reshapedVertices,
        inferenceTime: inferenceTime
    };
}
