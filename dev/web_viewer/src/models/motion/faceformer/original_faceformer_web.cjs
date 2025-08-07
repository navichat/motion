// Original FaceFormer Web Implementation (CommonJS)
const fs = require('fs');

class OriginalFaceFormerWeb {
    constructor() {
        this.weights = null;
        this.config = null;
        this.initialized = false;
        this.dataset = null;
    }

    async initialize(dataset = 'vocaset', weightsPath = null) {
        console.log(`🤖 Initializing Original FaceFormer (${dataset.toUpperCase()})...`);
        
        this.dataset = dataset.toLowerCase();
        const defaultPath = `./converted_weights/faceformer_${this.dataset}_weights.json`;
        const weightFile = weightsPath || defaultPath;
        
        try {
            // Load weights
            const weightData = JSON.parse(fs.readFileSync(weightFile, 'utf8'));
            this.weights = weightData.weights;
            this.config = weightData.config;
            
            console.log('✅ Original FaceFormer weights loaded:');
            console.log(`  Dataset: ${this.config.dataset}`);
            console.log(`  Feature dim: ${this.config.feature_dim}`);
            console.log(`  Vertex dim: ${this.config.vertice_dim}`);
            console.log(`  Subjects: ${this.config.num_subjects}`);
            console.log(`  Components: ${Object.keys(this.weights).length}`);
            
            this.initialized = true;
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize Original FaceFormer:', error);
            return false;
        }
    }

    // Multi-head attention implementation
    multiHeadAttention(query, key, value, weightsPrefix, numHeads = 4) {
        const inProjWeight = this.weights[`${weightsPrefix}.in_proj_weight`];
        const inProjBias = this.weights[`${weightsPrefix}.in_proj_bias`];
        const outProjWeight = this.weights[`${weightsPrefix}.out_proj.weight`];
        const outProjBias = this.weights[`${weightsPrefix}.out_proj.bias`];
        
        const featureDim = this.config.feature_dim;
        const headDim = featureDim / numHeads;
        
        // Combined QKV projection
        const qkvInput = query; // For self-attention, Q=K=V
        const qkv = this.linearLayer(qkvInput, inProjWeight, inProjBias);
        
        // Split into Q, K, V
        const q = qkv.slice(0, featureDim);
        const k = qkv.slice(featureDim, featureDim * 2);
        const v = qkv.slice(featureDim * 2, featureDim * 3);
        
        // Reshape for multi-head attention
        const qHeads = this.reshapeForHeads(q, numHeads, headDim);
        const kHeads = this.reshapeForHeads(k, numHeads, headDim);
        const vHeads = this.reshapeForHeads(v, numHeads, headDim);
        
        // Scaled dot-product attention per head
        const attentionHeads = [];
        for (let h = 0; h < numHeads; h++) {
            const attention = this.scaledDotProductAttention(
                qHeads[h], kHeads[h], vHeads[h], headDim
            );
            attentionHeads.push(attention);
        }
        
        // Concatenate heads
        const concatenated = attentionHeads.flat();
        
        // Output projection
        const output = this.linearLayer(concatenated, outProjWeight, outProjBias);
        
        return output;
    }

    scaledDotProductAttention(q, k, v, headDim) {
        // Simplified attention for 1D case
        const scale = 1.0 / Math.sqrt(headDim);
        
        // Attention weights (simplified for single token)
        let attentionScore = 0;
        for (let i = 0; i < headDim; i++) {
            attentionScore += q[i] * k[i];
        }
        attentionScore *= scale;
        
        const attentionWeight = Math.exp(attentionScore); // Simplified softmax
        
        // Apply attention to values
        const output = new Array(headDim);
        for (let i = 0; i < headDim; i++) {
            output[i] = v[i] * attentionWeight;
        }
        
        return output;
    }

    reshapeForHeads(tensor, numHeads, headDim) {
        const heads = [];
        for (let h = 0; h < numHeads; h++) {
            const start = h * headDim;
            const end = start + headDim;
            heads.push(tensor.slice(start, end));
        }
        return heads;
    }

    // Transformer decoder layer
    transformerDecoderLayer(input, memory) {
        const layerPrefix = 'transformer_decoder.layers.0';
        
        // Self-attention
        const norm1Input = this.layerNorm(input, `${layerPrefix}.norm1`);
        const selfAttn = this.multiHeadAttention(
            norm1Input, norm1Input, norm1Input, 
            `${layerPrefix}.self_attn`
        );
        const residual1 = this.addVectors(input, selfAttn);
        
        // Cross-attention with memory
        const norm2Input = this.layerNorm(residual1, `${layerPrefix}.norm2`);
        const crossAttn = this.multiHeadAttention(
            norm2Input, memory, memory,
            `${layerPrefix}.multihead_attn`
        );
        const residual2 = this.addVectors(residual1, crossAttn);
        
        // Feed-forward network
        const norm3Input = this.layerNorm(residual2, `${layerPrefix}.norm3`);
        const linear1Out = this.linearLayer(
            norm3Input,
            this.weights[`${layerPrefix}.linear1.weight`],
            this.weights[`${layerPrefix}.linear1.bias`]
        );
        const reluOut = linear1Out.map(x => Math.max(0, x)); // ReLU activation
        const linear2Out = this.linearLayer(
            reluOut,
            this.weights[`${layerPrefix}.linear2.weight`],
            this.weights[`${layerPrefix}.linear2.bias`]
        );
        const residual3 = this.addVectors(residual2, linear2Out);
        
        return residual3;
    }

    // Layer normalization
    layerNorm(input, weightPrefix) {
        const weight = this.weights[`${weightPrefix}.weight`];
        const bias = this.weights[`${weightPrefix}.bias`];
        
        // Calculate mean and variance
        const mean = input.reduce((sum, x) => sum + x, 0) / input.length;
        const variance = input.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / input.length;
        const std = Math.sqrt(variance + 1e-5); // eps = 1e-5
        
        // Normalize and apply learned parameters
        const output = new Array(input.length);
        for (let i = 0; i < input.length; i++) {
            output[i] = ((input[i] - mean) / std) * weight[i] + bias[i];
        }
        
        return output;
    }

    // Linear layer
    linearLayer(input, weight, bias) {
        const outputSize = weight.length;
        const inputSize = weight[0].length;
        const output = new Array(outputSize);
        
        for (let i = 0; i < outputSize; i++) {
            let sum = 0;
            for (let j = 0; j < inputSize; j++) {
                sum += input[j] * weight[i][j];
            }
            output[i] = sum + (bias ? bias[i] : 0);
        }
        
        return output;
    }

    // Positional encoding
    applyPositionalEncoding(input, position) {
        const pe = this.weights['PPE.pe'][0]; // Shape: [625/630, feature_dim]
        const featureDim = this.config.feature_dim;
        
        const output = new Array(featureDim);
        const peRow = pe[position % pe.length];
        
        for (let i = 0; i < featureDim; i++) {
            output[i] = input[i] + peRow[i];
        }
        
        return output;
    }

    // Vector addition
    addVectors(a, b) {
        const result = new Array(a.length);
        for (let i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    // Main generation function
    async generateVertices(audioFeatures, template, subjectId = 0) {
        if (!this.initialized) {
            throw new Error('FaceFormer not initialized');
        }

        console.log('🎭 Generating vertices with original FaceFormer...');
        
        const seqLen = audioFeatures.length;
        const featureDim = this.config.feature_dim;
        const verticeDim = this.config.vertice_dim;
        
        // Create one-hot encoding for subject
        const oneHot = new Array(this.config.num_subjects).fill(0);
        oneHot[subjectId] = 1.0;
        
        // Get style embedding
        const styleEmbedding = this.linearLayer(
            oneHot,
            this.weights['obj_vector.weight'],
            null // obj_vector has no bias
        );
        
        // Process audio features
        const processedAudio = [];
        for (let t = 0; t < seqLen; t++) {
            const audioFrame = audioFeatures[t]; // [768] from Wav2Vec2
            const mappedAudio = this.linearLayer(
                audioFrame,
                this.weights['audio_feature_map.weight'],
                this.weights['audio_feature_map.bias']
            );
            processedAudio.push(mappedAudio);
        }
        
        // Auto-regressive generation
        const generatedVertices = [];
        let currentEmbedding = [...styleEmbedding]; // Start with style
        
        for (let t = 0; t < seqLen; t++) {
            console.log(`  Generating frame ${t + 1}/${seqLen}...`);
            
            // Apply positional encoding
            const positionedInput = this.applyPositionalEncoding(currentEmbedding, t);
            
            // Transformer decoder
            const decoderOutput = this.transformerDecoderLayer(
                positionedInput,
                processedAudio[t]
            );
            
            // Map to vertex space
            const vertexDelta = this.linearLayer(
                decoderOutput,
                this.weights['vertice_map_r.weight'],
                this.weights['vertice_map_r.bias']
            );
            
            // Add to template
            const vertices = new Array(verticeDim);
            for (let i = 0; i < verticeDim; i++) {
                vertices[i] = template[i] + vertexDelta[i];
            }
            
            generatedVertices.push(vertices);
            
            // Prepare next input (if not last frame)
            if (t < seqLen - 1) {
                // Map current vertices back to feature space
                const vertexInput = new Array(verticeDim);
                for (let i = 0; i < verticeDim; i++) {
                    vertexInput[i] = vertices[i] - template[i];
                }
                
                const mappedVertex = this.linearLayer(
                    vertexInput,
                    this.weights['vertice_map.weight'],
                    this.weights['vertice_map.bias']
                );
                
                // Add style embedding
                currentEmbedding = this.addVectors(mappedVertex, styleEmbedding);
            }
        }
        
        console.log(`✅ Generated ${generatedVertices.length} vertex frames`);
        return generatedVertices;
    }

    // Get model information
    getModelInfo() {
        return {
            initialized: this.initialized,
            dataset: this.dataset,
            config: this.config,
            hasWeights: !!this.weights,
            totalWeights: this.weights ? Object.keys(this.weights).length : 0
        };
    }
}

// Wav2Vec2 Feature Extractor (mock for web)
class Wav2Vec2WebExtractor {
    constructor() {
        this.sampleRate = 16000;
        this.initialized = false;
    }

    async initialize() {
        console.log('🎵 Initializing Wav2Vec2 Web Extractor...');
        // In a real implementation, you would load the Wav2Vec2 model
        // For now, we'll create a mock that generates realistic features
        this.initialized = true;
        console.log('✅ Wav2Vec2 extractor ready (mock mode)');
        return true;
    }

    async extractFeatures(audioBuffer, frameCount) {
        if (!this.initialized) {
            await this.initialize();
        }

        console.log(`🎵 Extracting audio features for ${frameCount} frames...`);
        
        // Mock feature extraction - in reality you'd use ONNX Runtime with Wav2Vec2
        const features = [];
        for (let i = 0; i < frameCount; i++) {
            const frame = new Array(768);
            for (let j = 0; j < 768; j++) {
                // Generate realistic audio features (small random values)
                frame[j] = (Math.random() - 0.5) * 0.1;
            }
            features.push(frame);
        }
        
        console.log(`✅ Extracted ${features.length} audio feature frames`);
        return features;
    }
}

// Complete Original FaceFormer Web System
class OriginalFaceFormerSystem {
    constructor() {
        this.faceformer = new OriginalFaceFormerWeb();
        this.audioExtractor = new Wav2Vec2WebExtractor();
        this.initialized = false;
    }

    async initialize(dataset = 'vocaset') {
        console.log('🚀 Initializing Original FaceFormer Web System...');
        
        // Initialize components
        const faceformerSuccess = await this.faceformer.initialize(dataset);
        const audioSuccess = await this.audioExtractor.initialize();
        
        this.initialized = faceformerSuccess && audioSuccess;
        
        if (this.initialized) {
            console.log('🎉 Original FaceFormer System ready!');
            console.log('📊 Model info:', this.faceformer.getModelInfo());
        } else {
            console.error('❌ Failed to initialize FaceFormer system');
        }
        
        return this.initialized;
    }

    async generateFromAudio(audioBuffer, template, subjectId = 0, frameCount = null) {
        if (!this.initialized) {
            throw new Error('System not initialized');
        }

        // Determine frame count
        const targetFrames = frameCount || Math.min(50, Math.floor(audioBuffer.length / 1000));
        
        console.log(`🎬 Generating animation (${targetFrames} frames)...`);
        
        // Extract audio features
        const audioFeatures = await this.audioExtractor.extractFeatures(audioBuffer, targetFrames);
        
        // Generate vertices
        const vertices = await this.faceformer.generateVertices(audioFeatures, template, subjectId);
        
        return {
            vertices: vertices,
            frameCount: vertices.length,
            dataset: this.faceformer.dataset,
            subjectId: subjectId,
            generated: new Date().toISOString()
        };
    }

    getSystemStatus() {
        return {
            initialized: this.initialized,
            faceformer: this.faceformer.getModelInfo(),
            audioExtractor: {
                initialized: this.audioExtractor.initialized,
                sampleRate: this.audioExtractor.sampleRate
            }
        };
    }
}

module.exports = {
    OriginalFaceFormerWeb,
    Wav2Vec2WebExtractor,
    OriginalFaceFormerSystem
};
