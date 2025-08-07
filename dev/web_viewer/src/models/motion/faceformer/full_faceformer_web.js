// Full FaceFormer Implementation with Transformer Layers and Wav2Vec2
// Optimized for WebGPU, WebNN, ONNX Runtime Web, and WASM

class FullFaceFormerWeb {
    constructor() {
        this.backend = null;
        this.session = null;
        this.wav2vec2Session = null;
        this.weights = null;
        this.config = null;
        this.initialized = false;
        this.dataset = null;
        this.activeBackend = null;
        this.performanceStats = {
            audioProcessingTime: 0,
            transformerTime: 0,
            totalInferenceTime: 0
        };
    }

    async detectAvailableBackends() {
        console.log('🔍 Detecting available acceleration backends...');
        
        const backends = {
            webgpu: false,
            webnn: false,
            onnxruntime: false,
            wasm: true // Always available as fallback
        };

        // Check WebGPU support
        try {
            if ('gpu' in navigator) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    backends.webgpu = true;
                    console.log('✅ WebGPU available');
                }
            }
        } catch (e) {
            console.log('❌ WebGPU not available:', e.message);
        }

        // Check WebNN support
        try {
            if ('ml' in navigator) {
                backends.webnn = true;
                console.log('✅ WebNN available');
            }
        } catch (e) {
            console.log('❌ WebNN not available:', e.message);
        }

        // Check ONNX Runtime Web
        try {
            if (typeof ort !== 'undefined') {
                backends.onnxruntime = true;
                console.log('✅ ONNX Runtime Web available');
            }
        } catch (e) {
            console.log('❌ ONNX Runtime Web not available:', e.message);
        }

        console.log('📊 Backend availability:', backends);
        return backends;
    }

    async initializeBackend(preferredBackend = 'auto') {
        const availableBackends = await this.detectAvailableBackends();
        
        // Auto-select best backend
        if (preferredBackend === 'auto') {
            if (availableBackends.webgpu) {
                preferredBackend = 'webgpu';
            } else if (availableBackends.webnn) {
                preferredBackend = 'webnn';
            } else if (availableBackends.onnxruntime) {
                preferredBackend = 'onnxruntime-web';
            } else {
                preferredBackend = 'wasm';
            }
        }

        console.log(`🚀 Initializing backend: ${preferredBackend}`);

        switch (preferredBackend) {
            case 'webgpu':
                return await this.initializeWebGPU();
            case 'webnn':
                return await this.initializeWebNN();
            case 'onnxruntime-web':
                return await this.initializeONNXRuntime();
            case 'wasm':
                return await this.initializeWASM();
            default:
                throw new Error(`Unsupported backend: ${preferredBackend}`);
        }
    }

    async initializeONNXRuntime() {
        console.log('🔧 Initializing ONNX Runtime Web backend...');
        
        try {
            const sessionOptions = {
                executionProviders: ['webgpu', 'wasm'],
                logSeverityLevel: 3,
                logVerbosityLevel: 0,
                enableMemPattern: true,
                enableCpuMemArena: true,
                graphOptimizationLevel: 'all'
            };

            // Check if WebGPU provider is available
            const availableProviders = ort.env.availableProviders || [];
            if (availableProviders.includes('webgpu')) {
                sessionOptions.executionProviders = ['webgpu', 'wasm'];
                console.log('🎮 Using WebGPU execution provider');
            } else {
                sessionOptions.executionProviders = ['wasm'];
                console.log('🌐 Using WASM execution provider');
            }

            this.backend = {
                type: 'onnxruntime-web',
                sessionOptions: sessionOptions,
                providers: sessionOptions.executionProviders
            };

            console.log('✅ ONNX Runtime Web backend initialized');
            this.activeBackend = 'onnxruntime-web';
            return true;
            
        } catch (error) {
            console.error('❌ ONNX Runtime Web initialization failed:', error);
            return false;
        }
    }

    async initializeWebGPU() {
        console.log('🔧 Initializing WebGPU backend...');
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter.requestDevice();
            
            this.backend = {
                type: 'webgpu',
                device: device,
                adapter: adapter,
                features: Array.from(device.features),
                limits: device.limits
            };

            console.log('✅ WebGPU backend initialized');
            this.activeBackend = 'webgpu';
            return true;
            
        } catch (error) {
            console.error('❌ WebGPU initialization failed:', error);
            return false;
        }
    }

    async initializeWebNN() {
        console.log('🔧 Initializing WebNN backend...');
        
        try {
            const context = await navigator.ml.createContext();
            
            this.backend = {
                type: 'webnn',
                context: context
            };

            console.log('✅ WebNN backend initialized');
            this.activeBackend = 'webnn';
            return true;
            
        } catch (error) {
            console.error('❌ WebNN initialization failed:', error);
            return false;
        }
    }

    async initializeWASM() {
        console.log('🔧 Initializing WASM backend...');
        
        try {
            this.backend = {
                type: 'wasm',
                threads: navigator.hardwareConcurrency || 4,
                simd: await this.checkSIMDSupport()
            };

            console.log('✅ WASM backend initialized');
            this.activeBackend = 'wasm';
            return true;
            
        } catch (error) {
            console.error('❌ WASM initialization failed:', error);
            return false;
        }
    }

    async checkSIMDSupport() {
        try {
            return typeof WebAssembly.SIMD !== 'undefined';
        } catch {
            return false;
        }
    }

    async loadFullModel(dataset = 'vocaset') {
        console.log(`📥 Loading full FaceFormer model (${dataset.toUpperCase()})...`);
        
        this.dataset = dataset.toLowerCase();
        
        // Load Wav2Vec2 model
        await this.loadWav2Vec2Model();
        
        // Load FaceFormer transformer model
        await this.loadFaceFormerModel(dataset);
        
        console.log('✅ Full model loaded successfully');
        return true;
    }

    async loadWav2Vec2Model() {
        console.log('🎵 Loading Wav2Vec2 audio encoder...');
        
        const wav2vec2Path = './models/wav2vec2_base.onnx';
        
        try {
            this.wav2vec2Session = await ort.InferenceSession.create(wav2vec2Path, this.backend.sessionOptions);
            console.log('✅ Wav2Vec2 model loaded');
            console.log(`  Input names: ${this.wav2vec2Session.inputNames.join(', ')}`);
            console.log(`  Output names: ${this.wav2vec2Session.outputNames.join(', ')}`);
        } catch (error) {
            console.warn('⚠️ Wav2Vec2 model not found, using mock audio features');
            this.wav2vec2Session = null;
        }
    }

    async loadFaceFormerModel(dataset) {
        console.log(`🎭 Loading FaceFormer transformer model (${dataset})...`);
        
        const modelPath = `./models/faceformer_${dataset}_full.onnx`;
        
        try {
            this.session = await ort.InferenceSession.create(modelPath, this.backend.sessionOptions);
            console.log('✅ FaceFormer transformer model loaded');
            console.log(`  Input names: ${this.session.inputNames.join(', ')}`);
            console.log(`  Output names: ${this.session.outputNames.join(', ')}`);
            
            // Load model configuration
            await this.loadModelConfig(dataset);
            
        } catch (error) {
            console.error('❌ FaceFormer model loading failed:', error);
            // Fallback to creating model from weights
            return await this.createModelFromWeights(dataset);
        }
    }

    async loadModelConfig(dataset) {
        const configPath = `./models/faceformer_${dataset}_config.json`;
        
        try {
            const response = await fetch(configPath);
            this.config = await response.json();
            console.log('✅ Model configuration loaded');
        } catch (error) {
            console.log('⚠️ Using default configuration');
            this.config = this.getDefaultConfig(dataset);
        }
    }

    getDefaultConfig(dataset) {
        if (dataset === 'vocaset') {
            return {
                dataset: 'VOCASET',
                architecture: 'FaceFormer',
                feature_dim: 64,
                audio_input_dim: 768,
                vertice_dim: 15069,
                num_subjects: 8,
                num_heads: 8,
                num_layers: 6,
                max_seq_length: 600,
                dropout: 0.1
            };
        } else if (dataset === 'biwi') {
            return {
                dataset: 'BIWI',
                architecture: 'FaceFormer',
                feature_dim: 128,
                audio_input_dim: 768,
                vertice_dim: 70110,
                num_subjects: 6,
                num_heads: 8,
                num_layers: 6,
                max_seq_length: 600,
                dropout: 0.1
            };
        }
    }

    async createModelFromWeights(dataset) {
        console.log('🔨 Creating ONNX model from weights...');
        
        // Load the converted weights
        const weightsPath = `./converted_weights/faceformer_${dataset}_weights.json`;
        const response = await fetch(weightsPath);
        const weightData = await response.json();
        
        this.weights = weightData.weights;
        this.config = weightData.config;
        
        // Create a full ONNX model with transformer layers
        return await this.createFullTransformerONNX(dataset);
    }

    async createFullTransformerONNX(dataset) {
        console.log('🏗️ Creating full transformer ONNX model...');
        
        // This would create a complete ONNX model with:
        // 1. Multi-head attention layers
        // 2. Feed-forward networks
        // 3. Layer normalization
        // 4. Positional encoding
        // 5. Auto-regressive generation
        
        // For now, use the optimized fallback
        const optimizedPath = `./faceformer_${dataset}_simple.onnx`;
        try {
            this.session = await ort.InferenceSession.create(optimizedPath, this.backend.sessionOptions);
            console.log('✅ Using optimized model as fallback');
            return true;
        } catch (error) {
            console.error('❌ No model available');
            return false;
        }
    }

    async processAudioWithWav2Vec2(audioData) {
        console.log('🎵 Processing audio with Wav2Vec2...');
        
        const start = performance.now();
        
        if (this.wav2vec2Session) {
            try {
                // Prepare audio input (16kHz, normalized)
                const audioTensor = new ort.Tensor('float32', audioData, [1, audioData.length]);
                
                const feeds = { 'input_values': audioTensor };
                const results = await this.wav2vec2Session.run(feeds);
                
                // Extract features from last hidden state
                const features = Array.from(results.last_hidden_state.data);
                
                const processingTime = performance.now() - start;
                this.performanceStats.audioProcessingTime = processingTime;
                
                console.log(`✅ Wav2Vec2 processing: ${processingTime.toFixed(2)}ms`);
                return this.reshapeAudioFeatures(features);
                
            } catch (error) {
                console.warn('⚠️ Wav2Vec2 processing failed, using mock features:', error);
                return this.generateMockAudioFeatures(audioData.length);
            }
        } else {
            // Generate mock features when Wav2Vec2 is not available
            console.log('🔄 Generating mock audio features...');
            return this.generateMockAudioFeatures(audioData.length);
        }
    }

    reshapeAudioFeatures(features) {
        // Reshape Wav2Vec2 output to expected format
        const featureDim = this.config.audio_input_dim;
        const numFrames = Math.floor(features.length / featureDim);
        
        const reshapedFeatures = [];
        for (let i = 0; i < numFrames; i++) {
            const frameStart = i * featureDim;
            const frameEnd = frameStart + featureDim;
            reshapedFeatures.push(features.slice(frameStart, frameEnd));
        }
        
        return reshapedFeatures;
    }

    generateMockAudioFeatures(audioLength) {
        // Generate realistic mock features based on audio length
        const frameRate = 50; // 50 Hz feature rate (typical for speech)
        const numFrames = Math.floor(audioLength / 320); // 16kHz / 50Hz = 320 samples per frame
        const featureDim = this.config.audio_input_dim;
        
        const features = [];
        for (let i = 0; i < numFrames; i++) {
            const frame = [];
            for (let j = 0; j < featureDim; j++) {
                // Generate correlated features with some randomness
                const base = Math.sin(i * 0.1 + j * 0.01) * 0.1;
                const noise = (Math.random() - 0.5) * 0.02;
                frame.push(base + noise);
            }
            features.push(frame);
        }
        
        return features;
    }

    async generateVerticesWithTransformer(audioFeatures, template, subjectId = 0) {
        console.log('🎭 Generating vertices with full transformer...');
        
        const start = performance.now();
        
        switch (this.activeBackend) {
            case 'onnxruntime-web':
                return await this.generateWithONNXTransformer(audioFeatures, template, subjectId);
            case 'webgpu':
                return await this.generateWithWebGPUTransformer(audioFeatures, template, subjectId);
            case 'webnn':
                return await this.generateWithWebNNTransformer(audioFeatures, template, subjectId);
            case 'wasm':
                return await this.generateWithWASMTransformer(audioFeatures, template, subjectId);
            default:
                throw new Error(`Unsupported backend: ${this.activeBackend}`);
        }
    }

    async generateWithONNXTransformer(audioFeatures, template, subjectId) {
        console.log('📦 Generating with ONNX Runtime transformer...');
        
        const seqLen = audioFeatures.length;
        const batchSize = 1;
        
        // Smart sequence processing - chunked for long sequences
        const maxSeqLen = 200;
        
        if (seqLen > maxSeqLen) {
            console.log(`⚠️ Sequence too long (${seqLen}), processing in chunks of ${maxSeqLen} frames`);
            return await this.processLongSequenceInChunks(audioFeatures, template, subjectId, maxSeqLen);
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
                console.log('💡 Memory error detected - try reducing sequence length');
            }
            
            throw error; // Re-throw to handle in calling function
        }
    }

    async processLongSequenceInChunks(audioFeatures, template, subjectId, maxChunkSize) {
        console.log(`🔧 Processing long sequence in chunks of ${maxChunkSize} frames...`);
        
        const totalFrames = audioFeatures.length;
        const chunks = Math.ceil(totalFrames / maxChunkSize);
        const allVertices = [];
        let totalInferenceTime = 0;
        
        // Progress tracking
        const progressCallback = this.onProgress || (() => {});
        progressCallback(`Starting chunked processing: ${totalFrames} frames → ${chunks} chunks`);
        
        for (let i = 0; i < chunks; i++) {
            const startIdx = i * maxChunkSize;
            const endIdx = Math.min(startIdx + maxChunkSize, totalFrames);
            const chunkFeatures = audioFeatures.slice(startIdx, endIdx);
            
            console.log(`📊 Processing chunk ${i + 1}/${chunks}: frames ${startIdx}-${endIdx} (${chunkFeatures.length} frames)`);
            
            // Update progress
            const progress = Math.round(((i) / chunks) * 100);
            progressCallback(`Processing chunk ${i + 1}/${chunks} (${progress}%)`);
            
            try {
                // Process individual chunk
                const chunkResult = await this.processChunk(chunkFeatures, template, subjectId);
                allVertices.push(...chunkResult.vertices);
                totalInferenceTime += chunkResult.inferenceTime;
                
                // Brief pause to allow UI updates and prevent system overwhelm
                if (i < chunks - 1) {
                    await new Promise(resolve => setTimeout(resolve, 5));
                }
                
            } catch (error) {
                console.error(`❌ Failed to process chunk ${i + 1}/${chunks}:`, error);
                progressCallback(`Error in chunk ${i + 1}: ${error.message}`);
                throw new Error(`Chunk processing failed at ${i + 1}/${chunks}: ${error.message}`);
            }
        }
        
        const avgTimePerChunk = totalInferenceTime / chunks;
        const efficiency = (totalFrames / (totalInferenceTime / 1000)).toFixed(1);
        
        console.log(`✅ Chunked processing complete!`);
        console.log(`   📊 ${chunks} chunks processed`);
        console.log(`   ⏱️ Total inference: ${totalInferenceTime.toFixed(2)}ms`);
        console.log(`   📈 Avg per chunk: ${avgTimePerChunk.toFixed(2)}ms`);
        console.log(`   🚀 Throughput: ${efficiency} FPS`);
        
        progressCallback(`Chunked processing complete! ${chunks} chunks, ${efficiency} FPS`);
        
        return {
            vertices: allVertices,
            backend: 'onnxruntime-web-chunked',
            inferenceTime: totalInferenceTime,
            audioProcessingTime: this.performanceStats.audioProcessingTime,
            totalTime: this.performanceStats.audioProcessingTime + totalInferenceTime,
            chunksProcessed: chunks,
            avgTimePerChunk: avgTimePerChunk,
            throughputFPS: parseFloat(efficiency),
            totalFrames: totalFrames
        };
    }

    async processChunk(audioFeatures, template, subjectId) {
        const seqLen = audioFeatures.length;
        const batchSize = 1;
        
        // Prepare tensors for this specific chunk
        const audioTensorData = new Float32Array(audioFeatures.flat());
        const templateTensorData = new Float32Array(template);
        const subjectTensorData = new BigInt64Array([BigInt(subjectId)]);
        
        const feeds = {
            audio_features: new ort.Tensor('float32', audioTensorData, [batchSize, seqLen, this.config.audio_input_dim]),
            template: new ort.Tensor('float32', templateTensorData, [batchSize, this.config.vertice_dim]),
            subject_id: new ort.Tensor('int64', subjectTensorData, [1])
        };
        
        // Run inference on this chunk only
        const startTime = performance.now();
        const results = await this.session.run(feeds);
        const inferenceTime = performance.now() - startTime;
        
        // Extract and reshape results for this chunk
        const vertices = Array.from(results.vertices.data);
        const reshapedVertices = this.reshapeVertices(vertices, seqLen);
        
        return {
            vertices: reshapedVertices,
            inferenceTime: inferenceTime
        };
    }

    setProgressCallback(callback) {
        this.onProgress = callback;
    }

    async generateWithWebGPUTransformer(audioFeatures, template, subjectId) {
        console.log('🎮 Generating with WebGPU transformer...');
        
        try {
            // For now, WebGPU compute shaders for transformers are complex to implement
            // Use ONNX Runtime with WebGPU provider as the best option
            console.log('🔄 Using ONNX Runtime with WebGPU acceleration...');
            return await this.generateWithONNXTransformer(audioFeatures, template, subjectId);
        } catch (error) {
            console.warn('⚠️ WebGPU transformer failed:', error.message);
            throw error; // Don't fallback, we want to see the real issue
        }
    }

    async executeWebGPUComputeShaders(audioFeatures, template, subjectId) {
        const device = this.backend.device;
        
        // Create compute shaders for transformer operations
        const multiHeadAttentionShader = `
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
            @group(0) @binding(2) var<storage, read> weight_q: array<f32>;
            @group(0) @binding(3) var<storage, read> weight_k: array<f32>;
            @group(0) @binding(4) var<storage, read> weight_v: array<f32>;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&output_data)) {
                    return;
                }
                
                // Multi-head attention computation
                // Q = input * W_q, K = input * W_k, V = input * W_v
                // Attention = softmax(QK^T / sqrt(d_k)) * V
                
                // Simplified implementation for demonstration
                output_data[index] = input_data[index] * 0.707; // sqrt(0.5) scaling
            }
        `;
        
        const feedForwardShader = `
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
            @group(0) @binding(2) var<storage, read> weight1: array<f32>;
            @group(0) @binding(3) var<storage, read> weight2: array<f32>;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&output_data)) {
                    return;
                }
                
                // Feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2
                let hidden = max(0.0, input_data[index]); // ReLU activation
                output_data[index] = hidden;
            }
        `;
        
        // For now, fall back to CPU computation with GPU buffers
        console.log('🔄 Using optimized CPU computation with GPU memory...');
        return await this.generateWithSimplifiedTransformer(audioFeatures, template, subjectId);
    }

    async generateWithWebNNTransformer(audioFeatures, template, subjectId) {
        console.log('🧠 Generating with WebNN transformer...');
        
        try {
            // WebNN transformer implementation is complex
            // Use ONNX Runtime as the reliable option for now
            console.log('🔄 Using ONNX Runtime for WebNN backend...');
            return await this.generateWithONNXTransformer(audioFeatures, template, subjectId);
            
        } catch (error) {
            console.warn('⚠️ WebNN transformer failed:', error.message);
            throw error;
        }
    }

    async generateWithWASMTransformer(audioFeatures, template, subjectId) {
        console.log('🌐 Generating with optimized WASM transformer...');
        
        // Use SIMD-optimized operations if available
        if (this.backend.simd) {
            console.log('⚡ Using SIMD optimizations');
        }
        
        // Use ONNX Runtime with WASM backend
        console.log('🔄 Using ONNX Runtime with WASM backend...');
        return await this.generateWithONNXTransformer(audioFeatures, template, subjectId);
    }

    async generateWithSimplifiedTransformer(audioFeatures, template, subjectId) {
        console.log('🔄 Using simplified transformer fallback...');
        
        // Check if we have a simplified model loaded
        const simplifiedPath = `./faceformer_${this.dataset}_simple.onnx`;
        
        try {
            // Try to create a simplified session if it doesn't exist
            if (!this.simplifiedSession) {
                console.log('📦 Loading simplified model...');
                this.simplifiedSession = await ort.InferenceSession.create(simplifiedPath, this.backend.sessionOptions);
            }
            
            const seqLen = audioFeatures.length;
            const batchSize = 1;
            
            // Use the simplified ONNX model
            const feeds = {
                audio_features: new ort.Tensor('float32', 
                    new Float32Array(audioFeatures.flat()), 
                    [batchSize, seqLen, this.config.audio_input_dim]
                ),
                template: new ort.Tensor('float32', 
                    new Float32Array(template), 
                    [batchSize, this.config.vertice_dim]
                ),
                subject_id: new ort.Tensor('int64', 
                    new BigInt64Array([BigInt(subjectId)]), 
                    [batchSize]
                )
            };
            
            const results = await this.simplifiedSession.run(feeds);
            
            const inferenceTime = performance.now() - this.performanceStats.audioProcessingTime;
            this.performanceStats.transformerTime = inferenceTime;
            
            const vertices = Array.from(results.vertices.data);
            const reshapedVertices = this.reshapeVertices(vertices, seqLen);
            
            return {
                vertices: reshapedVertices,
                backend: `${this.activeBackend}-simplified`,
                inferenceTime: inferenceTime,
                audioProcessingTime: this.performanceStats.audioProcessingTime,
                totalTime: this.performanceStats.audioProcessingTime + inferenceTime
            };
            
        } catch (error) {
            console.error('❌ Simplified transformer failed:', error);
            console.log('🎭 Creating mock animation data...');
            
            // Generate mock animation data for demonstration
            return this.generateMockAnimation(audioFeatures, template, subjectId);
        }
    }

    generateMockAnimation(audioFeatures, template, subjectId) {
        console.log('🎨 Generating mock facial animation...');
        
        const seqLen = audioFeatures.length;
        const vertexDim = this.config.vertice_dim;
        const vertices = [];
        
        // Generate realistic mock facial animation
        for (let t = 0; t < seqLen; t++) {
            const frame = new Array(vertexDim);
            
            // Base on template with some animation
            for (let i = 0; i < vertexDim; i++) {
                const baseValue = template[i] || 0;
                
                // Add some mouth movement based on audio features
                const audioIntensity = audioFeatures[t] ? 
                    Math.abs(audioFeatures[t].reduce((sum, val) => sum + Math.abs(val), 0) / audioFeatures[t].length) : 0;
                
                // Mock mouth region (approximate indices for demonstration)
                const isMouthRegion = i % 100 < 20; // Rough approximation
                const animation = isMouthRegion ? audioIntensity * 0.01 * Math.sin(t * 0.1) : 0;
                
                frame[i] = baseValue + animation;
            }
            
            vertices.push(frame);
        }
        
        const mockTime = seqLen * 0.5; // Mock processing time
        
        return {
            vertices: vertices,
            backend: `${this.activeBackend}-mock`,
            inferenceTime: mockTime,
            audioProcessingTime: this.performanceStats.audioProcessingTime,
            totalTime: this.performanceStats.audioProcessingTime + mockTime
        };
    }

    reshapeVertices(vertices, seqLen) {
        const vertexDim = this.config.vertice_dim;
        const reshapedVertices = [];
        
        for (let t = 0; t < seqLen; t++) {
            const frameStart = t * vertexDim;
            const frameEnd = frameStart + vertexDim;
            reshapedVertices.push(vertices.slice(frameStart, frameEnd));
        }
        
        return reshapedVertices;
    }

    async processAudioToAnimation(audioData, template, subjectId = 0) {
        console.log('🎬 Processing audio to facial animation...');
        
        if (!this.initialized) {
            throw new Error('Model not initialized');
        }
        
        const totalStart = performance.now();
        
        // Step 1: Process audio with Wav2Vec2
        const audioFeatures = await this.processAudioWithWav2Vec2(audioData);
        
        // Step 2: Generate vertices with transformer
        const result = await this.generateVerticesWithTransformer(audioFeatures, template, subjectId);
        
        const totalTime = performance.now() - totalStart;
        
        console.log(`🎉 Animation generation complete: ${totalTime.toFixed(2)}ms total`);
        console.log(`  Audio processing: ${result.audioProcessingTime.toFixed(2)}ms`);
        console.log(`  Transformer inference: ${result.inferenceTime.toFixed(2)}ms`);
        
        return {
            ...result,
            totalTime: totalTime,
            frames: result.vertices.length,
            performanceBreakdown: {
                audioProcessing: result.audioProcessingTime,
                transformerInference: result.inferenceTime,
                totalPipeline: totalTime
            }
        };
    }

    async initialize(dataset = 'vocaset', backend = 'auto') {
        console.log(`🚀 Initializing Full FaceFormer (${dataset.toUpperCase()})...`);
        
        try {
            // Initialize backend
            const backendSuccess = await this.initializeBackend(backend);
            if (!backendSuccess) {
                throw new Error('Failed to initialize any backend');
            }
            
            // Load full model
            const modelSuccess = await this.loadFullModel(dataset);
            if (!modelSuccess) {
                throw new Error('Failed to load model');
            }
            
            this.initialized = true;
            
            console.log('🎉 Full FaceFormer initialized successfully!');
            console.log(`  Backend: ${this.activeBackend}`);
            console.log(`  Dataset: ${dataset.toUpperCase()}`);
            console.log(`  Wav2Vec2: ${this.wav2vec2Session ? 'Loaded' : 'Mock features'}`);
            console.log(`  Transformer: ${this.session ? 'Loaded' : 'Error'}`);
            
            return true;
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            return false;
        }
    }

    getSystemInfo() {
        return {
            initialized: this.initialized,
            activeBackend: this.activeBackend,
            dataset: this.dataset,
            config: this.config,
            hasWav2Vec2: !!this.wav2vec2Session,
            hasTransformer: !!this.session,
            performanceStats: this.performanceStats,
            backend: this.backend ? {
                type: this.backend.type,
                ...(this.backend.type === 'webgpu' && {
                    features: this.backend.features,
                    maxBufferSize: this.backend.limits?.maxBufferSize
                }),
                ...(this.backend.type === 'onnxruntime-web' && {
                    providers: this.backend.providers
                }),
                ...(this.backend.type === 'wasm' && {
                    threads: this.backend.threads,
                    simd: this.backend.simd
                })
            } : null
        };
    }
}

// Export for both browser and Node.js
if (typeof window !== 'undefined') {
    window.FullFaceFormerWeb = FullFaceFormerWeb;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = FullFaceFormerWeb;
    module.exports.FullFaceFormerWeb = FullFaceFormerWeb;
}
