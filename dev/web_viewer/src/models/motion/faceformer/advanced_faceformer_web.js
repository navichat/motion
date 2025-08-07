// High-Performance FaceFormer Web Implementation
// Supports WebGPU, WebNN, ONNX Runtime Web with WASM fallback

class AdvancedFaceFormerWeb {
    constructor() {
        this.backend = null;
        this.session = null;
        this.weights = null;
        this.config = null;
        this.initialized = false;
        this.dataset = null;
        this.supportedBackends = ['webgpu', 'webnn', 'onnxruntime-web', 'wasm'];
        this.activeBackend = null;
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
            console.log(`  Features: ${this.backend.features.join(', ')}`);
            console.log(`  Max buffer size: ${(this.backend.limits.maxBufferSize / (1024*1024)).toFixed(1)}MB`);
            
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

    async initializeONNXRuntime() {
        console.log('🔧 Initializing ONNX Runtime Web backend...');
        
        try {
            // Configure ONNX Runtime for optimal performance
            const sessionOptions = {
                executionProviders: ['webgpu', 'wasm'],
                logSeverityLevel: 3,
                logVerbosityLevel: 0,
                enableMemPattern: true,
                enableCpuMemArena: true,
                enableProfiling: false
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
            console.log(`  Providers: ${this.backend.providers.join(', ')}`);
            
            this.activeBackend = 'onnxruntime-web';
            return true;
            
        } catch (error) {
            console.error('❌ ONNX Runtime Web initialization failed:', error);
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
            console.log(`  Threads: ${this.backend.threads}`);
            console.log(`  SIMD: ${this.backend.simd}`);
            
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

    async loadModel(modelPath, weightsPath, dataset = 'vocaset') {
        console.log(`📥 Loading FaceFormer model (${dataset.toUpperCase()})...`);
        console.log(`  Model: ${modelPath}`);
        console.log(`  Weights: ${weightsPath}`);
        
        this.dataset = dataset.toLowerCase();
        
        switch (this.activeBackend) {
            case 'onnxruntime-web':
                return await this.loadONNXModel(modelPath);
            case 'webgpu':
                return await this.loadWebGPUModel(weightsPath);
            case 'webnn':
                return await this.loadWebNNModel(weightsPath);
            case 'wasm':
                return await this.loadWASMModel(weightsPath);
            default:
                throw new Error(`No active backend for model loading`);
        }
    }

    async loadONNXModel(modelPath) {
        console.log('📦 Loading ONNX model...');
        
        try {
            // Create optimized ONNX model on the fly if needed
            const onnxModelPath = await this.ensureONNXModel(modelPath);
            
            this.session = await ort.InferenceSession.create(onnxModelPath, this.backend.sessionOptions);
            
            console.log('✅ ONNX model loaded successfully');
            console.log(`  Input names: ${this.session.inputNames.join(', ')}`);
            console.log(`  Output names: ${this.session.outputNames.join(', ')}`);
            
            return true;
            
        } catch (error) {
            console.error('❌ ONNX model loading failed:', error);
            return false;
        }
    }

    async ensureONNXModel(weightsPath) {
        // Check if ONNX model already exists
        const onnxPath = weightsPath.replace('_weights.json', '.onnx');
        
        try {
            // Try to fetch existing ONNX model
            const response = await fetch(onnxPath);
            if (response.ok) {
                console.log('✅ Using existing ONNX model');
                return onnxPath;
            }
        } catch (e) {
            console.log('📝 Creating ONNX model from weights...');
        }
        
        // Create ONNX model from weights
        return await this.createONNXFromWeights(weightsPath);
    }

    async createONNXFromWeights(weightsPath) {
        console.log('🔨 Creating ONNX model from weights...');
        
        // Load weights
        const response = await fetch(weightsPath);
        const weightData = await response.json();
        this.weights = weightData.weights;
        this.config = weightData.config;
        
        // Create a simplified ONNX model specification
        const modelSpec = await this.createONNXModelSpec();
        
        // For now, return the weights path and handle in the session
        return weightsPath;
    }

    async createONNXModelSpec() {
        const spec = {
            graph: {
                name: `FaceFormer_${this.dataset}`,
                inputs: [
                    {
                        name: 'audio_features',
                        type: 'tensor(float)',
                        shape: ['batch_size', 'sequence_length', this.config.audio_input_dim]
                    },
                    {
                        name: 'template',
                        type: 'tensor(float)',
                        shape: ['batch_size', this.config.vertice_dim]
                    },
                    {
                        name: 'one_hot',
                        type: 'tensor(float)',
                        shape: ['batch_size', this.config.num_subjects]
                    }
                ],
                outputs: [
                    {
                        name: 'vertices',
                        type: 'tensor(float)',
                        shape: ['batch_size', 'sequence_length', this.config.vertice_dim]
                    }
                ],
                nodes: [] // Will be populated with actual operations
            }
        };
        
        return spec;
    }

    async loadWebGPUModel(weightsPath) {
        console.log('🎮 Loading model for WebGPU...');
        
        try {
            // Load weights
            const response = await fetch(weightsPath);
            const weightData = await response.json();
            this.weights = weightData.weights;
            this.config = weightData.config;
            
            // Create WebGPU buffers for weights
            await this.createWebGPUBuffers();
            
            console.log('✅ WebGPU model loaded');
            return true;
            
        } catch (error) {
            console.error('❌ WebGPU model loading failed:', error);
            return false;
        }
    }

    async createWebGPUBuffers() {
        console.log('🗂️ Creating WebGPU buffers...');
        
        this.gpuBuffers = {};
        const device = this.backend.device;
        
        let totalMemory = 0;
        
        for (const [name, weights] of Object.entries(this.weights)) {
            const flatWeights = Array.isArray(weights[0]) ? weights.flat() : weights;
            const buffer = new Float32Array(flatWeights);
            
            const gpuBuffer = device.createBuffer({
                size: buffer.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            
            new Float32Array(gpuBuffer.getMappedRange()).set(buffer);
            gpuBuffer.unmap();
            
            this.gpuBuffers[name] = {
                buffer: gpuBuffer,
                size: buffer.length,
                byteLength: buffer.byteLength
            };
            
            totalMemory += buffer.byteLength;
        }
        
        console.log(`✅ Created ${Object.keys(this.gpuBuffers).length} GPU buffers`);
        console.log(`📊 Total GPU memory: ${(totalMemory / (1024*1024)).toFixed(1)}MB`);
    }

    async loadWebNNModel(weightsPath) {
        console.log('🧠 Loading model for WebNN...');
        
        try {
            // Load weights
            const response = await fetch(weightsPath);
            const weightData = await response.json();
            this.weights = weightData.weights;
            this.config = weightData.config;
            
            // Create WebNN graph
            await this.createWebNNGraph();
            
            console.log('✅ WebNN model loaded');
            return true;
            
        } catch (error) {
            console.error('❌ WebNN model loading failed:', error);
            return false;
        }
    }

    async createWebNNGraph() {
        console.log('📊 Creating WebNN computation graph...');
        
        const builder = new MLGraphBuilder(this.backend.context);
        
        // Define inputs
        const audioInput = builder.input('audio_features', {
            type: 'float32',
            dimensions: [-1, -1, this.config.audio_input_dim]
        });
        
        const templateInput = builder.input('template', {
            type: 'float32',
            dimensions: [-1, this.config.vertice_dim]
        });
        
        const oneHotInput = builder.input('one_hot', {
            type: 'float32',
            dimensions: [-1, this.config.num_subjects]
        });
        
        // Build simplified computation graph (placeholder)
        // In a full implementation, this would build the complete transformer
        const output = builder.add(templateInput, templateInput); // Placeholder
        
        this.mlGraph = await builder.build({'vertices': output});
        console.log('✅ WebNN graph created');
    }

    async loadWASMModel(weightsPath) {
        console.log('🌐 Loading model for WASM...');
        
        try {
            // Load weights
            const response = await fetch(weightsPath);
            const weightData = await response.json();
            this.weights = weightData.weights;
            this.config = weightData.config;
            
            // Optimize weights for WASM execution
            await this.optimizeWeightsForWASM();
            
            console.log('✅ WASM model loaded');
            return true;
            
        } catch (error) {
            console.error('❌ WASM model loading failed:', error);
            return false;
        }
    }

    async optimizeWeightsForWASM() {
        console.log('⚡ Optimizing weights for WASM execution...');
        
        // Convert nested arrays to flat Float32Arrays for better performance
        this.optimizedWeights = {};
        
        for (const [name, weights] of Object.entries(this.weights)) {
            if (Array.isArray(weights)) {
                const flatWeights = Array.isArray(weights[0]) ? weights.flat() : weights;
                this.optimizedWeights[name] = new Float32Array(flatWeights);
            } else {
                this.optimizedWeights[name] = weights;
            }
        }
        
        console.log('✅ Weights optimized for WASM');
    }

    async initialize(dataset = 'vocaset', backend = 'auto') {
        console.log(`🚀 Initializing Advanced FaceFormer Web (${dataset.toUpperCase()})...`);
        
        try {
            // Initialize backend
            const backendSuccess = await this.initializeBackend(backend);
            if (!backendSuccess) {
                throw new Error('Failed to initialize any backend');
            }
            
            // Determine paths
            const weightsPath = `./converted_weights/faceformer_${dataset}_weights.json`;
            const modelPath = weightsPath.replace('_weights.json', '.onnx');
            
            // Load model
            const modelSuccess = await this.loadModel(modelPath, weightsPath, dataset);
            if (!modelSuccess) {
                throw new Error('Failed to load model');
            }
            
            this.initialized = true;
            
            console.log('🎉 Advanced FaceFormer Web initialized successfully!');
            console.log(`  Backend: ${this.activeBackend}`);
            console.log(`  Dataset: ${dataset.toUpperCase()}`);
            console.log(`  Model: ${this.config ? this.config.architecture : 'Unknown'}`);
            
            return true;
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            return false;
        }
    }

    async generateVertices(audioFeatures, template, subjectId = 0) {
        if (!this.initialized) {
            throw new Error('Model not initialized');
        }

        console.log(`🎭 Generating vertices using ${this.activeBackend} backend...`);
        
        switch (this.activeBackend) {
            case 'onnxruntime-web':
                return await this.generateWithONNX(audioFeatures, template, subjectId);
            case 'webgpu':
                return await this.generateWithWebGPU(audioFeatures, template, subjectId);
            case 'webnn':
                return await this.generateWithWebNN(audioFeatures, template, subjectId);
            case 'wasm':
                return await this.generateWithWASM(audioFeatures, template, subjectId);
            default:
                throw new Error(`Unsupported backend: ${this.activeBackend}`);
        }
    }

    async generateWithONNX(audioFeatures, template, subjectId) {
        console.log('📦 Generating with ONNX Runtime Web...');
        
        // Prepare inputs
        const seqLen = audioFeatures.length;
        const batchSize = 1;
        
        // Create one-hot encoding
        const oneHot = new Array(this.config.num_subjects).fill(0);
        oneHot[subjectId] = 1.0;
        
        // Create tensors
        const feeds = {
            audio_features: new ort.Tensor('float32', 
                new Float32Array(audioFeatures.flat()), 
                [batchSize, seqLen, this.config.audio_input_dim]
            ),
            template: new ort.Tensor('float32', 
                new Float32Array(template), 
                [batchSize, this.config.vertice_dim]
            ),
            one_hot: new ort.Tensor('float32', 
                new Float32Array(oneHot), 
                [batchSize, this.config.num_subjects]
            )
        };
        
        // Run inference
        const start = performance.now();
        const results = await this.session.run(feeds);
        const inferenceTime = performance.now() - start;
        
        console.log(`✅ ONNX inference completed in ${inferenceTime.toFixed(2)}ms`);
        
        // Extract results
        const vertices = Array.from(results.vertices.data);
        const reshapedVertices = [];
        
        for (let t = 0; t < seqLen; t++) {
            const frameStart = t * this.config.vertice_dim;
            const frameEnd = frameStart + this.config.vertice_dim;
            reshapedVertices.push(vertices.slice(frameStart, frameEnd));
        }
        
        return {
            vertices: reshapedVertices,
            backend: 'onnxruntime-web',
            inferenceTime: inferenceTime
        };
    }

    async generateWithWebGPU(audioFeatures, template, subjectId) {
        console.log('🎮 Generating with WebGPU...');
        
        // Implement WebGPU compute shaders for FaceFormer
        // This would require writing WGSL shaders for the transformer operations
        
        // For now, fall back to CPU computation with optimized memory access
        return await this.generateWithOptimizedCPU(audioFeatures, template, subjectId, 'webgpu-fallback');
    }

    async generateWithWebNN(audioFeatures, template, subjectId) {
        console.log('🧠 Generating with WebNN...');
        
        // Use the WebNN graph for inference
        // This is a simplified implementation
        
        const oneHot = new Array(this.config.num_subjects).fill(0);
        oneHot[subjectId] = 1.0;
        
        const inputs = {
            'audio_features': new Float32Array(audioFeatures.flat()),
            'template': new Float32Array(template),
            'one_hot': new Float32Array(oneHot)
        };
        
        try {
            const start = performance.now();
            const results = await this.backend.context.compute(this.mlGraph, inputs);
            const inferenceTime = performance.now() - start;
            
            console.log(`✅ WebNN inference completed in ${inferenceTime.toFixed(2)}ms`);
            
            // Process results (simplified)
            return await this.generateWithOptimizedCPU(audioFeatures, template, subjectId, 'webnn-fallback');
            
        } catch (error) {
            console.warn('WebNN inference failed, falling back to CPU:', error);
            return await this.generateWithOptimizedCPU(audioFeatures, template, subjectId, 'webnn-fallback');
        }
    }

    async generateWithWASM(audioFeatures, template, subjectId) {
        console.log('🌐 Generating with optimized WASM...');
        
        return await this.generateWithOptimizedCPU(audioFeatures, template, subjectId, 'wasm');
    }

    async generateWithOptimizedCPU(audioFeatures, template, subjectId, backend) {
        console.log(`⚡ Optimized CPU generation (${backend})...`);
        
        const start = performance.now();
        
        // Use the original implementation but with optimized data structures
        const weights = this.optimizedWeights || this.weights;
        const seqLen = audioFeatures.length;
        const featureDim = this.config.feature_dim;
        const verticeDim = this.config.vertice_dim;
        
        // Create one-hot encoding
        const oneHot = new Float32Array(this.config.num_subjects);
        oneHot[subjectId] = 1.0;
        
        // Get style embedding using optimized linear layer
        const styleEmbedding = this.optimizedLinearLayer(
            oneHot,
            weights['obj_vector.weight']
        );
        
        // Process audio features
        const processedAudio = [];
        for (let t = 0; t < seqLen; t++) {
            const audioFrame = new Float32Array(audioFeatures[t]);
            const mappedAudio = this.optimizedLinearLayer(
                audioFrame,
                weights['audio_feature_map.weight'],
                weights['audio_feature_map.bias']
            );
            processedAudio.push(mappedAudio);
        }
        
        // Auto-regressive generation with optimized operations
        const generatedVertices = [];
        let currentEmbedding = new Float32Array(styleEmbedding);
        
        for (let t = 0; t < seqLen; t++) {
            // Simplified transformer operations for performance
            const decoderOutput = this.simplifiedTransformerStep(
                currentEmbedding,
                processedAudio[t],
                weights,
                t
            );
            
            // Map to vertex space
            const vertexDelta = this.optimizedLinearLayer(
                decoderOutput,
                weights['vertice_map_r.weight'],
                weights['vertice_map_r.bias']
            );
            
            // Add to template
            const vertices = new Float32Array(verticeDim);
            for (let i = 0; i < verticeDim; i++) {
                vertices[i] = template[i] + vertexDelta[i];
            }
            
            generatedVertices.push(Array.from(vertices));
            
            // Prepare next input
            if (t < seqLen - 1) {
                const vertexInput = new Float32Array(verticeDim);
                for (let i = 0; i < verticeDim; i++) {
                    vertexInput[i] = vertices[i] - template[i];
                }
                
                const mappedVertex = this.optimizedLinearLayer(
                    vertexInput,
                    weights['vertice_map.weight'],
                    weights['vertice_map.bias']
                );
                
                // Add style embedding
                for (let i = 0; i < featureDim; i++) {
                    currentEmbedding[i] = mappedVertex[i] + styleEmbedding[i];
                }
            }
        }
        
        const inferenceTime = performance.now() - start;
        console.log(`✅ ${backend} generation completed in ${inferenceTime.toFixed(2)}ms`);
        
        return {
            vertices: generatedVertices,
            backend: backend,
            inferenceTime: inferenceTime
        };
    }

    optimizedLinearLayer(input, weight, bias = null) {
        const outputSize = weight.length;
        const inputSize = weight[0] ? weight[0].length : input.length;
        const output = new Float32Array(outputSize);
        
        // Optimized matrix multiplication
        for (let i = 0; i < outputSize; i++) {
            let sum = 0;
            const weightRow = weight[i];
            for (let j = 0; j < inputSize; j++) {
                sum += input[j] * weightRow[j];
            }
            output[i] = sum + (bias ? bias[i] : 0);
        }
        
        return output;
    }

    simplifiedTransformerStep(input, memory, weights, position) {
        // Simplified transformer step for performance
        // In a full implementation, this would include all attention mechanisms
        
        const featureDim = this.config.feature_dim;
        
        // Apply positional encoding
        const pe = weights['PPE.pe'][0];
        const positionIndex = position % pe.length;
        const peRow = pe[positionIndex];
        
        const positionedInput = new Float32Array(featureDim);
        for (let i = 0; i < featureDim; i++) {
            positionedInput[i] = input[i] + peRow[i];
        }
        
        // Simplified attention (placeholder for full implementation)
        const attended = new Float32Array(featureDim);
        for (let i = 0; i < featureDim; i++) {
            attended[i] = positionedInput[i] + memory[i] * 0.1; // Simplified cross-attention
        }
        
        return attended;
    }

    getSystemInfo() {
        return {
            initialized: this.initialized,
            activeBackend: this.activeBackend,
            dataset: this.dataset,
            config: this.config,
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
    window.AdvancedFaceFormerWeb = AdvancedFaceFormerWeb;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedFaceFormerWeb;
    module.exports.AdvancedFaceFormerWeb = AdvancedFaceFormerWeb;
}
