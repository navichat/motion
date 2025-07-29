// Main Thread AI Model Manager
// This runs AI models in the main thread to avoid ONNX Runtime worker issues

class MainThreadModelManager {
    constructor() {
        this.models = new Map();
        this.isInitialized = false;
        this.ort = null;
    }
    
    async initialize() {
        if (this.isInitialized) return true;
        
        try {
            console.log('[Main Thread] Initializing ONNX Runtime...');
            
            // Import ONNX Runtime with fallbacks
            const versions = ['1.19.0', '1.18.0', '1.17.3'];
            
            for (const version of versions) {
                try {
                    const ortModule = await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/ort.min.js`);
                    this.ort = ortModule.default || ortModule;
                    
                    // Configure for main thread
                    this.ort.env.wasm.numThreads = 1;
                    this.ort.env.wasm.simd = false;
                    this.ort.env.wasm.proxy = false;
                    
                    console.log(`[Main Thread] ONNX Runtime ${version} loaded successfully`);
                    this.isInitialized = true;
                    return true;
                    
                } catch (versionError) {
                    console.log(`[Main Thread] ONNX Runtime ${version} failed:`, versionError.message);
                }
            }
            
            console.error('[Main Thread] All ONNX Runtime versions failed');
            return false;
            
        } catch (error) {
            console.error('[Main Thread] Failed to initialize ONNX Runtime:', error);
            return false;
        }
    }
    
    async loadModel(modelType, modelPath) {
        if (this.models.has(modelType)) {
            return this.models.get(modelType);
        }
        
        await this.initialize();
        
        if (!this.ort) {
            console.warn(`[Main Thread] ONNX Runtime not available, using mock for ${modelType}`);
            return this.createMockModel(modelType);
        }
        
        try {
            console.log(`[Main Thread] Loading ${modelType} from ${modelPath}`);
            
            const session = await this.ort.InferenceSession.create(modelPath, {
                executionProviders: ['cpu'],
                graphOptimizationLevel: 'disabled'
            });
            
            console.log(`[Main Thread] Successfully loaded ${modelType}`);
            this.models.set(modelType, session);
            return session;
            
        } catch (error) {
            console.error(`[Main Thread] Failed to load ${modelType}:`, error);
            const mockModel = this.createMockModel(modelType);
            this.models.set(modelType, mockModel);
            return mockModel;
        }
    }
    
    createMockModel(modelType) {
        return {
            isMock: true,
            modelType: modelType,
            run: async (inputs) => {
                // Return realistic mock outputs based on model type
                if (modelType === 'TinyLlama' || modelType === 'DiabloGPT') {
                    return {
                        logits: new this.ort.Tensor('float32', new Float32Array(50257).fill(0.1), [1, 50257])
                    };
                } else if (modelType === 'Whisper') {
                    return {
                        output: new this.ort.Tensor('float32', new Float32Array(100).fill(0.5), [1, 100])
                    };
                } else {
                    return {
                        output: new this.ort.Tensor('float32', new Float32Array(10).fill(0.3), [1, 10])
                    };
                }
            }
        };
    }
    
    async runInference(modelType, modelPath, inputs) {
        const model = await this.loadModel(modelType, modelPath);
        
        try {
            const result = await model.run(inputs);
            
            return {
                success: true,
                result: result,
                isMock: model.isMock || false,
                modelType: modelType
            };
            
        } catch (error) {
            console.error(`[Main Thread] Inference failed for ${modelType}:`, error);
            return {
                success: false,
                error: error.message,
                modelType: modelType
            };
        }
    }
}

// Global instance
window.mainThreadModelManager = new MainThreadModelManager();

console.log('[Main Thread] Main Thread Model Manager initialized');
