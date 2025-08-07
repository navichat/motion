// Enhanced FaceFormer Web Generator (CommonJS version)
const { FaceFormerWeightLoader, SimpleFaceFormerJS } = require('./weight_loader.cjs');

class FaceFormerWebGeneratorEnhanced {
    constructor() {
        this.session = null;
        this.useExactWeights = false;
        this.weightLoader = null;
        this.customModel = null;
        this.initialized = false;
    }

    async initialize(options = {}) {
        console.log('🤖 Initializing Enhanced FaceFormer with Python weight support...');
        
        const {
            modelPath = './faceformer_minimal.onnx',
            pythonWeightsPath = './faceformer_python_weights.json',
            useExactWeights = true
        } = options;

        try {
            // Try to load exact Python weights first
            if (useExactWeights) {
                console.log('🔄 Loading exact Python weights...');
                this.weightLoader = new FaceFormerWeightLoader();
                const weightsLoaded = await this.weightLoader.loadPythonWeights(pythonWeightsPath);
                
                if (weightsLoaded) {
                    console.log('✅ Loaded exact Python weights, initializing custom model...');
                    this.customModel = new SimpleFaceFormerJS();
                    this.weightLoader.applyToCustomModel(this.customModel);
                    this.useExactWeights = true;
                    console.log('🎯 Enhanced FaceFormer initialized with exact Python weights!');
                } else {
                    console.warn('⚠️ Failed to load Python weights, falling back to ONNX...');
                    await this.initializeONNX(modelPath);
                }
            } else {
                await this.initializeONNX(modelPath);
            }

            this.initialized = true;
            return true;

        } catch (error) {
            console.error('❌ Failed to initialize Enhanced FaceFormer:', error);
            // Fallback to basic ONNX initialization
            try {
                await this.initializeONNX(modelPath);
                this.initialized = true;
                return true;
            } catch (fallbackError) {
                console.error('❌ Fallback initialization also failed:', fallbackError);
                return false;
            }
        }
    }

    async initializeONNX(modelPath) {
        console.log('📁 Initializing ONNX session:', modelPath);
        
        // This would use the actual ONNX Runtime in a real environment
        if (typeof ort !== 'undefined') {
            this.session = await ort.InferenceSession.create(modelPath);
            console.log('✅ ONNX session initialized');
        } else {
            console.warn('⚠️ ONNX Runtime not available, using mock session');
            this.session = {
                run: async () => {
                    throw new Error('ONNX Runtime not available');
                }
            };
        }
        this.useExactWeights = false;
    }

    async generateFrame(audioFeatures, verticeEmb, oneHot, template) {
        if (!this.initialized) {
            throw new Error('Generator not initialized. Call initialize() first.');
        }

        try {
            if (this.useExactWeights && this.customModel) {
                return await this.generateWithExactWeights(audioFeatures, verticeEmb, oneHot, template);
            } else {
                return await this.generateWithONNX(audioFeatures, verticeEmb, oneHot, template);
            }
        } catch (error) {
            console.error('❌ Frame generation failed:', error);
            throw error;
        }
    }

    async generateWithExactWeights(audioFeatures, verticeEmb, oneHot, template) {
        const inputs = {
            audio_features: audioFeatures,
            vertice_emb: verticeEmb,
            one_hot: oneHot,
            template: template
        };

        const outputs = this.customModel.forward(inputs);
        
        return {
            vertices: outputs.vertices,
            embedding: outputs.embedding,
            usingExactWeights: true,
            consistency: 'perfect'
        };
    }

    async generateWithONNX(audioFeatures, verticeEmb, oneHot, template) {
        console.log('📁 Generating frame with ONNX session...');
        
        // Prepare tensors
        const feeds = {
            audio_features: new ort.Tensor('float32', new Float32Array(audioFeatures), [1, 1, audioFeatures.length]),
            vertice_emb: new ort.Tensor('float32', new Float32Array(verticeEmb), [1, 1, verticeEmb.length]),
            one_hot: new ort.Tensor('float32', new Float32Array(oneHot), [1, oneHot.length]),
            template: new ort.Tensor('float32', new Float32Array(template), [1, 1, template.length])
        };

        const results = await this.session.run(feeds);
        
        return {
            vertices: Array.from(results.new_vertice_out.data),
            embedding: Array.from(results.updated_vertice_emb.data),
            usingExactWeights: false,
            consistency: 'onnx_based'
        };
    }

    getStatus() {
        return {
            initialized: this.initialized,
            usingExactWeights: this.useExactWeights,
            hasCustomModel: !!this.customModel,
            hasONNXSession: !!this.session,
            hasWeightLoader: !!this.weightLoader
        };
    }

    async benchmark(iterations = 10) {
        if (!this.initialized) {
            throw new Error('Generator not initialized');
        }

        console.log(`🏃 Running benchmark (${iterations} iterations)...`);
        
        // Generate test data
        const audioFeatures = new Array(768).fill(0.01);
        const verticeEmb = new Array(64).fill(0.01);
        const oneHot = [1, 0, 0];
        const template = new Array(15069).fill(0.01);

        const times = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.generateFrame(audioFeatures, verticeEmb, oneHot, template);
            const end = performance.now();
            times.push(end - start);
        }

        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);

        console.log(`📊 Benchmark Results (${this.useExactWeights ? 'Exact Weights' : 'ONNX'}):`);
        console.log(`  Average: ${avgTime.toFixed(2)}ms`);
        console.log(`  Min: ${minTime.toFixed(2)}ms`);
        console.log(`  Max: ${maxTime.toFixed(2)}ms`);

        return {
            iterations,
            averageMs: avgTime,
            minMs: minTime,
            maxMs: maxTime,
            usingExactWeights: this.useExactWeights
        };
    }
}

// Convenience function for creating enhanced generator
async function createEnhancedFaceFormer(options = {}) {
    const generator = new FaceFormerWebGeneratorEnhanced();
    const success = await generator.initialize(options);
    
    if (success) {
        console.log('🎉 Enhanced FaceFormer ready!');
        return generator;
    } else {
        throw new Error('Failed to initialize Enhanced FaceFormer');
    }
}

module.exports = { 
    FaceFormerWebGeneratorEnhanced,
    createEnhancedFaceFormer
};
