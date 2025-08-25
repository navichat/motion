// Browser-compatible FaceFormer Web Generator - FIXED VERSION
// Uses the new minimal FaceFormer model that actually works

class FaceFormerWebGeneratorFixed {
    constructor(modelPath = './faceformer/faceformer_minimal.onnx') {
        this.modelPath = modelPath;
        this.session = null;
        this.weightLoader = null;
        this.useLoadedWeights = false;
    }

    async initialize(useExactWeights = true) {
        console.log('🤖 Loading FaceFormer Minimal model...');
        
        // Check if onnxruntime is available
        if (typeof ort === 'undefined') {
            throw new Error('ONNXRuntime not available. Please include onnxruntime-web.');
        }
        
        try {
            // Try to load with exact Python weights first
            if (useExactWeights) {
                const exactModelPath = './faceformer_python_weights.onnx';
                try {
                    this.session = await ort.InferenceSession.create(exactModelPath, {
                        executionProviders: ['wasm']
                    });
                    console.log('✅ Loaded FaceFormer with exact Python weights!');
                    this.useLoadedWeights = true;
                } catch (weightError) {
                    console.warn('⚠️ Could not load model with exact weights, falling back to original model');
                    console.warn('Weight loading error:', weightError.message);
                }
            }
            
            // Fallback to original model if weight loading failed
            if (!this.session) {
                this.session = await ort.InferenceSession.create(this.modelPath, {
                    executionProviders: ['wasm']  // Use WebAssembly backend for browsers
                });
                console.log('✅ Minimal FaceFormer model loaded successfully!');
                
                // Try to load weight loader for potential manual weight application
                try {
                    this.weightLoader = new FaceFormerWeightLoader();
                    const loaded = await this.weightLoader.loadPythonWeights('./faceformer_python_weights.json');
                    if (loaded) {
                        console.log('✅ Python weights loaded for potential manual application');
                    }
                } catch (weightLoaderError) {
                    console.warn('⚠️ Could not load Python weights:', weightLoaderError.message);
                }
            }
            
            // Print model info
            const inputNames = this.session.inputNames || [];
            const outputNames = this.session.outputNames || [];
            console.log('📋 Model inputs:', inputNames);
            console.log('📋 Model outputs:', outputNames);
            console.log('🔧 Using loaded weights:', this.useLoadedWeights);
            
            return true;
        } catch (error) {
            console.error('❌ Failed to load FaceFormer model:', error);
            throw error;
        }
    }

    async generateSequence(audioFeatures, template, oneHot, maxFrames = 100) {
        if (!this.session) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        console.log('🎭 Starting facial animation generation...');
        console.log(`📊 Input data: audio=${audioFeatures.length}, template=${template.length}, oneHot=${oneHot.length}`);
        
        try {
            // The minimal model expects very specific shapes:
            // - audio_features: [1, 1, 768]
            // - vertice_emb: [1, 1, 64] 
            // - one_hot: [1, 3]
            // - template: [1, 1, 15069]
            
            const batchSize = 1;
            
            // Process audio features - take single frame
            let processedAudio;
            if (audioFeatures.length >= 768) {
                // Take first 768 features as single frame
                processedAudio = audioFeatures.slice(0, 768);
            } else {
                // Pad with zeros if too short
                processedAudio = new Array(768).fill(0);
                for (let i = 0; i < Math.min(audioFeatures.length, 768); i++) {
                    processedAudio[i] = audioFeatures[i];
                }
            }
            
            // Process template - ensure correct size
            let processedTemplate;
            if (template.length >= 15069) {
                processedTemplate = template.slice(0, 15069);
            } else {
                processedTemplate = new Array(15069).fill(0);
                for (let i = 0; i < Math.min(template.length, 15069); i++) {
                    processedTemplate[i] = template[i];
                }
            }
            
            // Process one-hot encoding
            let processedOneHot;
            if (oneHot.length >= 3) {
                processedOneHot = oneHot.slice(0, 3);
            } else {
                processedOneHot = [1, 0, 0]; // Default to first subject
            }
            
            // Create initial tensors
            const audioTensor = new ort.Tensor('float32', new Float32Array(processedAudio), [batchSize, 1, 768]);
            const templateTensor = new ort.Tensor('float32', new Float32Array(processedTemplate), [batchSize, 1, 15069]);
            const oneHotTensor = new ort.Tensor('float32', new Float32Array(processedOneHot), [batchSize, 3]);
            
            console.log('✅ Input tensors created successfully');
            console.log(`📐 Shapes: audio=${audioTensor.dims}, template=${templateTensor.dims}, oneHot=${oneHotTensor.dims}`);
            
            // Initialize vertex embedding (64-dimensional)
            let currentVerticeEmb = new Array(64).fill(0.1); // Start with small positive values
            
            const generatedVertices = [];
            const testMaxFrames = Math.min(maxFrames, 50); // Limit for testing
            
            console.log(`🎬 Generating ${testMaxFrames} animation frames...`);
            
            for (let i = 0; i < testMaxFrames; i++) {
                try {
                    // Create current embedding tensor
                    const verticeEmbTensor = new ort.Tensor('float32', new Float32Array(currentVerticeEmb), [batchSize, 1, 64]);
                    
                    // Prepare model inputs
                    const feeds = {
                        audio_features: audioTensor,
                        vertice_emb: verticeEmbTensor,
                        one_hot: oneHotTensor,
                        template: templateTensor
                    };
                    
                    // Run inference
                    const results = await this.session.run(feeds);
                    
                    // Extract results
                    if (!results.new_vertice_out || !results.updated_vertice_emb) {
                        console.warn(`⚠️ Missing outputs at frame ${i}:`, Object.keys(results));
                        break;
                    }
                    
                    const newVerticeOut = Array.from(results.new_vertice_out.data);
                    const updatedVerticeEmb = Array.from(results.updated_vertice_emb.data);
                    
                    // Store generated frame
                    generatedVertices.push(newVerticeOut);
                    
                    // Update embedding for next iteration
                    currentVerticeEmb = updatedVerticeEmb;
                    
                    // Progress logging
                    if ((i + 1) % 10 === 0 || i < 5) {
                        console.log(`📈 Generated frame ${i + 1}/${testMaxFrames}`);
                    }
                    
                } catch (frameError) {
                    console.error(`❌ Error generating frame ${i + 1}:`, frameError);
                    break;
                }
            }
            
            console.log(`✅ Generation complete! Created ${generatedVertices.length} frames`);
            
            if (generatedVertices.length === 0) {
                throw new Error('No frames were generated successfully');
            }
            
            return generatedVertices;
            
        } catch (error) {
            console.error('❌ FaceFormer generation error:', error);
            throw error;
        }
    }

    // Test the model with minimal inputs
    async testModel() {
        if (!this.session) {
            throw new Error('Model not initialized');
        }
        
        console.log('🧪 Testing FaceFormer model...');
        
        try {
            // Create minimal test inputs
            const audioTensor = new ort.Tensor('float32', new Float32Array(768).fill(0.01), [1, 1, 768]);
            const verticeEmbTensor = new ort.Tensor('float32', new Float32Array(64).fill(0.01), [1, 1, 64]);
            const oneHotTensor = new ort.Tensor('float32', new Float32Array([1, 0, 0]), [1, 3]);
            const templateTensor = new ort.Tensor('float32', new Float32Array(15069).fill(0.01), [1, 1, 15069]);
            
            const feeds = {
                audio_features: audioTensor,
                vertice_emb: verticeEmbTensor,
                one_hot: oneHotTensor,
                template: templateTensor
            };
            
            const results = await this.session.run(feeds);
            
            console.log('✅ Model test successful!');
            console.log('📊 Output shapes:');
            Object.keys(results).forEach(key => {
                console.log(`  ${key}: ${JSON.stringify(results[key].dims)}`);
            });
            
            return true;
            
        } catch (error) {
            console.error('❌ Model test failed:', error);
            return false;
        }
    }
}

// Export for use in other modules (Node.js)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FaceFormerWebGeneratorFixed };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.FaceFormerWebGeneratorFixed = FaceFormerWebGeneratorFixed;
}
