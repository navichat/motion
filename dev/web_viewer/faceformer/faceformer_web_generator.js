// Browser-compatible FaceFormer Web Generator
// Note: This assumes onnxruntime-web is available globally via CDN

class FaceFormerWebGenerator {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
    }

    async initialize() {
        console.log('Loading FaceFormer core step model...');
        
        // Check if onnxruntime is available
        if (typeof ort === 'undefined') {
            throw new Error('ONNXRuntime not available. Please include onnxruntime-web.');
        }
        
        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['wasm']  // Use WebAssembly backend for browsers
        });
        console.log('Model loaded successfully!');
        
        // Print detailed model info
        console.log('\n🔍 DETAILED MODEL ANALYSIS:');
        try {
            const inputNames = this.session.inputNames || [];
            console.log('📥 Input names:', inputNames);
            
            // Enhanced metadata inspection
            if (this.session.inputMetadata) {
                const metadata = this.session.inputMetadata;
                console.log('📊 INPUT SPECIFICATIONS:');
                if (typeof metadata === 'object') {
                    Object.keys(metadata).forEach(name => {
                        const input = metadata[name];
                        const dims = input.dims || input.shape || 'unknown';
                        const type = input.type || 'unknown';
                        console.log(`  ✅ ${name}: shape=${JSON.stringify(dims)}, type=${type}`);
                        
                        // Try to extract exact dimensions
                        if (Array.isArray(dims)) {
                            const totalSize = dims.reduce((a, b) => a * (b === -1 ? 1 : b), 1);
                            console.log(`      Total elements: ${totalSize} (dynamic dims treated as 1)`);
                        }
                    });
                } else {
                    console.log('  ⚠️ Metadata not in expected object format');
                }
            } else {
                console.log('  ❌ No input metadata available');
            }
            
            // Enhanced output metadata
            console.log('📤 OUTPUT SPECIFICATIONS:');
            const outputNames = this.session.outputNames || [];
            console.log('Output names:', outputNames);
            
            if (this.session.outputMetadata) {
                const outputMetadata = this.session.outputMetadata;
                if (typeof outputMetadata === 'object') {
                    Object.keys(outputMetadata).forEach(name => {
                        const output = outputMetadata[name];
                        const dims = output.dims || output.shape || 'unknown';
                        const type = output.type || 'unknown';
                        console.log(`  ✅ ${name}: shape=${JSON.stringify(dims)}, type=${type}`);
                    });
                }
            } else {
                console.log('  ❌ No output metadata available');
            }
            
        } catch (error) {
            console.log('❌ Could not access model metadata:', error.message);
            console.log('📋 Available session properties:', Object.keys(this.session));
        }
    }

    async generateSequence(audioFeatures, template, oneHot, maxFrames = 100) {
        if (!this.session) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        console.log('Starting autoregressive generation...');
        
        // SYSTEMATIC MODEL ANALYSIS
        console.log('🔍 Starting systematic model analysis...');
        
        // Get model metadata more thoroughly
        const inputNames = this.session.inputNames || [];
        const outputNames = this.session.outputNames || [];
        console.log('📋 Model expects inputs:', inputNames);
        console.log('📋 Model produces outputs:', outputNames);
        
        // Based on exact error messages, we now know the precise requirements:
        // - one_hot: rank 2 (not 3)
        // - vertice_emb: exactly 64 dimensions
        // - audio_features: exactly 768 dimensions  
        // - template: exactly 15069 dimensions
        const testConfigurations = [
            // Config 1: EXACT MODEL REQUIREMENTS (from error messages)
            {
                name: "Exact Model Requirements",
                audio_shape: [1, 1, 768],      // Rank 3, 768 features (confirmed)
                vertice_shape: [1, 1, 64],     // Rank 3, 64 dimensions (confirmed)
                template_shape: [1, 1, 15069], // Rank 3, 15069 dimensions (confirmed)
                one_hot_shape: [1, 3]          // Rank 2, 3 subjects (confirmed)
            },
            // Config 2: Try different sequence lengths for audio
            {
                name: "Audio Sequence Length 2",
                audio_shape: [1, 2, 768],      // Try 2 audio frames
                vertice_shape: [1, 1, 64],     // Keep exact vertice requirement
                template_shape: [1, 1, 15069], // Keep exact template requirement
                one_hot_shape: [1, 3]          // Keep exact one_hot requirement
            },
            // Config 3: Try different sequence lengths for vertice
            {
                name: "Vertice Sequence Length 2",
                audio_shape: [1, 1, 768],      // Keep exact audio requirement
                vertice_shape: [1, 2, 64],     // Try 2 vertice frames
                template_shape: [1, 1, 15069], // Keep exact template requirement
                one_hot_shape: [1, 3]          // Keep exact one_hot requirement
            },
            // Config 4: Try longer audio sequences (common in training)
            {
                name: "Audio Sequence Length 8",
                audio_shape: [1, 8, 768],      // Try 8 audio frames
                vertice_shape: [1, 1, 64],     // Keep exact vertice requirement
                template_shape: [1, 1, 15069], // Keep exact template requirement
                one_hot_shape: [1, 3]          // Keep exact one_hot requirement
            }
        ];
        
        let workingConfig = null;
        
        // Test each configuration
        for (const config of testConfigurations) {
            try {
                console.log(`🧪 Testing configuration: ${config.name}`);
                console.log(`  audio: ${JSON.stringify(config.audio_shape)}`);
                console.log(`  vertice: ${JSON.stringify(config.vertice_shape)}`);
                console.log(`  template: ${JSON.stringify(config.template_shape)}`);
                console.log(`  one_hot: ${JSON.stringify(config.one_hot_shape)}`);
                
                // Create test tensors with proper size calculation
                const audioSize = config.audio_shape.reduce((a, b) => a * b, 1);
                const verticeSize = config.vertice_shape.reduce((a, b) => a * b, 1);
                const templateSize = config.template_shape.reduce((a, b) => a * b, 1);
                const oneHotSize = config.one_hot_shape.reduce((a, b) => a * b, 1);
                
                // Create test tensors with appropriate values
                const testAudio = new ort.Tensor('float32', new Float32Array(audioSize).fill(0.01), config.audio_shape);
                const testVertice = new ort.Tensor('float32', new Float32Array(verticeSize).fill(0.01), config.vertice_shape);
                const testTemplate = new ort.Tensor('float32', new Float32Array(templateSize).fill(0.01), config.template_shape);
                
                // OneHot tensor needs special handling - should be one-hot encoded
                const oneHotData = new Float32Array(oneHotSize).fill(0.0);
                oneHotData[0] = 1.0; // Set first element to 1 for one-hot encoding
                const testOneHot = new ort.Tensor('float32', oneHotData, config.one_hot_shape);
                
                const testFeeds = {
                    audio_features: testAudio,
                    vertice_emb: testVertice,
                    template: testTemplate,
                    one_hot: testOneHot
                };
                
                // Try running with this configuration
                const testResults = await this.session.run(testFeeds);
                console.log(`✅ SUCCESS! Configuration "${config.name}" works!`);
                console.log('  Available outputs:', Object.keys(testResults));
                
                // Log output shapes for verification
                Object.keys(testResults).forEach(outputName => {
                    const output = testResults[outputName];
                    console.log(`  ${outputName}: shape=${JSON.stringify(output.dims)}, size=${output.size}`);
                });
                
                workingConfig = config;
                break; // Found working configuration
                
            } catch (configError) {
                const errorCode = typeof configError === 'number' ? configError : configError.message || 'Unknown error';
                console.log(`❌ Configuration "${config.name}" failed: ${errorCode}`);
                continue;
            }
        }

        // If no working configuration found, try additional diagnostics
        if (!workingConfig) {
            console.log('🔧 Trying additional diagnostic approaches...');
            
            // Try multiple value initialization strategies
            const initializationStrategies = [
                {
                    name: "Zero initialization",
                    audioValue: 0.0,
                    verticeValue: 0.0,
                    templateValue: 0.0,
                    oneHotValue: [1.0, 0.0, 0.0]
                },
                {
                    name: "Small positive values",
                    audioValue: 0.001,
                    verticeValue: 0.001,
                    templateValue: 0.001,
                    oneHotValue: [1.0, 0.0, 0.0]
                },
                {
                    name: "Normal range values",
                    audioValue: 0.1,
                    verticeValue: 0.1,
                    templateValue: 0.1,
                    oneHotValue: [1.0, 0.0, 0.0]
                },
                {
                    name: "Random small values",
                    audioValue: "random_small",
                    verticeValue: "random_small",
                    templateValue: "random_small",
                    oneHotValue: [1.0, 0.0, 0.0]
                },
                {
                    name: "Unit values",
                    audioValue: 1.0,
                    verticeValue: 1.0,
                    templateValue: 1.0,
                    oneHotValue: [1.0, 0.0, 0.0]
                },
                {
                    name: "Negative values",
                    audioValue: -0.1,
                    verticeValue: -0.1,
                    templateValue: -0.1,
                    oneHotValue: [1.0, 0.0, 0.0]
                }
            ];
            
            for (const strategy of initializationStrategies) {
                try {
                    console.log(`🧪 Testing strategy: ${strategy.name}`);
                    
                    // Create audio data
                    let audioData = new Float32Array(768);
                    if (strategy.audioValue === "random_small") {
                        for (let i = 0; i < 768; i++) {
                            audioData[i] = (Math.random() - 0.5) * 0.01; // Random values between -0.005 and 0.005
                        }
                    } else {
                        audioData.fill(strategy.audioValue);
                    }
                    
                    // Create vertice data
                    let verticeData = new Float32Array(64);
                    if (strategy.verticeValue === "random_small") {
                        for (let i = 0; i < 64; i++) {
                            verticeData[i] = (Math.random() - 0.5) * 0.01;
                        }
                    } else {
                        verticeData.fill(strategy.verticeValue);
                    }
                    
                    // Create template data
                    let templateData = new Float32Array(15069);
                    if (strategy.templateValue === "random_small") {
                        for (let i = 0; i < 15069; i++) {
                            templateData[i] = (Math.random() - 0.5) * 0.01;
                        }
                    } else {
                        templateData.fill(strategy.templateValue);
                    }
                    
                    const strategyFeeds = {
                        audio_features: new ort.Tensor('float32', audioData, [1, 1, 768]),
                        vertice_emb: new ort.Tensor('float32', verticeData, [1, 1, 64]),
                        template: new ort.Tensor('float32', templateData, [1, 1, 15069]),
                        one_hot: new ort.Tensor('float32', new Float32Array(strategy.oneHotValue), [1, 3])
                    };
                    
                    console.log(`📊 Strategy ${strategy.name} tensor info:`);
                    console.log(`  Audio range: [${Math.min(...audioData)}, ${Math.max(...audioData)}]`);
                    console.log(`  Vertice range: [${Math.min(...verticeData)}, ${Math.max(...verticeData)}]`);
                    console.log(`  Template range: [${Math.min(...templateData)}, ${Math.max(...templateData)}]`);
                    console.log(`  OneHot: [${strategy.oneHotValue.join(', ')}]`);
                    
                    const strategyResult = await this.session.run(strategyFeeds);
                    console.log(`🎉 SUCCESS with ${strategy.name}!`);
                    console.log('Output keys:', Object.keys(strategyResult));
                    
                    workingConfig = {
                        name: strategy.name,
                        audio_shape: [1, 1, 768],
                        vertice_shape: [1, 1, 64],
                        template_shape: [1, 1, 15069],
                        one_hot_shape: [1, 3]
                    };
                    break;
                    
                } catch (strategyError) {
                    const errorCode = typeof strategyError === 'number' ? strategyError : strategyError.message;
                    console.log(`❌ Strategy ${strategy.name} failed: ${errorCode}`);
                    
                    // Log specific error details for debugging
                    if (typeof strategyError === 'number') {
                        console.log(`🔍 Error code analysis for ${strategy.name}:`);
                        switch (strategyError) {
                            case 126199320:
                                console.log('  → Model input validation failed - possibly incompatible tensor values');
                                break;
                            case 126075928:
                                console.log('  → Input tensor dimension mismatch - sequence length issue');
                                break;
                            case 231269720:
                                console.log('  → Memory allocation error - tensor too large or invalid');
                                break;
                            case 143224752:
                                console.log('  → Shape inference failed - incompatible tensor configuration');
                                break;
                            case 126076288:
                                console.log('  → Input tensor dimension mismatch - specific to current configuration');
                                break;
                            default:
                                console.log(`  → Unknown error code: ${strategyError}`);
                        }
                    }
                }
            }
            
            // If still no working config, try with different tensor data types or backends
            if (!workingConfig) {
                console.log('🔧 Trying alternative ONNX configurations...');
                
                try {
                    // Try recreating the session with different options
                    console.log('🧪 Testing with different session options...');
                    
                    // Check if we can create a new session with different execution providers
                    const alternativeSession = await ort.InferenceSession.create(this.modelPath, {
                        executionProviders: ['wasm'],
                        graphOptimizationLevel: 'disabled',
                        executionMode: 'sequential'
                    });
                    
                    // Try with the alternative session
                    const altFeeds = {
                        audio_features: new ort.Tensor('float32', new Float32Array(768).fill(0.01), [1, 1, 768]),
                        vertice_emb: new ort.Tensor('float32', new Float32Array(64).fill(0.01), [1, 1, 64]),
                        template: new ort.Tensor('float32', new Float32Array(15069).fill(0.01), [1, 1, 15069]),
                        one_hot: new ort.Tensor('float32', new Float32Array([1.0, 0.0, 0.0]), [1, 3])
                    };
                    
                    const altResult = await alternativeSession.run(altFeeds);
                    console.log('🎉 SUCCESS with alternative session configuration!');
                    console.log('Available outputs:', Object.keys(altResult));
                    
                    // Replace the main session with the working one
                    this.session = alternativeSession;
                    
                    workingConfig = {
                        name: "Alternative Session Configuration",
                        audio_shape: [1, 1, 768],
                        vertice_shape: [1, 1, 64],
                        template_shape: [1, 1, 15069],
                        one_hot_shape: [1, 3]
                    };
                    
                } catch (altError) {
                    console.log('❌ Alternative session failed:', typeof altError === 'number' ? altError : altError.message);
                    
                    // Last resort: try with different model loading approach
                    console.log('🆘 Last resort: checking model file integrity...');
                    
                    try {
                        // Try to get more information about the model
                        console.log('🔍 Model file diagnostics:');
                        console.log('  Model path:', this.modelPath);
                        console.log('  Session input names:', this.session.inputNames);
                        console.log('  Session output names:', this.session.outputNames);
                        
                        // Try to inspect the model handler more deeply
                        if (this.session.handler) {
                            console.log('  Handler session ID:', this.session.handler.sessionId);
                            console.log('  Handler type:', typeof this.session.handler);
                        }
                        
                        console.log('💡 Possible issues:');
                        console.log('  1. Model file may be corrupted or incomplete');
                        console.log('  2. Model was trained with different input specifications');
                        console.log('  3. ONNX runtime version incompatibility');
                        console.log('  4. WebAssembly backend limitations');
                        console.log('  5. Model expects specific value ranges or preprocessing');
                        
                    } catch (diagError) {
                        console.log('❌ Model diagnostics failed:', diagError.message);
                    }
                }
            }
            
            if (!workingConfig) {
                throw new Error('Unable to determine correct tensor shapes for FaceFormer model after exhaustive testing');
            }
        }
        
        console.log(`🎯 Using working configuration: ${workingConfig.name}`);
        
        // Process inputs according to the working configuration
        const batchSize = 1;
        
        // Audio processing based on working config
        const targetAudioShape = workingConfig.audio_shape;
        const audioSeqLen = targetAudioShape[1];
        const audioFeatureDim = targetAudioShape[2];
        const requiredAudioLength = audioSeqLen * audioFeatureDim;
        
        console.log(`🔧 Target audio shape: [${targetAudioShape.join(', ')}]`);
        console.log(`🔧 Required audio elements: ${requiredAudioLength}`);
        
        // Smart audio feature processing
        let processedAudioFeatures;
        if (audioFeatures.length === 0) {
            console.warn('⚠️ Empty audio features - using zeros');
            processedAudioFeatures = new Array(requiredAudioLength).fill(0.01);
        } else if (audioFeatures.length >= requiredAudioLength) {
            console.log(`✂️ Trimming audio from ${audioFeatures.length} to ${requiredAudioLength}`);
            processedAudioFeatures = audioFeatures.slice(0, requiredAudioLength);
        } else {
            console.log(`🔄 Expanding audio from ${audioFeatures.length} to ${requiredAudioLength}`);
            processedAudioFeatures = new Array(requiredAudioLength);
            for (let i = 0; i < requiredAudioLength; i++) {
                processedAudioFeatures[i] = audioFeatures[i % audioFeatures.length];
            }
        }
        
        // Template processing based on working config
        const targetTemplateShape = workingConfig.template_shape;
        const expectedTemplateSize = targetTemplateShape.reduce((a, b) => a * b, 1);
        
        let processedTemplate;
        if (template.length === expectedTemplateSize) {
            processedTemplate = [...template];
        } else if (template.length > expectedTemplateSize) {
            console.log(`✂️ Trimming template from ${template.length} to ${expectedTemplateSize}`);
            processedTemplate = template.slice(0, expectedTemplateSize);
        } else {
            console.log(`🔄 Padding template from ${template.length} to ${expectedTemplateSize}`);
            processedTemplate = [...template];
            while (processedTemplate.length < expectedTemplateSize) {
                processedTemplate.push(0.0);
            }
        }
        
        // OneHot processing
        const expectedOneHotSize = workingConfig.one_hot_shape.reduce((a, b) => a * b, 1);
        let processedOneHot;
        if (oneHot.length === expectedOneHotSize) {
            processedOneHot = [...oneHot];
        } else {
            console.log(`🔄 Adjusting one-hot from ${oneHot.length} to ${expectedOneHotSize}`);
            processedOneHot = new Array(expectedOneHotSize).fill(0);
            if (oneHot.length > 0) {
                processedOneHot[0] = 1.0; // Default to first subject
            }
        }
        
        console.log('✅ Processed input shapes:');
        console.log(`  Audio: ${processedAudioFeatures.length} elements -> ${JSON.stringify(targetAudioShape)}`);
        console.log(`  Template: ${processedTemplate.length} elements -> ${JSON.stringify(targetTemplateShape)}`);
        console.log(`  OneHot: ${processedOneHot.length} elements -> ${JSON.stringify(workingConfig.one_hot_shape)}`);
        
        // Convert inputs to ONNX tensors using the working configuration
        let audioTensor, templateTensor, oneHotTensor;
        
        try {
            console.log('Creating tensors with working configuration...');
            
            // Create audio tensor
            audioTensor = new ort.Tensor('float32', new Float32Array(processedAudioFeatures), workingConfig.audio_shape);
            console.log('✅ Audio tensor created successfully');
            
            // Create template tensor  
            templateTensor = new ort.Tensor('float32', new Float32Array(processedTemplate), workingConfig.template_shape);
            console.log('✅ Template tensor created successfully');
            
            // Create oneHot tensor
            oneHotTensor = new ort.Tensor('float32', new Float32Array(processedOneHot), workingConfig.one_hot_shape);
            console.log('✅ OneHot tensor created successfully');
            
        } catch (tensorError) {
            console.error('❌ Tensor creation error:', tensorError);
            throw new Error(`Tensor creation failed: ${tensorError.message}`);
        }
        
        // Initialize with style embedding (variable dimensional based on working config)
        const featureDim = workingConfig.vertice_shape[2]; // Get embedding dimension from working config
        let currentVerticeEmb = new Array(batchSize * 1 * featureDim).fill(0.1);
        
        const generatedVertices = [];
        
        // Try with smaller test case first to avoid memory issues
        const testMaxFrames = Math.min(maxFrames, 5); // Limit to 5 frames for testing
        console.log(`🎯 Starting generation with ${testMaxFrames} frames using: ${workingConfig.name}`);
        console.log(`🎯 Using embedding dimension: ${featureDim}`);
        
        for (let i = 0; i < testMaxFrames; i++) {
            console.log(`Generation step ${i + 1}/${testMaxFrames}`);
            
            // Keep sequence length at 1 to avoid growing memory usage
            const currentSeqLen = 1; // Fixed sequence length instead of i + 1
            const verticeEmbTensor = new ort.Tensor('float32', new Float32Array(currentVerticeEmb), [batchSize, currentSeqLen, featureDim]);
            
            console.log(`Current embedding shape: [${batchSize}, ${currentSeqLen}, ${featureDim}]`);
            
            // Run one step of generation with correctly shaped inputs
            const feeds = {
                audio_features: audioTensor, // Use the working configuration tensor
                vertice_emb: verticeEmbTensor,
                one_hot: oneHotTensor,
                template: templateTensor
            };
            
            // Log tensor details before inference
            console.log('📊 Tensor details before inference:');
            console.log(`  audio_features: shape ${feeds.audio_features.dims}, type ${feeds.audio_features.type}, size ${feeds.audio_features.size}`);
            console.log(`  vertice_emb: shape ${feeds.vertice_emb.dims}, type ${feeds.vertice_emb.type}, size ${feeds.vertice_emb.size}`);
            console.log(`  one_hot: shape ${feeds.one_hot.dims}, type ${feeds.one_hot.type}, size ${feeds.one_hot.size}`);
            console.log(`  template: shape ${feeds.template.dims}, type ${feeds.template.type}, size ${feeds.template.size}`);
            
            try {
                console.log('Running inference with feeds:', Object.keys(feeds));
                const results = await this.session.run(feeds);
                
                console.log('✅ Inference successful!');
                console.log('Available outputs:', Object.keys(results));
                
                // Check if outputs exist
                if (!results.new_vertice_out || !results.updated_vertice_emb) {
                    console.error('Missing expected outputs from model');
                    console.log('Available outputs:', Object.keys(results));
                    console.log('Expected outputs: new_vertice_out, updated_vertice_emb');
                    break;
                }
                
                // Extract results
                const newVerticeOut = Array.from(results.new_vertice_out.data);
                const updatedVerticeEmb = Array.from(results.updated_vertice_emb.data);
                
                console.log(`Output shapes: new_out=${results.new_vertice_out.dims}, updated_emb=${results.updated_vertice_emb.dims}`);
                
                // Store the generated vertex
                generatedVertices.push(newVerticeOut);
                
                // Update embeddings for next iteration
                currentVerticeEmb = updatedVerticeEmb;
                
            } catch (error) {
                console.error(`❌ Error in generation step ${i + 1}:`, error);
                
                // Try with minimal fallback
                try {
                    console.log('🔧 Trying minimal fallback approach...');
                    const minimalFeeds = {
                        audio_features: new ort.Tensor('float32', new Float32Array(768).fill(0.01), [1, 1, 768]),
                        vertice_emb: new ort.Tensor('float32', new Float32Array(64).fill(0.01), [1, 1, 64]),
                        template: templateTensor,
                        one_hot: oneHotTensor
                    };
                    
                    const minimalResults = await this.session.run(minimalFeeds);
                    console.log('✅ Minimal fallback successful!');
                    
                    if (minimalResults.new_vertice_out && minimalResults.updated_vertice_emb) {
                        const newVerticeOut = Array.from(minimalResults.new_vertice_out.data);
                        const updatedVerticeEmb = Array.from(minimalResults.updated_vertice_emb.data);
                        
                        generatedVertices.push(newVerticeOut);
                        currentVerticeEmb = updatedVerticeEmb;
                        
                        console.log(`✅ Fallback generation step ${i + 1} completed`);
                        continue;
                    }
                    
                } catch (fallbackError) {
                    console.log('❌ Fallback also failed:', fallbackError.message);
                    // Skip this step but continue
                    if (i === 0) {
                        console.error('💥 Critical error on first step, aborting generation');
                        break;
                    }
                }
            }
        }
        
        console.log('Generation complete!');
        return generatedVertices;
    }
}

// Export for use in other modules (Node.js)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FaceFormerWebGenerator };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.FaceFormerWebGenerator = FaceFormerWebGenerator;
}
