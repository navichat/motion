/**
 * ONNX Runtime Web Compatibility Fixer
 * Addresses wire type 4 and other compatibility issues
 */

// Enhanced error handling and compatibility fixes for ONNX Runtime Web
class ONNXRuntimeFixer {
    static getCompatibleONNXVersion() {
        // Different ONNX Runtime Web versions for compatibility
        return [
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js', // More stable
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js', // Older but more compatible
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js', // Current version
        ];
    }

    static async createSession(modelPath, options = {}) {
        const compatibleOptions = {
            executionProviders: [
                {
                    name: 'wasm',
                    deviceType: 'cpu',
                    // Disable problematic features that might cause wire type issues
                    enableCpuMemArena: false,
                    enableMemPattern: false,
                },
                'cpu'
            ],
            // Disable graph optimization that might cause issues
            graphOptimizationLevel: 'disabled',
            // Set specific session options for compatibility
            sessionOptions: {
                enableCpuMemArena: false,
                enableMemPattern: false,
                enableProfiling: false,
                executionMode: 'sequential',
                ...options.sessionOptions
            },
            ...options
        };

        try {
            // First attempt with standard loading
            console.log(`[ONNX Fixer] Attempting to load model: ${modelPath}`);
            const session = await ort.InferenceSession.create(modelPath, compatibleOptions);
            console.log(`[ONNX Fixer] ✅ Model loaded successfully with standard method`);
            return session;

        } catch (error) {
            const errorMessage = error.message || error.toString();
            console.warn(`[ONNX Fixer] Standard loading failed: ${errorMessage}`);

            if (errorMessage.includes('wire type 4') || 
                errorMessage.includes('invalid wire type') ||
                errorMessage.includes('protobuf')) {
                
                console.log(`[ONNX Fixer] 🔧 Applying wire type compatibility fixes...`);
                
                // Try with even more conservative settings
                const fallbackOptions = {
                    executionProviders: ['cpu'], // CPU only
                    graphOptimizationLevel: 'disabled',
                    sessionOptions: {
                        enableCpuMemArena: false,
                        enableMemPattern: false,
                        enableProfiling: false,
                        executionMode: 'sequential',
                        logSeverityLevel: 4, // Minimal logging
                        logVerbosityLevel: 0
                    }
                };

                try {
                    console.log(`[ONNX Fixer] Retrying with fallback settings...`);
                    const session = await ort.InferenceSession.create(modelPath, fallbackOptions);
                    console.log(`[ONNX Fixer] ✅ Model loaded with fallback settings`);
                    return session;
                } catch (fallbackError) {
                    console.error(`[ONNX Fixer] Fallback also failed: ${fallbackError.message}`);
                    
                    // Try loading as buffer (alternative approach)
                    return await this.loadAsBuffer(modelPath, fallbackOptions);
                }
            }
            
            throw error;
        }
    }

    static async loadAsBuffer(modelPath, options) {
        try {
            console.log(`[ONNX Fixer] 🔄 Attempting buffer-based loading...`);
            
            // Fetch model as array buffer
            const response = await fetch(modelPath);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const arrayBuffer = await response.arrayBuffer();
            console.log(`[ONNX Fixer] Model buffer loaded: ${arrayBuffer.byteLength} bytes`);
            
            // Create session from buffer
            const session = await ort.InferenceSession.create(arrayBuffer, options);
            console.log(`[ONNX Fixer] ✅ Model loaded from buffer successfully`);
            return session;
            
        } catch (bufferError) {
            console.error(`[ONNX Fixer] Buffer loading failed: ${bufferError.message}`);
            throw new Error(`All loading methods failed. Original error: ${bufferError.message}`);
        }
    }

    static async runInference(session, inputs, options = {}) {
        try {
            // Standard inference
            const results = await session.run(inputs, options);
            return results;
            
        } catch (error) {
            console.warn(`[ONNX Fixer] Inference failed, trying fallback: ${error.message}`);
            
            // Try inference with different options
            const fallbackOptions = {
                ...options,
                runOptions: {
                    logSeverityLevel: 4,
                    logVerbosityLevel: 0,
                    ...options.runOptions
                }
            };
            
            return await session.run(inputs, fallbackOptions);
        }
    }

    static createCompatibleInputs(inputData, inputNames, expectedShapes) {
        const inputs = {};
        
        inputNames.forEach((name, index) => {
            const data = inputData[index] || inputData[name];
            const expectedShape = expectedShapes[index] || expectedShapes[name];
            
            if (data) {
                // Ensure data is in the correct format
                let tensorData = data;
                
                if (Array.isArray(data)) {
                    tensorData = new Float32Array(data.flat());
                }
                
                inputs[name] = new ort.Tensor('float32', tensorData, expectedShape);
            }
        });
        
        return inputs;
    }
}

// Export for use in workers
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXRuntimeFixer;
}
if (typeof window !== 'undefined') {
    window.ONNXRuntimeFixer = ONNXRuntimeFixer;
}
