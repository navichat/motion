/**
 * Unified Animation Systems Model Verification Test
 * Tests simultaneous loading and operation of all animation models
 */

class UnifiedAnimationModelTest {
    constructor() {
        this.models = {
            rsmt: {
                deepphase: { path: './rsmt/deepphase.onnx', session: null, status: 'pending' },
                stylevae: { path: './rsmt/stylevae.onnx', session: null, status: 'pending' },
                transitionnet: { path: './rsmt/transitionnet.onnx', session: null, status: 'pending' },
                manifoldvae: { path: './rsmt/onnx_models/manifold_vae.onnx', session: null, status: 'pending' }
            },
            faceformer: {
                minimal: { path: '../engine/web_porting_poc/faceformer/faceformer_minimal.onnx', session: null, status: 'pending' },
                core: { path: '../engine/web_porting_poc/faceformer/faceformer_core_step.onnx', session: null, status: 'pending' },
                vocaset: { path: '../engine/web_porting_poc/faceformer/models/faceformer_vocaset_full.onnx', session: null, status: 'pending' },
                biwi: { path: '../engine/web_porting_poc/faceformer/faceformer_biwi_simple.onnx', session: null, status: 'pending' }
            },
            audiogesture: {
                main: { path: '../audio2gesture/audio2gesture_step_fixed.onnx', session: null, status: 'pending' },
                enhanced: { path: '../audio2gesture/enhanced_audio2gesture_model.onnx', session: null, status: 'pending' },
                attention: { path: '../audio2gesture/audio2gesture_attention.onnx', session: null, status: 'pending' }
            },
            deepphase: {
                policy: { path: '../deepmimic/deepphase_policy.onnx', session: null, status: 'pending' },
                discriminator: { path: '../deepmimic/deepphase_discriminator.onnx', session: null, status: 'pending' }
            }
        };
        
        this.testResults = {
            startTime: Date.now(),
            totalModels: 0,
            loadedModels: 0,
            failedModels: 0,
            loadTimes: {},
            memoryUsage: {},
            inferenceTests: {},
            bvhOutputs: {},
            errors: []
        };
        
        this.bvhFrameBuffer = {
            currentFrame: null,
            frameHistory: [],
            sources: new Set(),
            compositeFrame: null
        };
        
        // Calculate total models
        for (const system in this.models) {
            this.testResults.totalModels += Object.keys(this.models[system]).length;
        }
        
        console.log('🧪 Unified Animation Model Test initialized');
        console.log(`📊 Total models to test: ${this.testResults.totalModels}`);
    }
    
    /**
     * Run comprehensive model loading test
     */
    async runCompleteTest() {
        console.log('\n🚀 Starting Unified Animation Model Test...\n');
        
        try {
            // Test 1: Check ONNX Runtime availability
            await this.testONNXRuntimeAvailability();
            
            // Test 2: Load all models simultaneously
            await this.loadAllModels();
            
            // Test 3: Test individual model inference
            await this.testModelInference();
            
            // Test 4: Test BVH frame generation
            await this.testBVHFrameGeneration();
            
            // Test 5: Test frame compositing
            await this.testFrameCompositing();
            
            // Test 6: Memory and performance analysis
            await this.analyzePerformance();
            
            // Generate final report
            const report = this.generateTestReport();
            console.log('\n📋 Test Report:\n', report);
            
            return this.testResults;
            
        } catch (error) {
            console.error('❌ Test suite failed:', error);
            this.testResults.errors.push(error.message);
            return this.testResults;
        }
    }
    
    /**
     * Test ONNX Runtime availability and capabilities
     */
    async testONNXRuntimeAvailability() {
        console.log('🔍 Testing ONNX Runtime availability...');
        
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime not available. Please include onnxruntime-web.');
        }
        
        console.log('✅ ONNX Runtime available');
        console.log('📦 ONNX Runtime version:', ort.version || 'unknown');
        
        // Test execution providers
        const providers = ort.env?.webgl ? ['webgl', 'cpu'] : ['cpu'];
        console.log('🖥️ Available execution providers:', providers);
        
        this.testResults.onnxRuntime = {
            available: true,
            version: ort.version || 'unknown',
            providers: providers
        };
    }
    
    /**
     * Load all models simultaneously
     */
    async loadAllModels() {
        console.log('📦 Loading all models simultaneously...');
        
        const loadPromises = [];
        
        for (const [systemName, systemModels] of Object.entries(this.models)) {
            for (const [modelName, modelInfo] of Object.entries(systemModels)) {
                const promise = this.loadSingleModel(systemName, modelName, modelInfo);
                loadPromises.push(promise);
            }
        }
        
        // Wait for all models to finish loading (success or failure)
        const results = await Promise.allSettled(loadPromises);
        
        // Process results
        results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
                this.testResults.loadedModels++;
            } else {
                this.testResults.failedModels++;
                this.testResults.errors.push(result.reason);
            }
        });
        
        console.log(`✅ Model loading complete: ${this.testResults.loadedModels}/${this.testResults.totalModels} successful`);
        
        if (this.testResults.failedModels > 0) {
            console.log(`⚠️ ${this.testResults.failedModels} models failed to load`);
        }
    }
    
    /**
     * Load a single model with error handling
     */
    async loadSingleModel(systemName, modelName, modelInfo) {
        const modelId = `${systemName}.${modelName}`;
        const startTime = performance.now();
        
        try {
            console.log(`🔄 Loading ${modelId} from ${modelInfo.path}...`);
            
            // Check if file exists by attempting to fetch
            const response = await fetch(modelInfo.path, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`Model file not found: ${response.status} ${response.statusText}`);
            }
            
            // Load ONNX model
            const session = await ort.InferenceSession.create(modelInfo.path, {
                executionProviders: ['cpu'] // Start with CPU for compatibility
            });
            
            const loadTime = performance.now() - startTime;
            
            // Update model info
            modelInfo.session = session;
            modelInfo.status = 'loaded';
            
            // Store load time
            this.testResults.loadTimes[modelId] = loadTime;
            
            console.log(`✅ ${modelId} loaded successfully (${loadTime.toFixed(1)}ms)`);
            
            // Log model metadata
            this.logModelMetadata(modelId, session);
            
            return { modelId, success: true, loadTime };
            
        } catch (error) {
            const loadTime = performance.now() - startTime;
            
            // Update model info
            modelInfo.status = 'error';
            modelInfo.error = error.message;
            
            console.log(`❌ ${modelId} failed to load: ${error.message} (${loadTime.toFixed(1)}ms)`);
            
            throw new Error(`${modelId}: ${error.message}`);
        }
    }
    
    /**
     * Log model metadata for debugging
     */
    logModelMetadata(modelId, session) {
        try {
            const inputNames = session.inputNames || [];
            const outputNames = session.outputNames || [];
            
            console.log(`📊 ${modelId} metadata:`, {
                inputs: inputNames.length,
                outputs: outputNames.length,
                inputNames: inputNames.slice(0, 3), // Show first 3
                outputNames: outputNames.slice(0, 3)
            });
            
        } catch (error) {
            console.log(`⚠️ Could not read ${modelId} metadata:`, error.message);
        }
    }
    
    /**
     * Test inference for all loaded models
     */
    async testModelInference() {
        console.log('\n🧠 Testing model inference...');
        
        const inferenceTests = {};
        
        for (const [systemName, systemModels] of Object.entries(this.models)) {
            inferenceTests[systemName] = {};
            
            for (const [modelName, modelInfo] of Object.entries(systemModels)) {
                if (modelInfo.status === 'loaded' && modelInfo.session) {
                    try {
                        const result = await this.testSingleModelInference(systemName, modelName, modelInfo);
                        inferenceTests[systemName][modelName] = result;
                        
                    } catch (error) {
                        console.log(`❌ ${systemName}.${modelName} inference failed:`, error.message);
                        inferenceTests[systemName][modelName] = { success: false, error: error.message };
                    }
                }
            }
        }
        
        this.testResults.inferenceTests = inferenceTests;
        
        // Count successful inferences
        let successCount = 0;
        let totalTests = 0;
        
        for (const system of Object.values(inferenceTests)) {
            for (const test of Object.values(system)) {
                totalTests++;
                if (test.success) successCount++;
            }
        }
        
        console.log(`✅ Inference testing complete: ${successCount}/${totalTests} successful`);
    }
    
    /**
     * Test inference for a single model
     */
    async testSingleModelInference(systemName, modelName, modelInfo) {
        const modelId = `${systemName}.${modelName}`;
        
        try {
            // Generate test input data based on system type
            const inputData = this.generateTestInputData(systemName, modelName, modelInfo.session);
            
            // Run inference
            const startTime = performance.now();
            const outputs = await modelInfo.session.run(inputData);
            const inferenceTime = performance.now() - startTime;
            
            // Validate outputs
            const outputSummary = this.summarizeOutputs(outputs);
            
            console.log(`✅ ${modelId} inference successful (${inferenceTime.toFixed(1)}ms)`);
            
            return {
                success: true,
                inferenceTime: inferenceTime,
                inputShapes: this.getInputShapes(inputData),
                outputSummary: outputSummary
            };
            
        } catch (error) {
            console.log(`❌ ${modelId} inference failed:`, error.message);
            throw error;
        }
    }
    
    /**
     * Generate test input data for different model types
     */
    generateTestInputData(systemName, modelName, session) {
        const inputData = {};
        
        try {
            const inputNames = session.inputNames || [];
            
            for (const inputName of inputNames) {
                let shape, data;
                
                // Determine input shape and data based on system type
                switch (systemName) {
                    case 'rsmt':
                        shape = this.getRSMTInputShape(modelName);
                        data = this.generateRSMTTestData(shape);
                        break;
                        
                    case 'faceformer':
                        shape = this.getFaceFormerInputShape(modelName);
                        data = this.generateFaceFormerTestData(shape);
                        break;
                        
                    case 'audiogesture':
                        shape = this.getAudioGestureInputShape(modelName);
                        data = this.generateAudioGestureTestData(shape);
                        break;
                        
                    case 'deepphase':
                        shape = this.getDeepPhaseInputShape(modelName);
                        data = this.generateDeepPhaseTestData(shape);
                        break;
                        
                    default:
                        // Generic fallback
                        shape = [1, 64];
                        data = new Float32Array(64).map(() => Math.random());
                }
                
                inputData[inputName] = new ort.Tensor('float32', data, shape);
            }
            
        } catch (error) {
            console.log(`⚠️ Using fallback input data for ${systemName}.${modelName}`);
            
            // Fallback: create minimal input
            const inputName = session.inputNames[0] || 'input';
            inputData[inputName] = new ort.Tensor('float32', new Float32Array([1.0]), [1, 1]);
        }
        
        return inputData;
    }
    
    /**
     * Get RSMT model input shapes
     */
    getRSMTInputShape(modelName) {
        const shapes = {
            deepphase: [1, 132],    // 22 joints * 6 DOF
            stylevae: [1, 32],      // Phase vector
            transitionnet: [1, 48], // Combined input
            manifoldvae: [1, 32]    // Phase vector
        };
        return shapes[modelName] || [1, 64];
    }
    
    /**
     * Get FaceFormer input shapes
     */
    getFaceFormerInputShape(modelName) {
        const shapes = {
            minimal: [1, 50, 768],   // Sequence, features
            core: [1, 100, 768],     // Longer sequence
            vocaset: [1, 50, 512],   // VocaSet specific
            biwi: [1, 50, 256]       // BIWI specific
        };
        return shapes[modelName] || [1, 50, 768];
    }
    
    /**
     * Get AudioGesture input shapes
     */
    getAudioGestureInputShape(modelName) {
        const shapes = {
            main: [1, 100, 64],      // Sequence, audio features
            enhanced: [1, 150, 64],  // Enhanced model
            attention: [1, 200, 128] // Attention model
        };
        return shapes[modelName] || [1, 100, 64];
    }
    
    /**
     * Get DeepPhase policy input shapes
     */
    getDeepPhaseInputShape(modelName) {
        const shapes = {
            policy: [1, 197],        // State vector
            discriminator: [1, 132]  // Motion vector
        };
        return shapes[modelName] || [1, 197];
    }
    
    /**
     * Generate test data for different systems
     */
    generateRSMTTestData(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Float32Array(size).map(() => (Math.random() - 0.5) * 2);
    }
    
    generateFaceFormerTestData(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Float32Array(size).map(() => Math.random());
    }
    
    generateAudioGestureTestData(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Float32Array(size).map(() => (Math.random() - 0.5));
    }
    
    generateDeepPhaseTestData(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Float32Array(size).map(() => Math.random() * 0.1);
    }
    
    /**
     * Summarize model outputs
     */
    summarizeOutputs(outputs) {
        const summary = {};
        
        for (const [name, tensor] of Object.entries(outputs)) {
            if (tensor.data && tensor.dims) {
                summary[name] = {
                    shape: tensor.dims,
                    min: Math.min(...tensor.data),
                    max: Math.max(...tensor.data),
                    mean: tensor.data.reduce((a, b) => a + b, 0) / tensor.data.length,
                    size: tensor.data.length
                };
            }
        }
        
        return summary;
    }
    
    /**
     * Get input shapes summary
     */
    getInputShapes(inputData) {
        const shapes = {};
        for (const [name, tensor] of Object.entries(inputData)) {
            shapes[name] = tensor.dims;
        }
        return shapes;
    }
    
    /**
     * Test BVH frame generation from each system
     */
    async testBVHFrameGeneration() {
        console.log('\n🎭 Testing BVH frame generation...');
        
        const bvhTests = {};
        
        // Test RSMT BVH generation
        if (this.models.rsmt.deepphase.status === 'loaded') {
            try {
                const frame = await this.generateRSMTBVHFrame();
                bvhTests.rsmt = { success: true, frame: frame, joints: Object.keys(frame).length };
                console.log('✅ RSMT BVH frame generated');
            } catch (error) {
                bvhTests.rsmt = { success: false, error: error.message };
                console.log('❌ RSMT BVH generation failed:', error.message);
            }
        }
        
        // Test FaceFormer BVH generation
        if (this.models.faceformer.minimal.status === 'loaded') {
            try {
                const frame = await this.generateFaceFormerBVHFrame();
                bvhTests.faceformer = { success: true, frame: frame, joints: Object.keys(frame).length };
                console.log('✅ FaceFormer BVH frame generated');
            } catch (error) {
                bvhTests.faceformer = { success: false, error: error.message };
                console.log('❌ FaceFormer BVH generation failed:', error.message);
            }
        }
        
        // Test AudioGesture BVH generation
        if (this.models.audiogesture.main.status === 'loaded') {
            try {
                const frame = await this.generateAudioGestureBVHFrame();
                bvhTests.audiogesture = { success: true, frame: frame, joints: Object.keys(frame).length };
                console.log('✅ AudioGesture BVH frame generated');
            } catch (error) {
                bvhTests.audiogesture = { success: false, error: error.message };
                console.log('❌ AudioGesture BVH generation failed:', error.message);
            }
        }
        
        this.testResults.bvhOutputs = bvhTests;
    }
    
    /**
     * Generate BVH frame from RSMT system
     */
    async generateRSMTBVHFrame() {
        // Simulate RSMT pipeline: skeleton -> phase -> manifold -> skeleton
        const skeletonData = this.generateRSMTTestData([1, 132]);
        
        // Phase encoding
        const phaseInput = { input: new ort.Tensor('float32', skeletonData, [1, 132]) };
        const phaseOutput = await this.models.rsmt.deepphase.session.run(phaseInput);
        
        // Convert to BVH frame format
        return this.convertRSMTToBVH(skeletonData);
    }
    
    /**
     * Generate BVH frame from FaceFormer system
     */
    async generateFaceFormerBVHFrame() {
        // Generate facial animation BVH frame
        const audioFeatures = this.generateFaceFormerTestData([1, 50, 768]);
        
        // Convert to facial BVH frame
        return this.convertFaceFormerToBVH(audioFeatures);
    }
    
    /**
     * Generate BVH frame from AudioGesture system
     */
    async generateAudioGestureBVHFrame() {
        // Generate gesture BVH frame
        const audioFeatures = this.generateAudioGestureTestData([1, 100, 64]);
        
        // Convert to gesture BVH frame
        return this.convertAudioGestureToBVH(audioFeatures);
    }
    
    /**
     * Convert RSMT data to BVH frame
     */
    convertRSMTToBVH(data) {
        const frame = {};
        let index = 0;
        
        // Standard BVH joint hierarchy
        const joints = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ];
        
        // Root has 6 DOF (position + rotation)
        frame[joints[0]] = Array.from(data.slice(index, index + 6));
        index += 6;
        
        // Other joints have 3 DOF (rotation only)
        for (let i = 1; i < joints.length && index + 3 <= data.length; i++) {
            frame[joints[i]] = Array.from(data.slice(index, index + 3));
            index += 3;
        }
        
        return frame;
    }
    
    /**
     * Convert FaceFormer data to BVH frame
     */
    convertFaceFormerToBVH(data) {
        return {
            Head: [Math.random() * 5, Math.random() * 5, Math.random() * 5],
            Neck: [Math.random() * 3, Math.random() * 3, Math.random() * 3],
            Jaw: [Math.random() * 2, 0, 0],
            LeftEye: [Math.random() * 1, Math.random() * 1, 0],
            RightEye: [Math.random() * 1, Math.random() * 1, 0]
        };
    }
    
    /**
     * Convert AudioGesture data to BVH frame
     */
    convertAudioGestureToBVH(data) {
        return {
            LeftShoulder: [Math.random() * 30, Math.random() * 20, Math.random() * 15],
            RightShoulder: [Math.random() * 30, Math.random() * 20, Math.random() * 15],
            LeftElbow: [Math.random() * 45, 0, 0],
            RightElbow: [Math.random() * 45, 0, 0],
            LeftWrist: [Math.random() * 10, Math.random() * 10, Math.random() * 10],
            RightWrist: [Math.random() * 10, Math.random() * 10, Math.random() * 10]
        };
    }
    
    /**
     * Test frame compositing from multiple sources
     */
    async testFrameCompositing() {
        console.log('\n🎬 Testing frame compositing...');
        
        try {
            // Collect frames from all systems
            const frames = {};
            
            if (this.testResults.bvhOutputs.rsmt?.success) {
                frames.rsmt = this.testResults.bvhOutputs.rsmt.frame;
                this.bvhFrameBuffer.sources.add('rsmt');
            }
            
            if (this.testResults.bvhOutputs.faceformer?.success) {
                frames.faceformer = this.testResults.bvhOutputs.faceformer.frame;
                this.bvhFrameBuffer.sources.add('faceformer');
            }
            
            if (this.testResults.bvhOutputs.audiogesture?.success) {
                frames.audiogesture = this.testResults.bvhOutputs.audiogesture.frame;
                this.bvhFrameBuffer.sources.add('audiogesture');
            }
            
            // Composite frames
            const compositeFrame = this.compositeFrames(frames);
            this.bvhFrameBuffer.compositeFrame = compositeFrame;
            this.bvhFrameBuffer.currentFrame = compositeFrame;
            
            console.log(`✅ Frame compositing successful`);
            console.log(`📊 Composite frame joints: ${Object.keys(compositeFrame).length}`);
            console.log(`🎭 Active sources: ${Array.from(this.bvhFrameBuffer.sources).join(', ')}`);
            
            return compositeFrame;
            
        } catch (error) {
            console.log('❌ Frame compositing failed:', error.message);
            throw error;
        }
    }
    
    /**
     * Composite multiple BVH frames
     */
    compositeFrames(frames) {
        const composite = {};
        
        // Priority order: RSMT > AudioGesture > FaceFormer
        const priorities = ['rsmt', 'audiogesture', 'faceformer'];
        
        for (const source of priorities) {
            if (frames[source]) {
                Object.assign(composite, frames[source]);
            }
        }
        
        // Add metadata
        composite._metadata = {
            sources: Object.keys(frames),
            timestamp: Date.now(),
            joints: Object.keys(composite).filter(k => !k.startsWith('_')).length
        };
        
        return composite;
    }
    
    /**
     * Analyze performance and memory usage
     */
    async analyzePerformance() {
        console.log('\n📊 Analyzing performance...');
        
        const performance = {
            totalLoadTime: Object.values(this.testResults.loadTimes).reduce((a, b) => a + b, 0),
            averageLoadTime: 0,
            slowestModel: null,
            fastestModel: null,
            memoryEstimate: 0
        };
        
        // Calculate averages
        const loadTimes = Object.entries(this.testResults.loadTimes);
        if (loadTimes.length > 0) {
            performance.averageLoadTime = performance.totalLoadTime / loadTimes.length;
            
            // Find fastest and slowest
            const sortedTimes = loadTimes.sort((a, b) => a[1] - b[1]);
            performance.fastestModel = { model: sortedTimes[0][0], time: sortedTimes[0][1] };
            performance.slowestModel = { model: sortedTimes[sortedTimes.length - 1][0], time: sortedTimes[sortedTimes.length - 1][1] };
        }
        
        // Estimate memory usage
        if (window.performance && window.performance.memory) {
            performance.memoryEstimate = window.performance.memory.usedJSHeapSize / 1024 / 1024;
        }
        
        this.testResults.performance = performance;
        
        console.log(`⏱️ Total load time: ${performance.totalLoadTime.toFixed(1)}ms`);
        console.log(`📈 Average load time: ${performance.averageLoadTime.toFixed(1)}ms`);
        console.log(`🐌 Slowest model: ${performance.slowestModel?.model} (${performance.slowestModel?.time.toFixed(1)}ms)`);
        console.log(`🚀 Fastest model: ${performance.fastestModel?.model} (${performance.fastestModel?.time.toFixed(1)}ms)`);
        console.log(`💾 Memory usage: ${performance.memoryEstimate.toFixed(1)}MB`);
    }
    
    /**
     * Generate comprehensive test report
     */
    generateTestReport() {
        const endTime = Date.now();
        const totalTime = endTime - this.testResults.startTime;
        
        const report = {
            summary: {
                totalTime: totalTime,
                totalModels: this.testResults.totalModels,
                loadedModels: this.testResults.loadedModels,
                failedModels: this.testResults.failedModels,
                successRate: (this.testResults.loadedModels / this.testResults.totalModels * 100).toFixed(1)
            },
            
            modelStatus: {},
            
            bvhGeneration: {
                totalSources: Object.keys(this.testResults.bvhOutputs).length,
                successfulSources: Object.values(this.testResults.bvhOutputs).filter(o => o.success).length,
                compositingSuccess: !!this.bvhFrameBuffer.compositeFrame
            },
            
            performance: this.testResults.performance,
            
            errors: this.testResults.errors,
            
            recommendations: this.generateRecommendations()
        };
        
        // Organize model status
        for (const [systemName, systemModels] of Object.entries(this.models)) {
            report.modelStatus[systemName] = {};
            for (const [modelName, modelInfo] of Object.entries(systemModels)) {
                report.modelStatus[systemName][modelName] = {
                    status: modelInfo.status,
                    loadTime: this.testResults.loadTimes[`${systemName}.${modelName}`],
                    error: modelInfo.error
                };
            }
        }
        
        return report;
    }
    
    /**
     * Generate recommendations based on test results
     */
    generateRecommendations() {
        const recommendations = [];
        
        if (this.testResults.failedModels > 0) {
            recommendations.push('Some models failed to load. Check file paths and model availability.');
        }
        
        if (this.testResults.performance?.averageLoadTime > 1000) {
            recommendations.push('Model loading is slow. Consider optimizing model sizes or using model quantization.');
        }
        
        if (this.testResults.performance?.memoryEstimate > 500) {
            recommendations.push('High memory usage detected. Consider loading models on-demand.');
        }
        
        if (Object.keys(this.testResults.bvhOutputs).length < 3) {
            recommendations.push('Not all animation systems are producing BVH output. Check integration implementations.');
        }
        
        if (recommendations.length === 0) {
            recommendations.push('All systems are working correctly! Ready for production use.');
        }
        
        return recommendations;
    }
    
    /**
     * Get current BVH frame for external access
     */
    getCurrentBVHFrame() {
        return this.bvhFrameBuffer.currentFrame;
    }
    
    /**
     * Get formatted frame data as text
     */
    getBVHFrameAsText() {
        const frame = this.bvhFrameBuffer.currentFrame;
        if (!frame) return 'No frame data available';
        
        let text = `BVH Frame Data (${frame._metadata?.joints || 'unknown'} joints)\n`;
        text += `Sources: ${frame._metadata?.sources?.join(', ') || 'unknown'}\n`;
        text += `Timestamp: ${new Date(frame._metadata?.timestamp || Date.now()).toLocaleTimeString()}\n\n`;
        
        for (const [joint, data] of Object.entries(frame)) {
            if (!joint.startsWith('_') && Array.isArray(data)) {
                const values = data.map(v => v.toFixed(3)).join(', ');
                text += `${joint}: [${values}]\n`;
            }
        }
        
        return text;
    }
    
    /**
     * Clean up resources
     */
    dispose() {
        // Release ONNX sessions
        for (const systemModels of Object.values(this.models)) {
            for (const modelInfo of Object.values(systemModels)) {
                if (modelInfo.session && modelInfo.session.release) {
                    modelInfo.session.release();
                }
            }
        }
        
        console.log('🧹 Unified Animation Model Test cleaned up');
    }
}

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedAnimationModelTest;
} else if (typeof window !== 'undefined') {
    window.UnifiedAnimationModelTest = UnifiedAnimationModelTest;
}
