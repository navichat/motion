/**
 * RealJobFactory creates diverse computational jobs for realistic queue testing
 */

class RealJobFactory {
    constructor() {
        // Initialize with base job types that always work
        this.jobTypes = [
            'WASMMatrix', 'WASMPrime', 'WASMFractal',
            // AI Model Jobs that can fallback to CPU/WASM
            'Whisper', 'VAD', 'TinyLlama', 'DiabloGPT'
        ];
        
        this.jobCounter = 0;
        this.aiModelFactory = new AIModelJobFactory();
        
        // Capabilities (will be populated asynchronously)
        this.capabilities = {
            webgpu: false,
            webnn: false,
            onnxWebGL: false,
            onnxWasm: true // WASM is always available
        };
        
        // Start capability detection (non-blocking)
        this.capabilityPromise = this.detectCapabilities();
    }

    async detectCapabilities() {
        console.log('🔧 Starting hardware capability detection...');
        
        const capabilities = {
            webgpu: false,
            webnn: false,
            onnxWebGL: false,
            onnxWasm: true
        };

        // Test WebGPU with actual GPU adapter request
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    // Try to create a device to verify it actually works
                    const device = await adapter.requestDevice();
                    if (device) {
                        capabilities.webgpu = true;
                        console.log('✅ WebGPU: Available and working');
                        device.destroy(); // Clean up
                    }
                }
            } catch (error) {
                console.log('❌ WebGPU: Failed adapter/device test:', error.message);
            }
        } else {
            console.log('❌ WebGPU: Navigator.gpu not available');
        }

        // Test WebNN if available
        if (typeof navigator !== 'undefined' && navigator.ml) {
            try {
                // Try to create a simple WebNN context
                const context = await navigator.ml.createContext();
                if (context) {
                    capabilities.webnn = true;
                    console.log('✅ WebNN: Available and working');
                }
            } catch (error) {
                console.log('❌ WebNN: Failed context test:', error.message);
            }
        } else {
            console.log('❌ WebNN: Navigator.ml not available');
        }

        // Test ONNX Runtime providers (if ONNX Runtime is loaded)
        if (typeof ort !== 'undefined') {
            try {
                const providers = ort.env.availableProviders || [];
                console.log('🔧 ONNX Runtime providers:', providers);
                
                // Test WebGL provider with a dummy session
                if (providers.includes('webgl')) {
                    try {
                        // Create a minimal model to test WebGL provider
                        await this.testONNXProvider('webgl');
                        capabilities.onnxWebGL = true;
                        console.log('✅ ONNX WebGL: Available and working');
                    } catch (error) {
                        console.log('❌ ONNX WebGL: Failed test:', error.message);
                    }
                }
                
                // WASM provider should always work
                if (providers.includes('wasm')) {
                    console.log('✅ ONNX WASM: Available');
                }
            } catch (error) {
                console.log('❌ ONNX Runtime: Error checking providers:', error.message);
            }
        } else {
            console.log('❌ ONNX Runtime: Not loaded globally');
        }

        // Update capabilities
        this.capabilities = capabilities;
        
        // Update job types based on detected capabilities
        this.updateJobTypes();
        
        console.log('🔧 Final capabilities:', this.capabilities);
        console.log('🔧 Available job types:', this.jobTypes);
        
        return capabilities;
    }

    async testONNXProvider(provider) {
        // Create a minimal identity model to test the provider
        const modelData = new Uint8Array([
            // Minimal ONNX model bytes (identity operation)
            8, 1, 18, 12, 10, 10, 18, 8, 10, 1, 120, 18, 3, 121, 58, 1
        ]);
        
        const session = await ort.InferenceSession.create(modelData, {
            executionProviders: [provider]
        });
        
        // Test with dummy input
        const input = new ort.Tensor('float32', [1.0], [1]);
        const output = await session.run({ x: input });
        
        session.release();
        return output;
    }

    updateJobTypes() {
        // FORCE ALL AI MODELS TO BE AVAILABLE - Always include all models for comprehensive testing
        this.jobTypes = [
            // Base computational jobs
            'WASMMatrix', 'WASMPrime', 'WASMFractal',
            // Core AI models - ALWAYS AVAILABLE
            'TinyLlama', 'DiabloGPT', 'Whisper', 'VAD',
            // Advanced AI models - FORCE AVAILABILITY 
            'Kokoro', 'SpeechT5', 'FaceFormer', 'RSMT', 'DeepMimic', 'Audio2Gesture',
            // KNN/Vector Search models - ALWAYS AVAILABLE
            'CloseVectorJob', 'HNSWJob', 'UnifiedKNNJob'
        ];
        
        // Add WebGPU jobs if available
        if (this.capabilities.webgpu) {
            this.jobTypes.push('WebGPUMatrix', 'WebGPUImage', 'WebGPUParticle');
        }
        
        // Add WebNN-specific jobs if available
        if (this.capabilities.webnn) {
            this.jobTypes.push('WebNNImageClassification', 'WebNNTextProcessing', 'WebNNAudioProcessing');
        }
        
        console.log('� FORCED ALL AI models to be available for comprehensive testing:', this.jobTypes);
        console.log('🎭 Total AI models available:', this.jobTypes.filter(t => 
            ['TinyLlama', 'DiabloGPT', 'Whisper', 'VAD', 'Kokoro', 'SpeechT5', 
             'FaceFormer', 'RSMT', 'DeepMimic', 'Audio2Gesture'].includes(t)).length);
    }

    async createRealisticWorkload(jobCount = 50) {
        // Wait for capability detection to complete
        await this.capabilityPromise;
        
        const jobs = [];
        
        for (let i = 0; i < jobCount; i++) {
            const job = this.createRandomJob();
            jobs.push({
                job: job,
                priority: this.generateRealisticPriority(job.type),
                scheduledTime: this.generateScheduledTime(),
                options: {
                    maxRetries: 2,
                    timeout: job.duration * 2
                }
            });
        }
        
        return jobs;
    }

    createRandomJob() {
        const jobType = this.jobTypes[Math.floor(Math.random() * this.jobTypes.length)];
        console.log('🔧 createRandomJob selected jobType:', jobType, 'from available:', this.jobTypes);
        const complexity = Math.floor(Math.random() * 3) + 1; // 1-3
        const id = `job_${Date.now()}_${this.jobCounter++}`;
        
        // Handle AI Model jobs first - these will have varied parameters
        if (['DeepMimic', 'FaceFormer', 'Audio2Gesture', 'RSMT', 
             'Whisper', 'VAD', 'TinyLlama', 'DiabloGPT', 'Kokoro', 'SpeechT5'].includes(jobType)) {
            const aiJob = this.aiModelFactory.createJob(jobType, { complexity });
            
            // Ensure AI jobs have their parameter data available for workers
            aiJob.jobData = {
                ...aiJob, // Include all job properties
                useRealInference: true,
                parametersForValidation: true
            };
            
            return aiJob;
        }
        
        // Handle KNN/Vector Search jobs
        if (['CloseVectorJob', 'HNSWJob', 'UnifiedKNNJob'].includes(jobType)) {
            const knnClass = window[jobType];
            if (knnClass) {
                const knnJob = new knnClass(id, {
                    dimensions: 128 + Math.floor(Math.random() * 384), // 128-512 dimensions
                    vectorCount: 1000 + Math.floor(Math.random() * 9000), // 1k-10k vectors
                    queryK: 5 + Math.floor(Math.random() * 15), // top 5-20 results
                    complexity: complexity
                });
                
                knnJob.jobData = {
                    ...knnJob,
                    useRealKNNSearch: true,
                    parametersForValidation: true
                };
                
                console.log(`🔍 Created KNN job: ${jobType} with ${knnJob.params.vectorCount} vectors, ${knnJob.params.dimensions} dimensions`);
                return knnJob;
            } else {
                console.warn(`⚠️ KNN class ${jobType} not available, falling back to WASMMatrix`);
                const fallbackJob = new WASMMatrixJob(id, 256, complexity);
                fallbackJob.jobData = { ...fallbackJob };
                return fallbackJob;
            }
        }
        
        switch (jobType) {
            case 'WASMMatrix':
                const matrixJob = new WASMMatrixJob(id, 
                    128 + Math.random() * 256, // Size 128-384
                    complexity);
                matrixJob.jobData = { ...matrixJob };
                return matrixJob;
                    
            case 'WASMPrime':
                const primeJob = new WASMPrimeJob(id,
                    50000 + Math.random() * 100000, // Limit 50k-150k
                    complexity);
                primeJob.jobData = { ...primeJob };
                return primeJob;
                    
            case 'WASMFractal':
                const fractalJob = new WASMFractalJob(id,
                    256 + Math.random() * 256, // Size 256-512
                    50 + Math.random() * 100, // Iterations 50-150
                    complexity);
                fractalJob.jobData = { ...fractalJob };
                return fractalJob;
                    
            case 'WebGPUMatrix':
                const gpuMatrixJob = new WebGPUMatrixJob(id,
                    256 + Math.random() * 512, // Size 256-768
                    complexity);
                gpuMatrixJob.jobData = { ...gpuMatrixJob };
                return gpuMatrixJob;
                    
            case 'WebGPUImage':
                const imageJob = new WebGPUImageJob(id,
                    512 + Math.random() * 512, // Width 512-1024
                    512 + Math.random() * 512, // Height 512-1024
                    complexity);
                imageJob.jobData = { ...imageJob };
                return imageJob;
                    
            case 'WebGPUParticle':
                const particleJob = new WebGPUParticleJob(id,
                    10000 + Math.random() * 40000, // Particles 10k-50k
                    50 + Math.random() * 100, // Steps 50-150
                    complexity);
                particleJob.jobData = { ...particleJob };
                return particleJob;
                    
            case 'WebNNImageClassification':
                const classificationJob = new WebNNImageClassificationJob(id,
                    8 + Math.random() * 24, // Batch size 8-32
                    224, // Standard ImageNet size
                    complexity);
                classificationJob.jobData = { ...classificationJob };
                return classificationJob;
                    
            case 'WebNNTextProcessing':
                const textJob = new WebNNTextProcessingJob(id,
                    256 + Math.random() * 256, // Sequence length 256-512
                    4 + Math.random() * 12, // Batch size 4-16
                    complexity);
                textJob.jobData = { ...textJob };
                return textJob;
                    
            case 'WebNNAudioProcessing':
                const audioJob = new WebNNAudioProcessingJob(id,
                    8000 + Math.random() * 16000, // Audio length 0.5-1.5s
                    4 + Math.random() * 8, // Batch size 4-12
                    complexity);
                audioJob.jobData = { ...audioJob };
                return audioJob;
                    
            default:
                const defaultJob = new WASMMatrixJob(id, 256, 1);
                defaultJob.jobData = { ...defaultJob };
                return defaultJob;
        }
    }

    generateRealisticPriority(jobType) {
        // Assign realistic priorities based on job types
        const priorityMaps = {
            // Real-time AI models (highest priority = lower number)
            'VAD': () => Math.floor(Math.random() * 2), // 0-1 (highest)
            'Kokoro': () => Math.floor(Math.random() * 2), // 0-1 (real-time TTS)
            'FaceFormer': () => 1 + Math.floor(Math.random() * 2), // 1-2
            'RSMT': () => 2 + Math.floor(Math.random() * 2), // 2-3
            
            // Real-time jobs (higher priority = lower number)
            'WebNNAudioProcessing': () => Math.floor(Math.random() * 3), // 0-2 (highest)
            'WebGPUParticle': () => Math.floor(Math.random() * 3), // 0-2 (real-time sim)
            
            // Interactive AI models
            'Whisper': () => 3 + Math.floor(Math.random() * 2), // 3-4
            'Audio2Gesture': () => 4 + Math.floor(Math.random() * 2), // 4-5
            
            // Interactive jobs
            'WebNNImageClassification': () => 2 + Math.floor(Math.random() * 3), // 2-4
            'WebGPUImage': () => 2 + Math.floor(Math.random() * 3), // 2-4
            
            // Batch AI models
            'TinyLlama': () => 5 + Math.floor(Math.random() * 2), // 5-6
            'DeepMimic': () => 6 + Math.floor(Math.random() * 2), // 6-7
            'DiabloGPT': () => 7 + Math.floor(Math.random() * 2), // 7-8
            
            // Batch processing jobs
            'WebNNTextProcessing': () => 4 + Math.floor(Math.random() * 3), // 4-6
            'WebGPUMatrix': () => 4 + Math.floor(Math.random() * 3), // 4-6
            
            // Background computation jobs
            'WASMMatrix': () => 6 + Math.floor(Math.random() * 3), // 6-8
            'WASMPrime': () => 7 + Math.floor(Math.random() * 3), // 7-9
            'WASMFractal': () => 7 + Math.floor(Math.random() * 3), // 7-9
        };
        
        const priorityFn = priorityMaps[jobType] || (() => Math.floor(Math.random() * 10));
        return priorityFn();
    }

    generateScheduledTime() {
        // Some jobs are immediate, others are scheduled for future
        const now = Date.now();
        const delay = Math.random();
        
        if (delay < 0.7) {
            return null; // Immediate execution (70%)
        } else if (delay < 0.9) {
            return now + Math.random() * 5000; // 0-5 seconds delay (20%)
        } else {
            return now + 5000 + Math.random() * 10000; // 5-15 seconds delay (10%)
        }
    }

    async createStressTestWorkload(intensity = 'medium') {
        // Wait for capability detection to complete
        await this.capabilityPromise;
        
        const intensitySettings = {
            light: { jobCount: 20, maxComplexity: 1 },
            medium: { jobCount: 50, maxComplexity: 2 },
            heavy: { jobCount: 100, maxComplexity: 3 },
            extreme: { jobCount: 200, maxComplexity: 3 }
        };
        
        const settings = intensitySettings[intensity] || intensitySettings.medium;
        const jobs = [];
        
        for (let i = 0; i < settings.jobCount; i++) {
            const job = this.createRandomJob();
            // Override complexity for stress test
            if (job.complexity !== undefined) {
                job.complexity = Math.min(job.complexity, settings.maxComplexity);
            }
            
            jobs.push({
                job: job,
                priority: this.generateRealisticPriority(job.type),
                scheduledTime: this.generateScheduledTime(),
                options: {
                    maxRetries: 1, // Fewer retries for stress test
                    timeout: job.duration * 1.5
                }
            });
        }
        
        return jobs;
    }

    createMLPipelineWorkload() {
        // Create a realistic ML pipeline workload
        const jobs = [];
        
        // Data preprocessing jobs (high priority)
        for (let i = 0; i < 5; i++) {
            jobs.push({
                job: new WebNNImageClassificationJob(`preprocess_${i}`, 16, 224, 1),
                priority: 1,
                scheduledTime: null
            });
        }
        
        // Feature extraction (medium priority)
        for (let i = 0; i < 8; i++) {
            jobs.push({
                job: new WebNNTextProcessingJob(`feature_${i}`, 512, 8, 2),
                priority: 3,
                scheduledTime: null
            });
        }
        
        // Model training/inference (mixed priority)
        for (let i = 0; i < 10; i++) {
            const isTraining = i < 3;
            jobs.push({
                job: new WebGPUMatrixJob(`model_${i}`, 512, isTraining ? 3 : 1),
                priority: isTraining ? 2 : 5,
                scheduledTime: null
            });
        }
        
        // Audio processing (real-time, highest priority)
        for (let i = 0; i < 6; i++) {
            jobs.push({
                job: new WebNNAudioProcessingJob(`audio_${i}`, 16000, 4, 1),
                priority: 0,
                scheduledTime: i * 2000 // Staggered every 2 seconds
            });
        }
        
        // Background computation (lowest priority)
        for (let i = 0; i < 15; i++) {
            const jobTypes = [WASMPrimeJob, WASMFractalJob, WASMMatrixJob];
            const JobClass = jobTypes[i % jobTypes.length];
            jobs.push({
                job: new JobClass(`background_${i}`, 100000, 2),
                priority: 8,
                scheduledTime: null
            });
        }
        
        return jobs;
    }

    // Static method for creating AI model jobs
    static createJob(jobType, options = {}) {
        const factory = new RealJobFactory();
        if (['DeepMimic', 'FaceFormer', 'Audio2Gesture', 'RSMT', 
             'Whisper', 'VAD', 'TinyLlama', 'DiabloGPT', 'Kokoro'].includes(jobType)) {
            return factory.aiModelFactory.createJob(jobType, options);
        } else {
            return factory.createRandomJob();
        }
    }
}

// Export factory
window.RealJobFactory = RealJobFactory;

// Convenience function for testing
window.createRealisticWorkload = async function(intensity = 'medium') {
    console.log('🔧 Global createRealisticWorkload called with intensity:', intensity);
    const factory = new RealJobFactory();
    
    // Wait for capability detection and then log results
    await factory.capabilityPromise;
    console.log('🔧 Factory capabilities detected:', factory.capabilities);
    console.log('🔧 Factory jobTypes:', factory.jobTypes);
    
    return await factory.createStressTestWorkload(intensity);
};

window.createMLPipelineWorkload = function() {
    const factory = new RealJobFactory();
    return factory.createMLPipelineWorkload();
};
