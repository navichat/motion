/**
 * AI Model Jobs - Real AI model inference tasks for the task manager
 */

// Base AI Model Job class
class AIModelJob {
    constructor(id, modelType, backend = 'cpu', complexity = 1) {
        this.id = id;
        this.type = modelType;
        this.jobType = modelType;
        this.backend = backend;
        this.complexity = complexity;
        this.duration = this.getEstimatedDuration(modelType, complexity);
        this.resourceRequirements = this.getResourceRequirements(modelType, backend);
        this.useRealInference = true;
        this.modelPaths = this.getModelPaths(modelType);
        this.metadata = {
            description: this.getModelDescription(modelType),
            flopsEstimate: this.getModelFLOPS(modelType)
        };
    }

    getEstimatedDuration(modelType, complexity) {
        const baseDurations = {
            'DeepMimic': 2000,      // Complex physics simulation
            'FaceFormer': 150,      // Real-time facial animation
            'Audio2Gesture': 800,   // Full body gesture generation
            'RSMT': 300,           // Motion transition
            'Kokoro': 100,         // Real-time speech synthesis
            'Whisper': 500,        // Speech recognition
            'VAD': 50,             // Voice activity detection
            'TinyLlama': 400,      // Language model inference
            'DiabloGPT': 600       // Conversational AI
        };
        return (baseDurations[modelType] || 500) * complexity;
    }

    getResourceRequirements(modelType, backend) {
        const baseMemory = {
            'DeepMimic': 512,      // MB
            'FaceFormer': 128,     // MB
            'Audio2Gesture': 256,  // MB
            'RSMT': 164,          // MB
            'Kokoro': 64,         // MB
            'Whisper': 200,       // MB
            'VAD': 32,            // MB
            'TinyLlama': 96,      // MB
            'DiabloGPT': 384      // MB
        };

        return {
            cpu: backend === 'cpu' ? 100 : 25,
            gpu: backend === 'gpu' ? 100 : 0,
            webnn: backend === 'webnn' ? 100 : 0,
            memory: baseMemory[modelType] || 128
        };
    }

    getModelPaths(modelType) {
        const paths = {
            'DeepMimic': './deepmimic/compatible_humanoid3d_humanoid3d_walk.onnx',
            'FaceFormer': './faceformer/faceformer_core_step.onnx',
            'Audio2Gesture': '../../audio2gesture_step_fixed.onnx',
            'RSMT': {
                deepPhase: './rsmt/deepphase.onnx',
                styleVAE: './rsmt/stylevae.onnx',
                transitionNet: './rsmt/transitionnet.onnx'
            },
            'Kokoro': './kokoro.js/dist/',
            'Whisper': './models/whisper.onnx',
            'VAD': './models/vad.onnx',
            'TinyLlama': './models/tinyllama.onnx',
            'DiabloGPT': './models/diablogpt.onnx'
        };
        return paths[modelType] || null;
    }

    getModelDescription(modelType) {
        const descriptions = {
            'DeepMimic': 'Physics-based character animation with reinforcement learning',
            'FaceFormer': 'Real-time facial animation from audio',
            'Audio2Gesture': 'Full-body gesture generation from speech audio',
            'RSMT': 'Real-time Stylized Motion Transition',
            'Kokoro': 'Real-time emotional speech synthesis',
            'Whisper': 'Automatic speech recognition and transcription',
            'VAD': 'Voice Activity Detection for real-time processing',
            'TinyLlama': 'Lightweight language model for text generation',
            'DiabloGPT': 'Conversational AI model for dialogue generation'
        };
        return descriptions[modelType] || 'AI Model Task';
    }

    getModelFLOPS(modelType) {
        const baseFLOPS = {
            'DeepMimic': 2.5e12,    // 2.5 TFLOPS
            'FaceFormer': 0.8e12,   // 800 GFLOPS
            'Audio2Gesture': 1.5e12, // 1.5 TFLOPS
            'RSMT': 0.6e12,         // 600 GFLOPS
            'Kokoro': 0.3e12,       // 300 GFLOPS
            'Whisper': 1.2e12,      // 1.2 TFLOPS
            'VAD': 0.1e12,          // 100 GFLOPS
            'TinyLlama': 0.5e12,    // 500 GFLOPS
            'DiabloGPT': 2.0e12     // 2.0 TFLOPS
        };
        return baseFLOPS[modelType] || 1.0e12;
    }

    async execute(progressCallback, shouldStop) {
        // This is just a placeholder - actual execution is handled by workers
        console.log(`Executing ${this.type} AI model job ${this.id} on ${this.backend} backend`);
        
        const startTime = Date.now();
        const steps = Math.max(5, Math.floor(this.complexity * 3));
        
        for (let step = 0; step < steps && !shouldStop(); step++) {
            const progress = Math.round(((step + 1) / steps) * 100);
            const elapsed = Date.now() - startTime;
            
            if (progressCallback) {
                progressCallback(progress, {
                    step: step + 1,
                    totalSteps: steps,
                    elapsed,
                    modelType: this.type,
                    backend: this.backend
                });
            }
            
            await new Promise(resolve => setTimeout(resolve, this.duration / steps));
        }
        
        return {
            success: true,
            executionTime: Date.now() - startTime,
            modelType: this.type,
            backend: this.backend,
            output: `Mock ${this.type} output`
        };
    }
}

// Specific AI Model Job classes
class DeepMimicJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'DeepMimic', 'gpu', complexity);
        this.motionType = 'walking'; // could be walking, running, jumping, etc.
        this.characterModel = 'humanoid3d';
    }
}

class FaceFormerJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'FaceFormer', selectedBackend, complexity);
        this.audioLength = 1.0; // seconds
        this.facialLandmarks = 68;
        console.log(`🎭 FaceFormer job created with ${selectedBackend} backend`);
    }
}

class Audio2GestureJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'Audio2Gesture', 'gpu', complexity);
        this.audioLength = 2.0; // seconds
        this.gestureFrames = 60; // 30 FPS
    }
}

class RSMTJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'RSMT', selectedBackend, complexity);
        this.motionStyle = 'casual'; // casual, energetic, formal, etc.
        this.transitionDuration = 0.5; // seconds
        console.log(`🏃 RSMT job created with ${selectedBackend} backend`);
    }
}

class KokoroJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'Kokoro', selectedBackend, complexity);
        this.text = 'Hello, this is a test of emotional speech synthesis.';
        this.emotion = 'neutral'; // neutral, happy, sad, angry, etc.
        this.voice = 'default';
        console.log(`🗣️ Kokoro job created with ${selectedBackend} backend`);
    }
}

class WhisperJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'Whisper', 'gpu', complexity);
        this.audioLength = 10.0; // seconds
        this.language = 'en';
    }
}

class VADJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'VAD', 'cpu', complexity);
        this.audioLength = 1.0; // seconds
        this.threshold = 0.5;
    }
}

class TinyLlamaJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'TinyLlama', selectedBackend, complexity);
        this.prompt = 'Generate a creative story about...';
        this.maxTokens = 256;
        console.log(`🦙 TinyLlama job created with ${selectedBackend} backend`);
    }
}

class DiabloGPTJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'DiabloGPT', 'gpu', complexity);
        this.conversation = ['Hello, how are you today?'];
        this.maxResponseLength = 128;
    }
}

// AI Model Job Factory
class AIModelJobFactory {
    constructor() {
        this.jobCounter = 0;
    }

    createJob(modelType, options = {}) {
        const id = `ai_${modelType.toLowerCase()}_${Date.now()}_${this.jobCounter++}`;
        const complexity = options.complexity || 1;

        switch (modelType) {
            case 'DeepMimic':
                return new DeepMimicJob(id, complexity);
            case 'FaceFormer':
                return new FaceFormerJob(id, complexity);
            case 'Audio2Gesture':
                return new Audio2GestureJob(id, complexity);
            case 'RSMT':
                return new RSMTJob(id, complexity);
            case 'Kokoro':
                return new KokoroJob(id, complexity);
            case 'Whisper':
                return new WhisperJob(id, complexity);
            case 'VAD':
                return new VADJob(id, complexity);
            case 'TinyLlama':
                return new TinyLlamaJob(id, complexity);
            case 'DiabloGPT':
                return new DiabloGPTJob(id, complexity);
            default:
                console.warn(`Unknown AI model type: ${modelType}`);
                return new AIModelJob(id, modelType, 'cpu', complexity);
        }
    }

    createRandomAIJob() {
        const modelTypes = [
            'DeepMimic', 'FaceFormer', 'Audio2Gesture', 'RSMT', 
            'Kokoro', 'Whisper', 'VAD', 'TinyLlama', 'DiabloGPT'
        ];
        const modelType = modelTypes[Math.floor(Math.random() * modelTypes.length)];
        const complexity = Math.floor(Math.random() * 3) + 1;
        
        return this.createJob(modelType, { complexity });
    }

    createAIModelWorkload(count = 10) {
        const jobs = [];
        for (let i = 0; i < count; i++) {
            const job = this.createRandomAIJob();
            jobs.push({
                job: job,
                priority: this.getAIPriority(job.type),
                scheduledTime: Date.now() + Math.random() * 5000, // Stagger 0-5 seconds
                options: {
                    maxRetries: 2,
                    timeout: job.duration * 2
                }
            });
        }
        return jobs;
    }

    getAIPriority(modelType) {
        // Real-time models get higher priority (lower number)
        const priorities = {
            'VAD': 0,           // Highest - real-time
            'Kokoro': 0,        // Highest - real-time TTS
            'FaceFormer': 1,    // High - real-time face
            'RSMT': 2,          // High - real-time motion
            'Whisper': 3,       // Medium - ASR
            'Audio2Gesture': 4, // Medium - gesture generation
            'TinyLlama': 5,     // Medium-low - text generation
            'DeepMimic': 6,     // Low - physics simulation
            'DiabloGPT': 7      // Lowest - conversation
        };
        return priorities[modelType] || 5;
    }
}

// Export for global use
if (typeof window !== 'undefined') {
    window.AIModelJob = AIModelJob;
    window.DeepMimicJob = DeepMimicJob;
    window.FaceFormerJob = FaceFormerJob;
    window.Audio2GestureJob = Audio2GestureJob;
    window.RSMTJob = RSMTJob;
    window.KokoroJob = KokoroJob;
    window.WhisperJob = WhisperJob;
    window.VADJob = VADJob;
    window.TinyLlamaJob = TinyLlamaJob;
    window.DiabloGPTJob = DiabloGPTJob;
    window.AIModelJobFactory = AIModelJobFactory;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AIModelJob,
        DeepMimicJob,
        FaceFormerJob,
        Audio2GestureJob,
        RSMTJob,
        KokoroJob,
        WhisperJob,
        VADJob,
        TinyLlamaJob,
        DiabloGPTJob,
        AIModelJobFactory
    };
}
