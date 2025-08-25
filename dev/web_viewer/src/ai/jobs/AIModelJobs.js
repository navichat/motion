/**
 * AI Model Jobs - Real AI model inference tasks for the task manager
 */

// Base AI Model Job class
class AIModelJob {
    constructor(id, modelType, backend = 'webnn', complexity = 1) {
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
            'Audio2Gesture': 850,   // Full body gesture generation
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
            'DeepMimic': './models/deepmimic.onnx',
            'FaceFormer': './models/faceformer.onnx',
            'Audio2Gesture': './models/audio2gesture.onnx',
            'RSMT': {
                deepPhase: './models/rsmt_deepphase.onnx',
                styleVAE: './models/rsmt_stylevae.onnx',
                transitionNet: './models/rsmt_transitionnet.onnx'
            },
            'Kokoro': './models/kokoro.onnx',
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
        console.log(`Executing ${this.type} AI model job ${this.id} on ${this.backend} backend`);
        const startTime = Date.now();

        try {
            // Use ONNX runtime for model inference
            const session = await ort.InferenceSession.create(this.modelPaths, {
                executionProviders: [this.backend],
                graphOptimizationLevel: 'all'
            });

            // Create dummy input tensors (replace with real data if available)
            const inputs = {};
            for (const input of session.inputNames) {
                const dummyData = new Float32Array(1);
                inputs[input] = new ort.Tensor('float32', dummyData, [1]);
            }

            // Run inference
            const outputs = await session.run(inputs);

            // Process output (example, replace with actual logic)
            const result = {};
            for (const key in outputs) {
                result[key] = outputs[key].data;
            }

            return {
                success: true,
                executionTime: Date.now() - startTime,
                modelOutput: result,
                modelType: this.type,
                backend: this.backend
            };
        } catch (error) {
            console.error(`Error executing ${this.type} model:`, error);
            return {
                success: false,
                error: error.message,
                executionTime: Date.now() - startTime,
                modelType: this.type,
                backend: this.backend
            };
        }
    }
}

// Specific AI Model Job classes
class DeepMimicJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'DeepMimic', selectedBackend, complexity);
        
        // Vary motion types for different outputs
        const motionTypes = ['walking', 'running', 'jumping', 'dancing', 'fighting', 'climbing', 'swimming', 'crawling'];
        this.motionType = motionTypes[Math.floor(Math.random() * motionTypes.length)];
        
        // Vary character models
        const characterModels = ['humanoid3d', 'athlete', 'child', 'elderly', 'robot', 'creature'];
        this.characterModel = characterModels[Math.floor(Math.random() * characterModels.length)];
        
        // Add physics simulation parameters
        this.physicsParams = {
            gravity: 9.8 + (Math.random() - 0.5) * 2.0, // 8.8-10.8 m/s²
            friction: 0.3 + Math.random() * 0.4,        // 0.3-0.7
            damping: 0.1 + Math.random() * 0.2,         // 0.1-0.3
            stiffness: 800 + Math.random() * 400,       // 800-1200
            mass: 60 + Math.random() * 40,              // 60-100 kg
            height: 1.6 + Math.random() * 0.4,          // 1.6-2.0 m
            agility: Math.random(),                     // 0-1
            balance: 0.7 + Math.random() * 0.3,         // 0.7-1.0
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🏃 DeepMimic job created with ${selectedBackend} backend, motion: ${this.motionType}, character: ${this.characterModel}`);
    }
}

class FaceFormerJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'FaceFormer', selectedBackend, complexity);
        
        // Vary audio input characteristics
        this.audioLength = 0.5 + Math.random() * 2.0; // 0.5-2.5 seconds
        this.facialLandmarks = 68;
        
        // Add facial animation parameters
        this.animationParams = {
            expressiveness: 0.3 + Math.random() * 0.7,  // 0.3-1.0
            lipSyncAccuracy: 0.8 + Math.random() * 0.2, // 0.8-1.0
            emotionalRange: Math.random(),               // 0-1.0
            eyeMovement: 0.5 + Math.random() * 0.5,     // 0.5-1.0
            browAnimation: Math.random() * 0.8,          // 0-0.8
            jawMovement: 0.7 + Math.random() * 0.3,     // 0.7-1.0
            cheekDeformation: Math.random() * 0.6,       // 0-0.6
            noseFlare: Math.random() * 0.3,             // 0-0.3
            audioSampleRate: 16000 + Math.floor(Math.random() * 32000), // 16-48kHz
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🎭 FaceFormer job created with ${selectedBackend} backend, audio: ${this.audioLength.toFixed(2)}s, expression: ${this.animationParams.expressiveness.toFixed(2)}`);
    }
}

class Audio2GestureJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'Audio2Gesture', selectedBackend, complexity);
        
        // Vary audio and gesture characteristics
        this.audioLength = 1.0 + Math.random() * 3.0; // 1-4 seconds
        this.gestureFrames = Math.floor(this.audioLength * 30); // 30 FPS
        
        // Add gesture generation parameters
        this.gestureParams = {
            amplitude: 0.3 + Math.random() * 0.7,       // 0.3-1.0 gesture size
            frequency: 0.5 + Math.random() * 2.0,       // 0.5-2.5 Hz gesture speed
            naturalness: 0.6 + Math.random() * 0.4,     // 0.6-1.0
            synchronization: 0.8 + Math.random() * 0.2, // 0.8-1.0 audio sync
            handDominance: Math.random() > 0.5 ? 'right' : 'left',
            bodyInvolvement: Math.random() * 0.8,        // 0-0.8 full body vs hands
            culturalStyle: ['western', 'eastern', 'expressive', 'subtle'][Math.floor(Math.random() * 4)],
            emotionalIntensity: Math.random(),           // 0-1.0
            gestureComplexity: 1 + Math.floor(Math.random() * 4), // 1-4 complexity levels
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🎵 Audio2Gesture job created with ${selectedBackend} backend, duration: ${this.audioLength.toFixed(2)}s, style: ${this.gestureParams.culturalStyle}`);
    }
}

class RSMTJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'RSMT', selectedBackend, complexity);
        
        // Vary motion styles and transition parameters
        const motionStyles = ['casual', 'energetic', 'formal', 'graceful', 'athletic', 'robotic', 'flowing', 'sharp'];
        this.motionStyle = motionStyles[Math.floor(Math.random() * motionStyles.length)];
        
        this.transitionDuration = 0.2 + Math.random() * 1.0; // 0.2-1.2 seconds
        
        // Add motion transition parameters
        this.transitionParams = {
            blendWeight: 0.3 + Math.random() * 0.4,     // 0.3-0.7 transition blend
            smoothness: 0.7 + Math.random() * 0.3,      // 0.7-1.0
            preserveRhythm: Math.random() > 0.3,        // 70% chance
            adaptToTerrain: Math.random() > 0.5,        // 50% chance
            energyConservation: 0.5 + Math.random() * 0.5, // 0.5-1.0
            styleIntensity: Math.random(),               // 0-1.0
            motionQuality: 0.8 + Math.random() * 0.2,   // 0.8-1.0
            transitionType: ['linear', 'ease-in', 'ease-out', 'elastic'][Math.floor(Math.random() * 4)],
            jointPriority: Math.random() > 0.6 ? 'upper' : 'lower', // body focus
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🎬 RSMT job created with ${selectedBackend} backend, style: ${this.motionStyle}, duration: ${this.transitionDuration.toFixed(2)}s`);
    }
}

class KokoroJob extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'Kokoro', selectedBackend, complexity);
        
        // Vary the input text to create different outputs
        const texts = [
            'Hello, this is a test of emotional speech synthesis.',
            'The weather today is absolutely beautiful and sunny.',
            'I am excited to demonstrate artificial intelligence capabilities.',
            'Technology continues to advance at an incredible pace.',
            'Virtual avatars will transform digital communication.',
            'Natural language processing enables human-computer interaction.',
            'Machine learning models can generate realistic speech patterns.',
            'Innovation drives progress in computational linguistics.'
        ];
        
        this.text = texts[Math.floor(Math.random() * texts.length)] + ` Complexity ${complexity}.`;
        
        // Vary emotional parameters
        const emotions = ['neutral', 'happy', 'confident', 'calm', 'energetic', 'thoughtful'];
        this.emotion = emotions[Math.floor(Math.random() * emotions.length)];
        
        // Vary voice characteristics
        const voices = ['default', 'warm', 'bright', 'deep', 'clear'];
        this.voice = voices[Math.floor(Math.random() * voices.length)];
        
        // Add unique speech parameters for variation
        this.speechParams = {
            pitch: 0.8 + Math.random() * 0.4, // 0.8-1.2
            rate: 0.9 + Math.random() * 0.2,  // 0.9-1.1
            volume: 0.7 + Math.random() * 0.3, // 0.7-1.0
            emphasis: Math.random() * 0.5,     // 0-0.5
            breathiness: Math.random() * 0.3,  // 0-0.3
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🗣️ Kokoro job created with ${selectedBackend} backend, emotion: ${this.emotion}, voice: ${this.voice}`);
    }
}

class SpeechT5Job extends AIModelJob {
    constructor(id, complexity = 1, backend = null) {
        // Try WebGPU as fallback if WebNN is not available and no backend specified
        const selectedBackend = backend || (window.navigator?.ml ? 'webnn' : 'gpu');
        super(id, 'SpeechT5', selectedBackend, complexity);
        
        // Vary the input text dramatically for different outputs
        const speechTexts = [
            'This is SpeechT5 text-to-speech synthesis test.',
            'Advanced neural networks enable realistic voice generation.',
            'Artificial intelligence transforms communication technology.',
            'Digital avatars require sophisticated speech synthesis.',
            'Machine learning creates natural sounding voices.',
            'Real-time processing enables interactive conversations.',
            'Voice synthesis quality depends on model architecture.',
            'Neural speech generation advances human-computer interaction.'
        ];
        
        this.text = speechTexts[Math.floor(Math.random() * speechTexts.length)] + ` Test ${complexity}.`;
        
        // Vary speaker characteristics significantly
        const speakers = ['default', 'female1', 'male1', 'child', 'elderly', 'professional', 'casual', 'narrator'];
        this.speaker = speakers[Math.floor(Math.random() * speakers.length)];
        
        // Vary speed and other parameters
        this.speed = 0.7 + Math.random() * 0.6; // 0.7-1.3
        
        // Add unique synthesis parameters
        this.synthesisParams = {
            temperature: 0.5 + Math.random() * 0.5, // 0.5-1.0
            top_k: 20 + Math.floor(Math.random() * 30), // 20-50
            pitch_shift: -0.2 + Math.random() * 0.4, // -0.2 to 0.2
            energy_scale: 0.8 + Math.random() * 0.4, // 0.8-1.2
            duration_scale: 0.9 + Math.random() * 0.2, // 0.9-1.1
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🎙️ SpeechT5 job created with ${selectedBackend} backend, speaker: ${this.speaker}, speed: ${this.speed.toFixed(2)}`);
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
        
        // Vary prompts dramatically for diverse outputs
        const prompts = [
            'Generate a creative story about a futuristic city where',
            'Explain the importance of artificial intelligence in',
            'Write a detailed description of how virtual avatars',
            'Describe the scientific principles behind machine learning',
            'Create a narrative about the evolution of computer technology',
            'Discuss the potential impact of neural networks on',
            'Generate creative content about the intersection of art and',
            'Explain complex algorithms in simple terms for beginners who'
        ];
        
        this.prompt = prompts[Math.floor(Math.random() * prompts.length)] + ` [Complexity ${complexity}]`;
        
        // Vary generation parameters significantly
        this.maxTokens = 128 + Math.floor(Math.random() * 256); // 128-384 tokens
        
        // Add diverse generation parameters
        this.generationParams = {
            temperature: 0.3 + Math.random() * 0.9,    // 0.3-1.2 (creativity)
            top_p: 0.7 + Math.random() * 0.3,          // 0.7-1.0 (nucleus sampling)
            top_k: 20 + Math.floor(Math.random() * 50), // 20-70 (top-k sampling)
            repetition_penalty: 1.0 + Math.random() * 0.2, // 1.0-1.2
            length_penalty: 0.8 + Math.random() * 0.4,  // 0.8-1.2
            num_beams: 1 + Math.floor(Math.random() * 4), // 1-4 (beam search)
            seed: Math.floor(Math.random() * 1000000),   // Random seed
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🦙 TinyLlama job created with ${selectedBackend} backend, max tokens: ${this.maxTokens}, temp: ${this.generationParams.temperature.toFixed(2)}`);
    }
}

class DiabloGPTJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'DiabloGPT', 'gpu', complexity);
        
        // Vary conversation contexts for different outputs
        const conversationStarters = [
            ['Hello, how are you today?'],
            ['What do you think about artificial intelligence?'],
            ['Tell me about your favorite technology.'],
            ['How do you see the future of computing?'],
            ['What interests you most about virtual avatars?'],
            ['Describe your ideal human-AI collaboration.'],
            ['What are your thoughts on machine learning?'],
            ['How would you explain consciousness to an AI?']
        ];
        
        this.conversation = conversationStarters[Math.floor(Math.random() * conversationStarters.length)];
        
        // Vary response parameters
        this.maxResponseLength = 128 + Math.floor(Math.random() * 128); // 128-256
        
        // Add personality parameters
        this.personalityParams = {
            creativity: 0.3 + Math.random() * 0.7,    // 0.3-1.0
            analytical: 0.2 + Math.random() * 0.8,    // 0.2-1.0
            empathy: 0.4 + Math.random() * 0.6,       // 0.4-1.0
            humor: Math.random() * 0.8,               // 0-0.8
            formality: Math.random(),                 // 0-1.0
            enthusiasm: 0.2 + Math.random() * 0.8,    // 0.2-1.0
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🤖 DiabloGPT job created with personality: creativity=${this.personalityParams.creativity.toFixed(2)}, empathy=${this.personalityParams.empathy.toFixed(2)}`);
    }
}

class WASMMatrixJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'WASMMatrix', 'cpu', complexity);
        this.matrixSize = 512 * complexity;
        this.operations = ['multiply', 'transpose', 'inverse'];
        console.log(`🔢 WASMMatrix job created with CPU backend`);
    }
}

class WASMPrimeJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'WASMPrime', 'cpu', complexity);
        this.maxNumber = 10000 * complexity;
        this.algorithm = 'sieve';
        console.log(`🔍 WASMPrime job created with CPU backend`);
    }
}

class WASMFractalJob extends AIModelJob {
    constructor(id, complexity = 1) {
        super(id, 'WASMFractal', 'cpu', complexity);
        
        // Vary fractal parameters significantly to create unique outputs
        this.iterations = 500 + Math.floor(Math.random() * 1000) * complexity; // 500-1500 per complexity
        
        const fractalTypes = ['mandelbrot', 'julia', 'burning_ship', 'tricorn', 'multibrot'];
        this.fractalType = fractalTypes[Math.floor(Math.random() * fractalTypes.length)];
        
        this.resolution = 128 + Math.floor(Math.random() * 256); // 128-384
        
        // Vary mathematical parameters dramatically
        this.fractalParams = {
            centerX: -0.5 + Math.random() * 1.0,    // -0.5 to 0.5
            centerY: -0.5 + Math.random() * 1.0,    // -0.5 to 0.5 
            zoom: 0.5 + Math.random() * 3.0,        // 0.5 to 3.5
            escapeRadius: 2.0 + Math.random() * 2.0, // 2.0 to 4.0
            colorScheme: Math.floor(Math.random() * 8), // 0-7
            juliaConstantReal: -0.8 + Math.random() * 1.6, // -0.8 to 0.8
            juliaConstantImag: -0.8 + Math.random() * 1.6, // -0.8 to 0.8
            power: 2 + Math.floor(Math.random() * 4), // 2-5 for multibrot
            uniqueId: Date.now() + Math.random()
        };
        
        console.log(`🌀 WASMFractal job created: ${this.fractalType}, iterations: ${this.iterations}, resolution: ${this.resolution}x${this.resolution}`);
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
            case 'SpeechT5':
                return new SpeechT5Job(id, complexity);
            case 'Whisper':
                return new WhisperJob(id, complexity);
            case 'VAD':
                return new VADJob(id, complexity);
            case 'TinyLlama':
                return new TinyLlamaJob(id, complexity);
            case 'DiabloGPT':
                return new DiabloGPTJob(id, complexity);
            case 'WASMMatrix':
                return new WASMMatrixJob(id, complexity);
            case 'WASMPrime':
                return new WASMPrimeJob(id, complexity);
            case 'WASMFractal':
                return new WASMFractalJob(id, complexity);
            default:
                console.warn(`Unknown AI model type: ${modelType}`);
                return new AIModelJob(id, modelType, 'cpu', complexity);
        }
    }

    createRandomAIJob() {
        const modelTypes = [
            'DeepMimic', 'FaceFormer', 'Audio2Gesture', 'RSMT', 
            'Kokoro', 'SpeechT5', 'Whisper', 'VAD', 'TinyLlama', 'DiabloGPT',
            'WASMMatrix', 'WASMPrime', 'WASMFractal'
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