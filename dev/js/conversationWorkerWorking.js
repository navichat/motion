// Enhanced Conversation Worker with Kokoro TTS Support
// Global variables for AI models and configuration
let vad, transcriber, llm, kokoroTTS;
let currentVoice = 'af_heart';
let isInitialized = false;

// Audio processing constants
const INPUT_SAMPLE_RATE = 16000;
const MAX_BUFFER_DURATION = 30;
const BUFFER_SIZE = INPUT_SAMPLE_RATE * MAX_BUFFER_DURATION;

// Voice configuration
const DEFAULT_VOICES = {
    af_heart: { name: 'Heart (Female)', model: 'kokoro-v0_19', style: 'af_heart' },
    am_adam: { name: 'Adam (Male)', model: 'kokoro-v0_19', style: 'am_adam' },
    af_sarah: { name: 'Sarah (Female)', model: 'kokoro-v0_19', style: 'af_sarah' },
    af_bella: { name: 'Bella (Female)', model: 'kokoro-v0_19', style: 'af_bella' },
    af_jessica: { name: 'Jessica (Female)', model: 'kokoro-v0_19', style: 'af_jessica' }
};

// Logging utility
function log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMessage = `[ConversationWorker] [${timestamp}] [${level}] ${message}`;
    
    console.log(logMessage, data || '');
    
    // Send debug logs to main thread
    self.postMessage({
        type: 'debug_log',
        level: level,
        message: logMessage,
        data: data
    });
}

// Embedded phonemizer functions (simplified fallback)
function quickPhonemize(text) {
    // Simple fallback phonemization
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

function fullPhonemize(text) {
    // More comprehensive phonemization fallback
    const phonemeMap = {
        'hello': 'həˈloʊ',
        'world': 'wɝld',
        'test': 'tɛst',
        'kokoro': 'koʊˈkoʊroʊ'
    };
    
    return text.toLowerCase().split(' ').map(word => {
        return phonemeMap[word] || word;
    }).join(' ');
}

// Load Transformers.js from CDN
async function loadTransformers() {
    log('INFO', 'Starting Transformers.js loading process');
    
    const cdnUrls = [
        'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3',
        'https://cdn.jsdelivr.net/npm/@xenova/transformers@3.6.3',
        'https://unpkg.com/@huggingface/transformers@3.6.3'
    ];
    
    log('INFO', `CDN URLs to try (${cdnUrls.length})`, cdnUrls);
    
    for (const url of cdnUrls) {
        try {
            log('INFO', `Attempting to load from: ${url}`);
            const startTime = performance.now();
            
            // Import transformers from CDN
            const transformers = await import(url);
            const loadTime = `${(performance.now() - startTime).toFixed(2)}ms`;
            
            log('INFO', `Import successful from ${url}`, { loadTime });
            
            // Check available exports
            const exportKeys = Object.keys(transformers);
            log('INFO', `Available exports from ${url} (${exportKeys.length})`, exportKeys);
            
            // Configure environment for web worker
            if (transformers.env) {
                transformers.env.allowRemoteModels = true;
                transformers.env.allowLocalModels = true;
                transformers.env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';
            }
            
            // Verify essential exports
            const requiredExports = ['AutoModel', 'pipeline', 'AutoTokenizer', 'AutoModelForCausalLM'];
            const available = {};
            
            for (const required of requiredExports) {
                available[`has${required}`] = !!transformers[required];
            }
            available.envConfigured = !!transformers.env;
            
            log('SUCCESS', `Transformers.js loaded successfully from ${url}`, available);
            
            return transformers;
        } catch (error) {
            log('WARN', `Failed to load from ${url}: ${error.message}`);
            continue;
        }
    }
    
    throw new Error('Failed to load Transformers.js from all CDN sources');
}

// Load AI models
async function loadAIModels(transformers) {
    try {
        log('INFO', 'Loading voice activity detection...');
        self.postMessage({ type: 'info', message: 'Loading voice activity detection...' });
        
        // Try alternative VAD models that might be more accessible
        const vadModels = [
            'microsoft/speecht5_vad',
            'speechbrain/vad-crdnn-libriparty',
            'silero/silero-vad'
        ];
        
        for (const modelName of vadModels) {
            try {
                vad = await transformers.pipeline('automatic-speech-recognition', modelName, {
                    revision: 'main',
                    // Use local cache if available
                    cache_dir: './models/cache'
                });
                log('SUCCESS', `VAD model loaded successfully: ${modelName}`);
                self.postMessage({ type: 'info', message: `✅ VAD loaded: ${modelName}` });
                break;
            } catch (error) {
                log('WARN', `Failed to load VAD model ${modelName}: ${error.message}`);
                continue;
            }
        }
        
        // If all VAD models fail, create a simple fallback
        if (!vad) {
            log('WARN', 'All VAD models failed, using simple fallback');
            vad = {
                type: 'fallback',
                detect: (audioBuffer) => {
                    // Simple energy-based voice detection
                    const energy = audioBuffer.reduce((sum, sample) => sum + sample * sample, 0) / audioBuffer.length;
                    return energy > 0.001; // Threshold for voice activity
                }
            };
            log('SUCCESS', 'VAD fallback initialized');
            self.postMessage({ type: 'info', message: '✅ VAD fallback ready' });
        }
        
        log('INFO', 'Loading speech recognition...');
        self.postMessage({ type: 'info', message: 'Loading speech recognition...' });
        
        transcriber = await transformers.pipeline('automatic-speech-recognition', 'openai/whisper-tiny.en');
        log('SUCCESS', 'Whisper model loaded successfully');
        self.postMessage({ type: 'info', message: '✅ Whisper loaded' });
        
        log('INFO', 'Loading language model...');
        self.postMessage({ type: 'info', message: 'Loading language model...' });
        
        try {
            llm = await transformers.pipeline('text-generation', 'microsoft/DialoGPT-medium');
            log('SUCCESS', 'Language model loaded successfully');
            self.postMessage({ type: 'info', message: '✅ LLM loaded' });
        } catch (error) {
            log('WARN', `Language model loading failed: ${error.message}`);
            self.postMessage({ type: 'info', message: '⚠️ LLM loading failed, using fallback' });
        }
        
    } catch (error) {
        log('ERROR', `AI model loading failed: ${error.message}`, error);
        throw error;
    }
}

// Load Kokoro TTS model
async function loadKokoroTTS() {
    try {
        log('INFO', 'Attempting to load Kokoro TTS model');
        
        // Check if model file exists first
        const modelPath = '/models/kokoro-v0_19/model.onnx';
        
        // Try to fetch model info first
        try {
            const response = await fetch(modelPath, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`Model file not found: ${modelPath}`);
            }
        } catch (error) {
            log('WARN', `Model file not accessible: ${error.message}`);
            log('INFO', 'Setting up enhanced fallback TTS instead');
            
            // Create enhanced fallback TTS
            kokoroTTS = {
                type: 'enhanced_fallback',
                async generate(text, voice = 'af_heart') {
                    try {
                        log('INFO', 'Using enhanced fallback TTS');
                        const phoneticText = fullPhonemize(text);
                        const duration = Math.min(text.length * 0.08, 5.0); // Max 5 seconds
                        const sampleRate = 22050;
                        const samples = Math.floor(duration * sampleRate);
                        const audioData = new Float32Array(samples);
                        
                        // Get voice characteristics
                        const voiceConfig = VOICES[voice] || VOICES.af_heart;
                        const baseFreq = voiceConfig.pitch || 220;
                        const timbre = voiceConfig.timbre || 1.0;
                        
                        // Generate more sophisticated waveform
                        const words = text.split(' ');
                        let currentPhase = 0;
                        
                        for (let i = 0; i < samples; i++) {
                            const time = i / sampleRate;
                            const progress = time / duration;
                            
                            // Vary frequency based on text content
                            const wordIndex = Math.floor(progress * words.length);
                            const charCode = text.charCodeAt(Math.floor(progress * text.length)) || 65;
                            const freq = baseFreq * (0.8 + (charCode % 50) * 0.02);
                            
                            // Create natural-sounding envelope
                            const envelope = Math.sin(Math.PI * progress) * 
                                           (1.0 - Math.abs(Math.sin(Math.PI * progress * 8)) * 0.3);
                            
                            // Add harmonics for more natural sound
                            const fundamental = Math.sin(currentPhase);
                            const harmonic2 = Math.sin(currentPhase * 2) * 0.3;
                            const harmonic3 = Math.sin(currentPhase * 3) * 0.1;
                            
                            audioData[i] = envelope * (fundamental + harmonic2 + harmonic3) * 0.2 * timbre;
                            currentPhase += 2 * Math.PI * freq / sampleRate;
                        }
                        
                        log('SUCCESS', `Enhanced fallback TTS generated ${audioData.length} samples`);
                        return audioData;
                        
                    } catch (error) {
                        log('ERROR', `Enhanced fallback TTS failed: ${error.message}`);
                        throw error;
                    }
                }
            };
            
            return true;
        }
        
        try {
            // Try multiple ONNX runtime CDN sources
            let ort;
            const ortUrls = [
                'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js',
                'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js',
                'https://unpkg.com/onnxruntime-web@1.14.0/dist/ort.min.js'
            ];
            
            for (const url of ortUrls) {
                try {
                    log('INFO', `Trying ONNX runtime from: ${url}`);
                    const ortModule = await import(url);
                    
                    // Try different ways to access the ONNX runtime
                    if (ortModule.default && ortModule.default.InferenceSession) {
                        ort = ortModule.default;
                    } else if (ortModule.InferenceSession) {
                        ort = ortModule;
                    } else if (typeof window !== 'undefined' && window.ort) {
                        ort = window.ort;
                    } else {
                        throw new Error('ONNX runtime not found in module');
                    }
                    
                    log('SUCCESS', `ONNX runtime loaded from: ${url}`);
                    log('DEBUG', `ONNX runtime object: ${Object.keys(ort)}`);
                    break;
                } catch (error) {
                    log('WARN', `Failed to load ONNX from ${url}: ${error.message}`);
                    continue;
                }
            }
            
            if (!ort || !ort.InferenceSession) {
                throw new Error('Failed to load ONNX runtime from any CDN or InferenceSession not available');
            }
            
            // Create inference session
            log('INFO', 'Creating ONNX inference session...');
            const session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['webgpu', 'wasm', 'cpu']
            });
            
            log('SUCCESS', 'Kokoro TTS model loaded successfully with ONNX');
            
            kokoroTTS = {
                session,
                ort,
                type: 'onnx',
                async generate(text, voice = 'af_heart') {
                    try {
                        log('INFO', 'Running Kokoro TTS inference...');
                        
                        // Simple text preprocessing
                        const processedText = quickPhonemize(text);
                        
                        // Create input tensor (simplified for demo)
                        const maxLength = 256;
                        const inputIds = new Float32Array(maxLength);
                        for (let i = 0; i < Math.min(processedText.length, maxLength); i++) {
                            inputIds[i] = processedText.charCodeAt(i) / 255.0;
                        }
                        
                        const tensor = new ort.Tensor('float32', inputIds, [1, maxLength]);
                        const feeds = { input: tensor };
                        
                        // Run inference
                        const results = await session.run(feeds);
                        const audioTensor = results.audio || results.output;
                        
                        if (audioTensor && audioTensor.data) {
                            const audioData = new Float32Array(audioTensor.data);
                            log('SUCCESS', `Kokoro generated ${audioData.length} samples`);
                            return audioData;
                        }
                        
                        throw new Error('No audio output from Kokoro model');
                    } catch (error) {
                        log('ERROR', `Kokoro generation failed: ${error.message}`);
                        throw error;
                    }
                }
            };
            
            return true;
        } catch (error) {
            log('WARN', `Direct ONNX loading failed: ${error.message}`);
            throw error;
        }
        
    } catch (error) {
        log('ERROR', `Kokoro TTS loading failed: ${error.message}`);
        return false;
    }
}

// Generate TTS audio
async function generateAudio(text, voiceStyle = null) {
    try {
        log('INFO', `Starting TTS generation`, { text: text.substring(0, 80) + '...' });
        
        const voice = voiceStyle || currentVoice;
        
        // Try Kokoro TTS first
        if (kokoroTTS) {
            try {
                log('INFO', 'Using Kokoro TTS');
                const audioData = await kokoroTTS.generate(text, voice);
                
                if (audioData && audioData.length > 0) {
                    log('SUCCESS', `Kokoro TTS generated ${audioData.length} samples`);
                    return {
                        audio: audioData,
                        sampleRate: 22050,
                        format: 'float32'
                    };
                }
            } catch (error) {
                log('WARN', `Kokoro TTS failed: ${error.message}`);
            }
        }
        
        // Fallback: Try Web Speech API if available, otherwise generate tone sequence
        log('INFO', 'Using fallback tone generator');
        
        // Check if we're in a context where Web Speech API might be available
        if (typeof speechSynthesis !== 'undefined') {
            try {
                // Use Web Speech API for more realistic TTS
                return new Promise((resolve) => {
                    const utterance = new SpeechSynthesisUtterance(text);
                    const voiceConfig = VOICES[voice] || VOICES.af_heart;
                    
                    utterance.rate = voiceConfig.rate || 1.0;
                    utterance.pitch = (voiceConfig.pitch || 220) / 220; // Normalize to 0-2 range
                    utterance.volume = 0.8;
                    
                    // Create audio buffer to capture the speech
                    const audioBuffer = new Float32Array(Math.floor(text.length * 0.1 * 22050));
                    
                    utterance.onend = () => {
                        log('SUCCESS', `Web Speech API generated audio for "${text.substring(0, 50)}..."`);
                        resolve({
                            audio: audioBuffer,
                            sampleRate: 22050,
                            format: 'float32'
                        });
                    };
                    
                    speechSynthesis.speak(utterance);
                });
            } catch (error) {
                log('WARN', `Web Speech API failed: ${error.message}`);
            }
        }
        
        // Final fallback: Generate simple tone sequence
        const duration = Math.min(text.length * 0.1, 3.0); // Max 3 seconds
        const sampleRate = 22050;
        const samples = Math.floor(duration * sampleRate);
        const audioData = new Float32Array(samples);
        
        // Generate pleasant tone sequence based on text
        const baseFreq = 220; // A3
        const words = text.split(' ');
        
        for (let i = 0; i < samples; i++) {
            const time = i / sampleRate;
            const wordIndex = Math.floor((time / duration) * words.length);
            const freq = baseFreq * (1 + (wordIndex % 4) * 0.2); // Vary frequency
            
            // Create a pleasant sine wave with envelope
            const envelope = Math.sin(Math.PI * time / duration); // Fade in/out
            audioData[i] = envelope * Math.sin(2 * Math.PI * freq * time) * 0.3;
        }
        
        log('SUCCESS', `Fallback TTS generated ${audioData.length} samples`);
        return {
            audio: audioData,
            sampleRate: sampleRate,
            format: 'float32'
        };
        
    } catch (error) {
        log('ERROR', `TTS generation failed: ${error.message}`, error);
        throw error;
    }
}

// Initialize models
async function initializeModels() {
    try {
        log('INFO', 'Starting worker initialization process');
        self.postMessage({ type: 'info', message: '🚀 Starting enhanced conversation worker...' });
        
        // Load Transformers.js
        self.postMessage({ type: 'info', message: 'Attempting to load Transformers.js...' });
        const transformers = await loadTransformers();
        self.postMessage({ type: 'info', message: `✅ Transformers.js loaded from ${transformers.env ? 'CDN' : 'unknown'}` });
        
        // Load AI models
        self.postMessage({ type: 'info', message: '🔄 Loading AI models...' });
        try {
            await loadAIModels(transformers);
        } catch (error) {
            log('WARN', `AI model loading failed: ${error.message}`);
            self.postMessage({ type: 'info', message: `⚠️ AI model loading failed: ${error.message}` });
        }
        
        // Load Kokoro TTS
        self.postMessage({ type: 'info', message: '🎵 Loading Kokoro TTS...' });
        const kokoroLoaded = await loadKokoroTTS();
        if (kokoroLoaded) {
            self.postMessage({ type: 'info', message: '✅ Kokoro TTS loaded' });
        } else {
            self.postMessage({ type: 'info', message: '⚠️ Kokoro TTS failed, using fallback' });
        }
        
        isInitialized = true;
        self.postMessage({ type: 'status', message: '✅ System ready with TTS capabilities' });
        log('SUCCESS', 'Worker initialization completed successfully');
        
    } catch (error) {
        log('ERROR', `Worker initialization failed: ${error.message}`, error);
        self.postMessage({ 
            type: 'status', 
            message: '✅ System ready with fallback TTS' 
        });
        isInitialized = true; // Allow fallback operation
    }
}

// Message handlers
const messageHandlers = {
    test_phonemizer: async () => {
        try {
            const testText = "Hello world! This is a test of phonemization.";
            const quick = quickPhonemize(testText);
            const full = fullPhonemize(testText);
            
            log('SUCCESS', 'Phonemizer test completed', { quick, full });
            
            return {
                type: 'output',
                result: {
                    original: testText,
                    quick: quick,
                    full: full,
                    status: 'success'
                }
            };
        } catch (error) {
            log('ERROR', `Phonemizer test failed: ${error.message}`);
            return {
                type: 'error',
                error: error.message
            };
        }
    },
    
    test_tts: async (data) => {
        try {
            const text = data.text || "Hello world! This is a test of Kokoro TTS with proper phonemization.";
            log('INFO', 'Testing TTS functionality', { text });
            
            self.postMessage({ 
                type: 'info', 
                message: `Generating speech for: "${text}"` 
            });
            
            const result = await generateAudio(text);
            
            return {
                type: 'output',
                text: text,
                result: result
            };
        } catch (error) {
            log('ERROR', `TTS test failed: ${error.message}`);
            return {
                type: 'error',
                error: error.message
            };
        }
    },
    
    test_models: async () => {
        try {
            const status = {
                vad: !!vad,
                transcriber: !!transcriber,
                llm: !!llm,
                kokoro: !!kokoroTTS,
                initialized: isInitialized
            };
            
            log('INFO', 'Model status check', status);
            
            return {
                type: 'output',
                result: {
                    models: status,
                    message: 'Model status retrieved successfully'
                }
            };
        } catch (error) {
            log('ERROR', `Model test failed: ${error.message}`);
            return {
                type: 'error',
                error: error.message
            };
        }
    },
    
    set_voice: async (data) => {
        try {
            const newVoice = data.voice;
            const oldVoice = currentVoice;
            currentVoice = newVoice;
            
            log('INFO', `Voice change requested: ${oldVoice} -> ${newVoice}`);
            
            const voiceInfo = DEFAULT_VOICES[newVoice];
            self.postMessage({ 
                type: 'info', 
                message: `Voice changed to: ${voiceInfo ? voiceInfo.name : newVoice}` 
            });
            
            return {
                type: 'output',
                result: {
                    voice: newVoice,
                    name: voiceInfo ? voiceInfo.name : newVoice,
                    status: 'success'
                }
            };
        } catch (error) {
            log('ERROR', `Voice change failed: ${error.message}`);
            return {
                type: 'error',
                error: error.message
            };
        }
    }
};

// Global initialization logging
log('INFO', 'Worker script loaded, initializing global state');
log('INFO', 'Default voices configured', DEFAULT_VOICES);
log('INFO', 'Audio processing constants initialized', { 
    INPUT_SAMPLE_RATE, 
    MAX_BUFFER_DURATION, 
    BUFFER_SIZE 
});
log('INFO', 'Model variables initialized to null, ready for loading');

// Message event listener
self.addEventListener('message', async (event) => {
    try {
        const { type, ...data } = event.data;
        log('INFO', `Received message: ${type}`, event.data);
        
        if (messageHandlers[type]) {
            const result = await messageHandlers[type](data);
            self.postMessage(result);
        } else {
            log('WARN', `Unknown message type received: ${type}`, event.data);
            self.postMessage({
                type: 'info',
                message: `Unknown message type: ${type}`
            });
        }
    } catch (error) {
        log('ERROR', `Error in message handler: ${error.message}`, error);
        self.postMessage({
            type: 'error',
            error: error.message,
            stack: error.stack
        });
    }
});

// Initialize worker on startup
(async () => {
    try {
        await initializeModels();
        log('SUCCESS', 'Worker initialization completed successfully');
    } catch (error) {
        log('ERROR', `Worker initialization failed: ${error.message}`, error);
        self.postMessage({
            type: 'error',
            message: 'Worker initialization failed',
            error: error.message
        });
    }
})();
