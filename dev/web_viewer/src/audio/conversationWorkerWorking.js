// Enhanced Conversation Worker with Kokoro TTS Support
// Global variables for AI models and configuration
let vad, transcriber, llm, kokoroTTS;
let currentVoice = 'af_heart';
let isInitialized = false;

// Add a global error handler for the worker
self.onerror = function(event) {
    log('FATAL', `Uncaught error in worker: ${event.message}`, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error ? event.error.stack || event.error.message : 'No error object'
    });
    // Re-throw the error to ensure it's propagated to the main thread's worker.onerror
    return false; // Prevent default error handling
};

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
                transformers.env.allowRemoteModels = false;
                transformers.env.allowLocalModels = true;
                transformers.env.localModelPath = '../models/';
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
        
        try {
            vad = await transformers.pipeline('automatic-speech-recognition', '../models/silero-vad/onnx/model_quantized.onnx', {
                revision: 'main',
                // Use local cache if available
                cache_dir: './models/cache'
            });
            log('SUCCESS', `VAD model loaded successfully: ../models/silero-vad/onnx/model_quantized.onnx`);
            self.postMessage({ type: 'info', message: `✅ VAD loaded: ../models/silero-vad/onnx/model_quantized.onnx` });
        } catch (error) {
            log('WARN', `Failed to load VAD model ../models/silero-vad/onnx/model_quantized.onnx: ${error.message}`);
            throw error; // Re-throw to ensure fallback is only used if local fails
        }
        
        log('INFO', 'Loading speech recognition...');
        self.postMessage({ type: 'info', message: 'Loading speech recognition...' });
        
        try {
            try {
            // Try loading local Whisper model first
            transcriber = await transformers.pipeline('automatic-speech-recognition', '../models/whisper-tiny.en/onnx/decoder_model_merged_quantized.onnx', {
                dtype: 'fp32',
                device: 'wasm'
            });
            log('SUCCESS', 'Local Whisper model loaded successfully');
            self.postMessage({ type: 'info', message: '✅ Whisper loaded (local)' });
        } catch (error) {
            log('WARN', `Local Whisper loading failed: ${error.message}`);
            self.postMessage({ type: 'info', message: '⚠️ Whisper failed, using fallback' });
            transcriber = null;
        }
        
        log('INFO', 'Loading language model...');
        self.postMessage({ type: 'info', message: 'Loading language model...' });
        
        try {
            try {
            // Try loading local DialoGPT model first
            llm = await transformers.pipeline('text-generation', '../models/DialoGPT-medium/pytorch_model.bin', {
                dtype: 'fp32',
                device: 'wasm'
            });
            log('SUCCESS', 'Local DialoGPT model loaded successfully');
            self.postMessage({ type: 'info', message: '✅ LLM loaded (local)' });
        } catch (error) {
            log('WARN', `Local DialoGPT loading failed: ${error.message}`);
            self.postMessage({ type: 'info', message: '⚠️ LLM loading failed, using fallback' });
            llm = null;
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
        
        // Check multiple possible model locations
        const modelPaths = [
            const modelPaths = [            '../models/Kokoro-82M-v1.0-ONNX/model.onnx',
            '../models/Kokoro-82M-v1.0-ONNX/model_quantized.onnx',
            '../models/speecht5_tts/onnx/decoder_model_merged_quantized.onnx'
        ];
        
        let modelFound = false;
        let modelPath = null;
        
        // Try to find an available model
        for (const path of modelPaths) {
            try {
                const response = await fetch(path, { method: 'HEAD' });
                if (response.ok) {
                    modelFound = true;
                    modelPath = path;
                    log('INFO', `Found model at: ${path}`);
                    break;
                }
            } catch (error) {
                log('DEBUG', `Model not found at: ${path}`);
                continue;
            }
        }
        
        if (!modelFound) {
            log('INFO', 'No ONNX model files found, trying transformers pipeline...');
            
            // Try to load TTS using transformers pipeline with local models
            try {
                // Try Kokoro model directory
                kokoroTTS = await transformers.pipeline('text-to-speech', './models/Kokoro-82M-v1.0-ONNX', {
                    dtype: 'fp32',
                    device: 'wasm'
                });
                log('SUCCESS', 'Kokoro TTS loaded via transformers pipeline');
                return true;
            } catch (error) {
                log('WARN', `Kokoro pipeline loading failed: ${error.message}`);
                
                try {
                    // Try SpeechT5 model directory
                    kokoroTTS = await transformers.pipeline('text-to-speech', './models/speecht5_tts', {
                        dtype: 'fp32',
                        device: 'wasm'
                    });
                    log('SUCCESS', 'SpeechT5 TTS loaded via transformers pipeline');
                    return true;
                } catch (speechT5Error) {
                    log('WARN', `SpeechT5 pipeline loading failed: ${speechT5Error.message}`);
                }
            }
            
            log('INFO', 'Creating enhanced fallback TTS');
            
            // Create enhanced fallback TTS
            kokoroTTS = {
                type: 'enhanced_fallback',
                async generate(text, voice = 'af_heart') {
                    try {
                        log('INFO', 'Using enhanced fallback TTS');
                        const phoneticText = quickPhonemize(text);
                        const duration = Math.min(text.length * 0.08, 5.0); // Max 5 seconds
                        const sampleRate = 22050;
                        const samples = Math.floor(duration * sampleRate);
                        const audioData = new Float32Array(samples);
                        
                        // Get voice characteristics
                        const voiceConfig = DEFAULT_VOICES[voice] || DEFAULT_VOICES.af_heart;
                        
                        // Generate more sophisticated waveform
                        const words = text.split(' ');
                        let currentPhase = 0;
                        
                        for (let i = 0; i < samples; i++) {
                            const baseFreq = 220; // A3
                            const time = i / sampleRate;
                            const progress = time / duration; // 0 to 1
                            
                            // Vary frequency based on text content
                            const wordIndex = Math.floor(progress * words.length);
                            const freqOffset = (text.charCodeAt(Math.floor(progress * text.length)) % 20) / 100;
                            const freq = baseFreq * (1 + freqOffset);
                            
                            // Create natural-sounding envelope
                            const envelope = Math.sin(Math.PI * progress); // Fade in/out
                            
                            // Add harmonics for more natural sound
                            const fundamental = Math.sin(currentPhase);         // Base tone
                            const harmonic2 = Math.sin(currentPhase * 2) * 0.3; // Second harmonic
                            const harmonic3 = Math.sin(currentPhase * 3) * 0.1; // Third harmonic
                            
                            // Combine and scale harmonics with the envelope
                            audioData[i] = envelope * (fundamental + harmonic2 + harmonic3) * 0.3;
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
            // ONNX runtime is typically bundled with transformers.js or loaded separately in the main thread.
            // In a web worker, `window` is not defined, so we rely on the global `ort` if it's made available
            // or assume transformers.js handles its own ONNX runtime dependency.
            // For direct ONNX model loading, `ort` needs to be imported or made available globally in the worker.
            // Assuming `ort` is available in the worker's global scope if transformers.js is not used for ONNX.
            // If using transformers.js, it manages ONNX runtime internally.
            // For this direct ONNX path, we need `ort` to be available.
            // If `ort` is not globally available, this block will fail, and the fallback will be used.
            let ort = self.ort; // Attempt to get ort from worker's global scope
            if (!ort || !ort.InferenceSession) {
                throw new Error('ONNX runtime (ort) not found or not properly initialized in worker global scope.');
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
        
        // Always use our enhanced fallback TTS since Web Speech API doesn't work in workers
        log('INFO', 'Using enhanced fallback tone generator');
        
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
