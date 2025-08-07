/**
 * Modular Kokoro TTS System
 * 
 * This module provides a clean interface for text-to-speech using the Kokoro model
 * with proper phonemization and voice handling. Designed to integrate seamlessly
 * with the existing 3D avatar chat system.
 * 
 * Features:
 * - Proper text-to-phoneme conversion
 * - Voice selection and management
 * - Streaming support for real-time generation
 * - WebGPU and WASM backend support
 * - Caching for performance
 * 
 * Usage:
 * ```javascript
 * import { KokoroTTS } from './js/kokoro-tts.js';
 * 
 * const tts = new KokoroTTS();
 * await tts.initialize();
 * 
 * const audioBuffer = await tts.synthesize("Hello world!", { voice: "af_heart" });
 * ```
 */

import { phonemize } from '../kokoro.js/src/phonemize.js';
import { VOICES } from '../kokoro.js/src/voices.js';

// Configuration constants
const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;
const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const VOICE_DATA_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices";

// Cache for voice data and model
const VOICE_CACHE = new Map();
let MODEL_CACHE = null;
let TOKENIZER_CACHE = null;

/**
 * Text splitter for streaming support
 */
class TextSplitter {
    constructor() {
        this.buffer = [];
        this.closed = false;
    }
    
    push(text) {
        if (this.closed) return;
        this.buffer.push(text);
    }
    
    close() {
        this.closed = true;
    }
    
    *[Symbol.iterator]() {
        while (this.buffer.length > 0 || !this.closed) {
            if (this.buffer.length > 0) {
                yield this.buffer.shift();
            } else {
                // Wait for more data
                // In a real implementation, this would be async
                break;
            }
        }
    }
}

/**
 * Main Kokoro TTS class
 */
export class KokoroTTS {
    constructor(options = {}) {
        this.options = {
            dtype: options.dtype || 'fp32',
            device: options.device || 'wasm',
            modelPath: options.modelPath || './models/Kokoro-82M-v1.0-ONNX',
            useLocalModel: options.useLocalModel || true,
            ...options
        };
        
        this.model = null;
        this.tokenizer = null;
        this.initialized = false;
        this.voices = VOICES;
    }
    
    /**
     * Initialize the TTS system
     */
    async initialize() {
        if (this.initialized) return;
        
        try {
            // Load transformers.js
            const { AutoModel, AutoTokenizer, env, Tensor } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3');
            this.Tensor = Tensor;
            
            // Configure environment
            this.configureEnvironment(env);
            
            // Load model and tokenizer
            if (this.options.useLocalModel) {
                this.model = await this.loadLocalModel(AutoModel);
                this.tokenizer = await this.loadLocalTokenizer(AutoTokenizer);
            } else {
                this.model = await AutoModel.from_pretrained(MODEL_ID, {
                    dtype: this.options.dtype,
                    device: this.options.device
                });
                this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
            }
            
            this.initialized = true;
            console.log('✅ Kokoro TTS initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Kokoro TTS:', error);
            throw error;
        }
    }
    
    /**
     * Configure transformers.js environment
     */
    configureEnvironment(env) {
        env.allowLocalModels = true;
        env.localModelPath = './';
        env.remoteModelPath = 'https://huggingface.co/';
        env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3/dist/';
        
        // WebGPU configuration
        if (this.options.device === 'webgpu') {
            env.backends.onnx.executionProviders = ['webgpu', 'wasm'];
            env.backends.onnx.webgpu = {
                preferredLayout: 'NHWC',
                validateInputContent: false,
                contextId: 'kokoro-webgpu-context',
                powerPreference: 'high-performance'
            };
        }
    }
    
    /**
     * Load local model with proper quantization selection
     */
    async loadLocalModel(AutoModel) {
        // Check WebGPU support
        const webgpuSupported = await this.checkWebGPUSupport();
        
        // Select appropriate quantization
        const quantizationOptions = [
            { dtype: 'q4f16', modelFile: 'model_q4f16.onnx', name: '4-bit float16', requiresF16: true },
            { dtype: 'q8f16', modelFile: 'model_q8f16.onnx', name: '8-bit float16', requiresF16: true },
            { dtype: 'fp16', modelFile: 'model_fp16.onnx', name: '16-bit floating point', requiresF16: true },
            { dtype: 'fp32', modelFile: 'model.onnx', name: '32-bit floating point', requiresF16: false }
        ];
        
        const deviceOptions = [
            { device: 'webgpu', name: 'WebGPU', compatible: webgpuSupported.supported },
            { device: 'wasm', name: 'WASM', compatible: true }
        ];
        
        for (const deviceOption of deviceOptions) {
            if (!deviceOption.compatible) continue;
            
            let quantsToTry = quantizationOptions;
            if (deviceOption.device === 'webgpu' && !webgpuSupported.f16) {
                quantsToTry = quantizationOptions.filter(q => !q.requiresF16);
            }
            
            for (const quantOption of quantsToTry) {
                try {
                    console.log(`Trying ${quantOption.name} quantization with ${deviceOption.name}...`);
                    
                    const modelOptions = {
                        dtype: quantOption.dtype,
                        device: deviceOption.device
                    };
                    
                    const model = await AutoModel.from_pretrained(this.options.modelPath, modelOptions);
                    console.log(`✅ Successfully loaded model with ${quantOption.name} on ${deviceOption.name}`);
                    return model;
                    
                } catch (error) {
                    console.warn(`⚠️ Failed to load with ${quantOption.name} on ${deviceOption.name}:`, error.message);
                    continue;
                }
            }
        }
        
        throw new Error('Failed to load model with any quantization option');
    }
    
    /**
     * Load local tokenizer
     */
    async loadLocalTokenizer(AutoTokenizer) {
        return await AutoTokenizer.from_pretrained(this.options.modelPath);
    }
    
    /**
     * Check WebGPU support and f16 capabilities
     */
    async checkWebGPUSupport() {
        if (!navigator.gpu) {
            return { supported: false, f16: false };
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { supported: false, f16: false };
            }
            
            const f16Supported = adapter.features.has('shader-f16');
            
            const deviceDescriptor = {};
            if (f16Supported) {
                deviceDescriptor.requiredFeatures = ['shader-f16'];
            }
            
            const device = await adapter.requestDevice(deviceDescriptor);
            return { 
                supported: !!device, 
                f16: f16Supported,
                adapter: adapter,
                device: device
            };
            
        } catch (error) {
            console.warn('WebGPU check failed:', error);
            return { supported: false, f16: false };
        }
    }
    
    /**
     * Get available voices
     */
    getVoices() {
        return Object.keys(this.voices).map(id => ({
            id,
            ...this.voices[id]
        }));
    }
    
    /**
     * Validate voice selection
     */
    validateVoice(voice) {
        if (!this.voices.hasOwnProperty(voice)) {
            throw new Error(`Voice "${voice}" not found. Available voices: ${Object.keys(this.voices).join(', ')}`);
        }
        return voice.at(0); // Return language code ('a' for American, 'b' for British)
    }
    
    /**
     * Load voice data from cache or fetch
     */
    async getVoiceData(voice) {
        if (VOICE_CACHE.has(voice)) {
            return VOICE_CACHE.get(voice);
        }
        
        try {
            // Try local file first
            const localPath = `./models/Kokoro-82M-v1.0-ONNX/voices/${voice}.bin`;
            const response = await fetch(localPath);
            
            if (response.ok) {
                const buffer = await response.arrayBuffer();
                const data = new Float32Array(buffer);
                VOICE_CACHE.set(voice, data);
                return data;
            }
        } catch (error) {
            console.warn(`Failed to load local voice ${voice}:`, error);
        }
        
        // Fall back to remote
        const url = `${VOICE_DATA_URL}/${voice}.bin`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load voice data for ${voice}: ${response.status}`);
        }
        
        const buffer = await response.arrayBuffer();
        const data = new Float32Array(buffer);
        VOICE_CACHE.set(voice, data);
        return data;
    }
    
    /**
     * Convert text to phonemes
     */
    async textToPhonemes(text, language = 'a') {
        try {
            return await phonemize(text, language);
        } catch (error) {
            console.warn('Phonemization failed, using fallback:', error);
            return this.fallbackPhonemes(text);
        }
    }
    
    /**
     * Fallback phoneme conversion (basic implementation)
     */
    fallbackPhonemes(text) {
        // Simple phoneme mapping for common words
        const phonemeMap = {
            'hello': 'h ə l oʊ',
            'world': 'w ɝ l d',
            'this': 'ð ɪ s',
            'is': 'ɪ z',
            'a': 'ə',
            'test': 't ɛ s t',
            'of': 'ʌ v',
            'the': 'ð ə',
            'and': 'æ n d',
            'to': 't u',
            'you': 'j u',
            'I': 'aɪ',
            'we': 'w i',
            'they': 'ð eɪ',
            'can': 'k æ n',
            'will': 'w ɪ l',
            'have': 'h æ v',
            'are': 'ɑ r',
            'with': 'w ɪ θ',
            'for': 'f ɔ r',
            'on': 'ɑ n',
            'at': 'æ t',
            'by': 'b aɪ',
            'from': 'f r ʌ m',
            'up': 'ʌ p',
            'about': 'ə b aʊ t',
            'into': 'ɪ n t u',
            'over': 'oʊ v ɝ',
            'after': 'æ f t ɝ'
        };
        
        // Convert to lowercase and split into words
        const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
        
        // Convert each word to phonemes
        const phonemes = words.map(word => {
            if (phonemeMap[word]) {
                return phonemeMap[word];
            } else {
                // Basic letter-to-phoneme mapping
                return word.split('').map(char => {
                    const charMap = {
                        'a': 'æ', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɛ',
                        'f': 'f', 'g': 'ɡ', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
                        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ɑ',
                        'p': 'p', 'q': 'kw', 'r': 'r', 's': 's', 't': 't',
                        'u': 'ʌ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j',
                        'z': 'z'
                    };
                    return charMap[char] || char;
                }).join(' ');
            }
        });
        
        return phonemes.join(' ');
    }
    
    /**
     * Synthesize speech from text
     */
    async synthesize(text, options = {}) {
        if (!this.initialized) {
            await this.initialize();
        }
        
        const { voice = 'af_heart', speed = 1.0 } = options;
        
        try {
            // Validate voice
            const language = this.validateVoice(voice);
            
            // Convert text to phonemes
            const phonemes = await this.textToPhonemes(text, language);
            console.log(`Phonemes: ${phonemes}`);
            
            // Tokenize phonemes
            const { input_ids } = this.tokenizer(phonemes, { truncation: true });
            const tokenLength = input_ids.dims.at(-1);
            
            // Load voice data
            const voiceData = await this.getVoiceData(voice);
            
            // Calculate style vector based on token length
            const styleIndex = Math.min(Math.max(tokenLength - 2, 0), 509);
            const offset = styleIndex * STYLE_DIM;
            const styleVector = voiceData.slice(offset, offset + STYLE_DIM);
            
            // Prepare model inputs
            const inputs = {
                input_ids,
                style: new this.Tensor('float32', styleVector, [1, STYLE_DIM]),
                speed: new this.Tensor('float32', [speed], [1])
            };
            
            // Run inference
            const { waveform } = await this.model(inputs);
            
            // Create audio buffer
            const audioBuffer = this.createAudioBuffer(waveform.data);
            
            return {
                audioBuffer,
                sampleRate: SAMPLE_RATE,
                duration: waveform.data.length / SAMPLE_RATE,
                metadata: {
                    text,
                    phonemes,
                    voice,
                    speed,
                    tokens: tokenLength
                }
            };
            
        } catch (error) {
            console.error('Synthesis failed:', error);
            throw error;
        }
    }
    
    /**
     * Create Web Audio API AudioBuffer from waveform data
     */
    createAudioBuffer(waveformData) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = audioContext.createBuffer(1, waveformData.length, SAMPLE_RATE);
        const channelData = audioBuffer.getChannelData(0);
        
        // Copy waveform data to audio buffer
        if (waveformData instanceof Float32Array) {
            channelData.set(waveformData);
        } else {
            for (let i = 0; i < waveformData.length; i++) {
                channelData[i] = waveformData[i];
            }
        }
        
        return audioBuffer;
    }
    
    /**
     * Play audio buffer
     */
    async playAudio(audioBuffer) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
        
        return new Promise((resolve) => {
            source.onended = resolve;
        });
    }
    
    /**
     * Convert audio buffer to WAV blob
     */
    audioBufferToWav(audioBuffer) {
        const length = audioBuffer.length;
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, audioBuffer.sampleRate, true);
        view.setUint32(28, audioBuffer.sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert float32 to int16
        const channelData = audioBuffer.getChannelData(0);
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
        
        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }
    
    /**
     * Stream synthesis (for real-time generation)
     */
    async* streamSynthesize(text, options = {}) {
        if (!this.initialized) {
            await this.initialize();
        }
        
        const { voice = 'af_heart', speed = 1.0, chunkSize = 100 } = options;
        
        // Split text into chunks
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        
        for (const sentence of sentences) {
            if (sentence.trim().length > 0) {
                try {
                    const result = await this.synthesize(sentence.trim(), { voice, speed });
                    yield result;
                } catch (error) {
                    console.warn('Failed to synthesize sentence:', sentence, error);
                }
            }
        }
    }
}

/**
 * Convenience function to create and initialize TTS
 */
export async function createKokoroTTS(options = {}) {
    const tts = new KokoroTTS(options);
    await tts.initialize();
    return tts;
}

/**
 * Export voices for external use
 */
export { VOICES };

/**
 * Export utility functions
 */
export const KokoroUtils = {
    /**
     * Check if browser supports WebGPU
     */
    async checkWebGPUSupport() {
        if (!navigator.gpu) return false;
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            return !!adapter;
        } catch {
            return false;
        }
    },
    
    /**
     * Get recommended configuration for current browser
     */
    async getRecommendedConfig() {
        const webgpuSupported = await this.checkWebGPUSupport();
        
        return {
            device: webgpuSupported ? 'webgpu' : 'wasm',
            dtype: webgpuSupported ? 'fp32' : 'q8',
            useLocalModel: true
        };
    }
};
