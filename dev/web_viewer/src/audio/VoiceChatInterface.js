/**
 * VoiceChatInterface - Standalone voice chat interface with memory management
 * 
 * Features:
 * - Voice Activity Detection (VAD)
 * - Model eviction based on memory constraints
 * - Audio chunking and queueing
 * - Event-driven architecture
 * - Uses ResourceManager for dependency injection
 */

import { ResourceManager } from './ResourceManager.js';
import { WhisperModule } from './WhisperModule.js';
import { KokoroModule } from './KokoroModule.js';
import { LlamaModule } from './LlamaModule.js';
import { VoiceActivityDetector } from './VoiceActivityDetector.js';
import { AudioQueue } from './AudioQueue.js';

export class VoiceChatInterface extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            memoryThresholdMB: options.memoryThresholdMB || 512,
            vadSensitivity: options.vadSensitivity || 0.5,
            chunkSizeMs: options.chunkSizeMs || 1000,
            audioSampleRate: options.audioSampleRate || 22050,
            modelCacheTimeout: options.modelCacheTimeout || 30000, // 30 seconds
            device: options.device || 'wasm',
            ...options
        };

        // Initialize ResourceManager for dependency injection
        this.resourceManager = new ResourceManager({
            device: this.options.device,
            memoryThresholdMB: this.options.memoryThresholdMB,
            maxConcurrentModels: 2,
            modelCacheTimeout: this.options.modelCacheTimeout,
            audioSampleRate: this.options.audioSampleRate
        });

        // Core modules will be initialized with dependency injection
        this.vad = null;
        this.audioQueue = null;

        // State management
        this.isInitialized = false;
        this.isListening = false;
        this.isProcessing = false;
        this.isSpeaking = false;
        this.currentConversation = [];

        // Model loading failure tracking to prevent spam
        this.modelLoadFailures = {};
        this.maxRetryAttempts = 1; // Only try to load each model once

        // Event handlers
        this.setupEventHandlers();
        
        // Setup event handlers
        this.setupEventHandlers();
        
        // Memory monitoring
        this.memoryMonitor = null;
    }

    /**
     * Initialize the voice chat interface
     */
    async initialize() {
        try {
            this.emit('status', { type: 'init', message: 'Initializing voice chat interface...' });

            // Initialize ResourceManager first
            await this.resourceManager.initialize();
            this.emit('status', { type: 'init', message: 'Resource manager ready' });

            // Register AI models with ResourceManager
            this.resourceManager.registerModel('whisper', WhisperModule, {
                whisperModel: this.options.whisperModel || 'tiny',
                device: this.options.device
            });

            this.resourceManager.registerModel('llama', LlamaModule, {
                llamaModel: 'TinyLlama-1.1B-Chat-v1.0',
                device: this.options.device,
                systemPrompt: this.options.systemPrompt
            });

            this.resourceManager.registerModel('kokoro', KokoroModule, {
                kokoroModelPath: this.options.kokoroModelPath || './Kokoro-82M-v1.0-ONNX/',
                voice: this.options.voice || 'af_heart',
                device: this.options.device
            });

            // Initialize VAD with auto-detection for modern/legacy
            this.vad = this.resourceManager.createVAD({
                ...this.options,
                audioContext: this.resourceManager.audioContext
            });
            await this.vad.initialize();
            this.setupVADEventHandlers();
            this.emit('status', { type: 'init', message: 'Voice activity detection ready' });

            // Initialize audio queue with auto-detection for modern/legacy
            this.audioQueue = this.resourceManager.createAudioQueue({
                ...this.options,
                audioContext: this.resourceManager.audioContext
            });
            await this.audioQueue.initialize();
            this.setupAudioQueueEventHandlers();
            this.emit('status', { type: 'init', message: 'Audio queue ready' });

            // Start memory monitoring
            this.startMemoryMonitoring();

            this.isInitialized = true;
            this.emit('ready', { message: 'Voice chat interface ready' });

            return true;
        } catch (error) {
            this.emit('error', { type: 'init', error });
            throw error;
        }
    }

    /**
     * Start listening for voice input
     */
    async startListening() {
        if (!this.isInitialized) {
            throw new Error('Interface not initialized');
        }

        if (this.isListening) {
            return;
        }

        try {
            this.isListening = true;
            await this.vad.startListening();
            this.emit('listening', { status: 'started' });
        } catch (error) {
            this.isListening = false;
            this.emit('error', { type: 'listening', error });
            throw error;
        }
    }

    /**
     * Stop listening for voice input
     */
    async stopListening() {
        if (!this.isListening) {
            return;
        }

        try {
            this.isListening = false;
            await this.vad.stopListening();
            this.emit('listening', { status: 'stopped' });
        } catch (error) {
            this.emit('error', { type: 'listening', error });
            throw error;
        }
    }

    /**
     * Process text input (bypassing voice recognition)
     */
    async processTextInput(text) {
        if (this.isProcessing) {
            this.emit('warning', { message: 'Already processing input' });
            return;
        }

        try {
            this.isProcessing = true;
            this.emit('processing', { type: 'text', input: text });

            // Generate response using LLM
            const response = await this.generateResponse(text);
            
            // Convert to speech and queue audio
            await this.synthesizeSpeech(response);

            this.isProcessing = false;
        } catch (error) {
            this.isProcessing = false;
            this.emit('error', { type: 'processing', error });
        }
    }

    /**
     * Setup event handlers for modules
     */
    setupEventHandlers() {
        // ResourceManager events
        this.resourceManager.addEventListener('modelLoaded', (event) => {
            this.emit('modelLoaded', event.detail);
        });

        this.resourceManager.addEventListener('modelUnloaded', (event) => {
            this.emit('modelUnloaded', event.detail);
        });

        this.resourceManager.addEventListener('modelEvicted', (event) => {
            this.emit('modelEvicted', event.detail);
        });

        this.resourceManager.addEventListener('memoryPressure', (event) => {
            this.emit('memoryPressure', event.detail);
        });

        // VAD events (will be set up after VAD is initialized)
        // Audio queue events (will be set up after AudioQueue is initialized)
    }

    /**
     * Setup VAD event handlers (called after VAD initialization)
     */
    setupVADEventHandlers() {
        // VAD events
        this.vad.addEventListener('speechStart', () => {
            this.emit('speechDetected', { type: 'start' });
            this.handleSpeechStart();
        });

        this.vad.addEventListener('speechEnd', (event) => {
            this.emit('speechDetected', { type: 'end', audioData: event.detail.audioData });
            this.handleSpeechEnd(event.detail.audioData);
        });
    }

    /**
     * Setup AudioQueue event handlers (called after AudioQueue initialization)
     */
    setupAudioQueueEventHandlers() {
        // Audio queue events
        this.audioQueue.addEventListener('playbackStart', () => {
            this.isSpeaking = true;
            this.emit('speaking', { status: 'started' });
        });

        this.audioQueue.addEventListener('playbackEnd', () => {
            this.isSpeaking = false;
            this.emit('speaking', { status: 'ended' });
        });
    }

    /**
     * Handle speech start detection
     */
    async handleSpeechStart() {
        // Stop any current playback
        if (this.isSpeaking) {
            await this.audioQueue.stop();
        }
    }

    /**
     * Handle speech end detection and process audio
     */
    async handleSpeechEnd(audioData) {
        if (this.isProcessing) {
            return;
        }

        try {
            this.isProcessing = true;
            this.emit('processing', { type: 'speech', audioData });

            // Transcribe speech using Whisper
            const transcription = await this.transcribeSpeech(audioData);
            
            if (!transcription.trim()) {
                this.isProcessing = false;
                return;
            }

            this.emit('transcription', { text: transcription });

            // Generate response using LLM
            const response = await this.generateResponse(transcription);
            
            // Convert to speech and queue audio
            await this.synthesizeSpeech(response);

            this.isProcessing = false;
        } catch (error) {
            this.isProcessing = false;
            this.emit('error', { type: 'processing', error });
        }
    }

    /**
     * Transcribe speech using Whisper module
     */
    async transcribeSpeech(audioData) {
        try {
            const whisperModel = await this.ensureModelLoaded('whisper');
            
            if (whisperModel) {
                const result = await whisperModel.transcribe(audioData);
                return result.text || '';
            } else {
                // Model failed to load, use fallback immediately
                throw new Error('Whisper model not available, using fallback');
            }
        } catch (error) {
            // Don't emit as error if it's an expected fallback scenario
            if (error.message.includes('Transformers.js not available') || 
                error.message.includes('Pipeline function not available') ||
                error.message.includes('Whisper model not available')) {
                this.emit('info', { type: 'transcription_fallback', message: 'Using Web Speech API for transcription' });
            } else {
                this.emit('warning', { type: 'transcription_warning', message: `Whisper fallback: ${error.message}` });
            }
            
            // Fallback to Web Speech API if available
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                try {
                    return await this.transcribeWithWebSpeechAPI(audioData);
                } catch (webSpeechError) {
                    // Only emit as error if it's not a common speech recognition issue
                    if (webSpeechError.message.includes('no-speech') || 
                        webSpeechError.message.includes('audio-capture') ||
                        webSpeechError.message.includes('not-allowed')) {
                        this.emit('info', { 
                            type: 'speech_recognition_info', 
                            message: `Web Speech API: ${webSpeechError.message}` 
                        });
                    } else {
                        this.emit('error', { type: 'web_speech_transcription', error: webSpeechError });
                    }
                }
            }
            
            return '';
        }
    }

    /**
     * Fallback transcription using Web Speech API
     */
    async transcribeWithWebSpeechAPI(audioData) {
        return new Promise((resolve, reject) => {
            try {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'en-US';
                
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    resolve(transcript);
                };
                
                recognition.onerror = (event) => {
                    // Handle "no-speech" as a normal case, not an error
                    if (event.error === 'no-speech') {
                        resolve('');
                    } else {
                        reject(new Error(`Speech recognition error: ${event.error}`));
                    }
                };
                
                recognition.onend = () => {
                    // If no result was returned, resolve with empty string
                    resolve('');
                };
                
                // Note: Web Speech API doesn't directly accept Float32Array
                // This is a simplified fallback - in practice, you'd need to convert the audio
                recognition.start();
                
                // Timeout after 10 seconds
                setTimeout(() => {
                    recognition.stop();
                    resolve('');
                }, 10000);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Generate response using LLM
     */
    async generateResponse(inputText) {
        try {
            const llamaModel = await this.ensureModelLoaded('llama');
            
            if (llamaModel) {
                // Add to conversation history
                this.currentConversation.push({ role: 'user', content: inputText });

                const response = await llamaModel.generateResponse(this.currentConversation);
                
                // Add response to conversation history
                this.currentConversation.push({ role: 'assistant', content: response });

                // Limit conversation history
                if (this.currentConversation.length > 20) {
                    this.currentConversation = this.currentConversation.slice(-18);
                }

                this.emit('response', { input: inputText, output: response });
                return response;
            } else {
                // Model failed to load, use fallback immediately
                throw new Error('LLM model not available, using fallback');
            }
        } catch (error) {
            // Don't emit as error if it's an expected fallback scenario
            if (error.message.includes('Transformers.js not available') || 
                error.message.includes('Pipeline function not available') ||
                error.message.includes('LLM model not available')) {
                this.emit('info', { type: 'generation_fallback', message: 'Using built-in response system' });
            } else {
                this.emit('warning', { type: 'generation_warning', message: `LLM fallback: ${error.message}` });
            }
            
            // Fallback to built-in response system
            const fallbackResponse = this.generateBuiltInResponse(inputText);
            
            // Add to conversation history
            this.currentConversation.push({ role: 'user', content: inputText });
            this.currentConversation.push({ role: 'assistant', content: fallbackResponse });
            
            this.emit('response', { input: inputText, output: fallbackResponse });
            return fallbackResponse;
        }
    }

    /**
     * Generate fallback response using built-in system
     */
    generateBuiltInResponse(inputText) {
        const responses = {
            greeting: [
                "Hello! It's great to chat with you! [happy]",
                "Hi there! How can I help you today? [friendly]",
                "Welcome! I'm excited to talk with you! [excited]"
            ],
            how_are_you: [
                "I'm doing wonderful, thank you for asking! [cheerful]",
                "I'm fantastic! Thanks for checking in! [joyful]",
                "I'm great! How are you doing? [warm]"
            ],
            goodbye: [
                "Goodbye! It was lovely chatting with you! [warm]",
                "Farewell! Hope to chat again soon! [friendly]",
                "See you later! Take care! [caring]"
            ],
            help: [
                "I'm here to help! What can I do for you? [supportive]",
                "Let me assist you with that! [helpful]",
                "I'd be happy to help! What do you need? [eager]"
            ],
            default: [
                "That's really interesting! Tell me more about that. [curious]",
                "I see! That's quite fascinating. [thoughtful]",
                "Thanks for sharing that with me! [appreciative]",
                "That's a great point! What do you think about it? [engaged]",
                "I'm listening! Please continue. [attentive]"
            ]
        };

        const message = inputText.toLowerCase();
        
        if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
            return responses.greeting[Math.floor(Math.random() * responses.greeting.length)];
        } else if (message.includes('how are you') || message.includes('how do you feel')) {
            return responses.how_are_you[Math.floor(Math.random() * responses.how_are_you.length)];
        } else if (message.includes('bye') || message.includes('goodbye') || message.includes('see you')) {
            return responses.goodbye[Math.floor(Math.random() * responses.goodbye.length)];
        } else if (message.includes('help') || message.includes('assist') || message.includes('support')) {
            return responses.help[Math.floor(Math.random() * responses.help.length)];
        } else {
            return responses.default[Math.floor(Math.random() * responses.default.length)];
        }
    }

    /**
     * Synthesize speech and queue audio chunks
     */
    async synthesizeSpeech(text) {
        try {
            const kokoroModel = await this.ensureModelLoaded('kokoro');

            // Chunk the text for better audio processing
            const chunks = this.chunkText(text);
            this.emit('synthesis', { text, chunks: chunks.length });

            if (kokoroModel) {
                // Generate audio for each chunk and queue
                for (let i = 0; i < chunks.length; i++) {
                    const chunk = chunks[i];
                    const audioData = await kokoroModel.synthesize(chunk);
                    
                    if (audioData && audioData.length > 0) {
                        await this.audioQueue.enqueue(audioData);
                        this.emit('audioChunk', { 
                            chunk: i + 1, 
                            total: chunks.length, 
                            text: chunk 
                        });
                    }
                }

                // Start playback if not already playing
                if (!this.isSpeaking) {
                    await this.audioQueue.play();
                }
            } else {
                // Fallback to Web Speech API
                await this.synthesizeWithWebSpeechAPI(text);
            }
        } catch (error) {
            // Don't emit as error if it's an expected fallback scenario
            if (error.message.includes('Transformers.js not available') || 
                error.message.includes('Pipeline function not available') ||
                error.message.includes('Kokoro model not available')) {
                this.emit('info', { type: 'synthesis_fallback', message: 'Using Web Speech API for synthesis' });
            } else {
                this.emit('warning', { type: 'synthesis_warning', message: `Kokoro fallback: ${error.message}` });
            }
            
            // Fallback to Web Speech API
            try {
                await this.synthesizeWithWebSpeechAPI(text);
            } catch (webSpeechError) {
                this.emit('error', { type: 'web_speech_synthesis', error: webSpeechError });
            }
        }
    }

    /**
     * Fallback speech synthesis using Web Speech API
     */
    async synthesizeWithWebSpeechAPI(text) {
        return new Promise((resolve, reject) => {
            try {
                if (!('speechSynthesis' in window)) {
                    throw new Error('Web Speech API not available');
                }

                // Clean emotion markers
                const cleanText = text.replace(/\[[^\]]+\]/g, '').trim();
                
                if (!cleanText) {
                    resolve();
                    return;
                }

                const utterance = new SpeechSynthesisUtterance(cleanText);
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                utterance.volume = 0.8;

                utterance.onend = () => {
                    this.emit('speaking', { status: 'ended' });
                    resolve();
                };

                utterance.onerror = (event) => {
                    reject(new Error(`Speech synthesis error: ${event.error}`));
                };

                utterance.onstart = () => {
                    this.emit('speaking', { status: 'started' });
                };

                speechSynthesis.speak(utterance);
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Chunk text for optimal audio synthesis
     */
    chunkText(text) {
        // Remove emotion markers
        const cleanText = text.replace(/\[[^\]]+\]/g, '').trim();
        
        // Split by sentences first
        const sentences = cleanText.split(/(?<=[.!?])\s+(?=[A-Z])/g).filter(s => s.trim());
        
        const chunks = [];
        let currentChunk = '';
        
        for (const sentence of sentences) {
            if (currentChunk.length + sentence.length <= 128) {
                currentChunk += (currentChunk ? ' ' : '') + sentence;
            } else {
                if (currentChunk) {
                    chunks.push(currentChunk);
                }
                currentChunk = sentence.length <= 128 ? sentence : sentence.substring(0, 128);
            }
        }
        
        if (currentChunk) {
            chunks.push(currentChunk);
        }
        
        return chunks.filter(chunk => chunk.trim().length > 0);
    }

    /**
     * Ensure a specific model is loaded
     */
    async ensureModelLoaded(modelType) {
        try {
            // Use ResourceManager to get the model
            const model = await this.resourceManager.getModel(modelType);
            return model;
        } catch (loadError) {
            // Log the error and emit warning but don't throw - let fallback mechanisms handle it
            if (loadError.message.includes('Transformers.js not available') || 
                loadError.message.includes('Pipeline function not available')) {
                console.info(`${modelType} model using fallback:`, loadError.message);
                this.emit('info', { 
                    message: `${modelType} model using fallback due to missing dependencies. This is expected.` 
                });
            } else {
                console.warn(`Failed to load ${modelType} model:`, loadError.message);
                this.emit('warning', { 
                    type: 'model_load_failure',
                    message: `Failed to load ${modelType} model: ${loadError.message}. Will use fallback.` 
                });
            }
            
            return null;
        }
    }

    /**
     * Check memory usage and evict models if necessary
     */
    /**
     * Start memory monitoring (now handled by ResourceManager)
     */
    startMemoryMonitoring() {
        // ResourceManager handles memory monitoring
        console.log('Memory monitoring delegated to ResourceManager');
    }

    /**
     * Stop all processing and clean up
     */
    async stop() {
        await this.stopListening();
        
        if (this.audioQueue) {
            await this.audioQueue.stop();
        }
        
        if (this.resourceManager) {
            await this.resourceManager.cleanup();
        }
        
        if (this.memoryMonitor) {
            clearInterval(this.memoryMonitor);
            this.memoryMonitor = null;
        }
        
        this.isInitialized = false;
        this.emit('stopped');
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }

    /**
     * Get current status
     */
    getStatus() {
        const resourceStatus = this.resourceManager ? this.resourceManager.getStatus() : {};
        
        return {
            initialized: this.isInitialized,
            listening: this.isListening,
            processing: this.isProcessing,
            speaking: this.isSpeaking,
            modelsLoaded: resourceStatus.loadedModels || [],
            conversationLength: this.currentConversation.length,
            queueLength: this.audioQueue ? this.audioQueue.getQueueLength() : 0,
            memoryUsage: resourceStatus.memoryUsage || 0,
            audioContext: resourceStatus.audioContext || false,
            mlContext: resourceStatus.mlContext || false
        };
    }
}
