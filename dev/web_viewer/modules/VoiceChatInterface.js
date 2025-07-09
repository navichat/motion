/**
 * VoiceChatInterface - Standalone voice chat interface with memory management
 * 
 * Features:
 * - Voice Activity Detection (VAD)
 * - Model eviction based on memory constraints
 * - Audio chunking and queueing
 * - Event-driven architecture
 */

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
            ...options
        };

        // Core modules
        this.whisperModule = new WhisperModule(this.options);
        this.kokoroModule = new KokoroModule(this.options);
        this.llamaModule = new LlamaModule(this.options);
        this.vad = new VoiceActivityDetector(this.options);
        this.audioQueue = new AudioQueue(this.options);

        // State management
        this.isInitialized = false;
        this.isListening = false;
        this.isProcessing = false;
        this.isSpeaking = false;
        this.currentConversation = [];

        // Memory management
        this.lastUsed = {
            whisper: 0,
            kokoro: 0,
            llama: 0
        };

        // Event handlers
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

            // Initialize VAD first (lightweight)
            await this.vad.initialize();
            this.emit('status', { type: 'init', message: 'Voice activity detection ready' });

            // Initialize audio queue
            await this.audioQueue.initialize();
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
        // VAD events
        this.vad.addEventListener('speechStart', () => {
            this.emit('speechDetected', { type: 'start' });
            this.handleSpeechStart();
        });

        this.vad.addEventListener('speechEnd', (event) => {
            this.emit('speechDetected', { type: 'end', audioData: event.detail.audioData });
            this.handleSpeechEnd(event.detail.audioData);
        });

        // Audio queue events
        this.audioQueue.addEventListener('playbackStart', () => {
            this.isSpeaking = true;
            this.emit('speaking', { status: 'started' });
        });

        this.audioQueue.addEventListener('playbackEnd', () => {
            this.isSpeaking = false;
            this.emit('speaking', { status: 'ended' });
        });

        // Module loading events
        [this.whisperModule, this.kokoroModule, this.llamaModule].forEach(module => {
            module.addEventListener('loaded', (event) => {
                this.lastUsed[event.detail.module] = Date.now();
                this.emit('modelLoaded', event.detail);
            });

            module.addEventListener('unloaded', (event) => {
                this.emit('modelUnloaded', event.detail);
            });
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
        await this.ensureModelLoaded('whisper');
        this.lastUsed.whisper = Date.now();
        
        try {
            const result = await this.whisperModule.transcribe(audioData);
            return result.text || '';
        } catch (error) {
            this.emit('error', { type: 'transcription', error });
            return '';
        }
    }

    /**
     * Generate response using LLM
     */
    async generateResponse(inputText) {
        await this.ensureModelLoaded('llama');
        this.lastUsed.llama = Date.now();

        // Add to conversation history
        this.currentConversation.push({ role: 'user', content: inputText });

        try {
            const response = await this.llamaModule.generateResponse(this.currentConversation);
            
            // Add response to conversation history
            this.currentConversation.push({ role: 'assistant', content: response });

            // Limit conversation history
            if (this.currentConversation.length > 20) {
                this.currentConversation = this.currentConversation.slice(-18);
            }

            this.emit('response', { input: inputText, output: response });
            return response;
        } catch (error) {
            this.emit('error', { type: 'generation', error });
            // Fallback response
            return "I'm having trouble processing that right now. Could you try again?";
        }
    }

    /**
     * Synthesize speech and queue audio chunks
     */
    async synthesizeSpeech(text) {
        await this.ensureModelLoaded('kokoro');
        this.lastUsed.kokoro = Date.now();

        try {
            // Chunk the text for better audio processing
            const chunks = this.chunkText(text);
            this.emit('synthesis', { text, chunks: chunks.length });

            // Generate audio for each chunk and queue
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i];
                const audioData = await this.kokoroModule.synthesize(chunk);
                
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
        } catch (error) {
            this.emit('error', { type: 'synthesis', error });
        }
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
        const modules = {
            whisper: this.whisperModule,
            kokoro: this.kokoroModule,
            llama: this.llamaModule
        };

        const module = modules[modelType];
        if (!module) {
            throw new Error(`Unknown model type: ${modelType}`);
        }

        if (!module.isLoaded()) {
            // Check memory before loading
            await this.checkMemoryAndEvict(modelType);
            await module.load();
        }
    }

    /**
     * Check memory usage and evict models if necessary
     */
    async checkMemoryAndEvict(modelToLoad) {
        if (!('memory' in performance)) {
            return; // Memory API not available
        }

        const memInfo = performance.memory;
        const usedMB = memInfo.usedJSHeapSize / 1024 / 1024;
        
        if (usedMB > this.options.memoryThresholdMB) {
            this.emit('memoryPressure', { usedMB, threshold: this.options.memoryThresholdMB });
            
            // Find least recently used models to evict
            const modelPriority = ['kokoro', 'llama', 'whisper'];
            const loadedModules = [];
            
            if (this.whisperModule.isLoaded()) loadedModules.push({ type: 'whisper', lastUsed: this.lastUsed.whisper });
            if (this.kokoroModule.isLoaded()) loadedModules.push({ type: 'kokoro', lastUsed: this.lastUsed.kokoro });
            if (this.llamaModule.isLoaded()) loadedModules.push({ type: 'llama', lastUsed: this.lastUsed.llama });
            
            // Sort by last used (oldest first)
            loadedModules.sort((a, b) => a.lastUsed - b.lastUsed);
            
            // Evict models until memory is acceptable or we've evicted one model
            for (const module of loadedModules) {
                if (module.type === modelToLoad) continue; // Don't evict the model we're about to load
                
                await this.evictModel(module.type);
                break; // Evict one model at a time
            }
        }
    }

    /**
     * Evict a specific model from memory
     */
    async evictModel(modelType) {
        const modules = {
            whisper: this.whisperModule,
            kokoro: this.kokoroModule,
            llama: this.llamaModule
        };

        const module = modules[modelType];
        if (module && module.isLoaded()) {
            await module.unload();
            this.emit('modelEvicted', { type: modelType });
        }
    }

    /**
     * Start memory monitoring
     */
    startMemoryMonitoring() {
        if (this.memoryMonitor) {
            clearInterval(this.memoryMonitor);
        }

        this.memoryMonitor = setInterval(() => {
            this.checkIdleModels();
        }, 10000); // Check every 10 seconds
    }

    /**
     * Check for idle models and unload them
     */
    async checkIdleModels() {
        const now = Date.now();
        const timeout = this.options.modelCacheTimeout;

        if (this.whisperModule.isLoaded() && (now - this.lastUsed.whisper) > timeout) {
            await this.evictModel('whisper');
        }

        if (this.kokoroModule.isLoaded() && (now - this.lastUsed.kokoro) > timeout) {
            await this.evictModel('kokoro');
        }

        if (this.llamaModule.isLoaded() && (now - this.lastUsed.llama) > timeout) {
            await this.evictModel('llama');
        }
    }

    /**
     * Stop all processing and clean up
     */
    async stop() {
        await this.stopListening();
        await this.audioQueue.stop();
        
        // Clear memory monitoring
        if (this.memoryMonitor) {
            clearInterval(this.memoryMonitor);
            this.memoryMonitor = null;
        }

        // Unload all models
        await this.evictModel('whisper');
        await this.evictModel('kokoro');
        await this.evictModel('llama');

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
        return {
            initialized: this.isInitialized,
            listening: this.isListening,
            processing: this.isProcessing,
            speaking: this.isSpeaking,
            modelsLoaded: {
                whisper: this.whisperModule.isLoaded(),
                kokoro: this.kokoroModule.isLoaded(),
                llama: this.llamaModule.isLoaded()
            },
            conversationLength: this.currentConversation.length,
            queueLength: this.audioQueue.getQueueLength()
        };
    }
}
