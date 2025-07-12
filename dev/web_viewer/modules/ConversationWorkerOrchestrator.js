/**
 * ConversationWorkerOrchestrator.js
 * Main orchestrator for VRM conversation system using Web Workers and Transformers.js
 * Based on the conversational-webgpu example
 */

import { ConversationNeuralNetwork } from './ConversationNeuralNetwork.js';
import { CONVERSATION_CONSTANTS } from './ConversationConstants.js';

/**
 * Main orchestrator class for handling VRM conversations
 */
export class ConversationWorkerOrchestrator extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            modelName: 'microsoft/DialoGPT-medium',
            maxTokens: 100,
            temperature: 0.7,
            voiceEnabled: true,
            ...options
        };
        
        this.isInitialized = false;
        this.isConversing = false;
        this.isListening = false;
        this.isSpeaking = false;
        this.isProcessing = false;
        
        // Web Workers
        this.conversationWorker = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.mediaRecorder = null;
        
        // Audio processing
        this.audioWorkletNode = null;
        this.vadProcessor = null;
        this.playWorklet = null;
        
        // Neural network
        this.neuralNetwork = null;
        
        // Voice synthesis
        this.speechSynthesis = window.speechSynthesis;
        this.currentVoice = null;
        this.voices = [];
        
        // Conversation state
        this.conversationHistory = [];
        this.currentPersonality = 'conversational';
        
        console.log('🎤 ConversationWorkerOrchestrator initialized');
    }

    /**
     * Initialize the conversation system
     */
    async initialize() {
        try {
            console.log('🚀 Initializing conversation system...');
            
            // Initialize neural network
            try {
                this.neuralNetwork = new ConversationNeuralNetwork();
                await this.neuralNetwork.initialize();
            } catch (error) {
                console.warn('⚠️ Neural network initialization failed, using fallback:', error);
            }
            
            // Initialize audio context
            await this.initializeAudioContext();
            
            // Initialize conversation worker
            await this.initializeConversationWorker();
            
            // Initialize voice synthesis
            await this.initializeVoiceSynthesis();
            
            this.isInitialized = true;
            
            this.dispatchEvent(new CustomEvent('ready', {
                detail: { status: 'initialized' }
            }));
            
            console.log('✅ Conversation system ready');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize conversation system:', error);
            // Don't throw error, allow fallback mode
            this.isInitialized = true; // Allow basic functionality
            return false;
        }
    }

    /**
     * Initialize audio context and worklets
     */
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000,
                channelCount: 1
            });
            
            // Try to load audio worklets
            try {
                await this.audioContext.audioWorklet.addModule('./worklets/vad-processor.js');
                await this.audioContext.audioWorklet.addModule('./worklets/play-worklet.js');
                console.log('🎵 Audio worklets loaded');
            } catch (workletError) {
                console.warn('⚠️ Audio worklets not available, using fallback:', workletError);
            }
            
            console.log('🎵 Audio context initialized');
            
        } catch (error) {
            console.warn('⚠️ Audio context initialization failed:', error);
            // Continue without audio context for basic functionality
        }
    }

    /**
     * Initialize conversation worker
     */
    async initializeConversationWorker() {
        try {
            this.conversationWorker = new Worker('./workers/ConversationWorker.js', { type: 'module' });
            
            this.conversationWorker.onmessage = (event) => {
                this.handleWorkerMessage(event.data);
            };
            
            this.conversationWorker.onerror = (error) => {
                console.error('❌ Conversation worker error:', error);
                this.dispatchEvent(new CustomEvent('error', {
                    detail: { error: error.message }
                }));
            };
            
            // Initialize worker
            this.conversationWorker.postMessage({
                type: 'initialize',
                options: this.options
            });
            
            console.log('👷 Conversation worker initialized');
            
        } catch (error) {
            console.warn('⚠️ Worker initialization failed, using fallback:', error);
            // Continue with local processing as fallback
        }
    }

    /**
     * Initialize voice synthesis
     */
    async initializeVoiceSynthesis() {
        return new Promise((resolve) => {
            const loadVoices = () => {
                this.voices = this.speechSynthesis.getVoices();
                
                // Set default voice
                this.currentVoice = this.voices.find(voice => 
                    voice.name.includes('Female') || voice.name.includes('Samantha')
                ) || this.voices[0];
                
                console.log(`🎵 Voice synthesis ready with ${this.voices.length} voices`);
                resolve();
            };
            
            if (this.voices.length > 0) {
                loadVoices();
            } else {
                this.speechSynthesis.onvoiceschanged = loadVoices;
                // Fallback timeout
                setTimeout(loadVoices, 1000);
            }
        });
    }

    /**
     * Handle messages from conversation worker
     */
    handleWorkerMessage(data) {
        const { type, payload } = data;
        
        switch (type) {
            case 'ready':
                console.log('🤖 Conversation worker ready');
                break;
                
            case 'response':
                this.handleConversationResponse(payload);
                break;
                
            case 'error':
                console.error('❌ Worker error:', payload.error);
                this.dispatchEvent(new CustomEvent('error', {
                    detail: { error: payload.error }
                }));
                break;
                
            case 'processing':
                this.isProcessing = payload.processing;
                this.dispatchEvent(new CustomEvent('processing', {
                    detail: { processing: this.isProcessing }
                }));
                break;
                
            default:
                console.warn('Unknown worker message type:', type);
        }
    }

    /**
     * Handle conversation response
     */
    handleConversationResponse(response) {
        const { text, emotion, confidence } = response;
        
        // Add to conversation history
        this.conversationHistory.push({
            type: 'assistant',
            text,
            emotion,
            confidence,
            timestamp: Date.now()
        });
        
        // Dispatch response event
        this.dispatchEvent(new CustomEvent('assistantMessage', {
            detail: { text, emotion, confidence }
        }));
        
        // Speak the response if voice is enabled
        if (this.options.voiceEnabled) {
            this.speakText(text);
        }
    }

    /**
     * Start conversation
     */
    async startConversation() {
        if (!this.isInitialized) {
            throw new Error('Conversation system not initialized');
        }
        
        try {
            // Request microphone permission
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            // Set up audio processing
            await this.setupAudioProcessing();
            
            this.isConversing = true;
            this.startListening();
            
            this.dispatchEvent(new CustomEvent('conversationStarted', {
                detail: { status: 'started' }
            }));
            
            console.log('🎤 Conversation started');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to start conversation:', error);
            throw error;
        }
    }

    /**
     * Stop conversation
     */
    async stopConversation() {
        this.isConversing = false;
        this.stopListening();
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioWorkletNode) {
            this.audioWorkletNode.disconnect();
            this.audioWorkletNode = null;
        }
        
        this.dispatchEvent(new CustomEvent('conversationStopped', {
            detail: { status: 'stopped' }
        }));
        
        console.log('🔇 Conversation stopped');
        return true;
    }

    /**
     * Send text message directly
     */
    async sendTextMessage(text) {
        if (!text.trim()) return null;
        
        await this.handleUserMessage(text);
        
        return new Promise((resolve) => {
            const handleResponse = (event) => {
                this.removeEventListener('assistantMessage', handleResponse);
                resolve(event.detail);
            };
            
            this.addEventListener('assistantMessage', handleResponse);
        });
    }

    /**
     * Handle user message (from speech or text)
     */
    async handleUserMessage(text) {
        if (!text.trim()) return;
        
        // Add to conversation history
        this.conversationHistory.push({
            type: 'user',
            text,
            timestamp: Date.now()
        });
        
        // Dispatch user message event
        this.dispatchEvent(new CustomEvent('userMessage', {
            detail: { text }
        }));
        
        // Generate response
        await this.generateResponse(text);
    }

    /**
     * Generate AI response
     */
    async generateResponse(userText) {
        try {
            if (this.conversationWorker) {
                // Send to worker
                this.conversationWorker.postMessage({
                    type: 'generateResponse',
                    text: userText,
                    history: this.conversationHistory,
                    personality: this.currentPersonality
                });
            } else {
                // Fallback: use local neural network or simple response
                const response = {
                    text: this.generateFallbackResponse(userText),
                    emotion: 'neutral',
                    confidence: 0.5
                };
                
                this.handleConversationResponse(response);
            }
            
        } catch (error) {
            console.error('❌ Response generation failed:', error);
            
            // Fallback response
            this.handleConversationResponse({
                text: "I'm sorry, I'm having trouble understanding right now. Could you try again?",
                emotion: 'neutral',
                confidence: 0.5
            });
        }
    }

    /**
     * Generate simple fallback response
     */
    generateFallbackResponse(userText) {
        const lowerText = userText.toLowerCase();
        
        // Simple response patterns
        if (lowerText.includes('hello') || lowerText.includes('hi')) {
            return "Hello! It's nice to meet you. How are you today?";
        } else if (lowerText.includes('how are you')) {
            return "I'm doing well, thank you for asking! How about you?";
        } else if (lowerText.includes('bye') || lowerText.includes('goodbye')) {
            return "Goodbye! It was nice talking with you. Have a great day!";
        } else if (lowerText.includes('thank')) {
            return "You're welcome! I'm happy to help.";
        } else if (lowerText.includes('help')) {
            return "I'd be happy to help! What can I do for you?";
        } else {
            const responses = [
                "That's interesting! Tell me more about that.",
                "I understand. How does that make you feel?",
                "That's a great point. What do you think about it?",
                "I see what you mean. Can you elaborate?",
                "That sounds important to you. Why is that?",
                "Interesting perspective! What led you to think that way?"
            ];
            
            return responses[Math.floor(Math.random() * responses.length)];
        }
    }

    /**
     * Set up audio processing pipeline
     */
    async setupAudioProcessing() {
        if (!this.audioContext || !this.mediaStream) {
            this.setupMediaRecorderFallback();
            return;
        }
        
        try {
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create VAD processor if worklet is available
            if (this.audioContext.audioWorklet) {
                try {
                    this.vadProcessor = new AudioWorkletNode(this.audioContext, 'vad-processor');
                    
                    this.vadProcessor.port.onmessage = (event) => {
                        const { type, data } = event.data;
                        
                        if (type === 'speech-start') {
                            this.onSpeechStart();
                        } else if (type === 'speech-end') {
                            this.onSpeechEnd(data);
                        }
                    };
                    
                    source.connect(this.vadProcessor);
                    console.log('🎤 VAD processor connected');
                } catch (vadError) {
                    console.warn('⚠️ VAD processor failed, using fallback:', vadError);
                    this.setupMediaRecorderFallback();
                }
            } else {
                // Fallback: use media recorder
                this.setupMediaRecorderFallback();
            }
            
        } catch (error) {
            console.warn('⚠️ Audio processing setup failed, using fallback:', error);
            this.setupMediaRecorderFallback();
        }
    }

    /**
     * Fallback audio recording using MediaRecorder
     */
    setupMediaRecorderFallback() {
        if (!this.mediaStream) return;
        
        this.mediaRecorder = new MediaRecorder(this.mediaStream, {
            mimeType: 'audio/webm'
        });
        
        let audioChunks = [];
        
        this.mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        this.mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            
            try {
                const audioBuffer = await audioBlob.arrayBuffer();
                const audioData = new Uint8Array(audioBuffer);
                await this.processAudioData(audioData);
            } catch (error) {
                console.error('❌ Audio processing error:', error);
            }
        };
        
        console.log('🎤 MediaRecorder fallback setup complete');
    }

    /**
     * Start listening for speech
     */
    startListening() {
        this.isListening = true;
        
        this.dispatchEvent(new CustomEvent('listening', {
            detail: { isListening: true }
        }));
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'inactive') {
            this.mediaRecorder.start();
            
            // Stop recording after 5 seconds for fallback mode
            setTimeout(() => {
                if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.stop();
                }
            }, 5000);
        }
    }

    /**
     * Stop listening for speech
     */
    stopListening() {
        this.isListening = false;
        
        this.dispatchEvent(new CustomEvent('listening', {
            detail: { isListening: false }
        }));
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }

    /**
     * Handle speech start
     */
    onSpeechStart() {
        console.log('🎤 Speech detected');
        this.dispatchEvent(new CustomEvent('speechStart'));
    }

    /**
     * Handle speech end
     */
    async onSpeechEnd(audioData) {
        console.log('🎤 Speech ended, processing...');
        await this.processAudioData(audioData);
    }

    /**
     * Process audio data for speech recognition
     */
    async processAudioData(audioData) {
        if (!this.isConversing) return;
        
        try {
            this.isProcessing = true;
            
            this.dispatchEvent(new CustomEvent('processing', {
                detail: { processing: true }
            }));
            
            // For fallback, we'll just simulate speech recognition
            const text = "I heard you speaking but speech recognition is not available in fallback mode. Please use text input instead.";
            
            // Simulate a delay
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            this.handleUserMessage(text);
            
        } catch (error) {
            console.error('❌ Audio processing failed:', error);
        } finally {
            this.isProcessing = false;
            
            this.dispatchEvent(new CustomEvent('processing', {
                detail: { processing: false }
            }));
        }
    }

    /**
     * Speak text using voice synthesis
     */
    async speakText(text) {
        if (!this.options.voiceEnabled || !text.trim()) return;
        
        try {
            this.isSpeaking = true;
            
            this.dispatchEvent(new CustomEvent('speaking', {
                detail: { isSpeaking: true }
            }));
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            if (this.currentVoice) {
                utterance.voice = this.currentVoice;
            }
            
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 0.8;
            
            utterance.onend = () => {
                this.isSpeaking = false;
                
                this.dispatchEvent(new CustomEvent('speaking', {
                    detail: { isSpeaking: false }
                }));
            };
            
            utterance.onerror = (error) => {
                console.error('❌ Speech synthesis error:', error);
                this.isSpeaking = false;
                
                this.dispatchEvent(new CustomEvent('speaking', {
                    detail: { isSpeaking: false }
                }));
            };
            
            this.speechSynthesis.speak(utterance);
            
        } catch (error) {
            console.error('❌ Text-to-speech failed:', error);
            this.isSpeaking = false;
        }
    }

    /**
     * Set voice for speech synthesis
     */
    setVoice(voiceName) {
        const voice = this.voices.find(v => 
            v.name.includes(voiceName) || 
            v.lang.includes(voiceName)
        );
        
        if (voice) {
            this.currentVoice = voice;
            console.log(`🎵 Voice changed to: ${voice.name}`);
        } else {
            console.warn(`⚠️ Voice not found: ${voiceName}`);
        }
    }

    /**
     * Set conversation personality
     */
    setPersonality(personality) {
        this.currentPersonality = personality;
        console.log(`🎭 Personality changed to: ${personality}`);
        
        this.dispatchEvent(new CustomEvent('personalityChanged', {
            detail: { personality }
        }));
    }

    /**
     * Reset conversation
     */
    resetConversation() {
        this.conversationHistory = [];
        
        this.dispatchEvent(new CustomEvent('conversationReset', {
            detail: { status: 'reset' }
        }));
        
        console.log('🔄 Conversation reset');
    }

    /**
     * Get conversation history
     */
    getConversationHistory() {
        return [...this.conversationHistory];
    }

    /**
     * Cleanup resources
     */
    dispose() {
        this.stopConversation();
        
        if (this.conversationWorker) {
            this.conversationWorker.terminate();
            this.conversationWorker = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        console.log('🧹 ConversationWorkerOrchestrator disposed');
    }
}
