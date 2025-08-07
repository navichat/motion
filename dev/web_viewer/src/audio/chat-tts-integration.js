/**
 * Chat Integration Example for Kokoro TTS
 * 
 * This demonstrates how to integrate the modular Kokoro TTS system
 * with the existing 3D avatar chat features.
 */

import { KokoroTTS, createKokoroTTS, VOICES, KokoroUtils } from './kokoro-tts.js';

class ChatTTSManager {
    constructor() {
        this.tts = null;
        this.currentVoice = 'af_heart';
        this.speaking = false;
        this.queue = [];
        this.onSpeakingStart = null;
        this.onSpeakingEnd = null;
        this.onAudioGenerated = null;
    }
    
    /**
     * Initialize the TTS system
     */
    async initialize() {
        try {
            // Get recommended configuration
            const config = await KokoroUtils.getRecommendedConfig();
            console.log('Recommended TTS config:', config);
            
            // Create TTS instance
            this.tts = await createKokoroTTS(config);
            
            console.log('✅ Chat TTS Manager initialized');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize Chat TTS Manager:', error);
            return false;
        }
    }
    
    /**
     * Get available voices with metadata
     */
    getVoices() {
        return this.tts ? this.tts.getVoices() : [];
    }
    
    /**
     * Set current voice
     */
    setVoice(voiceId) {
        if (this.tts && VOICES[voiceId]) {
            this.currentVoice = voiceId;
            console.log(`Voice changed to: ${voiceId} (${VOICES[voiceId].name})`);
        } else {
            console.warn(`Voice not found: ${voiceId}`);
        }
    }
    
    /**
     * Speak text with avatar synchronization
     */
    async speak(text, options = {}) {
        if (!this.tts) {
            console.error('TTS not initialized');
            return;
        }
        
        const speakOptions = {
            voice: options.voice || this.currentVoice,
            speed: options.speed || 1.0,
            interrupt: options.interrupt || false,
            ...options
        };
        
        // Handle interruption
        if (speakOptions.interrupt && this.speaking) {
            await this.stop();
        }
        
        // Queue the text if currently speaking
        if (this.speaking && !speakOptions.interrupt) {
            this.queue.push({ text, options: speakOptions });
            return;
        }
        
        try {
            this.speaking = true;
            
            // Notify speaking start
            if (this.onSpeakingStart) {
                this.onSpeakingStart(text, speakOptions);
            }
            
            // Generate audio
            const result = await this.tts.synthesize(text, speakOptions);
            
            // Notify audio generated (for lip sync, etc.)
            if (this.onAudioGenerated) {
                this.onAudioGenerated(result);
            }
            
            // Play audio
            await this.tts.playAudio(result.audioBuffer);
            
            console.log(`✅ Spoke: "${text}" (${result.duration.toFixed(2)}s)`);
            
            // Notify speaking end
            if (this.onSpeakingEnd) {
                this.onSpeakingEnd(text, speakOptions);
            }
            
            this.speaking = false;
            
            // Process queue
            if (this.queue.length > 0) {
                const next = this.queue.shift();
                await this.speak(next.text, next.options);
            }
            
        } catch (error) {
            console.error('Speech synthesis failed:', error);
            this.speaking = false;
            
            // Continue with queue even if current fails
            if (this.queue.length > 0) {
                const next = this.queue.shift();
                await this.speak(next.text, next.options);
            }
        }
    }
    
    /**
     * Stop current speech
     */
    async stop() {
        if (this.speaking) {
            // Clear queue
            this.queue = [];
            
            // Stop audio context (simplified)
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.suspend();
                await audioContext.resume();
            } catch (error) {
                console.warn('Failed to stop audio context:', error);
            }
            
            this.speaking = false;
            
            if (this.onSpeakingEnd) {
                this.onSpeakingEnd('', { interrupted: true });
            }
        }
    }
    
    /**
     * Clear speech queue
     */
    clearQueue() {
        this.queue = [];
    }
    
    /**
     * Get current speaking state
     */
    isSpeaking() {
        return this.speaking;
    }
    
    /**
     * Get queue length
     */
    getQueueLength() {
        return this.queue.length;
    }
    
    /**
     * Process chat message with TTS
     */
    async processChatMessage(message, options = {}) {
        // Extract text from message (handle different formats)
        let text = '';
        if (typeof message === 'string') {
            text = message;
        } else if (message.content) {
            text = message.content;
        } else if (message.text) {
            text = message.text;
        }
        
        if (!text.trim()) return;
        
        // Clean up text for TTS
        text = this.cleanTextForTTS(text);
        
        // Determine voice from message context
        const voice = options.voice || this.getVoiceForMessage(message) || this.currentVoice;
        
        // Speak the message
        await this.speak(text, { ...options, voice });
    }
    
    /**
     * Clean text for better TTS synthesis
     */
    cleanTextForTTS(text) {
        return text
            // Remove URLs
            .replace(/https?:\/\/[^\s]+/g, '')
            // Remove markdown formatting
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/`(.*?)`/g, '$1')
            // Remove excessive punctuation
            .replace(/[.]{2,}/g, '.')
            .replace(/[!]{2,}/g, '!')
            .replace(/[?]{2,}/g, '?')
            // Clean up whitespace
            .replace(/\s+/g, ' ')
            .trim();
    }
    
    /**
     * Get voice for message based on context
     */
    getVoiceForMessage(message) {
        // This could be enhanced to select voice based on:
        // - Message sender
        // - Message type
        // - Character/avatar assigned
        // - User preferences
        
        if (message.sender === 'assistant') {
            return 'af_heart'; // Default assistant voice
        } else if (message.sender === 'user') {
            return null; // Don't read user messages by default
        }
        
        return this.currentVoice;
    }
    
    /**
     * Generate audio for lip sync without playing
     */
    async generateAudioForLipSync(text, options = {}) {
        if (!this.tts) {
            console.error('TTS not initialized');
            return null;
        }
        
        try {
            const result = await this.tts.synthesize(text, {
                voice: options.voice || this.currentVoice,
                speed: options.speed || 1.0
            });
            
            return {
                audioBuffer: result.audioBuffer,
                duration: result.duration,
                phonemes: result.metadata.phonemes,
                wavBlob: this.tts.audioBufferToWav(result.audioBuffer)
            };
            
        } catch (error) {
            console.error('Audio generation for lip sync failed:', error);
            return null;
        }
    }
}

/**
 * Example usage with existing chat system
 */
class ChatIntegrationExample {
    constructor() {
        this.ttsManager = new ChatTTSManager();
        this.avatarController = null; // Would be your 3D avatar controller
        this.initialized = false;
    }
    
    async initialize() {
        // Initialize TTS
        const success = await this.ttsManager.initialize();
        if (!success) {
            console.error('Failed to initialize TTS');
            return false;
        }
        
        // Set up event handlers
        this.setupTTSEventHandlers();
        
        this.initialized = true;
        console.log('✅ Chat Integration initialized');
        return true;
    }
    
    setupTTSEventHandlers() {
        // Speaking start - trigger avatar mouth animation
        this.ttsManager.onSpeakingStart = (text, options) => {
            console.log('🎤 Speaking started:', text);
            
            // Start avatar mouth animation
            if (this.avatarController) {
                this.avatarController.startSpeaking();
            }
        };
        
        // Speaking end - stop avatar mouth animation
        this.ttsManager.onSpeakingEnd = (text, options) => {
            console.log('🤐 Speaking ended');
            
            // Stop avatar mouth animation
            if (this.avatarController) {
                this.avatarController.stopSpeaking();
            }
        };
        
        // Audio generated - could be used for lip sync
        this.ttsManager.onAudioGenerated = (result) => {
            console.log('🎵 Audio generated:', result.duration.toFixed(2) + 's');
            
            // Generate lip sync data
            if (this.avatarController) {
                this.avatarController.generateLipSync(result.audioBuffer, result.metadata.phonemes);
            }
        };
    }
    
    /**
     * Handle incoming chat message
     */
    async handleChatMessage(message) {
        if (!this.initialized) {
            console.warn('Chat integration not initialized');
            return;
        }
        
        // Process message with TTS
        await this.ttsManager.processChatMessage(message, {
            interrupt: message.priority === 'high'
        });
    }
    
    /**
     * Change avatar voice
     */
    changeVoice(voiceId) {
        this.ttsManager.setVoice(voiceId);
    }
    
    /**
     * Get voice options for UI
     */
    getVoiceOptions() {
        return this.ttsManager.getVoices().map(voice => ({
            id: voice.id,
            name: voice.name,
            language: voice.language,
            gender: voice.gender,
            quality: voice.overallGrade
        }));
    }
}

// Export for use in other modules
export { ChatTTSManager, ChatIntegrationExample };

// Example usage
export async function initializeChatTTS() {
    const chatIntegration = new ChatIntegrationExample();
    await chatIntegration.initialize();
    return chatIntegration;
}

// Standalone TTS for quick testing
export async function quickTTS(text, voice = 'af_heart') {
    const tts = await createKokoroTTS();
    const result = await tts.synthesize(text, { voice });
    await tts.playAudio(result.audioBuffer);
    return result;
}
