/**
 * Example usage of the VoiceChatInterface module
 * This replaces the original monolithic voice chat functionality
 */

import { VoiceChatInterface } from './modules/VoiceChatInterface.js';

// Configuration options
const voiceChatOptions = {
    memoryThresholdMB: 512,           // Evict models when memory usage exceeds this
    vadSensitivity: 0.6,              // Voice activity detection sensitivity (0-1)
    chunkSizeMs: 1000,                // Audio chunk size for TTS
    audioSampleRate: 22050,           // Audio sample rate
    modelCacheTimeout: 30000,         // Unload unused models after 30 seconds
    
    // Model-specific options
    whisperModel: 'tiny',             // 'tiny', 'base', 'small'
    llamaModel: 'Xenova/TinyLlama-1.1B-Chat-v0.4',
    kokoroModelPath: './Kokoro-82M-v1.0-ONNX/',
    voice: 'af_heart',
    
    // Generation parameters
    maxTokens: 150,
    temperature: 0.8,
    topP: 0.9,
    
    // VAD parameters
    minSpeechDuration: 300,           // Minimum speech duration to process (ms)
    maxSpeechDuration: 8000,          // Maximum speech duration
    silenceThreshold: 150,            // Silence duration to end speech (ms)
    
    systemPrompt: "You are a helpful AI assistant. Keep responses concise and engaging. Include emotion markers like [happy], [thoughtful], [excited] in your responses."
};

// Global instance
let voiceChat = null;

/**
 * Initialize the voice chat system
 */
async function initializeVoiceChat() {
    try {
        updateDebugInfo('🚀 Initializing modular voice chat interface...');
        
        // Make transformers.js globally available if possible
        try {
            if (typeof pipeline !== 'undefined' && typeof env !== 'undefined') {
                window.transformers = { pipeline, env };
                updateDebugInfo('✅ Transformers.js made globally available');
            }
        } catch (error) {
            updateDebugInfo('⚠️ Transformers.js not available - models will use fallbacks');
        }
        
        // Make Kokoro TTS globally available if possible
        try {
            if (window.kokoroTTS) {
                window.KokoroTTS = { from_pretrained: () => window.kokoroTTS };
                updateDebugInfo('✅ Kokoro TTS made globally available');
            }
        } catch (error) {
            updateDebugInfo('⚠️ Kokoro TTS not available - will use Web Speech API');
        }
        
        // Create voice chat instance
        voiceChat = new VoiceChatInterface(voiceChatOptions);
        
        // Set up event listeners
        setupVoiceChatEvents();
        
        // Initialize the system
        await voiceChat.initialize();
        
        updateDebugInfo('✅ Voice chat interface ready!');
        return true;
        
    } catch (error) {
        console.error('Voice chat initialization error:', error);
        updateDebugInfo(`❌ Voice chat initialization failed: ${error.message}`);
        return false;
    }
}

/**
 * Set up event listeners for the voice chat interface
 */
function setupVoiceChatEvents() {
    // Status events
    voiceChat.addEventListener('status', (event) => {
        updateDebugInfo(`📊 ${event.detail.message}`);
    });
    
    voiceChat.addEventListener('ready', (event) => {
        updateDebugInfo('🎉 Voice chat ready for interaction');
        updateSystemStatus('Voice chat ready');
    });
    
    // Model loading events
    voiceChat.addEventListener('modelLoaded', (event) => {
        const { module, model } = event.detail;
        updateDebugInfo(`📥 ${module} model loaded: ${model}`);
        updateModelLoadingStatus(module, 'loaded');
    });
    
    voiceChat.addEventListener('modelUnloaded', (event) => {
        const { module } = event.detail;
        updateDebugInfo(`📤 ${module} model unloaded (memory management)`);
        updateModelLoadingStatus(module, 'unloaded');
    });
    
    voiceChat.addEventListener('modelEvicted', (event) => {
        const { type } = event.detail;
        updateDebugInfo(`🗑️ ${type} model evicted due to memory pressure`);
    });
    
    // Voice activity detection events
    voiceChat.addEventListener('listening', (event) => {
        const { status } = event.detail;
        updateDebugInfo(`👂 Voice listening: ${status}`);
        updateListeningIndicator(status === 'started');
    });
    
    voiceChat.addEventListener('speechDetected', (event) => {
        const { type } = event.detail;
        if (type === 'start') {
            updateDebugInfo('🎤 Speech detected - listening...');
            showSpeechIndicator(true);
        } else {
            updateDebugInfo('🔇 Speech ended - processing...');
            showSpeechIndicator(false);
        }
    });
    
    // Processing events
    voiceChat.addEventListener('processing', (event) => {
        const { type } = event.detail;
        updateDebugInfo(`⚙️ Processing ${type} input...`);
        showProcessingIndicator(true);
    });
    
    voiceChat.addEventListener('transcription', (event) => {
        const { text } = event.detail;
        updateDebugInfo(`📝 Transcribed: "${text}"`);
        addToConversation('user', text);
    });
    
    voiceChat.addEventListener('response', (event) => {
        const { input, output } = event.detail;
        updateDebugInfo(`💬 Generated response for: "${input.substring(0, 30)}..."`);
        addToConversation('assistant', output);
        showProcessingIndicator(false);
    });
    
    // Audio synthesis events
    voiceChat.addEventListener('synthesis', (event) => {
        const { text, chunks } = event.detail;
        updateDebugInfo(`🎵 Synthesizing speech: ${chunks} chunks`);
    });
    
    voiceChat.addEventListener('audioChunk', (event) => {
        const { chunk, total } = event.detail;
        updateDebugInfo(`🎶 Audio chunk ${chunk}/${total} ready`);
    });
    
    voiceChat.addEventListener('speaking', (event) => {
        const { status } = event.detail;
        updateDebugInfo(`🗣️ Speaking: ${status}`);
        updateSpeakingIndicator(status === 'started');
    });
    
    // Error events
    voiceChat.addEventListener('error', (event) => {
        const { type, error } = event.detail;
        
        // Don't log as error if it's an expected fallback scenario
        if (error.message && (error.message.includes('Transformers.js not available') || 
                              error.message.includes('Pipeline function not available'))) {
            console.info(`Voice chat using fallback for ${type}:`, error.message);
            updateDebugInfo(`ℹ️ Using fallback for ${type}: ${error.message}`);
        } else {
            console.error(`Voice chat error (${type}):`, error);
            updateDebugInfo(`❌ Error in ${type}: ${error.message}`);
        }
    });
    
    // Info events for fallback scenarios
    voiceChat.addEventListener('info', (event) => {
        const { type, message } = event.detail;
        console.info(`Voice chat info (${type}):`, message);
        updateDebugInfo(`ℹ️ ${message}`);
    });
    
    // Warning events
    voiceChat.addEventListener('warning', (event) => {
        const { type, message } = event.detail;
        console.warn(`Voice chat warning (${type}):`, message);
        updateDebugInfo(`⚠️ ${message}`);
    });
    
    // Memory management events
    voiceChat.addEventListener('memoryPressure', (event) => {
        const { usedMB, threshold } = event.detail;
        updateDebugInfo(`⚠️ Memory pressure: ${usedMB.toFixed(1)}MB (threshold: ${threshold}MB)`);
    });
}

/**
 * Start voice chat listening
 */
async function startVoiceChat() {
    if (!voiceChat) {
        updateDebugInfo('❌ Voice chat not initialized');
        return;
    }
    
    try {
        await voiceChat.startListening();
        updateDebugInfo('👂 Started listening for voice input');
        
        // Update UI
        const startBtn = document.getElementById('start-voice-btn');
        const stopBtn = document.getElementById('stop-voice-btn');
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        
    } catch (error) {
        updateDebugInfo(`❌ Failed to start listening: ${error.message}`);
    }
}

/**
 * Stop voice chat listening
 */
async function stopVoiceChat() {
    if (!voiceChat) {
        return;
    }
    
    try {
        await voiceChat.stopListening();
        updateDebugInfo('🔇 Stopped listening for voice input');
        
        // Update UI
        const startBtn = document.getElementById('start-voice-btn');
        const stopBtn = document.getElementById('stop-voice-btn');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
    } catch (error) {
        updateDebugInfo(`❌ Failed to stop listening: ${error.message}`);
    }
}

/**
 * Send text message (bypassing voice input)
 */
async function sendTextMessage() {
    const messageInput = document.getElementById('message-input');
    if (!messageInput || !voiceChat) {
        return;
    }
    
    const text = messageInput.value.trim();
    if (!text) {
        return;
    }
    
    try {
        messageInput.value = '';
        await voiceChat.processTextInput(text);
    } catch (error) {
        updateDebugInfo(`❌ Failed to process text: ${error.message}`);
    }
}

/**
 * Test TTS functionality
 */
async function testTTS(text = 'Hello! I am your AI avatar assistant.') {
    if (!voiceChat) {
        updateDebugInfo('❌ Voice chat not initialized');
        return;
    }
    
    try {
        updateDebugInfo('🔬 Testing TTS functionality...');
        await voiceChat.processTextInput(text);
        updateDebugInfo('✅ TTS test completed successfully');
    } catch (error) {
        console.error('TTS test error:', error);
        updateDebugInfo(`❌ TTS test failed: ${error.message}`);
    }
}

/**
 * Get voice chat status
 */
function getVoiceChatStatus() {
    if (!voiceChat) {
        return { initialized: false };
    }
    
    return voiceChat.getStatus();
}

/**
 * Update UI indicators
 */
function updateListeningIndicator(listening) {
    const indicator = document.getElementById('listening-indicator');
    if (indicator) {
        indicator.textContent = listening ? '👂 Listening' : '🔇 Not listening';
        indicator.className = listening ? 'status-active' : 'status-inactive';
    }
}

function showSpeechIndicator(active) {
    const indicator = document.getElementById('speech-indicator');
    if (indicator) {
        indicator.textContent = active ? '🎤 Speaking detected' : '';
        indicator.style.display = active ? 'block' : 'none';
    }
}

function showProcessingIndicator(active) {
    const indicator = document.getElementById('processing-indicator');
    if (indicator) {
        indicator.textContent = active ? '⚙️ Processing...' : '';
        indicator.style.display = active ? 'block' : 'none';
    }
}

function updateSpeakingIndicator(speaking) {
    const indicator = document.getElementById('speaking-indicator');
    if (indicator) {
        indicator.textContent = speaking ? '🗣️ Speaking' : '';
        indicator.className = speaking ? 'status-active' : 'status-inactive';
    }
}

function updateModelLoadingStatus(module, status) {
    const statusEl = document.getElementById(`${module}-model-status`);
    if (statusEl) {
        statusEl.textContent = status;
        statusEl.className = status === 'loaded' ? 'status-loaded' : 'status-unloaded';
    }
}

/**
 * Cleanup voice chat on page unload
 */
window.addEventListener('beforeunload', () => {
    if (voiceChat) {
        voiceChat.stop();
    }
});

// Export functions for global access
window.initializeVoiceChat = initializeVoiceChat;
window.startVoiceChat = startVoiceChat;
window.stopVoiceChat = stopVoiceChat;
window.sendTextMessage = sendTextMessage;
window.testTTS = testTTS;
window.getVoiceChatStatus = getVoiceChatStatus;

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeVoiceChat);
} else {
    // DOM already loaded
    setTimeout(initializeVoiceChat, 100);
}
