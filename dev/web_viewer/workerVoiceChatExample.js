/**
 * Worker-based Voice Chat Example
 * Uses dedicated Web Worker for ML processing like conversational-webgpu
 */

import { WorkerVoiceChatInterface } from './modules/WorkerVoiceChatInterface.js';

// Configuration options
const voiceChatOptions = {
  device: 'wasm', // Use WASM for better compatibility
  vadSensitivity: 0.5,
  audioSampleRate: 16000,
  outputSampleRate: 24000,
  systemPrompt: "You're a helpful and conversational voice assistant. Keep your responses short, clear, and casual."
};

// Global instance
let voiceChat = null;
let isInitializing = false;

/**
 * Initialize the worker-based voice chat system
 */
async function initializeWorkerVoiceChat() {
  if (isInitializing) return;
  isInitializing = true;
  
  try {
    updateDebugInfo('🚀 Initializing worker-based voice chat interface...');
    updateDebugInfo('💡 Using Web Worker for ML processing (like conversational-webgpu)');
    
    // Create voice chat instance
    voiceChat = new WorkerVoiceChatInterface(voiceChatOptions);
    
    // Set up event listeners
    setupWorkerVoiceChatEvents();
    
    // Initialize the system
    await voiceChat.initialize();
    
    updateDebugInfo('✅ Worker-based voice chat interface ready!');
    updateSystemStatus('Voice chat ready');
    
  } catch (error) {
    console.error('Worker voice chat initialization error:', error);
    updateDebugInfo(`❌ Worker voice chat initialization failed: ${error.message}`);
    updateSystemStatus('Initialization failed');
  } finally {
    isInitializing = false;
  }
}

/**
 * Set up event listeners for the worker-based voice chat interface
 */
function setupWorkerVoiceChatEvents() {
  // Status and initialization events
  voiceChat.addEventListener('status', (event) => {
    updateDebugInfo(`📊 ${event.detail.message}`);
  });
  
  voiceChat.addEventListener('initialized', () => {
    updateDebugInfo('🎉 Voice chat core initialized');
  });
  
  voiceChat.addEventListener('ready', (event) => {
    updateDebugInfo('🎉 All models loaded and ready!');
    updateDebugInfo(`🧠 Loaded models: ${event.detail.models.join(', ')}`);
    updateSystemStatus('All models ready');
    enableUI();
  });
  
  // Model loading events
  voiceChat.addEventListener('modelLoaded', (event) => {
    const { model, loadedModels } = event.detail;
    updateDebugInfo(`🧠 Model loaded: ${model} (${loadedModels.length} total)`);
    updateModelStatus(model, 'loaded');
  });
  
  // Voice activity events
  voiceChat.addEventListener('listeningStart', () => {
    updateDebugInfo('🎤 Listening started');
    updateSystemStatus('Listening for speech...');
    updateListeningUI(true);
  });
  
  voiceChat.addEventListener('listeningStop', () => {
    updateDebugInfo('🎤 Listening stopped');
    updateSystemStatus('Voice chat ready');
    updateListeningUI(false);
  });
  
  voiceChat.addEventListener('speechDetected', (event) => {
    const { type, duration } = event.detail;
    if (type === 'start') {
      updateDebugInfo('🗣️ Speech detected - start');
      updateSystemStatus('Processing speech...');
    } else if (type === 'end') {
      updateDebugInfo(`🗣️ Speech detected - end (${duration?.toFixed(2)}s)`);
    }
  });
  
  // Processing events
  voiceChat.addEventListener('processingStart', (event) => {
    const { input, type } = event.detail;
    updateDebugInfo(`⚙️ Processing ${type} input...`);
    updateSystemStatus('Processing...');
  });
  
  // Message events
  voiceChat.addEventListener('userMessage', (event) => {
    const { text, type } = event.detail;
    updateDebugInfo(`👤 User (${type}): "${text}"`);
    addToConversation('user', text);
  });
  
  voiceChat.addEventListener('transcription', (event) => {
    const { text, confidence } = event.detail;
    updateDebugInfo(`🎯 Transcription (${(confidence * 100).toFixed(1)}%): "${text}"`);
  });
  
  voiceChat.addEventListener('assistantMessage', (event) => {
    const { text } = event.detail;
    updateDebugInfo(`🤖 Assistant: "${text}"`);
    addToConversation('assistant', text);
    updateSystemStatus('Speaking...');
  });
  
  // Speech synthesis events
  voiceChat.addEventListener('speakingStart', () => {
    updateDebugInfo('🔊 Speaking started');
    updateSpeakingUI(true);
  });
  
  voiceChat.addEventListener('speakingEnd', () => {
    updateDebugInfo('🔊 Speaking ended');
    updateSpeakingUI(false);
    updateSystemStatus('Voice chat ready');
  });
  
  // Conversation events
  voiceChat.addEventListener('conversationReset', () => {
    updateDebugInfo('🔄 Conversation reset');
    clearConversation();
  });
  
  // Error handling
  voiceChat.addEventListener('error', (event) => {
    const { type, error, details } = event.detail || {};
    const errorMessage = error?.message || error?.toString() || error || 'Unknown error';
    console.error(`Voice chat error (${type || 'unknown'}):`, error);
    updateDebugInfo(`❌ Error (${type || 'unknown'}): ${errorMessage}`);
    
    if (details) {
      console.error('Error details:', details);
    }
    
    // Update UI based on error type
    if (type === 'ml') {
      updateSystemStatus('Model error - check console');
    } else if (type === 'worker') {
      updateSystemStatus('Worker error - check models');
    } else if (type === 'audio') {
      updateSystemStatus('Audio error - check permissions');
    } else {
      updateSystemStatus('Error occurred');
    }
  });
}

/**
 * Enable UI controls when system is ready
 */
function enableUI() {
  const startBtn = document.getElementById('start-listening-btn');
  const stopBtn = document.getElementById('stop-listening-btn');
  const textInput = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-text-btn');
  const resetBtn = document.getElementById('reset-conversation-btn');
  
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = false;
  if (textInput) textInput.disabled = false;
  if (sendBtn) sendBtn.disabled = false;
  if (resetBtn) resetBtn.disabled = false;
}

/**
 * Update listening UI state
 */
function updateListeningUI(isListening) {
  const startBtn = document.getElementById('start-listening-btn');
  const stopBtn = document.getElementById('stop-listening-btn');
  
  if (startBtn) {
    startBtn.textContent = isListening ? 'Listening...' : 'Start Listening';
    startBtn.style.backgroundColor = isListening ? '#28a745' : '#007bff';
  }
  
  if (stopBtn) {
    stopBtn.disabled = !isListening;
  }
}

/**
 * Update speaking UI state
 */
function updateSpeakingUI(isSpeaking) {
  const statusEl = document.getElementById('system-status');
  if (statusEl && isSpeaking) {
    statusEl.style.color = '#17a2b8';
  } else if (statusEl) {
    statusEl.style.color = '';
  }
}

/**
 * Update model status in UI
 */
function updateModelStatus(model, status) {
  const modelStatusEl = document.getElementById('model-status');
  if (modelStatusEl) {
    const current = modelStatusEl.textContent || '';
    modelStatusEl.textContent = `${current} ${model}:${status}`;
  }
}

/**
 * Clear conversation display
 */
function clearConversation() {
  const conversationLog = document.getElementById('conversation-log');
  if (conversationLog) {
    conversationLog.innerHTML = '';
  }
}

// Button event handlers (will be called from HTML)
window.startListening = async function() {
  if (!voiceChat) {
    updateDebugInfo('❌ Voice chat not initialized');
    return;
  }
  
  try {
    await voiceChat.startListening();
  } catch (error) {
    updateDebugInfo(`❌ Failed to start listening: ${error.message}`);
  }
};

window.stopListening = async function() {
  if (!voiceChat) return;
  
  try {
    await voiceChat.stopListening();
  } catch (error) {
    updateDebugInfo(`❌ Failed to stop listening: ${error.message}`);
  }
};

window.sendTextMessage = async function() {
  if (!voiceChat) {
    updateDebugInfo('❌ Voice chat not initialized');
    return;
  }
  
  const textInput = document.getElementById('text-input');
  const text = textInput?.value?.trim();
  
  if (!text) return;
  
  try {
    textInput.value = '';
    await voiceChat.processTextInput(text);
  } catch (error) {
    updateDebugInfo(`❌ Failed to process text: ${error.message}`);
  }
};

window.resetConversation = function() {
  if (!voiceChat) return;
  
  voiceChat.resetConversation();
};

window.getVoiceChatStatus = function() {
  if (!voiceChat) return null;
  
  const status = voiceChat.getStatus();
  console.log('Voice chat status:', status);
  updateDebugInfo(`📊 Status: ${JSON.stringify(status, null, 2)}`);
  return status;
};

window.testTTS = function(message = "Hello! This is a test of the text-to-speech system.") {
  if (!voiceChat) {
    updateDebugInfo('❌ Voice chat not initialized');
    return;
  }
  
  try {
    updateDebugInfo(`🔊 Testing TTS: "${message}"`);
    voiceChat.synthesizeSpeech(message);
  } catch (error) {
    updateDebugInfo(`❌ TTS test failed: ${error.message}`);
  }
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
  updateDebugInfo('🔧 DOM loaded, initializing worker-based voice chat...');
  initializeWorkerVoiceChat().catch(error => {
    console.error('Failed to initialize worker voice chat:', error);
    updateDebugInfo(`❌ Initialization failed: ${error.message}`);
  });
});

// Auto-initialize if DOM already loaded
if (document.readyState === 'loading') {
  // DOM not ready yet
} else {
  updateDebugInfo('🔧 Initializing worker-based voice chat...');
  initializeWorkerVoiceChat().catch(error => {
    console.error('Failed to initialize worker voice chat:', error);
    updateDebugInfo(`❌ Initialization failed: ${error.message}`);
  });
}

console.log('Worker-based voice chat example loaded');
