/**
 * Conversation Worker - ML Processing
 * Handles all neural network operations for voice conversation
 * Based on conversational-webgpu/src/worker.js
 */

import {
  MAX_BUFFER_DURATION,
  INPUT_SAMPLE_RATE,
  MIN_SPEECH_DURATION_SAMPLES,
} from "./ConversationConstants.js";

import { ConversationNeuralNetwork } from "./ConversationNeuralNetwork.js";

// Global state
let isInitialized = false;
let currentDevice = 'webgpu';
let currentVoice = 'af_heart';

// Neural Network instance
let neuralNetwork = null;

// Audio processing state
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;
let isRecording = false;
let isPlaying = false;

/**
 * Initialize all ML models
 */
async function initializeModels(config) {
  currentDevice = config.device || 'webgpu';
  currentVoice = config.voice || 'af_heart';
  
  self.postMessage({ type: "info", message: `Using device: "${currentDevice}"` });
  self.postMessage({
    type: "info",
    message: "Loading models...",
    duration: "until_next",
  });

  try {
    neuralNetwork = new ConversationNeuralNetwork(currentDevice);
    const initResult = await neuralNetwork.initialize(config);
    
    isInitialized = true;
    
    self.postMessage({
      type: "status",
      status: "ready",
      message: "Ready!",
      voices: initResult.voices
    });
    
  } catch (error) {
    self.postMessage({ 
      type: "error", 
      error: `Model initialization failed: ${error.message}` 
    });
    throw error;
  }
}

/**
 * Process speech-to-speech pipeline
 */
async function speechToSpeech(buffer, userInput = null) {
  if (isPlaying) return;
  
  isPlaying = true;
  
  try {
    // 1. Transcribe audio (if not text input)
    let transcriptionResult;
    if (userInput) {
      transcriptionResult = { text: userInput, isEmpty: false };
    } else {
      transcriptionResult = await neuralNetwork.transcribeAudio(buffer);
      
      if (transcriptionResult.isEmpty) {
        isPlaying = false;
        return;
      }
    }
    
    self.postMessage({ 
      type: "transcription", 
      text: transcriptionResult.text 
    });
    
    // 2. Generate response with LLM
    const responseResult = await neuralNetwork.generateResponse(transcriptionResult.text);
    
    if (responseResult.text.trim()) {
      self.postMessage({
        type: "response",
        text: responseResult.text,
        userText: transcriptionResult.text
      });
      
      // 3. Generate speech audio (if TTS available)
      try {
        const audioData = await neuralNetwork.generateSpeech(responseResult.text, currentVoice);
        
        self.postMessage({
          type: "output",
          result: { audio: audioData.audio }
        });
      } catch (ttsError) {
        self.postMessage({ 
          type: "info", 
          message: `TTS failed: ${ttsError.message}` 
        });
      }
    }
    
  } catch (error) {
    self.postMessage({ 
      type: "error", 
      error: `Speech processing failed: ${error.message}` 
    });
  } finally {
    isPlaying = false;
  }
}

/**
 * Process incoming audio chunks for VAD
 */
async function processAudioChunk(buffer) {
  if (!isInitialized) return;
  
  // Add to buffer
  const remainingSpace = BUFFER.length - bufferPointer;
  const bytesToCopy = Math.min(buffer.length, remainingSpace);
  
  BUFFER.set(buffer.subarray(0, bytesToCopy), bufferPointer);
  bufferPointer += bytesToCopy;
  
  // Perform VAD
  const vadResult = await neuralNetwork.performVAD(buffer, isRecording);
  
  if (vadResult.isSpeech && !isRecording) {
    // Start recording
    isRecording = true;
    self.postMessage({ 
      type: "status", 
      status: "recording_start",
      message: "Started recording speech..." 
    });
    
  } else if (!vadResult.isSpeech && isRecording) {
    // End recording and process
    isRecording = false;
    self.postMessage({ 
      type: "status", 
      status: "recording_end",
      message: "Processing speech..." 
    });
    
    // Extract recorded audio
    const recordedAudio = BUFFER.slice(0, bufferPointer);
    bufferPointer = 0; // Reset buffer
    
    // Process the speech
    if (recordedAudio.length > MIN_SPEECH_DURATION_SAMPLES) {
      await speechToSpeech(recordedAudio);
    }
  }
}

/**
 * Handle worker messages
 */
self.onmessage = async function(event) {
  const { type, ...data } = event.data;
  
  try {
    switch (type) {
      case 'initialize':
        await initializeModels(data.config || {});
        break;
        
      case 'audio_chunk':
        await processAudioChunk(data.buffer);
        break;
        
      case 'text_input':
        if (data.text && data.text.trim()) {
          await speechToSpeech(null, data.text.trim());
        }
        break;
        
      case 'set_voice':
        currentVoice = data.voice;
        self.postMessage({ 
          type: "info", 
          message: `Voice changed to: ${currentVoice}` 
        });
        break;
        
      case 'stop_listening':
        isRecording = false;
        bufferPointer = 0;
        break;
        
      case 'reset_conversation':
        if (neuralNetwork) {
          neuralNetwork.resetConversation();
        }
        self.postMessage({ 
          type: "info", 
          message: "Conversation reset" 
        });
        break;
        
      case 'interrupt':
        // This needs to be handled by the neural network if it supports interruption
        // For now, we'll just log a message
        self.postMessage({
          type: "info",
          message: "Interruption requested, but not fully supported by current NN implementation."
        });
        break;
        
      default:
        self.postMessage({ 
          type: "error", 
          error: `Unknown message type: ${type}` 
        });
    }
    
  } catch (error) {
    self.postMessage({ 
      type: "error", 
      error: error.message,
      messageType: type 
    });
  }
};

// Global error handlers
self.addEventListener('error', (event) => {
  self.postMessage({
    type: 'error',
    error: `Worker error: ${event.message || 'Unknown error'}`,
    details: {
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno
    }
  });
});

self.addEventListener('unhandledrejection', (event) => {
  self.postMessage({
    type: 'error',
    error: `Unhandled promise rejection: ${event.reason}`,
    details: event.reason
  });
});

// Initialization message
self.postMessage({ 
  type: 'info', 
  message: 'Conversation worker initialized and ready for setup' 
});
