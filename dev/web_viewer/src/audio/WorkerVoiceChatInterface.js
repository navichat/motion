/**
 * Worker-based Voice Chat Interface
 * Based on conversational-webgpu App.jsx patterns
 * Uses Web Worker for all ML processing
 */

import { BrowserCompatibility } from './BrowserCompatibility.js';
import { ModernVoiceActivityDetector } from './ModernVoiceActivityDetector.js';
import { VoiceActivityDetector } from './VoiceActivityDetector.js';
import { ModernAudioQueue } from './ModernAudioQueue.js';
import { AudioQueue } from './AudioQueue.js';
import { INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE } from './constants.js';

export class WorkerVoiceChatInterface extends EventTarget {
  constructor(options = {}) {
    super();
    
    this.options = {
      device: options.device || 'wasm',
      audioSampleRate: INPUT_SAMPLE_RATE,
      outputSampleRate: OUTPUT_SAMPLE_RATE,
      vadSensitivity: options.vadSensitivity || 0.5,
      systemPrompt: options.systemPrompt || null,
      ...options
    };

    // Core components
    this.worker = null;
    this.audioContext = null;
    this.vad = null;
    this.audioQueue = null;
    
    // State
    this.isInitialized = false;
    this.isListening = false;
    this.isSpeaking = false;
    this.isProcessing = false;
    this.modelsReady = false;
    
    // Model status tracking
    this.loadedModels = new Set();
    this.modelErrors = new Map();
    
    this.setupWorker();
  }

  /**
   * Set up the ML worker
   */
  setupWorker() {
    this.worker = new Worker(new URL('./ml-worker.js', import.meta.url), {
      type: 'module'
    });
    
    this.worker.onmessage = (event) => {
      this.handleWorkerMessage(event.data);
    };
    
    this.worker.onerror = (event) => {
      console.error('Worker error:', event);
      // Extract meaningful error information from the error event
      const errorMessage = event.error?.message || 
                          event.message || 
                          `Worker error: ${event.filename || 'unknown file'}:${event.lineno || 'unknown line'}` ||
                          'Unknown worker error';
      this.emit('error', { 
        type: 'worker', 
        error: errorMessage,
        details: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          error: event.error
        }
      });
    };
  }

  /**
   * Handle messages from the worker
   */
  handleWorkerMessage(data) {
    const { type } = data;
    
    switch (type) {
      case 'workerReady':
        this.emit('status', { message: 'ML Worker ready' });
        break;
        
      case 'status':
        this.emit('status', { message: data.message || data.status });
        break;
        
      case 'modelLoaded':
        this.loadedModels.add(data.model);
        this.emit('modelLoaded', { 
          model: data.model,
          loadedModels: Array.from(this.loadedModels)
        });
        break;
        
      case 'ready':
        this.modelsReady = true;
        this.emit('ready', { 
          message: data.message,
          models: data.models
        });
        break;
        
      case 'transcription':
        this.isProcessing = false;
        this.emit('transcription', { 
          text: data.text,
          confidence: data.confidence
        });
        this.handleTranscription(data.text);
        break;
        
      case 'response':
        this.emit('response', { 
          text: data.text,
          conversationLength: data.conversationLength
        });
        this.handleResponse(data.text);
        break;
        
      case 'vadResult':
        this.emit('vadResult', { isVoice: data.isVoice });
        break;
        
      case 'error':
        this.modelErrors.set(data.messageType || 'unknown', data.error);
        this.emit('error', { 
          type: 'ml', 
          error: data.error,
          details: data.details
        });
        break;
        
      case 'conversationReset':
        this.emit('conversationReset');
        break;
    }
  }

  /**
   * Initialize the voice chat interface
   */
  async initialize() {
    try {
      this.emit('status', { message: 'Initializing voice chat interface...' });
      
      // Create audio context
      this.audioContext = BrowserCompatibility.createAudioContext({
        sampleRate: this.options.audioSampleRate
      });
      
      // Initialize VAD
      const features = BrowserCompatibility.detectFeatures();
      if (features.audioWorklet) {
        this.vad = new ModernVoiceActivityDetector({
          audioContext: this.audioContext,
          ...this.options
        });
      } else {
        this.vad = new VoiceActivityDetector({
          audioContext: this.audioContext,
          ...this.options
        });
      }
      
      await this.vad.initialize();
      this.setupVADEventHandlers();
      
      // Initialize audio queue
      if (features.audioWorklet) {
        this.audioQueue = new ModernAudioQueue({
          audioContext: this.audioContext,
          sampleRate: this.options.outputSampleRate
        });
      } else {
        this.audioQueue = new AudioQueue({
          audioContext: this.audioContext,
          sampleRate: this.options.outputSampleRate
        });
      }
      
      await this.audioQueue.initialize();
      this.setupAudioQueueEventHandlers();
      
      // Initialize worker with ML models
      this.worker.postMessage({
        type: 'initialize',
        data: {
          device: this.options.device,
          systemPrompt: this.options.systemPrompt
        }
      });
      
      this.isInitialized = true;
      this.emit('initialized');
      
    } catch (error) {
      this.emit('error', { type: 'initialization', error });
      throw error;
    }
  }

  /**
   * Set up VAD event handlers
   */
  setupVADEventHandlers() {
    this.vad.addEventListener('speechStart', () => {
      this.handleSpeechStart();
    });
    
    this.vad.addEventListener('speechEnd', (event) => {
      this.handleSpeechEnd(event.detail.audioData);
    });
    
    this.vad.addEventListener('error', (event) => {
      this.emit('error', { type: 'vad', error: event.detail.error });
    });
  }

  /**
   * Set up audio queue event handlers
   */
  setupAudioQueueEventHandlers() {
    this.audioQueue.addEventListener('playbackStart', () => {
      this.isSpeaking = true;
      this.emit('speakingStart');
    });
    
    this.audioQueue.addEventListener('playbackEnd', () => {
      this.isSpeaking = false;
      this.emit('speakingEnd');
    });
    
    this.audioQueue.addEventListener('error', (event) => {
      this.emit('error', { type: 'audio', error: event.detail.error });
    });
  }

  /**
   * Start listening for voice input
   */
  async startListening() {
    if (!this.isInitialized || this.isListening) return;
    
    try {
      await BrowserCompatibility.ensureAudioContextResumed(this.audioContext);
      await this.vad.startListening();
      this.isListening = true;
      this.emit('listeningStart');
      
    } catch (error) {
      this.emit('error', { type: 'startListening', error });
      throw error;
    }
  }

  /**
   * Stop listening
   */
  async stopListening() {
    if (!this.isListening) return;
    
    try {
      await this.vad.stopListening();
      this.isListening = false;
      this.emit('listeningStop');
      
    } catch (error) {
      this.emit('error', { type: 'stopListening', error });
    }
  }

  /**
   * Process text input directly
   */
  async processTextInput(text) {
    if (!this.modelsReady) {
      throw new Error('Models not ready');
    }
    
    this.isProcessing = true;
    this.emit('processingStart', { input: text, type: 'text' });
    
    this.worker.postMessage({
      type: 'generateResponse',
      data: { text }
    });
  }

  /**
   * Handle speech start
   */
  async handleSpeechStart() {
    this.emit('speechDetected', { type: 'start' });
    
    // Stop any current TTS playback
    if (this.isSpeaking) {
      await this.audioQueue.stop();
    }
  }

  /**
   * Handle speech end
   */
  async handleSpeechEnd(audioData) {
    this.emit('speechDetected', { type: 'end', duration: audioData.length / this.options.audioSampleRate });
    
    if (!this.modelsReady) {
      this.emit('error', { type: 'processing', error: 'Models not ready for transcription' });
      return;
    }
    
    this.isProcessing = true;
    this.emit('processingStart', { input: 'audio', type: 'speech' });
    
    // Send audio to worker for transcription
    this.worker.postMessage({
      type: 'processAudio',
      data: { audioBuffer: audioData }
    });
  }

  /**
   * Handle transcription result
   */
  async handleTranscription(text) {
    if (!text || text.trim().length === 0) {
      this.emit('status', { message: 'No speech detected in audio' });
      return;
    }
    
    this.emit('userMessage', { text, type: 'speech' });
    
    // Generate response
    this.worker.postMessage({
      type: 'generateResponse',
      data: { text }
    });
  }

  /**
   * Handle LLM response
   */
  async handleResponse(text) {
    this.emit('assistantMessage', { text });
    
    // Convert to speech using Web Speech API as fallback
    await this.synthesizeSpeech(text);
  }

  /**
   * Synthesize speech from text
   */
  async synthesizeSpeech(text) {
    try {
      // Use Web Speech API for TTS (fallback approach)
      if (window.speechSynthesis) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;
        
        utterance.onstart = () => {
          this.isSpeaking = true;
          this.emit('speakingStart');
        };
        
        utterance.onend = () => {
          this.isSpeaking = false;
          this.emit('speakingEnd');
        };
        
        utterance.onerror = (error) => {
          this.emit('error', { type: 'tts', error: error.error });
        };
        
        speechSynthesis.speak(utterance);
      } else {
        this.emit('error', { type: 'tts', error: 'Speech synthesis not available' });
      }
      
    } catch (error) {
      this.emit('error', { type: 'tts', error });
    }
  }

  /**
   * Reset conversation
   */
  resetConversation() {
    if (this.worker) {
      this.worker.postMessage({ type: 'resetConversation' });
    }
  }

  /**
   * Set system prompt
   */
  setSystemPrompt(prompt) {
    if (this.worker) {
      this.worker.postMessage({
        type: 'setSystemPrompt',
        data: { prompt }
      });
    }
  }

  /**
   * Get current status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      modelsReady: this.modelsReady,
      listening: this.isListening,
      speaking: this.isSpeaking,
      processing: this.isProcessing,
      loadedModels: Array.from(this.loadedModels),
      errors: Object.fromEntries(this.modelErrors),
      audioLevel: this.vad ? this.vad.getAudioLevel() : 0,
      queueLength: this.audioQueue ? this.audioQueue.getQueueLength() : 0
    };
  }

  /**
   * Cleanup resources
   */
  cleanup() {
    if (this.vad) {
      this.vad.cleanup();
    }
    
    if (this.audioQueue) {
      this.audioQueue.cleanup();
    }
    
    if (this.audioContext) {
      this.audioContext.close();
    }
    
    if (this.worker) {
      this.worker.terminate();
    }
    
    this.isInitialized = false;
    this.modelsReady = false;
  }

  /**
   * Emit custom events
   */
  emit(eventType, detail = {}) {
    this.dispatchEvent(new CustomEvent(eventType, { detail }));
  }
}
