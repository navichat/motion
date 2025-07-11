/**
 * Worker Orchestrator for Conversational Voice Chat
 * Manages communication between main thread and ML worker
 * Based on conversational-webgpu patterns
 */

import { INPUT_SAMPLE_RATE, AUDIO_CONFIG } from './ConversationConstants.js';

export class ConversationWorkerOrchestrator extends EventTarget {
  constructor(options = {}) {
    super();
    
    this.options = {
      device: 'webgpu',
      voice: 'af_heart',
      systemPrompt: 'conversational',
      ...options
    };
    
    this.worker = null;
    this.audioContext = null;
    this.micStream = null;
    this.vadProcessor = null;
    this.playbackNode = null;
    
    this.isReady = false;
    this.isRecording = false;
    this.isPlaying = false;
    this.voices = [];
    
    this.conversationHistory = [];
  }

  /**
   * Initialize the worker and audio system
   */
  async initialize() {
    try {
      this.emit('status', { message: 'Initializing conversation worker...' });
      
      // Create worker
      try {
        this.worker = new Worker(new URL('./ConversationWorker.js', import.meta.url), {
          type: 'module'
        });
      } catch (error) {
        console.error('Failed to create worker with import.meta.url, trying fallback:', error);
        this.worker = new Worker('./modules/ConversationWorker.js', {
          type: 'module'
        });
      }
      
      // Set up worker event handlers
      this.setupWorkerEvents();
      
      // Note: Audio initialization is deferred until user interaction
      // to comply with browser autoplay policies
      
      // Send initial configuration to worker
      this.worker.postMessage({
        type: 'initialize',
        config: this.options
      });
      
    } catch (error) {
      this.emit('error', { type: 'initialization', error: error.message });
      throw error;
    }
  }

  /**
   * Set up worker event handlers
   */
  setupWorkerEvents() {
    this.worker.onmessage = (event) => {
      this.handleWorkerMessage(event.data);
    };
    
    this.worker.onerror = (error) => {
      this.emit('error', { 
        type: 'worker', 
        error: `Worker error: ${error.message || 'Unknown error'}` 
      });
    };
  }

  /**
   * Handle messages from the worker
   */
  handleWorkerMessage(data) {
    if (data.error) {
      this.emit('error', { type: 'worker', error: data.error.message || data.error });
      return;
    }

    switch (data.type) {
      case 'status':
        if (data.status === 'ready') {
          this.isReady = true;
          this.voices = data.voices || [];
          this.emit('ready', { voices: this.voices });
        } else if (data.status === 'recording_start') {
          this.isRecording = true;
          this.emit('listening', { isListening: true });
        } else if (data.status === 'recording_end') {
          this.isRecording = false;
          this.emit('listening', { isListening: false });
        }
        this.emit('status', { message: data.message, status: data.status });
        break;
        
      case 'transcription':
        this.emit('transcription', { text: data.text });
        break;
        
      case 'response':
        this.conversationHistory.push({
          user: data.userText,
          assistant: data.text,
          timestamp: Date.now()
        });
        this.emit('response', { 
          text: data.text, 
          userText: data.userText,
          conversationLength: this.conversationHistory.length 
        });
        break;
        
      case 'output':
        if (!this.isPlaying && data.result?.audio) {
          this.playAudio(data.result.audio);
        }
        break;
        
      case 'info':
        this.emit('info', { message: data.message });
        break;
        
      default:
        console.log('Unknown worker message:', data);
    }
  }

  /**
   * Initialize audio context and processors
   */
  async initializeAudio() {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: INPUT_SAMPLE_RATE
    });
    
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
    
    // Load worklets
    await this.loadAudioWorklets();
    
    this.emit('status', { message: 'Audio system initialized' });
  }

  /**
   * Load audio worklets for VAD and playback
   */
  async loadAudioWorklets() {
    try {
      // Load VAD processor worklet
      const vadWorkletCode = `
        const MIN_CHUNK_SIZE = 512;
        let globalPointer = 0;
        let globalBuffer = new Float32Array(MIN_CHUNK_SIZE);

        class VADProcessor extends AudioWorkletProcessor {
          process(inputs, outputs, parameters) {
            const buffer = inputs[0][0];
            if (!buffer) return true;

            if (buffer.length > MIN_CHUNK_SIZE) {
              this.port.postMessage({ buffer });
            } else {
              const remaining = MIN_CHUNK_SIZE - globalPointer;
              if (buffer.length >= remaining) {
                globalBuffer.set(buffer.subarray(0, remaining), globalPointer);
                this.port.postMessage({ buffer: globalBuffer });
                globalBuffer.fill(0);
                globalBuffer.set(buffer.subarray(remaining), 0);
                globalPointer = buffer.length - remaining;
              } else {
                globalBuffer.set(buffer, globalPointer);
                globalPointer += buffer.length;
              }
            }
            return true;
          }
        }
        registerProcessor("conversation-vad-processor", VADProcessor);
      `;
      
      const vadBlob = new Blob([vadWorkletCode], { type: 'application/javascript' });
      const vadWorkletUrl = URL.createObjectURL(vadBlob);
      await this.audioContext.audioWorklet.addModule(vadWorkletUrl);
      URL.revokeObjectURL(vadWorkletUrl);
      
      // Load playback processor worklet
      const playbackWorkletCode = `
        class ConversationPlaybackProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.bufferQueue = [];
            this.currentChunkOffset = 0;
            this.hadData = false;

            this.port.onmessage = (event) => {
              const data = event.data;
              if (data instanceof Float32Array) {
                this.hadData = true;
                this.bufferQueue.push(data);
              } else if (data === "stop") {
                this.bufferQueue = [];
                this.currentChunkOffset = 0;
              }
            };
          }

          process(inputs, outputs) {
            const channel = outputs[0][0];
            if (!channel) return true;

            const numSamples = channel.length;
            let outputIndex = 0;

            if (this.hadData && this.bufferQueue.length === 0) {
              this.port.postMessage({ type: "playback_ended" });
              this.hadData = false;
            }

            while (outputIndex < numSamples) {
              if (this.bufferQueue.length > 0) {
                const currentChunk = this.bufferQueue[0];
                const remainingSamples = currentChunk.length - this.currentChunkOffset;
                const samplesToCopy = Math.min(remainingSamples, numSamples - outputIndex);

                channel.set(
                  currentChunk.subarray(this.currentChunkOffset, this.currentChunkOffset + samplesToCopy),
                  outputIndex
                );

                outputIndex += samplesToCopy;
                this.currentChunkOffset += samplesToCopy;

                if (this.currentChunkOffset >= currentChunk.length) {
                  this.bufferQueue.shift();
                  this.currentChunkOffset = 0;
                }
              } else {
                break;
              }
            }
            return true;
          }
        }
        registerProcessor("conversation-playback-processor", ConversationPlaybackProcessor);
      `;
      
      const playbackBlob = new Blob([playbackWorkletCode], { type: 'application/javascript' });
      const playbackWorkletUrl = URL.createObjectURL(playbackBlob);
      await this.audioContext.audioWorklet.addModule(playbackWorkletUrl);
      URL.revokeObjectURL(playbackWorkletUrl);
      
    } catch (error) {
      console.error('Failed to load audio worklets:', error);
      this.emit('error', { type: 'audio', error: 'Failed to load audio worklets' });
    }
  }

  /**
   * Start listening for voice input
   */
  async startListening() {
    if (!this.isReady) {
      throw new Error('System not ready');
    }
    
    try {
      // Initialize audio context on first use (user interaction required)
      if (!this.audioContext) {
        await this.initializeAudio();
      }
      
      // Get microphone access
      this.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: INPUT_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      // Create microphone source
      const micSource = this.audioContext.createMediaStreamSource(this.micStream);
      
      // Create VAD processor
      this.vadProcessor = new AudioWorkletNode(this.audioContext, 'conversation-vad-processor');
      
      // Handle VAD messages
      this.vadProcessor.port.onmessage = (event) => {
        if (event.data.buffer) {
          // Send audio buffer to worker for VAD processing
          this.worker.postMessage({
            type: 'audio_chunk',
            buffer: event.data.buffer
          });
        }
      };
      
      // Connect audio graph
      micSource.connect(this.vadProcessor);
      
      this.emit('status', { message: 'Started listening...' });
      
    } catch (error) {
      this.emit('error', { type: 'microphone', error: error.message });
      throw error;
    }
  }

  /**
   * Stop listening
   */
  stopListening() {
    if (this.micStream) {
      this.micStream.getTracks().forEach(track => track.stop());
      this.micStream = null;
    }
    
    if (this.vadProcessor) {
      this.vadProcessor.disconnect();
      this.vadProcessor = null;
    }
    
    this.worker.postMessage({ type: 'stop_listening' });
    this.emit('status', { message: 'Stopped listening' });
  }

  /**
   * Play audio through the worklet
   */
  playAudio(audioData) {
    if (!this.playbackNode) {
      this.playbackNode = new AudioWorkletNode(this.audioContext, 'conversation-playback-processor');
      this.playbackNode.connect(this.audioContext.destination);
      
      this.playbackNode.port.onmessage = (event) => {
        if (event.data.type === 'playback_ended') {
          this.isPlaying = false;
          this.emit('speaking', { isSpeaking: false });
        }
      };
    }
    
    this.isPlaying = true;
    this.emit('speaking', { isSpeaking: true });
    this.playbackNode.port.postMessage(audioData);
  }

  /**
   * Set voice for TTS
   */
  setVoice(voice) {
    this.options.voice = voice;
    this.worker.postMessage({
      type: 'set_voice',
      voice: voice
    });
  }

  /**
   * Reset conversation
   */
  resetConversation() {
    this.conversationHistory = [];
    this.worker.postMessage({ type: 'reset_conversation' });
    this.emit('conversation_reset');
  }

  /**
   * Get conversation history
   */
  getConversationHistory() {
    return [...this.conversationHistory];
  }

  /**
   * Send a text message (for testing or text input)
   */
  sendTextMessage(text) {
    this.worker.postMessage({
      type: 'text_input',
      text: text
    });
  }

  /**
   * Cleanup resources
   */
  dispose() {
    this.stopListening();
    
    if (this.playbackNode) {
      this.playbackNode.disconnect();
      this.playbackNode = null;
    }
    
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  /**
   * Emit events
   */
  emit(eventName, data) {
    this.dispatchEvent(new CustomEvent(eventName, { detail: data }));
  }
}
