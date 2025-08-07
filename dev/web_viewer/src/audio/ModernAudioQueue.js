/**
 * Modern AudioQueue using AudioWorklet for better performance
 * Based on conversational-webgpu example patterns
 */

import { BrowserCompatibility } from './BrowserCompatibility.js';
import { OUTPUT_SAMPLE_RATE } from './constants.js';
import AUDIO_WORKLET from './audio-worklet.js';

export class ModernAudioQueue extends EventTarget {
  constructor(options = {}) {
    super();
    
    this.options = {
      sampleRate: OUTPUT_SAMPLE_RATE,
      crossfadeDuration: options.crossfadeDuration || 50,
      maxQueueSize: options.maxQueueSize || 20,
      ...options
    };

    // Audio components
    this.audioContext = options.audioContext || null;
    this.needsOwnContext = !this.audioContext;
    this.workletNode = null;
    this.gainNode = null;
    this.analyser = null;

    // State
    this.isInitialized = false;
    this.isPlaying = false;
    this.queueLength = 0;
    this.features = null;
  }

  /**
   * Initialize the modern audio queue
   */
  async initialize() {
    try {
      this.features = BrowserCompatibility.detectFeatures();

      // Check AudioWorklet support
      if (!this.features.audioWorklet) {
        throw new Error('AudioWorklet not supported - falling back to legacy AudioQueue');
      }

      // Create or use injected audio context
      if (!this.audioContext) {
        this.audioContext = BrowserCompatibility.createAudioContext({
          sampleRate: this.options.sampleRate
        });
        this.needsOwnContext = true;
        console.log('🔊 Modern AudioQueue: Created new audio context');
      } else {
        this.needsOwnContext = false;
        console.log('🔊 Modern AudioQueue: Using injected audio context');
      }

      // Load AudioWorklet
      const blob = new Blob([`(${AUDIO_WORKLET.toString()})()`], {
        type: 'application/javascript'
      });
      const url = URL.createObjectURL(blob);
      
      try {
        await this.audioContext.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'buffered-audio-worklet-processor'
      );

      // Create gain node and analyser
      this.gainNode = this.audioContext.createGain();
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;

      // Connect audio pipeline
      this.workletNode.connect(this.gainNode);
      this.gainNode.connect(this.analyser);
      this.analyser.connect(this.audioContext.destination);

      // Handle worklet messages
      this.workletNode.port.onmessage = (event) => {
        this.handleWorkletMessage(event.data);
      };

      this.isInitialized = true;
      this.emit('initialized', { 
        sampleRate: this.audioContext.sampleRate,
        features: this.features 
      });

      console.log(`🔊 Modern AudioQueue initialized: ${this.audioContext.sampleRate}Hz (AudioWorklet)`);
      return true;

    } catch (error) {
      this.emit('error', { type: 'initialization', error });
      throw error;
    }
  }

  /**
   * Handle messages from AudioWorklet
   */
  handleWorkletMessage(data) {
    const { type } = data;
    
    switch (type) {
      case 'playback_ended':
        this.isPlaying = false;
        this.emit('playbackEnd');
        break;
        
      case 'chunk_completed':
        this.queueLength = data.remaining || 0;
        this.emit('chunkCompleted', { remaining: this.queueLength });
        break;
    }
  }

  /**
   * Add audio data to the playback queue
   */
  async enqueue(audioData) {
    if (!this.isInitialized) {
      throw new Error('AudioQueue not initialized');
    }

    try {
      // Convert to Float32Array if needed
      let processedData;
      if (audioData instanceof Float32Array) {
        processedData = audioData;
      } else if (Array.isArray(audioData)) {
        processedData = new Float32Array(audioData);
      } else {
        throw new Error('Invalid audio data format');
      }

      // Send to worklet
      this.workletNode.port.postMessage(processedData);
      this.queueLength++;

      this.emit('enqueued', { 
        queueLength: this.queueLength,
        audioLength: processedData.length / this.audioContext.sampleRate
      });

    } catch (error) {
      this.emit('error', { type: 'enqueue', error });
      throw error;
    }
  }

  /**
   * Start playback
   */
  async play() {
    if (!this.isInitialized) {
      throw new Error('AudioQueue not initialized');
    }

    if (this.isPlaying) return;

    try {
      // Ensure audio context is resumed
      await BrowserCompatibility.ensureAudioContextResumed(this.audioContext);

      this.isPlaying = true;
      this.emit('playbackStart');

    } catch (error) {
      this.isPlaying = false;
      this.emit('error', { type: 'play', error });
      throw error;
    }
  }

  /**
   * Stop playback and clear queue
   */
  async stop() {
    if (!this.workletNode) return;

    try {
      this.workletNode.port.postMessage({ type: 'stop' });
      this.isPlaying = false;
      this.queueLength = 0;
      this.emit('stopped');

    } catch (error) {
      this.emit('error', { type: 'stop', error });
    }
  }

  /**
   * Pause playback
   */
  async pause() {
    if (!this.workletNode || !this.isPlaying) return;

    try {
      this.workletNode.port.postMessage({ type: 'pause' });
      this.isPlaying = false;
      this.emit('paused');

    } catch (error) {
      this.emit('error', { type: 'pause', error });
    }
  }

  /**
   * Resume playback
   */
  async resume() {
    if (!this.workletNode || this.isPlaying) return;

    try {
      await BrowserCompatibility.ensureAudioContextResumed(this.audioContext);
      this.workletNode.port.postMessage({ type: 'resume' });
      this.isPlaying = true;
      this.emit('resumed');

    } catch (error) {
      this.emit('error', { type: 'resume', error });
    }
  }

  /**
   * Clear the queue without stopping playback
   */
  clearQueue() {
    if (!this.workletNode) return;

    const clearedItems = this.queueLength;
    this.workletNode.port.postMessage({ type: 'clear' });
    this.queueLength = 0;
    
    this.emit('queueCleared', { clearedItems });
  }

  /**
   * Set playback volume
   */
  setVolume(volume) {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  /**
   * Get current volume
   */
  getVolume() {
    return this.gainNode ? this.gainNode.gain.value : 1.0;
  }

  /**
   * Get current audio level for visualization
   */
  getAudioLevel() {
    if (!this.analyser) return 0;
    
    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteTimeDomainData(dataArray);
    
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const normalized = dataArray[i] / 128 - 1;
      sum += normalized * normalized;
    }
    
    return Math.sqrt(sum / dataArray.length);
  }

  /**
   * Get queue length
   */
  getQueueLength() {
    return this.queueLength;
  }

  /**
   * Get status information
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      playing: this.isPlaying,
      queueLength: this.queueLength,
      volume: this.getVolume(),
      audioLevel: this.getAudioLevel(),
      features: this.features,
      usesModernWorklet: true
    };
  }

  /**
   * Cleanup resources
   */
  cleanup() {
    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'stop' });
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.gainNode) {
      this.gainNode.disconnect();
      this.gainNode = null;
    }

    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }

    // Only close audio context if we created it
    if (this.audioContext && this.needsOwnContext) {
      this.audioContext.close();
      this.audioContext = null;
    } else if (this.audioContext) {
      this.audioContext = null;
    }

    this.isInitialized = false;
    this.isPlaying = false;
  }

  /**
   * Emit custom events
   */
  emit(eventType, detail = {}) {
    this.dispatchEvent(new CustomEvent(eventType, { detail }));
  }
}
