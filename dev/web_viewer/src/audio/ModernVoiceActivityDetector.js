/**
 * Modern VoiceActivityDetector using AudioWorklet
 * Based on conversational-webgpu example patterns
 */

import { BrowserCompatibility } from './BrowserCompatibility.js';
import { 
  INPUT_SAMPLE_RATE, 
  SPEECH_THRESHOLD, 
  EXIT_THRESHOLD,
  MIN_SILENCE_DURATION_SAMPLES,
  MIN_SPEECH_DURATION_SAMPLES,
  SPEECH_PAD_SAMPLES,
  AUDIO_WORKLET_OPTIONS,
  NEW_BUFFER_SIZE 
} from './constants.js';

export class ModernVoiceActivityDetector extends EventTarget {
  constructor(options = {}) {
    super();
    
    this.options = {
      sampleRate: INPUT_SAMPLE_RATE,
      threshold: SPEECH_THRESHOLD,
      exitThreshold: EXIT_THRESHOLD,
      minSilenceDuration: MIN_SILENCE_DURATION_SAMPLES,
      minSpeechDuration: MIN_SPEECH_DURATION_SAMPLES,
      speechPadding: SPEECH_PAD_SAMPLES,
      bufferSize: NEW_BUFFER_SIZE,
      ...options
    };

    // Audio components
    this.audioContext = options.audioContext || null;
    this.needsOwnContext = !this.audioContext;
    this.mediaStream = null;
    this.workletNode = null;
    this.source = null;
    this.analyser = null;

    // VAD state
    this.isListening = false;
    this.isSpeechActive = false;
    this.audioBuffer = [];
    this.speechBuffer = [];
    this.silenceCounter = 0;
    this.speechCounter = 0;

    // State tracking
    this.isInitialized = false;
    this.features = null;
  }

  /**
   * Initialize the modern VAD with AudioWorklet
   */
  async initialize() {
    try {
      this.features = BrowserCompatibility.detectFeatures();
      
      // Check AudioWorklet support
      if (!this.features.audioWorklet) {
        throw new Error('AudioWorklet not supported - falling back to legacy VAD');
      }

      // Create or use injected audio context
      if (!this.audioContext) {
        this.audioContext = BrowserCompatibility.createAudioContext({
          sampleRate: this.options.sampleRate
        });
        this.needsOwnContext = true;
        console.log('🎤 Modern VAD: Created new audio context');
      } else {
        this.needsOwnContext = false;
        console.log('🎤 Modern VAD: Using injected audio context');
      }

      // Get microphone access
      const audioSettings = BrowserCompatibility.getOptimalAudioSettings();
      this.mediaStream = await BrowserCompatibility.getUserMedia({
        audio: {
          channelCount: audioSettings.channelCount,
          echoCancellation: audioSettings.echoCancellation,
          noiseSuppression: audioSettings.noiseSuppression,
          autoGainControl: audioSettings.autoGainControl,
          sampleRate: this.options.sampleRate
        }
      });

      // Create audio analysis chain
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.source.connect(this.analyser);

      // Load AudioWorklet module
      const workletUrl = new URL('./vad-processor-worklet.js', import.meta.url);
      await this.audioContext.audioWorklet.addModule(workletUrl);

      // Create AudioWorklet node
      this.workletNode = new AudioWorkletNode(
        this.audioContext, 
        'vad-processor',
        AUDIO_WORKLET_OPTIONS
      );

      // Connect audio pipeline
      this.source.connect(this.workletNode);

      // Handle worklet messages
      this.workletNode.port.onmessage = (event) => {
        this.handleWorkletMessage(event.data);
      };

      this.isInitialized = true;
      this.emit('initialized', { 
        sampleRate: this.audioContext.sampleRate,
        features: this.features 
      });

      console.log(`🎤 Modern VAD initialized: ${this.audioContext.sampleRate}Hz (AudioWorklet)`);
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
    const { type, buffer } = data;
    
    if (type === 'audioData' && buffer) {
      this.processAudioData(buffer);
    }
  }

  /**
   * Process incoming audio data for VAD
   */
  processAudioData(audioData) {
    // Add to buffer
    this.audioBuffer.push(...audioData);

    // Simple energy-based VAD (can be enhanced with ML model)
    const energy = this.calculateEnergy(audioData);
    const isSpeech = energy > this.options.threshold;

    if (isSpeech && !this.isSpeechActive) {
      // Speech start detected
      this.handleSpeechStart();
    } else if (!isSpeech && this.isSpeechActive) {
      this.silenceCounter++;
      
      if (this.silenceCounter >= this.options.minSilenceDuration) {
        // Speech end detected
        this.handleSpeechEnd();
      }
    } else if (isSpeech && this.isSpeechActive) {
      // Continue speech
      this.silenceCounter = 0;
      this.speechCounter++;
    }

    // Add to speech buffer if active
    if (this.isSpeechActive) {
      this.speechBuffer.push(...audioData);
    }

    // Limit buffer sizes
    if (this.audioBuffer.length > this.options.sampleRate * 30) { // 30 seconds max
      this.audioBuffer = this.audioBuffer.slice(-this.options.sampleRate * 20); // Keep 20 seconds
    }
  }

  /**
   * Calculate audio energy for simple VAD
   */
  calculateEnergy(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  /**
   * Handle speech start
   */
  handleSpeechStart() {
    this.isSpeechActive = true;
    this.silenceCounter = 0;
    this.speechCounter = 0;
    this.speechBuffer = [];
    
    // Add padding from recent audio
    const paddingStart = Math.max(0, this.audioBuffer.length - this.options.speechPadding);
    this.speechBuffer.push(...this.audioBuffer.slice(paddingStart));
    
    this.emit('speechStart');
    console.log('🗣️ Speech started');
  }

  /**
   * Handle speech end
   */
  handleSpeechEnd() {
    if (this.speechBuffer.length < this.options.minSpeechDuration) {
      // Too short, ignore
      this.isSpeechActive = false;
      this.speechBuffer = [];
      return;
    }

    const speechData = new Float32Array(this.speechBuffer);
    this.isSpeechActive = false;
    this.speechBuffer = [];
    
    this.emit('speechEnd', { audioData: speechData });
    console.log(`🗣️ Speech ended (${(speechData.length / this.options.sampleRate).toFixed(2)}s)`);
  }

  /**
   * Start listening for voice activity
   */
  async startListening() {
    if (!this.isInitialized) {
      throw new Error('VAD not initialized');
    }

    if (this.isListening) return;

    try {
      // Ensure audio context is resumed
      await BrowserCompatibility.ensureAudioContextResumed(this.audioContext);

      // Start the worklet processor
      this.workletNode.port.postMessage({ type: 'start' });

      this.isListening = true;
      this.audioBuffer = [];
      this.isSpeechActive = false;

      this.emit('listening', { status: 'started' });
      console.log('🎤 Modern VAD listening started');

    } catch (error) {
      this.emit('error', { type: 'start_listening', error });
      throw error;
    }
  }

  /**
   * Stop listening
   */
  async stopListening() {
    if (!this.isListening) return;

    try {
      // Stop the worklet processor
      this.workletNode.port.postMessage({ type: 'stop' });

      this.isListening = false;

      // Handle any remaining speech
      if (this.isSpeechActive && this.speechBuffer.length >= this.options.minSpeechDuration) {
        const speechData = new Float32Array(this.speechBuffer);
        this.emit('speechEnd', { audioData: speechData });
      }

      this.emit('listening', { status: 'stopped' });
      console.log('🎤 Modern VAD listening stopped');

    } catch (error) {
      this.emit('error', { type: 'stop_listening', error });
    }
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
   * Get status information
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      listening: this.isListening,
      speechActive: this.isSpeechActive,
      audioBufferSize: this.audioBuffer.length,
      speechBufferSize: this.speechBuffer.length,
      audioLevel: this.getAudioLevel(),
      features: this.features,
      usesMoernWorklet: true
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

    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }

    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    // Only close audio context if we created it
    if (this.audioContext && this.needsOwnContext) {
      this.audioContext.close();
      this.audioContext = null;
    } else if (this.audioContext) {
      this.audioContext = null;
    }

    this.isInitialized = false;
    this.isListening = false;
  }

  /**
   * Emit custom events
   */
  emit(eventType, detail = {}) {
    this.dispatchEvent(new CustomEvent(eventType, { detail }));
  }
}
