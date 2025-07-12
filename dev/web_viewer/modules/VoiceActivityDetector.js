/**
 * VoiceActivityDetector - Simple voice activity detection using Web Audio API
 * Triggers speech start/end events for the voice chat interface
 * Uses ScriptProcessorNode with deprecation warning (AudioWorklet requires separate processor file)
 * Supports dependency injection for audio context
 */

import { BrowserCompatibility } from './BrowserCompatibility.js';

export class VoiceActivityDetector extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            sampleRate: options.sampleRate || options.audioSampleRate || 16000,
            sensitivity: options.vadSensitivity || 0.5,
            minSpeechDuration: options.minSpeechDuration || 300, // ms
            maxSpeechDuration: options.maxSpeechDuration || 8000, // ms
            silenceThreshold: options.silenceThreshold || 100, // ms of silence to end speech
            energyThreshold: options.energyThreshold || 0.01,
            ...options
        };

        // Use injected audio context if available, otherwise create new one
        this.audioContext = options.audioContext || null;
        this.needsOwnContext = !this.audioContext;
        
        this.mediaStream = null;
        this.processor = null;
        this.analyser = null;
        
        this.isListening = false;
        this.isSpeechActive = false;
        this.speechStartTime = 0;
        this.lastSpeechTime = 0;
        this.audioBuffer = [];
        
        this.silenceTimer = null;
        this.maxSpeechTimer = null;
    }

    /**
     * Initialize the voice activity detector
     */
    async initialize() {
        try {
            // Use injected audio context if available
            if (!this.audioContext) {
                console.log('🎤 VAD: Creating new audio context');
                this.audioContext = BrowserCompatibility.createAudioContext();
                this.needsOwnContext = true;
            } else {
                console.log('🎤 VAD: Using injected audio context');
                this.needsOwnContext = false;
            }
            
            // Note: AudioContext will be in 'suspended' state until user interaction
            // This is normal browser behavior and will be resumed when startListening() is called
            
            // Get optimal audio settings for this browser
            const audioSettings = BrowserCompatibility.getOptimalAudioSettings();
            
            // Request microphone access with browser-optimized constraints
            this.mediaStream = await BrowserCompatibility.getUserMedia({
                audio: {
                    channelCount: audioSettings.channelCount,
                    echoCancellation: audioSettings.echoCancellation,
                    noiseSuppression: audioSettings.noiseSuppression,
                    autoGainControl: audioSettings.autoGainControl
                }
            });

            // Update our sample rate to match the actual AudioContext sample rate
            this.options.sampleRate = this.audioContext.sampleRate;
            
            // Log the actual sample rate for debugging
            console.log(`🎤 VAD initialized with sample rate: ${this.audioContext.sampleRate}Hz`);
            console.log(`🎤 MediaStream tracks:`, this.mediaStream.getAudioTracks().map(t => ({
                label: t.label,
                sampleRate: t.getSettings().sampleRate || 'auto'
            })));

            // Create audio processing nodes
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.3;

            // Use ScriptProcessor (deprecated but widely supported)
            // Note: This will show a deprecation warning but provides better compatibility
            this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);
            
            // Connect audio graph
            source.connect(this.analyser);
            this.analyser.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            // Set up audio processing
            this.processor.onaudioprocess = (event) => {
                this.processAudioData(event);
            };

            this.emit('initialized');
            return true;
        } catch (error) {
            // Provide more specific error information for debugging
            if (error.name === 'DOMException' && error.message.includes('sample-rate')) {
                console.error('🔧 Sample rate mismatch detected. AudioContext sample rate:', this.audioContext?.sampleRate);
                console.error('🔧 Try refreshing the page or using a different browser');
            }
            this.emit('error', { type: 'initialization', error });
            throw error;
        }
    }

    /**
     * Start listening for voice activity
     */
    async startListening() {
        if (!this.audioContext) {
            throw new Error('VAD not initialized');
        }

        if (this.isListening) {
            return;
        }

        try {
            // Ensure audio context is resumed (required for mobile browsers)
            await BrowserCompatibility.ensureAudioContextResumed(this.audioContext);

            this.isListening = true;
            this.audioBuffer = [];
            this.isSpeechActive = false;

            this.emit('listening', { status: 'started' });
        } catch (error) {
            this.emit('error', { type: 'start_listening', error });
            throw error;
        }
    }

    /**
     * Stop listening for voice activity
     */
    async stopListening() {
        if (!this.isListening) {
            return;
        }

        this.isListening = false;
        
        // Clear timers
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        
        if (this.maxSpeechTimer) {
            clearTimeout(this.maxSpeechTimer);
            this.maxSpeechTimer = null;
        }

        // End any active speech
        if (this.isSpeechActive) {
            this.endSpeech();
        }

        this.emit('listening', { status: 'stopped' });
    }

    /**
     * Process audio data for voice activity detection
     */
    processAudioData(audioEvent) {
        if (!this.isListening) {
            return;
        }

        const inputBuffer = audioEvent.inputBuffer.getChannelData(0);
        const bufferLength = inputBuffer.length;
        
        // Store audio data for speech recognition
        this.audioBuffer.push(new Float32Array(inputBuffer));

        // Calculate audio energy
        let energy = 0;
        for (let i = 0; i < bufferLength; i++) {
            energy += inputBuffer[i] * inputBuffer[i];
        }
        energy = Math.sqrt(energy / bufferLength);

        // Check for voice activity
        const isVoiceDetected = energy > this.options.energyThreshold;
        const currentTime = Date.now();

        if (isVoiceDetected) {
            this.lastSpeechTime = currentTime;
            
            if (!this.isSpeechActive) {
                this.startSpeech();
            } else {
                // Clear silence timer since we detected voice
                if (this.silenceTimer) {
                    clearTimeout(this.silenceTimer);
                    this.silenceTimer = null;
                }
            }
        } else if (this.isSpeechActive) {
            // No voice detected, but speech is active
            // Start silence timer if not already started
            if (!this.silenceTimer) {
                this.silenceTimer = setTimeout(() => {
                    this.endSpeech();
                }, this.options.silenceThreshold);
            }
        }

        // Limit buffer size to prevent memory issues
        if (this.audioBuffer.length > this.options.sampleRate * 10) { // 10 seconds max
            this.audioBuffer.shift();
        }
    }

    /**
     * Start speech detection
     */
    startSpeech() {
        const currentTime = Date.now();
        
        this.isSpeechActive = true;
        this.speechStartTime = currentTime;
        this.lastSpeechTime = currentTime;
        
        // Clear existing audio buffer when starting new speech
        this.audioBuffer = [];

        // Set maximum speech duration timer
        this.maxSpeechTimer = setTimeout(() => {
            this.endSpeech();
        }, this.options.maxSpeechDuration);

        this.emit('speechStart', {
            timestamp: currentTime
        });
    }

    /**
     * End speech detection
     */
    endSpeech() {
        if (!this.isSpeechActive) {
            return;
        }

        const currentTime = Date.now();
        const speechDuration = currentTime - this.speechStartTime;

        // Clear timers
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        
        if (this.maxSpeechTimer) {
            clearTimeout(this.maxSpeechTimer);
            this.maxSpeechTimer = null;
        }

        this.isSpeechActive = false;

        // Only process speech if it meets minimum duration
        if (speechDuration >= this.options.minSpeechDuration) {
            // Convert audio buffer to continuous Float32Array
            const audioData = this.combineAudioBuffers();
            
            this.emit('speechEnd', {
                timestamp: currentTime,
                duration: speechDuration,
                audioData: audioData
            });
        }

        // Clear audio buffer
        this.audioBuffer = [];
    }

    /**
     * Combine audio buffers into a single Float32Array
     */
    combineAudioBuffers() {
        if (this.audioBuffer.length === 0) {
            return new Float32Array(0);
        }

        // Calculate total length
        let totalLength = 0;
        for (const buffer of this.audioBuffer) {
            totalLength += buffer.length;
        }

        // Combine buffers
        const combinedBuffer = new Float32Array(totalLength);
        let offset = 0;
        
        for (const buffer of this.audioBuffer) {
            combinedBuffer.set(buffer, offset);
            offset += buffer.length;
        }

        return combinedBuffer;
    }

    /**
     * Get current audio level (0-1)
     */
    getCurrentAudioLevel() {
        if (!this.analyser) {
            return 0;
        }

        const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(dataArray);
        
        let average = 0;
        for (let i = 0; i < dataArray.length; i++) {
            average += dataArray[i];
        }
        
        return (average / dataArray.length) / 256;
    }

    /**
     * Update sensitivity
     */
    setSensitivity(sensitivity) {
        this.options.sensitivity = Math.max(0, Math.min(1, sensitivity));
        this.options.energyThreshold = 0.005 + (this.options.sensitivity * 0.02);
    }

    /**
     * Cleanup resources
     */
    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopListening();

        if (this.processor) {
            this.processor.disconnect();
            this.processor.onaudioprocess = null;
            this.processor = null;
        }

        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }

        // Only close audio context if we created it ourselves
        if (this.audioContext && this.needsOwnContext) {
            this.audioContext.close();
            this.audioContext = null;
        } else if (this.audioContext) {
            // If using injected context, just clear our reference
            this.audioContext = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            initialized: !!this.audioContext,
            listening: this.isListening,
            speechActive: this.isSpeechActive,
            audioLevel: this.getCurrentAudioLevel(),
            sensitivity: this.options.sensitivity
        };
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }
}
