/**
 * VoiceActivityDetector - Simple voice activity detection using Web Audio API
 * Triggers speech start/end events for the voice chat interface
 * Uses AudioWorkletNode with fallback to deprecated ScriptProcessorNode
 */

export class VoiceActivityDetector extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            sampleRate: options.sampleRate || 16000,
            sensitivity: options.vadSensitivity || 0.5,
            minSpeechDuration: options.minSpeechDuration || 300, // ms
            maxSpeechDuration: options.maxSpeechDuration || 8000, // ms
            silenceThreshold: options.silenceThreshold || 100, // ms of silence to end speech
            energyThreshold: options.energyThreshold || 0.01,
            ...options
        };

        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.analyser = null;
        this.workletNode = null;
        
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
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.options.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.options.sampleRate
            });

            // Try AudioWorklet first, fall back to ScriptProcessor
            try {
                await this.initializeWithAudioWorklet();
            } catch (workletError) {
                console.warn('AudioWorklet not supported, falling back to ScriptProcessorNode');
                await this.initializeWithScriptProcessor();
            }

            this.emit('initialized');
            return true;
        } catch (error) {
            this.emit('error', { type: 'initialization', error });
            throw error;
        }
    }

    /**
     * Initialize with AudioWorklet (preferred method)
     */
    async initializeWithAudioWorklet() {
        // Load AudioWorklet processor
        const workletUrl = new URL('./vad-processor.js', import.meta.url);
        await this.audioContext.audioWorklet.addModule(workletUrl);

        // Create audio processing nodes
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.3;

        // Create AudioWorkletNode for real-time analysis
        this.workletNode = new AudioWorkletNode(this.audioContext, 'vad-processor');
        
        // Set up message handling from worklet
        this.workletNode.port.onmessage = (event) => {
            if (event.data.type === 'audioData') {
                this.processAudioData(event.data.audioData);
            }
        };

        // Connect audio graph
        source.connect(this.analyser);
        this.analyser.connect(this.workletNode);
        this.workletNode.connect(this.audioContext.destination);
    }

    /**
     * Fallback initialization using deprecated ScriptProcessorNode
     */
    async initializeWithScriptProcessor() {
        // Create audio processing nodes
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.3;

        // Create script processor for real-time analysis (deprecated but fallback)
        this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);
        
        // Connect audio graph
        source.connect(this.analyser);
        this.analyser.connect(this.processor);
        this.processor.connect(this.audioContext.destination);

        // Set up audio processing
        this.processor.onaudioprocess = (event) => {
            const inputBuffer = event.inputBuffer.getChannelData(0);
            this.processAudioData(inputBuffer);
        };
    }

    /**
     * Fallback initialization using deprecated ScriptProcessorNode
     */
    async initializeWithScriptProcessor() {
        try {
            // Request microphone access (if not already done)
            if (!this.mediaStream) {
                this.mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: this.options.sampleRate,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
            }

            // Create audio context (if not already done)
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: this.options.sampleRate
                });
            }

            // Create audio processing nodes
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.3;

            // Create script processor for real-time analysis (deprecated but fallback)
            this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);
            
            // Connect audio graph
            source.connect(this.analyser);
            this.analyser.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            // Set up audio processing
            this.processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                this.processAudioData(inputBuffer);
            };

            this.emit('initialized');
            return true;
        } catch (error) {
            this.emit('error', { type: 'initialization', error });
            throw error;
        }
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
            // Resume audio context if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

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
    processAudioData(audioData) {
        if (!this.isListening) {
            return;
        }

        let inputBuffer;
        
        // Handle both AudioWorklet data (Float32Array) and ScriptProcessor data
        if (audioData instanceof Float32Array) {
            // AudioWorklet data
            inputBuffer = audioData;
        } else if (audioData && audioData.inputBuffer) {
            // Legacy ScriptProcessor data
            inputBuffer = audioData.inputBuffer.getChannelData(0);
        } else {
            // Direct Float32Array
            inputBuffer = audioData;
        }

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
    cleanup() {
        this.stopListening();

        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }

        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
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
