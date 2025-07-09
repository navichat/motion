/**
 * WhisperModule - Speech-to-text using Whisper models
 * Supports model loading/unloading for memory management
 */

export class WhisperModule extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            modelSize: options.whisperModel || 'tiny',
            device: options.device || 'webgpu',
            quantized: options.quantized || true,
            ...options
        };

        this.model = null;
        this.processor = null;
        this.isModelLoaded = false;
        this.loadingPromise = null;
    }

    /**
     * Load the Whisper model
     */
    async load() {
        if (this.isModelLoaded) {
            return true;
        }

        if (this.loadingPromise) {
            return this.loadingPromise;
        }

        this.loadingPromise = this._loadModel();
        return this.loadingPromise;
    }

    async _loadModel() {
        try {
            this.emit('loading', { module: 'whisper', status: 'starting' });

            // Import transformers.js
            const { pipeline, env } = await import('@huggingface/transformers');
            
            // Configure for local models
            env.allowRemoteModels = false;
            env.allowLocalModels = true;
            env.localModelPath = './models/';

            const modelName = this._getModelName();
            
            this.emit('loading', { module: 'whisper', status: 'downloading', model: modelName });

            // Create speech recognition pipeline
            this.model = await pipeline('automatic-speech-recognition', modelName, {
                device: this.options.device,
                dtype: this.options.quantized ? 'q8' : 'fp16'
            });

            this.isModelLoaded = true;
            this.loadingPromise = null;
            
            this.emit('loaded', { 
                module: 'whisper', 
                model: modelName,
                device: this.options.device 
            });

            return true;
        } catch (error) {
            this.isModelLoaded = false;
            this.loadingPromise = null;
            this.emit('error', { module: 'whisper', error });
            throw error;
        }
    }

    /**
     * Unload the model to free memory
     */
    async unload() {
        if (!this.isModelLoaded) {
            return;
        }

        try {
            // Clear model references
            this.model = null;
            this.processor = null;
            this.isModelLoaded = false;

            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }

            this.emit('unloaded', { module: 'whisper' });
        } catch (error) {
            this.emit('error', { module: 'whisper', error });
        }
    }

    /**
     * Transcribe audio data
     */
    async transcribe(audioData) {
        if (!this.isModelLoaded) {
            throw new Error('Whisper model not loaded');
        }

        try {
            // Convert audio data to the format expected by Whisper
            const processedAudio = this._preprocessAudio(audioData);
            
            // Run transcription
            const result = await this.model(processedAudio, {
                task: 'transcribe',
                language: 'english', // Can be made configurable
                return_timestamps: false
            });

            return {
                text: result.text || '',
                confidence: result.confidence || 0
            };
        } catch (error) {
            this.emit('error', { module: 'whisper', operation: 'transcribe', error });
            throw error;
        }
    }

    /**
     * Preprocess audio data for Whisper
     */
    _preprocessAudio(audioData) {
        // Ensure audio is in the right format for Whisper
        // Whisper expects 16kHz mono audio
        
        if (audioData instanceof Float32Array) {
            return audioData;
        }
        
        if (audioData instanceof ArrayBuffer) {
            return new Float32Array(audioData);
        }
        
        if (Array.isArray(audioData)) {
            return new Float32Array(audioData);
        }
        
        throw new Error('Unsupported audio format');
    }

    /**
     * Get the model name based on size preference
     */
    _getModelName() {
        const models = {
            'tiny': 'Xenova/whisper-tiny.en',
            'base': 'Xenova/whisper-base.en',
            'small': 'Xenova/whisper-small.en'
        };

        return models[this.options.modelSize] || models.tiny;
    }

    /**
     * Check if model is loaded
     */
    isLoaded() {
        return this.isModelLoaded;
    }

    /**
     * Get model info
     */
    getModelInfo() {
        return {
            loaded: this.isModelLoaded,
            model: this._getModelName(),
            size: this.options.modelSize,
            device: this.options.device,
            quantized: this.options.quantized
        };
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }
}
