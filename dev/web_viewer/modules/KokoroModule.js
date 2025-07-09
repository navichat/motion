/**
 * KokoroModule - Text-to-speech using Kokoro TTS
 * Supports model loading/unloading for memory management
 */

export class KokoroModule extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            modelPath: options.kokoroModelPath || './Kokoro-82M-v1.0-ONNX/',
            voice: options.voice || 'af_heart',
            sampleRate: options.audioSampleRate || 22050,
            ...options
        };

        this.tts = null;
        this.isModelLoaded = false;
        this.loadingPromise = null;
    }

    /**
     * Load the Kokoro TTS model
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
            this.emit('loading', { module: 'kokoro', status: 'starting' });

            // Import Kokoro TTS
            const { KokoroTTS } = await import('../kokoro.web.js');
            
            this.emit('loading', { module: 'kokoro', status: 'initializing' });

            // Initialize Kokoro TTS with local model path
            this.tts = await KokoroTTS.from_pretrained(this.options.modelPath, {
                device: 'wasm', // Kokoro typically uses WASM
                dtype: 'fp32'
            });

            this.isModelLoaded = true;
            this.loadingPromise = null;
            
            this.emit('loaded', { 
                module: 'kokoro',
                modelPath: this.options.modelPath,
                voice: this.options.voice
            });

            return true;
        } catch (error) {
            this.isModelLoaded = false;
            this.loadingPromise = null;
            this.emit('error', { module: 'kokoro', error });
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
            this.tts = null;
            this.isModelLoaded = false;

            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }

            this.emit('unloaded', { module: 'kokoro' });
        } catch (error) {
            this.emit('error', { module: 'kokoro', error });
        }
    }

    /**
     * Synthesize speech from text
     */
    async synthesize(text, voice = null) {
        if (!this.isModelLoaded) {
            throw new Error('Kokoro TTS model not loaded');
        }

        try {
            // Clean the text (remove emotion markers, etc.)
            const cleanText = text.replace(/\[[^\]]+\]/g, '').trim();
            
            if (!cleanText) {
                return new Float32Array(0);
            }

            // Use specified voice or default
            const voiceToUse = voice || this.options.voice;
            
            this.emit('synthesizing', { 
                module: 'kokoro', 
                text: cleanText.substring(0, 50) + '...',
                voice: voiceToUse 
            });

            // Generate audio using Kokoro TTS
            const audioData = await this.tts.generate(cleanText, voiceToUse);
            
            if (!audioData || audioData.length === 0) {
                throw new Error('No audio data generated');
            }

            this.emit('synthesized', { 
                module: 'kokoro',
                audioLength: audioData.length,
                sampleRate: this.options.sampleRate
            });

            return audioData;
        } catch (error) {
            this.emit('error', { module: 'kokoro', operation: 'synthesize', error });
            throw error;
        }
    }

    /**
     * Get available voices
     */
    getAvailableVoices() {
        // Common Kokoro voices - this could be made dynamic based on model
        return [
            'af_heart',   // Afrikaans female
            'am_male',    // Amharic male
            'en_male',    // English male
            'en_female',  // English female
            'es_male',    // Spanish male
            'fr_female',  // French female
            'ja_male',    // Japanese male
            'zh_female'   // Chinese female
        ];
    }

    /**
     * Set default voice
     */
    setVoice(voice) {
        if (this.getAvailableVoices().includes(voice)) {
            this.options.voice = voice;
            return true;
        }
        return false;
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
            modelPath: this.options.modelPath,
            voice: this.options.voice,
            sampleRate: this.options.sampleRate,
            availableVoices: this.getAvailableVoices()
        };
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }
}
