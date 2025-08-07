/**
 * AudioQueue - Manages audio playback queue for voice synthesis
 * Supports dependency injection of audio context from ResourceManager
 */

import { BrowserCompatibility } from './BrowserCompatibility.js';

export class AudioQueue extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            sampleRate: options.audioSampleRate || 22050,
            crossfadeDuration: options.crossfadeDuration || 50, // ms
            maxQueueSize: options.maxQueueSize || 20,
            ...options
        };

        // Use injected audio context if available, otherwise create new one
        this.audioContext = options.audioContext || null;
        this.needsOwnContext = !this.audioContext;
        
        this.queue = [];
        this.isPlaying = false;
        this.currentSource = null;
        this.gainNode = null;
        this.playbackStartTime = 0;
    }

    /**
     * Initialize the audio queue
     */
    async initialize() {
        try {
            // Use injected audio context if available
            if (!this.audioContext) {
                console.log('🎵 AudioQueue: Creating new audio context');
                this.audioContext = BrowserCompatibility.createAudioContext({
                    sampleRate: this.options.sampleRate
                });
                this.needsOwnContext = true;
            } else {
                console.log('🎵 AudioQueue: Using injected audio context');
                this.needsOwnContext = false;
            }

            // Create gain node for volume control and crossfading
            this.gainNode = this.audioContext.createGain();
            this.gainNode.connect(this.audioContext.destination);

            this.emit('initialized');
            return true;
        } catch (error) {
            this.emit('error', { type: 'initialization', error });
            throw error;
        }
    }

    /**
     * Add audio data to the queue
     */
    async enqueue(audioData) {
        if (!this.audioContext) {
            throw new Error('Audio queue not initialized');
        }

        try {
            // Convert Float32Array to AudioBuffer
            const audioBuffer = this.createAudioBuffer(audioData);
            
            // Add to queue
            this.queue.push({
                buffer: audioBuffer,
                timestamp: Date.now()
            });

            // Limit queue size
            if (this.queue.length > this.options.maxQueueSize) {
                this.queue.shift();
                this.emit('warning', { message: 'Audio queue overflow, dropping oldest item' });
            }

            this.emit('enqueued', { 
                queueLength: this.queue.length,
                audioLength: audioBuffer.duration 
            });

        } catch (error) {
            this.emit('error', { type: 'enqueue', error });
            throw error;
        }
    }

    /**
     * Start playing the audio queue
     */
    async play() {
        if (!this.audioContext) {
            throw new Error('Audio queue not initialized');
        }

        if (this.isPlaying || this.queue.length === 0) {
            return;
        }

        try {
            // Resume audio context if suspended
            if (this.audioContext.state === 'suspended') {
                await BrowserCompatibility.ensureAudioContextResumed(this.audioContext);
            }

            this.isPlaying = true;
            this.playbackStartTime = this.audioContext.currentTime;
            
            this.emit('playbackStart');
            
            // Start playing the first item
            this.playNext();

        } catch (error) {
            this.isPlaying = false;
            this.emit('error', { type: 'play', error });
            throw error;
        }
    }

    /**
     * Play the next item in the queue
     */
    async playNext() {
        if (!this.isPlaying || this.queue.length === 0) {
            // End of queue reached
            this.isPlaying = false;
            this.currentSource = null;
            this.emit('playbackEnd');
            return;
        }

        try {
            const audioItem = this.queue.shift();
            const audioBuffer = audioItem.buffer;

            // Create audio source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.gainNode);

            this.currentSource = source;

            // Set up event handlers
            source.onended = () => {
                this.currentSource = null;
                // Small delay between chunks for natural speech rhythm
                setTimeout(() => this.playNext(), 50);
            };

            // Start playback
            source.start(0);

            this.emit('chunkStart', { 
                duration: audioBuffer.duration,
                remaining: this.queue.length 
            });

        } catch (error) {
            this.emit('error', { type: 'playNext', error });
            // Try to continue with next item
            setTimeout(() => this.playNext(), 100);
        }
    }

    /**
     * Stop playback and clear queue
     */
    async stop() {
        try {
            this.isPlaying = false;

            // Stop current source
            if (this.currentSource) {
                try {
                    this.currentSource.stop();
                } catch (e) {
                    // Ignore errors when stopping
                }
                this.currentSource = null;
            }

            // Clear queue
            this.queue = [];

            this.emit('stopped');
        } catch (error) {
            this.emit('error', { type: 'stop', error });
        }
    }

    /**
     * Pause playback (keeping queue intact)
     */
    async pause() {
        if (!this.isPlaying) {
            return;
        }

        try {
            this.isPlaying = false;

            // Stop current source
            if (this.currentSource) {
                try {
                    this.currentSource.stop();
                } catch (e) {
                    // Ignore errors when stopping
                }
                this.currentSource = null;
            }

            this.emit('paused');
        } catch (error) {
            this.emit('error', { type: 'pause', error });
        }
    }

    /**
     * Resume playback
     */
    async resume() {
        if (this.isPlaying || this.queue.length === 0) {
            return;
        }

        await this.play();
    }

    /**
     * Clear the queue without stopping current playback
     */
    clearQueue() {
        const clearedItems = this.queue.length;
        this.queue = [];
        
        this.emit('queueCleared', { clearedItems });
    }

    /**
     * Create AudioBuffer from Float32Array
     */
    createAudioBuffer(audioData) {
        // Use the actual audio context sample rate instead of the configured one
        const actualSampleRate = this.audioContext.sampleRate;
        
        const audioBuffer = this.audioContext.createBuffer(
            1, // mono
            audioData.length,
            actualSampleRate
        );

        const channelData = audioBuffer.getChannelData(0);
        channelData.set(audioData);

        return audioBuffer;
    }

    /**
     * Set playback volume (0.0 to 1.0)
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
     * Get queue length
     */
    getQueueLength() {
        return this.queue.length;
    }

    /**
     * Get total queued duration in seconds
     */
    getQueuedDuration() {
        return this.queue.reduce((total, item) => total + item.buffer.duration, 0);
    }

    /**
     * Get playback status
     */
    getStatus() {
        return {
            initialized: !!this.audioContext,
            playing: this.isPlaying,
            queueLength: this.queue.length,
            queuedDuration: this.getQueuedDuration(),
            volume: this.getVolume(),
            currentlyPlaying: !!this.currentSource
        };
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stop();

        if (this.gainNode) {
            this.gainNode.disconnect();
            this.gainNode = null;
        }

        // Only close audio context if we created it ourselves
        if (this.audioContext && this.needsOwnContext) {
            this.audioContext.close();
            this.audioContext = null;
        } else if (this.audioContext) {
            // If using injected context, just clear our reference
            this.audioContext = null;
        }
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }
}
