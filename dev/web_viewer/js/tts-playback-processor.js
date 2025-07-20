/**
 * Simple Audio Worklet for TTS Playback
 * Handles playing generated TTS audio data
 */

class TTSPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.audioBuffer = [];
        this.bufferPosition = 0;
        this.isPlaying = false;
        this.sampleRate = 16000;
        
        this.port.onmessage = (event) => {
            if (event.data.type === 'audio_data') {
                this.loadAudioData(event.data.data, event.data.sampleRate || 16000);
            }
        };
    }
    
    loadAudioData(audioData, sampleRate) {
        // Convert audio data to Float32Array if needed
        if (audioData instanceof Array) {
            audioData = new Float32Array(audioData);
        }
        
        this.audioBuffer = audioData;
        this.bufferPosition = 0;
        this.isPlaying = true;
        this.sampleRate = sampleRate;
        
        console.log(`TTS Playback: Loading ${audioData.length} samples at ${sampleRate}Hz`);
    }
    
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        
        if (this.isPlaying && this.audioBuffer.length > 0) {
            const channelData = output[0];
            
            for (let i = 0; i < channelData.length; i++) {
                if (this.bufferPosition < this.audioBuffer.length) {
                    channelData[i] = this.audioBuffer[this.bufferPosition];
                    this.bufferPosition++;
                } else {
                    // Finished playing
                    channelData[i] = 0;
                    if (this.isPlaying) {
                        this.isPlaying = false;
                        this.port.postMessage({ type: 'playback_ended' });
                    }
                }
            }
        } else {
            // No audio to play, output silence
            const channelData = output[0];
            for (let i = 0; i < channelData.length; i++) {
                channelData[i] = 0;
            }
        }
        
        return true;
    }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
