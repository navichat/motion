/**
 * Audio Playback Worklet - Buffered Audio Processing
 * Based on conversational-webgpu/src/play-worklet.js
 */

export default function() {
  class BufferedAudioWorkletProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.buffers = [];
      this.currentBuffer = null;
      this.currentPosition = 0;
      this.isPlaying = false;
      
      this.port.onmessage = (event) => {
        const audioData = event.data;
        if (audioData && audioData.length > 0) {
          this.buffers.push(new Float32Array(audioData));
          if (!this.isPlaying) {
            this.startPlayback();
          }
        }
      };
    }
    
    startPlayback() {
      if (this.buffers.length > 0) {
        this.currentBuffer = this.buffers.shift();
        this.currentPosition = 0;
        this.isPlaying = true;
      }
    }
    
    process(inputs, outputs, parameters) {
      const output = outputs[0];
      if (output.length === 0) return true;
      
      const outputChannel = output[0];
      
      if (!this.isPlaying || !this.currentBuffer) {
        // Output silence
        outputChannel.fill(0);
        return true;
      }
      
      for (let i = 0; i < outputChannel.length; i++) {
        if (this.currentPosition < this.currentBuffer.length) {
          outputChannel[i] = this.currentBuffer[this.currentPosition];
          this.currentPosition++;
        } else {
          // Current buffer finished
          if (this.buffers.length > 0) {
            // Start next buffer
            this.currentBuffer = this.buffers.shift();
            this.currentPosition = 0;
            outputChannel[i] = this.currentBuffer[this.currentPosition];
            this.currentPosition++;
          } else {
            // No more buffers
            outputChannel[i] = 0;
            this.isPlaying = false;
            
            // Notify main thread that playback ended
            this.port.postMessage({ type: 'playback_ended' });
          }
        }
      }
      
      return true;
    }
  }
  
  registerProcessor('buffered-audio-worklet-processor', BufferedAudioWorkletProcessor);
}
