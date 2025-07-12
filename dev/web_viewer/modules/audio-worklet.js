/**
 * Buffered Audio Worklet for TTS playback
 * Based on conversational-webgpu example
 */

export default () => {
  class BufferedAudioWorkletProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.bufferQueue = [];
      this.currentChunkOffset = 0;
      this.hadData = false;
      this.isPlaying = false;

      this.port.onmessage = (event) => {
        const data = event.data;
        
        if (data instanceof Float32Array) {
          this.hadData = true;
          this.bufferQueue.push(data);
          this.isPlaying = true;
        } else if (typeof data === 'object') {
          switch (data.type) {
            case 'stop':
              this.bufferQueue = [];
              this.currentChunkOffset = 0;
              this.isPlaying = false;
              this.hadData = false;
              break;
            case 'pause':
              this.isPlaying = false;
              break;
            case 'resume':
              this.isPlaying = true;
              break;
            case 'clear':
              this.bufferQueue = [];
              this.currentChunkOffset = 0;
              break;
          }
        }
      };
    }

    process(inputs, outputs) {
      const channel = outputs[0][0];
      if (!channel || !this.isPlaying) return true;

      const numSamples = channel.length;
      let outputIndex = 0;

      // Check if playback ended
      if (this.hadData && this.bufferQueue.length === 0) {
        this.port.postMessage({ type: "playback_ended" });
        this.hadData = false;
        this.isPlaying = false;
        return true;
      }

      // Fill output buffer
      while (outputIndex < numSamples && this.bufferQueue.length > 0) {
        const currentChunk = this.bufferQueue[0];
        const remainingSamples = currentChunk.length - this.currentChunkOffset;
        const samplesToCopy = Math.min(remainingSamples, numSamples - outputIndex);

        // Copy samples to output
        channel.set(
          currentChunk.subarray(
            this.currentChunkOffset,
            this.currentChunkOffset + samplesToCopy
          ),
          outputIndex
        );

        outputIndex += samplesToCopy;
        this.currentChunkOffset += samplesToCopy;

        // Move to next chunk if current is exhausted
        if (this.currentChunkOffset >= currentChunk.length) {
          this.bufferQueue.shift();
          this.currentChunkOffset = 0;
          
          // Notify about chunk completion
          this.port.postMessage({ 
            type: "chunk_completed",
            remaining: this.bufferQueue.length 
          });
        }
      }

      // Fill remaining samples with silence if needed
      if (outputIndex < numSamples) {
        channel.fill(0, outputIndex);
      }

      return true;
    }
  }

  registerProcessor("buffered-audio-worklet-processor", BufferedAudioWorkletProcessor);
};
