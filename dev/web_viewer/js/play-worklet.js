/**
 * Buffered Audio Worklet Processor for TTS playback
 * Manages audio output queue for streaming TTS audio
 */

class BufferedAudioWorkletProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferQueue = [];
        this.currentChunkOffset = 0;
        this.hadData = false;

        this.port.onmessage = (event) => {
            const data = event.data;
            
            if (data instanceof Float32Array) {
                // New audio data to play
                this.hadData = true;
                this.bufferQueue.push(data);
            } else if (data === "stop") {
                // Stop playback and clear queue
                this.bufferQueue = [];
                this.currentChunkOffset = 0;
                this.hadData = false;
            }
        };
    }

    process(inputs, outputs) {
        const channel = outputs[0][0];
        if (!channel) return true;

        const numSamples = channel.length;
        let outputIndex = 0;

        // Check if playback finished
        if (this.hadData && this.bufferQueue.length === 0 && this.currentChunkOffset === 0) {
            this.port.postMessage({ type: "playback_ended" });
            this.hadData = false;
        }

        // Fill output buffer
        while (outputIndex < numSamples) {
            if (this.bufferQueue.length > 0) {
                const currentChunk = this.bufferQueue[0];
                const remainingSamples = currentChunk.length - this.currentChunkOffset;
                const samplesToCopy = Math.min(remainingSamples, numSamples - outputIndex);

                // Copy audio data to output
                channel.set(
                    currentChunk.subarray(
                        this.currentChunkOffset,
                        this.currentChunkOffset + samplesToCopy,
                    ),
                    outputIndex,
                );

                this.currentChunkOffset += samplesToCopy;
                outputIndex += samplesToCopy;

                // Remove chunk if fully consumed
                if (this.currentChunkOffset >= currentChunk.length) {
                    this.bufferQueue.shift();
                    this.currentChunkOffset = 0;
                }
            } else {
                // No data available - fill with silence
                channel.fill(0, outputIndex);
                outputIndex = numSamples;
            }
        }

        return true; // Keep processor alive
    }
}

registerProcessor("buffered-audio-worklet-processor", BufferedAudioWorkletProcessor);
