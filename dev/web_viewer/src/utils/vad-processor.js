/**
 * AudioWorklet processor for Voice Activity Detection
 * This replaces the deprecated ScriptProcessorNode
 */

class VADProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 1024;
        this.sampleBuffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        if (input && input[0]) {
            const inputChannel = input[0];
            
            // Copy input to output (passthrough)
            if (output && output[0]) {
                output[0].set(inputChannel);
            }

            // Buffer audio data for analysis
            for (let i = 0; i < inputChannel.length; i++) {
                this.sampleBuffer[this.bufferIndex] = inputChannel[i];
                this.bufferIndex++;

                // When buffer is full, send it for analysis
                if (this.bufferIndex >= this.bufferSize) {
                    // Send audio data to main thread for analysis
                    this.port.postMessage({
                        type: 'audioData',
                        audioData: new Float32Array(this.sampleBuffer)
                    });

                    // Reset buffer
                    this.bufferIndex = 0;
                }
            }
        }

        return true; // Keep processor alive
    }
}

registerProcessor('vad-processor', VADProcessor);
