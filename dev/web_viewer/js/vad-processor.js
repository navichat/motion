/**
 * Voice Activity Detection Audio Worklet Processor
 * Processes incoming audio and buffers it for VAD analysis
 */

const MIN_CHUNK_SIZE = 512;
let globalPointer = 0;
let globalBuffer = new Float32Array(MIN_CHUNK_SIZE);

class VADProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        const buffer = inputs[0][0];
        if (!buffer) return true; // Keep processor alive even when no input

        if (buffer.length > MIN_CHUNK_SIZE) {
            // If the buffer is larger than the minimum chunk size, send the entire buffer
            this.port.postMessage({ buffer });
        } else {
            const remaining = MIN_CHUNK_SIZE - globalPointer;
            
            if (buffer.length >= remaining) {
                // Buffer will complete or exceed the global buffer
                globalBuffer.set(buffer.subarray(0, remaining), globalPointer);
                
                // Send the completed global buffer
                this.port.postMessage({ buffer: globalBuffer });
                
                // Reset and set overflow
                globalBuffer.fill(0);
                const overflow = buffer.subarray(remaining);
                globalBuffer.set(overflow, 0);
                globalPointer = overflow.length;
            } else {
                // Buffer fits completely in remaining space
                globalBuffer.set(buffer, globalPointer);
                globalPointer += buffer.length;
            }
        }

        return true; // Keep the processor alive
    }
}

registerProcessor("vad-processor", VADProcessor);
