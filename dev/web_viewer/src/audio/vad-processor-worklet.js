/**
 * Modern VAD AudioWorklet Processor
 * Based on conversational-webgpu example - replaces deprecated ScriptProcessorNode
 */

import { MIN_CHUNK_SIZE } from '../utils/constants.js';

let globalPointer = 0;
let globalBuffer = new Float32Array(MIN_CHUNK_SIZE);

class VADProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.isActive = false;
    
    // Handle messages from main thread
    this.port.onmessage = (event) => {
      const { type, data } = event.data;
      
      switch (type) {
        case 'start':
          this.isActive = true;
          break;
        case 'stop':
          this.isActive = false;
          // Reset buffers
          globalBuffer.fill(0);
          globalPointer = 0;
          break;
        case 'configure':
          // Handle configuration updates
          break;
      }
    };
  }

  process(inputs, outputs, parameters) {
    if (!this.isActive) return true;
    
    const buffer = inputs[0][0];
    if (!buffer) return true; // Buffer is null when stream ends

    if (buffer.length > MIN_CHUNK_SIZE) {
      // If buffer is larger than minimum chunk size, send entire buffer
      this.port.postMessage({ 
        type: 'audioData', 
        buffer: buffer.slice() // Clone to avoid transfer issues
      });
    } else {
      const remaining = MIN_CHUNK_SIZE - globalPointer;
      
      if (buffer.length >= remaining) {
        // Copy remaining space and send buffer
        globalBuffer.set(buffer.subarray(0, remaining), globalPointer);
        
        // Send the complete buffer
        this.port.postMessage({ 
          type: 'audioData', 
          buffer: globalBuffer.slice() 
        });
        
        // Reset and set remaining
        globalBuffer.fill(0);
        globalBuffer.set(buffer.subarray(remaining), 0);
        globalPointer = buffer.length - remaining;
      } else {
        // Copy buffer to global buffer
        globalBuffer.set(buffer, globalPointer);
        globalPointer += buffer.length;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor("vad-processor", VADProcessor);
