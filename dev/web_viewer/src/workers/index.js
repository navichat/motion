/**
 * Workers Module Index
 * Provides clean imports for all worker types
 */

// Compute Workers
export { default as CPUWorker } from './compute/CPUWorker.js';
export { default as GPUWorker } from './compute/GPUWorker.js';

// AI Workers  
export { default as WebNNWorker } from './ai/WebNNWorker.js';

// Note: These are Web Workers, so they need to be instantiated with new Worker()
// Example: const cpuWorker = new Worker('./src/workers/compute/CPUWorker.js');
