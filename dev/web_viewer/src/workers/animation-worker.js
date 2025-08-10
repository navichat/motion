/**
 * animation-worker.js
 *
 * Web Worker for offloading computationally intensive animation generation tasks.
 * Receives tasks from the main thread (via FibonacciScheduler) and executes them.
 */

// Import necessary adapters within the worker context
// Note: In a real setup, you might need a build step (e.g., Webpack) to bundle these imports for the worker.
import Audio2GestureAdapter from '../animation/adapters/Audio2GestureAdapter.js';
import BVHFileLoader from '../animation/adapters/BVHFileLoader.js';
import DeepMimicAdapter from '../animation/adapters/DeepMimicAdapter.js';
import FaceFormerAdapter from '../animation/adapters/FaceFormerAdapter.js';
import RSMTAdapter from '../animation/adapters/RSMTAdapter.js';

const adapters = {
    'audio2gesture': new Audio2GestureAdapter(),
    'bvh_file': new BVHFileLoader(),
    'deepmimic': new DeepMimicAdapter(),
    'faceformer': new FaceFormerAdapter(),
    'rsmt': new RSMTAdapter(),
};

self.onmessage = async (event) => {
    const { type, task } = event.data;

    if (type === 'startTask') {
        const { id, sourceType, inputData } = task;
        const adapter = adapters[sourceType];

        if (!adapter) {
            self.postMessage({ type: 'taskError', taskId: id, error: `Unknown source type: ${sourceType}` });
            return;
        }

        try {
            self.postMessage({ type: 'taskProgress', taskId: id, progress: 0 });
            // Assuming all adapters have a generateAnimation method that returns BVH frames
            const bvhFrames = await adapter.generateAnimation(inputData);
            self.postMessage({ type: 'taskProgress', taskId: id, progress: 1 });
            self.postMessage({ type: 'taskComplete', taskId: id, result: bvhFrames });
        } catch (error) {
            console.error(`Error processing task ${id} (${sourceType}):`, error);
            self.postMessage({ type: 'taskError', taskId: id, error: error.message });
        }
    }
};

console.log("Animation Web Worker initialized.");