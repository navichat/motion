// =================================================================================
//  ISOMORPHIC requestAnimationFrame IMPLEMENTATION
// =================================================================================

// 1. Check if we are in a Node.js environment.
const IS_NODE = typeof window === 'undefined';

// 2. Conditionally import Node's performance tools if we're on the server.
let performance;
if (IS_NODE) {
    try {
        // Use dynamic import for Node.js built-in modules
        const perf_hooks = await import('perf_hooks');
        performance = perf_hooks.performance;
    } catch (e) {
        console.error("Failed to import 'perf_hooks'. Performance timing will be less accurate.", e);
        // Fallback to Date.now() if perf_hooks fails for some reason
        performance = { now: () => Date.now() };
    }
} else {
    // In the browser, 'performance' is already globally available on the window object.
    performance = window.performance;
}


// 3. Define the Node.js versions of the functions.
const TARGET_FRAME_TIME = 1000 / 60; // Target a 60 FPS loop

/**
 * Node.js polyfill for requestAnimationFrame.
 * @param {function(number)} callback - The function to call, will receive a high-resolution timestamp.
 * @returns {NodeJS.Timeout} A timer ID that can be used with the cancel function.
 */
const nodeRequestAnimationFrame = (callback) => {
    // Schedule the callback to run after a delay equivalent to one frame.
    return setTimeout(() => {
        callback(performance.now());
    }, TARGET_FRAME_TIME);
};

/**
 * Node.js polyfill for cancelAnimationFrame.
 * @param {NodeJS.Timeout} id - The timer ID to cancel.
 */
const nodeCancelAnimationFrame = (id) => {
    clearTimeout(id);
};

// 4. Export the correct function based on the environment.
// This is the key part: your other code will import and use these,
// and they will automatically work everywhere.
export const requestAnimationFrame = IS_NODE ? nodeRequestAnimationFrame : window.requestAnimationFrame.bind(window);
export const cancelAnimationFrame = IS_NODE ? nodeCancelAnimationFrame : window.cancelAnimationFrame.bind(window);

// =================================================================================
