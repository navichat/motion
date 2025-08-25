/**
 * scheduler-worker.js
 *
 * Optional Web Worker for offloading the FibonacciScheduler logic itself.
 * This might be useful if the scheduler operations (insert, extractMin, decreaseKey)
 * become computationally expensive with a very large number of tasks.
 * For now, the FibonacciScheduler runs on the main thread and dispatches to animation-worker.
 * This file is a placeholder for future expansion if needed.
 */

self.onmessage = (event) => {
    const { type, payload } = event.data;

    switch (type) {
        case 'init':
            console.log("Scheduler Web Worker initialized.");
            // Initialize scheduler instance here if it were to run in a worker
            break;
        case 'addTask':
            // Logic to add task to scheduler's heap
            break;
        case 'changePriority':
            // Logic to change task priority
            break;
        // ... other scheduler operations
        default:
            console.warn(`Unknown message type received by scheduler worker: ${type}`);
    }
};