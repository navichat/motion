/**
 * FibonacciScheduler.js
 *
 * Implements a Fibonacci Heap-based task scheduler for managing and prioritizing
 * animation generation tasks. Supports preemption and dynamic priority changes.
 */
class FibonacciScheduler {
    constructor() {
        this.heap = []; // Placeholder for Fibonacci Heap implementation
        this.tasks = new Map(); // Map to store tasks by ID for quick access
        this.runningTask = null;
        this.worker = null; // Reference to the animation worker
    }

    /**
     * Initializes the scheduler with a Web Worker for task execution.
     * @param {Worker} worker - The Web Worker instance to dispatch tasks to.
     */
    init(worker) {
        this.worker = worker;
        this.worker.onmessage = this._handleWorkerMessage.bind(this);
        this.worker.onerror = this._handleWorkerError.bind(this);
    }

    /**
     * Adds a task to the scheduler.
     * @param {Object} task - The task object created by AnimationTaskFactory.
     */
    addTask(task) {
        console.log(`Adding task ${task.id} with priority ${task.priority}`);
        this.tasks.set(task.id, task);
        // In a real Fibonacci Heap, you'd insert the task based on its priority.
        // For this placeholder, we'll just add it to a simple array and sort.
        this.heap.push(task);
        this.heap.sort((a, b) => b.priority - a.priority); // Higher priority first
        this._scheduleNextTask();
    }

    /**
     * Attempts to schedule the next highest priority task.
     * If a task is already running, it checks for preemption.
     * @private
     */
    _scheduleNextTask() {
        if (this.heap.length === 0) {
            console.log("No tasks in queue.");
            this.runningTask = null;
            return;
        }

        const nextTask = this.heap[0]; // Highest priority task

        if (this.runningTask && nextTask.priority > this.runningTask.priority) {
            console.log(`Preempting task ${this.runningTask.id} with ${nextTask.id}`);
            // In a real scenario, you'd send a message to the worker to interrupt
            // For now, we'll just log and assume interruption.
            if (this.runningTask.onError) {
                this.runningTask.onError(new Error("Task preempted"));
            }
            this.runningTask.status = 'interrupted';
            this.runningTask = null; // Clear running task to allow new one to start
            // Re-add the interrupted task to the heap if it needs to be resumed later or handled differently
            // For simplicity, we'll just let the new task run.
        }

        if (!this.runningTask) {
            this.runningTask = this.heap.shift(); // Get and remove the highest priority task
            this.runningTask.status = 'running';
            console.log(`Starting task ${this.runningTask.id}`);
            // Dispatch task to worker
            if (this.worker) {
                this.worker.postMessage({ type: 'startTask', task: this.runningTask });
            } else {
                console.error("Worker not initialized for scheduler.");
                if (this.runningTask.onError) {
                    this.runningTask.onError(new Error("Worker not available"));
                }
                this.runningTask.status = 'error';
                this.runningTask = null;
                this._scheduleNextTask(); // Try next task
            }
        }
    }

    /**
     * Handles messages received from the animation worker.
     * @param {MessageEvent} event - The message event from the worker.
     * @private
     */
    _handleWorkerMessage(event) {
        const { type, taskId, result, error, progress } = event.data;
        const task = this.tasks.get(taskId);

        if (!task) {
            console.warn(`Received message for unknown task: ${taskId}`);
            return;
        }

        switch (type) {
            case 'taskComplete':
                console.log(`Task ${taskId} completed.`);
                task.status = 'completed';
                if (task.onComplete) {
                    task.onComplete(result);
                }
                this.runningTask = null;
                this._scheduleNextTask(); // Schedule next task
                break;
            case 'taskError':
                console.error(`Task ${taskId} error:`, error);
                task.status = 'error';
                if (task.onError) {
                    task.onError(error);
                }
                this.runningTask = null;
                this._scheduleNextTask(); // Schedule next task
                break;
            case 'taskProgress':
                if (task.onProgress) {
                    task.onProgress(progress);
                }
                break;
            default:
                console.warn(`Unknown message type from worker: ${type}`);
        }
    }

    /**
     * Handles errors from the animation worker.
     * @param {ErrorEvent} error - The error event from the worker.
     * @private
     */
    _handleWorkerError(error) {
        console.error("Worker error:", error);
        if (this.runningTask && this.runningTask.onError) {
            this.runningTask.onError(error);
            this.runningTask.status = 'error';
            this.runningTask = null;
            this._scheduleNextTask();
        }
    }

    /**
     * Changes the priority of an existing task.
     * @param {string} taskId - The ID of the task to update.
     * @param {number} newPriority - The new priority for the task.
     */
    changeTaskPriority(taskId, newPriority) {
        const task = this.tasks.get(taskId);
        if (task) {
            console.log(`Changing priority of task ${taskId} from ${task.priority} to ${newPriority}`);
            task.priority = newPriority;
            // In a real Fibonacci Heap, you'd call decreaseKey or increaseKey.
            // For this placeholder, we re-sort the heap.
            this.heap.sort((a, b) => b.priority - a.priority);
            this._scheduleNextTask(); // Re-evaluate scheduling after priority change
        }
    }

    /**
     * Removes a task from the scheduler.
     * @param {string} taskId - The ID of the task to remove.
     */
    removeTask(taskId) {
        console.log(`Removing task ${taskId}`);
        const task = this.tasks.get(taskId);
        if (task) {
            this.tasks.delete(taskId);
            this.heap = this.heap.filter(t => t.id !== taskId);
            if (this.runningTask && this.runningTask.id === taskId) {
                // If the running task is removed, stop it and schedule next
                // In a real scenario, send interrupt signal to worker
                this.runningTask = null;
                this._scheduleNextTask();
            }
        }
    }
}

export default FibonacciScheduler;