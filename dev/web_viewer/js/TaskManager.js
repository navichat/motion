/**
 * Advanced Task Management Engine
 * Uses Fibonacci Heap for efficient priority-based task scheduling
 * Supports preemption, worker pools, and resource allocation
 */

// Import dependencies (ensure these are loaded first)
// import { FibonacciHeap } from './FibonacciHeap.js';
// import { MockGPUJob, MockGPUJobFactory } from './MockGPUJobs.js';

class Task {
    constructor(job, priority = 0, scheduledTime = null, options = {}) {
        this.id = this._generateId();
        this.job = job;
        this.priority = priority; // Lower number = higher priority
        this.scheduledTime = scheduledTime || Date.now();
        this.createdTime = Date.now();
        this.startTime = null;
        this.endTime = null;
        this.status = 'queued'; // queued, running, completed, failed, cancelled, preempted
        this.worker = null;
        this.result = null;
        this.error = null;
        this.retryCount = 0;
        this.maxRetries = options.maxRetries || 3;
        this.timeout = options.timeout || 30000; // 30 seconds default
        this.canPreempt = options.canPreempt !== false; // Default to true
        this.resourceRequirements = options.resourceRequirements || { cpu: 1, gpu: 0, memory: 100 };
        this.dependencies = options.dependencies || [];
        this.callbacks = {
            onProgress: options.onProgress,
            onComplete: options.onComplete,
            onError: options.onError,
            onPreempt: options.onPreempt
        };
        this.metadata = options.metadata || {};
    }

    _generateId() {
        return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getEffectivePriority() {
        // Lower number = higher priority
        // Adjust priority based on wait time and retries
        const waitTime = Date.now() - this.createdTime;
        const agingBonus = Math.floor(waitTime / 10000); // +1 priority per 10 seconds
        const retryPenalty = this.retryCount * 2;
        return this.priority - agingBonus + retryPenalty;
    }

    canRun() {
        const now = Date.now();
        return this.scheduledTime <= now && this.status === 'queued';
    }

    isReadyToRun() {
        return this.canRun() && this.dependencies.every(dep => dep.status === 'completed');
    }
}

class WorkerPool {
    constructor(size = 4, workerType = 'cpu') {
        this.size = size;
        this.workerType = workerType;
        this.workers = [];
        this.availableWorkers = [];
        this.busyWorkers = new Map(); // worker -> task
        this.terminated = false;
        this._initializeWorkers();
    }

    _initializeWorkers() {
        for (let i = 0; i < this.size; i++) {
            const worker = this._createWorker(i);
            this.workers.push(worker);
            this.availableWorkers.push(worker);
        }
    }

    _createWorker(id) {
        // For now, create a mock worker. In real implementation, this would create actual Web Workers
        return {
            id: `${this.workerType}_worker_${id}`,
            type: this.workerType,
            busy: false,
            currentTask: null,
            terminate: () => { /* Mock terminate */ },
            postMessage: (data) => { /* Mock postMessage */ }
        };
    }

    getAvailableWorker() {
        return this.availableWorkers.pop();
    }

    releaseWorker(worker) {
        if (this.busyWorkers.has(worker)) {
            this.busyWorkers.delete(worker);
            worker.busy = false;
            worker.currentTask = null;
            this.availableWorkers.push(worker);
        }
    }

    assignWorker(worker, task) {
        worker.busy = true;
        worker.currentTask = task;
        this.busyWorkers.set(worker, task);
    }

    getStats() {
        return {
            type: this.workerType,
            total: this.size,
            available: this.availableWorkers.length,
            busy: this.busyWorkers.size,
            terminated: this.terminated
        };
    }

    terminate() {
        this.terminated = true;
        this.workers.forEach(worker => worker.terminate());
        this.workers.length = 0;
        this.availableWorkers.length = 0;
        this.busyWorkers.clear();
    }
}

class TaskManager {
    constructor(options = {}) {
        this.heap = new FibonacciHeap();
        this.tasks = new Map(); // taskId -> task
        this.taskNodes = new Map(); // taskId -> heapNode
        this.runningTasks = new Map(); // taskId -> task
        this.completedTasks = new Map(); // taskId -> task
        this.failedTasks = new Map(); // taskId -> task
        
        // Worker pools
        this.workerPools = {
            cpu: new WorkerPool(options.cpuWorkers || 2, 'cpu'),
            gpu: new WorkerPool(options.gpuWorkers || 1, 'gpu'),
            webnn: new WorkerPool(options.webnnWorkers || 1, 'webnn')
        };

        // Configuration
        this.maxConcurrentTasks = options.maxConcurrentTasks || 4;
        this.preemptionEnabled = options.preemptionEnabled !== false;
        this.schedulingInterval = options.schedulingInterval || 100; // ms
        this.running = false;
        this.schedulerTimer = null;

        // Statistics
        this.stats = {
            tasksScheduled: 0,
            tasksCompleted: 0,
            tasksFailed: 0,
            tasksPreempted: 0,
            totalExecutionTime: 0,
            averageWaitTime: 0
        };

        // Event handlers
        this.eventHandlers = {
            taskQueued: [],
            taskStarted: [],
            taskCompleted: [],
            taskFailed: [],
            taskPreempted: [],
            queueEmpty: [],
            queueFull: []
        };

        this._bindMethods();
    }

    _bindMethods() {
        this.scheduleTask = this.scheduleTask.bind(this);
        this.start = this.start.bind(this);
        this.stop = this.stop.bind(this);
        this._processQueue = this._processQueue.bind(this);
    }

    /**
     * Schedule a new task
     */
    scheduleTask(job, priority = 0, scheduledTime = null, options = {}) {
        const task = new Task(job, priority, scheduledTime, options);
        this.tasks.set(task.id, task);
        
        const effectivePriority = task.getEffectivePriority();
        const heapNode = this.heap.insert(effectivePriority, task);
        this.taskNodes.set(task.id, heapNode);
        
        this.stats.tasksScheduled++;
        this._emit('taskQueued', task);
        
        console.log(`Task ${task.id} queued with priority ${effectivePriority}`);
        return task.id;
    }

    /**
     * Cancel a task
     */
    cancelTask(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) return false;

        if (task.status === 'running') {
            // Interrupt running task
            task.status = 'cancelled';
            if (task.worker) {
                this._releaseWorker(task.worker);
                task.worker = null;
            }
            this.runningTasks.delete(taskId);
        } else if (task.status === 'queued') {
            // Remove from heap
            const heapNode = this.taskNodes.get(taskId);
            if (heapNode) {
                this.heap.delete(heapNode);
                this.taskNodes.delete(taskId);
            }
            task.status = 'cancelled';
        }

        this.tasks.delete(taskId);
        return true;
    }

    /**
     * Reprioritize a task
     */
    reprioritizeTask(taskId, newPriority) {
        const task = this.tasks.get(taskId);
        if (!task || task.status !== 'queued') return false;

        const heapNode = this.taskNodes.get(taskId);
        if (!heapNode) return false;

        const oldPriority = task.priority;
        task.priority = newPriority;
        const effectivePriority = task.getEffectivePriority();

        if (effectivePriority < heapNode.key) {
            this.heap.decreaseKey(heapNode, effectivePriority);
        } else {
            // Need to remove and re-insert for increased priority
            this.heap.delete(heapNode);
            const newNode = this.heap.insert(effectivePriority, task);
            this.taskNodes.set(taskId, newNode);
        }

        console.log(`Task ${taskId} reprioritized from ${oldPriority} to ${newPriority} (effective: ${effectivePriority})`);
        return true;
    }

    /**
     * Start the task manager
     */
    start() {
        if (this.running) return;
        
        this.running = true;
        console.log('Task Manager started');
        this._scheduleNextProcess();
    }

    /**
     * Stop the task manager
     */
    stop() {
        if (!this.running) return;
        
        this.running = false;
        if (this.schedulerTimer) {
            clearTimeout(this.schedulerTimer);
            this.schedulerTimer = null;
        }
        
        // Cancel all running tasks
        for (const task of this.runningTasks.values()) {
            this.cancelTask(task.id);
        }
        
        console.log('Task Manager stopped');
    }

    /**
     * Main scheduling loop
     */
    _scheduleNextProcess() {
        if (!this.running) return;
        
        this.schedulerTimer = setTimeout(() => {
            this._processQueue();
            this._scheduleNextProcess();
        }, this.schedulingInterval);
    }

    _processQueue() {
        // Update priorities for aging
        this._updateTaskPriorities();
        
        // Process tasks while we have available workers and tasks
        while (this.runningTasks.size < this.maxConcurrentTasks && !this.heap.isEmpty()) {
            const next = this.heap.peek();
            if (!next) break;
            
            const task = next.value;
            
            // Check if task is ready to run
            if (!task.isReadyToRun()) {
                break; // Tasks are ordered by priority, so stop here
            }

            // Get appropriate worker
            const worker = this._getAvailableWorker(task);
            if (!worker) {
                // Try preemption if enabled
                if (this.preemptionEnabled) {
                    const preemptedWorker = this._attemptPreemption(task);
                    if (preemptedWorker) {
                        this._runTask(task, preemptedWorker);
                    }
                }
                break; // No workers available
            }

            // Remove from heap and run
            this.heap.extractMin();
            this.taskNodes.delete(task.id);
            this._runTask(task, worker);
        }

        // Check if queue is empty
        if (this.heap.isEmpty() && this.runningTasks.size === 0) {
            this._emit('queueEmpty');
        }
    }

    _updateTaskPriorities() {
        // Periodically update priorities for tasks that have been waiting
        // This is a simplified aging mechanism
        for (const [taskId, heapNode] of this.taskNodes) {
            const task = heapNode.value;
            const newEffectivePriority = task.getEffectivePriority();
            
            if (newEffectivePriority < heapNode.key) {
                this.heap.decreaseKey(heapNode, newEffectivePriority);
            }
        }
    }

    _getAvailableWorker(task) {
        // Simple resource allocation - prefer specialized workers
        const requirements = task.resourceRequirements;
        
        if (requirements.gpu > 0) {
            return this.workerPools.gpu.getAvailableWorker();
        } else if (requirements.webnn > 0) {
            return this.workerPools.webnn.getAvailableWorker();
        } else {
            return this.workerPools.cpu.getAvailableWorker();
        }
    }

    _attemptPreemption(newTask) {
        if (!this.preemptionEnabled) return null;
        
        // Find a running task with lower priority that can be preempted
        let lowestPriorityTask = null;
        let lowestPriorityWorker = null;
        
        for (const [worker, task] of this.workerPools.cpu.busyWorkers) {
            if (task.canPreempt && task.getEffectivePriority() > newTask.getEffectivePriority()) {
                if (!lowestPriorityTask || task.getEffectivePriority() > lowestPriorityTask.getEffectivePriority()) {
                    lowestPriorityTask = task;
                    lowestPriorityWorker = worker;
                }
            }
        }

        if (lowestPriorityTask) {
            console.log(`Preempting task ${lowestPriorityTask.id} for higher priority task ${newTask.id}`);
            this._preemptTask(lowestPriorityTask, lowestPriorityWorker);
            this.stats.tasksPreempted++;
            return lowestPriorityWorker;
        }

        return null;
    }

    _preemptTask(task, worker) {
        task.status = 'preempted';
        task.job.interrupt();
        this.runningTasks.delete(task.id);
        this._releaseWorker(worker);
        
        // Re-queue the preempted task with slightly higher priority
        task.priority -= 1; // Higher priority for preempted tasks
        task.status = 'queued';
        const effectivePriority = task.getEffectivePriority();
        const heapNode = this.heap.insert(effectivePriority, task);
        this.taskNodes.set(task.id, heapNode);
        
        this._emit('taskPreempted', task);
    }

    async _runTask(task, worker) {
        task.status = 'running';
        task.startTime = Date.now();
        task.worker = worker;
        
        this.runningTasks.set(task.id, task);
        this._assignWorker(worker, task);
        
        console.log(`Starting task ${task.id} on worker ${worker.id}`);
        this._emit('taskStarted', task);

        try {
            // Set up progress callback
            const progressCallback = (progress, jobStats) => {
                if (task.callbacks.onProgress) {
                    task.callbacks.onProgress(progress, jobStats, task);
                }
            };

            // Set up cancellation check
            const shouldStop = () => task.status === 'cancelled' || task.status === 'preempted';

            // Execute the job
            const result = await task.job.execute(progressCallback, shouldStop);
            
            if (task.status === 'running') {
                task.status = 'completed';
                task.endTime = Date.now();
                task.result = result;
                
                this.runningTasks.delete(task.id);
                this.completedTasks.set(task.id, task);
                this._releaseWorker(worker);
                
                this.stats.tasksCompleted++;
                this.stats.totalExecutionTime += task.endTime - task.startTime;
                
                console.log(`Task ${task.id} completed in ${task.endTime - task.startTime}ms`);
                
                if (task.callbacks.onComplete) {
                    task.callbacks.onComplete(result, task);
                }
                this._emit('taskCompleted', task);
            }
            
        } catch (error) {
            if (task.status === 'running') {
                task.status = 'failed';
                task.endTime = Date.now();
                task.error = error.message;
                
                this.runningTasks.delete(task.id);
                this._releaseWorker(worker);
                
                // Retry logic
                if (task.retryCount < task.maxRetries) {
                    task.retryCount++;
                    task.status = 'queued';
                    task.priority += 1; // Lower priority for retries
                    
                    const effectivePriority = task.getEffectivePriority();
                    const heapNode = this.heap.insert(effectivePriority, task);
                    this.taskNodes.set(task.id, heapNode);
                    
                    console.log(`Task ${task.id} failed, retrying (${task.retryCount}/${task.maxRetries})`);
                } else {
                    this.failedTasks.set(task.id, task);
                    this.stats.tasksFailed++;
                    
                    console.log(`Task ${task.id} failed permanently: ${error.message}`);
                    
                    if (task.callbacks.onError) {
                        task.callbacks.onError(error, task);
                    }
                    this._emit('taskFailed', task);
                }
            }
        }
    }

    _assignWorker(worker, task) {
        if (worker.type === 'cpu') {
            this.workerPools.cpu.assignWorker(worker, task);
        } else if (worker.type === 'gpu') {
            this.workerPools.gpu.assignWorker(worker, task);
        } else if (worker.type === 'webnn') {
            this.workerPools.webnn.assignWorker(worker, task);
        }
    }

    _releaseWorker(worker) {
        if (worker.type === 'cpu') {
            this.workerPools.cpu.releaseWorker(worker);
        } else if (worker.type === 'gpu') {
            this.workerPools.gpu.releaseWorker(worker);
        } else if (worker.type === 'webnn') {
            this.workerPools.webnn.releaseWorker(worker);
        }
    }

    // Event system
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }

    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }

    _emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Get comprehensive statistics
     */
    getStats() {
        const queueSize = this.heap.size();
        const runningCount = this.runningTasks.size;
        
        return {
            queue: {
                size: queueSize,
                running: runningCount,
                completed: this.completedTasks.size,
                failed: this.failedTasks.size
            },
            workers: {
                cpu: this.workerPools.cpu.getStats(),
                gpu: this.workerPools.gpu.getStats(),
                webnn: this.workerPools.webnn.getStats()
            },
            performance: {
                tasksScheduled: this.stats.tasksScheduled,
                tasksCompleted: this.stats.tasksCompleted,
                tasksFailed: this.stats.tasksFailed,
                tasksPreempted: this.stats.tasksPreempted,
                averageExecutionTime: this.stats.tasksCompleted > 0 ? 
                    this.stats.totalExecutionTime / this.stats.tasksCompleted : 0
            },
            system: {
                running: this.running,
                preemptionEnabled: this.preemptionEnabled,
                maxConcurrentTasks: this.maxConcurrentTasks
            }
        };
    }

    /**
     * Get detailed queue information
     */
    getQueueInfo() {
        const queuedTasks = [];
        const runningTasks = [];
        
        // Get queued tasks (this is expensive but useful for debugging)
        for (const [taskId, task] of this.tasks) {
            if (task.status === 'queued') {
                queuedTasks.push({
                    id: task.id,
                    type: task.job.type,
                    priority: task.getEffectivePriority(),
                    waitTime: Date.now() - task.createdTime,
                    scheduledTime: task.scheduledTime
                });
            }
        }
        
        for (const [taskId, task] of this.runningTasks) {
            runningTasks.push({
                id: task.id,
                type: task.job.type,
                priority: task.getEffectivePriority(),
                runTime: Date.now() - task.startTime,
                progress: task.job.progress,
                worker: task.worker.id
            });
        }
        
        return {
            queued: queuedTasks.sort((a, b) => a.priority - b.priority),
            running: runningTasks
        };
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TaskManager, Task, WorkerPool };
} else if (typeof window !== 'undefined') {
    window.TaskManager = TaskManager;
    window.Task = Task;
    window.WorkerPool = WorkerPool;
}
