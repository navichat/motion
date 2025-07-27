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
        // Use job's resource requirements if available, otherwise use options or defaults
        this.resourceRequirements = job.resourceRequirements || options.resourceRequirements || { cpu: 1, gpu: 0, webnn: 0, memory: 100 };
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
    constructor(size = 4, workerType = 'cpu', manager, capabilities) {
        this.size = size;
        this.workerType = workerType;
        this.manager = manager;
        this.workers = [];
        this.availableWorkers = [];
        this.busyWorkers = new Map(); // worker -> task
        this.terminated = false;
        this.capabilities = capabilities || {}; // New: store worker capabilities
        this._initializeWorkers();
    }

    _initializeWorkers() {
        for (let i = 0; i < this.size; i++) {
            const worker = this._createWorker(i);
            if (worker.actualWorker) {
                worker.actualWorker.postMessage({
                    type: 'init',
                    capabilities: this.capabilities
                });
            }
            this.workers.push(worker);
            this.availableWorkers.push(worker);
        }
    }

    _createWorker(id) {
        try {
            let actualWorker = null;
            
            // Create actual workers based on type
            switch (this.workerType) {
                case 'cpu':
                    if (typeof Worker !== 'undefined') {
                        actualWorker = new Worker('./js/workers/cpu-worker-simple.js');
                    }
                    break;
                case 'gpu':
                    if (typeof Worker !== 'undefined') {
                        actualWorker = new Worker('./js/workers/gpu-worker-simple.js');
                    }
                    break;
                case 'webnn':
                    if (typeof Worker !== 'undefined') {
                        actualWorker = new Worker('./js/workers/webnn-worker-simple.js');
                    }
                    break;
                case 'wasm':
                    if (typeof Worker !== 'undefined') {
                        actualWorker = new Worker('./js/workers/wasm-worker-simple.js');
                    }
                    break;
                default:
                    console.warn(`Unknown worker type: ${this.workerType}`);
            }

            const workerWrapper = {
                id: `${this.workerType}_worker_${id}`,
                type: this.workerType,
                busy: false,
                currentTask: null,
                actualWorker: actualWorker,
                capabilities: {},
                terminate: () => {
                    if (actualWorker) {
                        actualWorker.terminate();
                    }
                },
                postMessage: (data) => {
                    if (actualWorker) {
                        actualWorker.postMessage(data);
                    }
                },
                addEventListener: (event, handler) => {
                    if (actualWorker) {
                        actualWorker.addEventListener(event, handler);
                    }
                }
            };

            if (actualWorker) {
                actualWorker.addEventListener('message', (event) => {
                    if (event.data.type === 'ready') {
                        workerWrapper.capabilities = event.data.capabilities;
                        console.log(`Worker ${workerWrapper.id} ready with capabilities:`, workerWrapper.capabilities);
                        this.manager.workerReady();
                    }
                });
                
                actualWorker.addEventListener('error', (error) => {
                    console.error(`Worker ${workerWrapper.id} error:`, error);
                });
                
                actualWorker.addEventListener('messageerror', (error) => {
                    console.error(`Worker ${workerWrapper.id} message error:`, error);
                });
            }

            return workerWrapper;
        } catch (error) {
            console.warn(`Failed to create ${this.workerType} worker:`, error);
            // Fallback to mock worker
            return {
                id: `${this.workerType}_worker_${id}_mock`,
                type: this.workerType,
                busy: false,
                currentTask: null,
                actualWorker: null,
                terminate: () => { /* Mock terminate */ },
                postMessage: (data) => { /* Mock postMessage */ },
                addEventListener: (event, handler) => { /* Mock addEventListener */ }
            };
        }
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
        
        // Worker pools (now includes WASM)
        this.workerPools = {
            cpu: new WorkerPool(options.cpuWorkers || 2, 'cpu', this, options.capabilities),
            gpu: new WorkerPool(options.gpuWorkers || 1, 'gpu', this, options.capabilities),
            webnn: new WorkerPool(options.webnnWorkers || 1, 'webnn', this, options.capabilities),
            wasm: new WorkerPool(options.wasmWorkers || 1, 'wasm', this, options.capabilities)
        };

        // Configuration
        this.maxConcurrentTasks = options.maxConcurrentTasks || 4;
        this.preemptionEnabled = options.preemptionEnabled !== false;
        this.schedulingInterval = options.schedulingInterval || 200; // Reduced for better responsiveness
        this.taskTimeout = options.taskTimeout || 60000; // Default 60 second timeout per task
        this.running = false;
        this.schedulerTimer = null;
        this.taskTimeouts = new Map(); // Track timeouts for running tasks

        // Add a logger for better observability
        this.logger = options.logger || ((message, type) => console.log(`[${type.toUpperCase()}] ${message}`));

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

        this.readyWorkers = 0;
        this.totalWorkers = 0;
        for (const pool of Object.values(this.workerPools)) {
            this.totalWorkers += pool.size;
        }

        this._bindMethods();
        this._startPromise = null;
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
        
        console.log(`Task ${task.id} queued with priority ${effectivePriority}. Heap size: ${this.heap.size()}`);
        
        // Trigger immediate processing if the manager is running
        if (this.running) {
            console.log(`🚀 Task ${task.id} queued, triggering immediate processing`);
            this._processQueue();
        }
        
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
        if (this.running) return Promise.resolve();
        if (this._startPromise) return this._startPromise;

        console.log('Task Manager starting...');

        this._startPromise = new Promise(resolve => {
            console.log('DEBUG: Creating start promise, totalWorkers:', this.totalWorkers);
            
            if (this.totalWorkers === 0) {
                console.log('No workers configured, starting processing immediately.');
                this.startProcessing();
                resolve();
                return;
            }

            // Check if all workers are already ready
            if (this.readyWorkers >= this.totalWorkers) {
                console.log('DEBUG: All workers already ready. Calling startProcessing directly.');
                this.startProcessing();
                resolve();
            } else {
                // Otherwise, wait for all workers to be ready
                const allWorkersReadyHandler = () => {
                console.log('*** DEBUG: allWorkersReadyHandler entered. Minimal. ***');
                this.startProcessing();
                resolve();
            };
                
                console.log('DEBUG: Setting up allWorkersReady event listener');
                this.on('allWorkersReady', allWorkersReadyHandler);
            }
        });

        // This part is crucial. We need to trigger the worker initialization
        // which in turn will lead to the 'allWorkersReady' event.
        // The WorkerPool constructor already sends the 'init' message.

        return this._startPromise;
    }

    workerReady() {
        this.readyWorkers++;
        console.log(`Worker ready. Total ready: ${this.readyWorkers}/${this.totalWorkers}`);
        console.log('DEBUG: workerReady called, current counts:', { ready: this.readyWorkers, total: this.totalWorkers });
        
        if (this.readyWorkers >= this.totalWorkers) {
            console.log('All workers are ready.');
            console.log('DEBUG: About to emit allWorkersReady event');
            this._emit('allWorkersReady');
            console.log('DEBUG: allWorkersReady event emitted');
        }
    }

    startProcessing() {
        console.log(`*** DEBUG: startProcessing - before setting this.running: ${this.running} ***`);
        this.running = true;
        console.log(`*** DEBUG: startProcessing - after setting this.running: ${this.running} ***`);
        console.log('Task Manager started');
        console.log('*** DEBUG: startProcessing entered. ***');
        this.logger('DEBUG: Calling _scheduleNextProcess from startProcessing', 'debug');
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
        
        console.log(`*** DEBUG: _scheduleNextProcess called. Setting timer. ***`);
        this.schedulerTimer = setTimeout(() => {
            console.log(`*** DEBUG: Inside setTimeout callback. this.running: ${this.running} ***`);
            this._processQueue();
            
            // Continue scheduling if we have tasks in any state or workers might finish soon
            if (!this.heap.isEmpty() || this.runningTasks.size > 0 || this.tasks.size > (this.completedTasks.size + this.failedTasks.size)) {
                console.log(`*** DEBUG: Continuing scheduling. Heap size: ${this.heap.size()}, Running tasks: ${this.runningTasks.size}, Total tasks: ${this.tasks.size} ***`);
                this._scheduleNextProcess();
            } else {
                console.log(`*** DEBUG: Stopping scheduler - no tasks in queue and no running tasks ***`);
            }
        }, this.schedulingInterval);
    }

    _processQueue() {
        console.log('*** DEBUG: Entering _processQueue. ***');
        this.logger(`Processing queue. Running tasks: ${this.runningTasks.size}, Max concurrent: ${this.maxConcurrentTasks}, Heap size: ${this.heap.size()}`, 'info');
        // Update priorities for aging
        this._updateTaskPriorities();
        
        // Defensive loop: catch heap errors
        try {
            // Process tasks while we have available workers and tasks
            let consecutiveSkips = 0;
            const maxSkips = this.heap.size(); // Prevent infinite loops
            
            while (this.runningTasks.size < this.maxConcurrentTasks && !this.heap.isEmpty() && consecutiveSkips < maxSkips) {
                console.log(`🔄 Processing loop - Running: ${this.runningTasks.size}, Max: ${this.maxConcurrentTasks}, Heap size: ${this.heap.size()}`);
                const next = this.heap.peek();
                if (!next) {
                    this.logger('Heap is empty, breaking loop.', 'debug');
                    break;
                }
                
                const task = next.value;
                console.log(`🔍 Evaluating task ${task.id} (${task.job.type}) with priority ${task.getEffectivePriority()}`);
                
                // Check if task is ready to run
                if (!task.isReadyToRun()) {
                    console.log(`⏸️ Task ${task.id} is not ready to run (status: ${task.status}, scheduled: ${new Date(task.scheduledTime).toLocaleTimeString()})`);
                    consecutiveSkips++;
                    
                    // If all tasks in heap are not ready, break to avoid infinite loop
                    if (consecutiveSkips >= maxSkips) {
                        console.log(`⚠️ All tasks in heap are not ready to run, breaking processing loop`);
                        break;
                    }
                    
                    // Remove from heap and try next task
                    this.heap.extractMin();
                    this.taskNodes.delete(task.id);
                    
                    // Re-insert the task back into the heap with a small delay
                    // This allows other ready tasks to be processed first
                    setTimeout(() => {
                        if (this.tasks.has(task.id) && task.status === 'queued' && this.running) {
                            console.log(`♻️ Re-inserting task ${task.id} back into queue`);
                            const newHeapNode = this.heap.insert(task.getEffectivePriority(), task);
                            this.taskNodes.set(task.id, newHeapNode);
                        }
                    }, 10); // Small delay to allow other processing
                    continue;
                } else {
                    consecutiveSkips = 0; // Reset skip counter when we find a ready task
                }

                console.log(`✅ Task ${task.id} is ready to run, looking for worker...`);
                // Get appropriate worker
                const worker = this._getAvailableWorker(task);
                if (!worker) {
                    console.log(`❌ No available worker for task ${task.id}. Attempting preemption.`);
                    // Try preemption if enabled
                    if (this.preemptionEnabled) {
                        const preemptedWorker = this._attemptPreemption(task);
                        if (preemptedWorker) {
                            this.logger(`Preempted worker ${preemptedWorker.id} for task ${task.id}`, 'info');
                            this._runTask(task, preemptedWorker);
                        } else {
                            this.logger(`No suitable worker to preempt for task ${task.id}`, 'debug');
                        }
                    }
                    break; // No workers available
                }

                console.log(`🎯 Found worker ${worker.id} for task ${task.id}, starting execution...`);
                // Remove from heap and run
                this.heap.extractMin();
                this.taskNodes.delete(task.id);
                this._runTask(task, worker);
            }
        } catch (err) {
            this.logger(`Error in _processQueue: ${err}`, 'error');
        }

        // Check if queue is empty
        if (this.heap.isEmpty() && this.runningTasks.size === 0) {
            this.logger('Queue is empty and no tasks are running.', 'info');
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
        const requirements = task.resourceRequirements || {};
        console.log(`🔍 Searching for worker for task ${task.id} (${task.job.type}) with requirements:`, requirements);

        const potentialPools = [];
        if (requirements.gpu) potentialPools.push(this.workerPools.gpu);
        if (requirements.webnn) potentialPools.push(this.workerPools.webnn);
        if (requirements.wasm) potentialPools.push(this.workerPools.wasm);
        potentialPools.push(this.workerPools.cpu); // Always consider CPU as a fallback

        console.log(`🔍 Potential pools for task ${task.id}:`, potentialPools.map(p => p.workerType));

        for (const pool of potentialPools) {
            console.log(`🔍 Checking pool: ${pool.workerType}. Available workers: ${pool.availableWorkers.length}`);
            for (const worker of pool.availableWorkers) {
                console.log(`🔍 Attempting to match task ${task.id} with worker ${worker.id} (type: ${worker.type})`);
                if (this._workerSatisfiesRequirements(worker, requirements)) {
                    console.log(`✅ Task ${task.id} (${task.job.type}) assigned to ${worker.type} worker: ${worker.id}`);
                    return worker;
                } else {
                    console.log(`❌ Worker ${worker.id} does not satisfy requirements for task ${task.id}`);
                }
            }
        }

        this.logger(`No available worker for task ${task.id} (${task.job.type})`, 'warn');
        return null;
    }

    _workerSatisfiesRequirements(worker, requirements) {
        if (!worker.capabilities) {
            console.log(`❌ Worker ${worker.id} has no capabilities object. Cannot satisfy requirements.`);
            return false;
        }
        console.log(`🔍 Checking worker ${worker.id} capabilities:`, worker.capabilities, 'against requirements:', requirements);

        if (requirements.gpu && !worker.capabilities.webgpu) {
            console.log(`❌ Worker ${worker.id} fails GPU requirement (needs webgpu:true, has webgpu:${worker.capabilities.webgpu})`);
            return false;
        }
        if (requirements.webnn && !worker.capabilities.webnn) {
            console.log(`❌ Worker ${worker.id} fails WebNN requirement (needs webnn:true, has webnn:${worker.capabilities.webnn})`);
            return false;
        }
        if (requirements.onnx && !worker.capabilities.onnx) {
            console.log(`❌ Worker ${worker.id} fails ONNX requirement (needs onnx:true, has onnx:${worker.capabilities.onnx})`);
            return false;
        }
        // If a WASM job, check if worker has WASM capability
        if (requirements.wasm && !worker.capabilities.wasm) {
            console.log(`❌ Worker ${worker.id} fails WASM requirement (needs wasm:true, has wasm:${worker.capabilities.wasm})`);
            return false;
        }
        console.log(`✅ Worker ${worker.id} satisfies all requirements.`);
        return true;
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
        
        // Set up task timeout
        const timeoutId = setTimeout(() => {
            console.log(`⏰ Task ${task.id} timed out after ${this.taskTimeout}ms`);
            this._handleTaskTimeout(task);
        }, this.taskTimeout);
        this.taskTimeouts.set(task.id, timeoutId);
        
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

            let result;
            const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('Task execution timeout')), task.timeout));

            if (worker.actualWorker) {
                result = await Promise.race([
                    this._executeTaskOnRealWorker(task, worker, progressCallback, shouldStop),
                    timeoutPromise
                ]);
            } else {
                // Fallback to job's execute method (for mock jobs)
                result = await Promise.race([
                    task.job.execute(progressCallback, shouldStop),
                    timeoutPromise
                ]);
            }
            
            if (task.status === 'running') {
                task.status = 'completed';
                task.endTime = Date.now();
                task.result = result;
                
                // Clear timeout
                if (this.taskTimeouts.has(task.id)) {
                    clearTimeout(this.taskTimeouts.get(task.id));
                    this.taskTimeouts.delete(task.id);
                }
                
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
                
                // Clear timeout
                if (this.taskTimeouts.has(task.id)) {
                    clearTimeout(this.taskTimeouts.get(task.id));
                    this.taskTimeouts.delete(task.id);
                }
                
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

    _handleTaskTimeout(task) {
        console.log(`⏰ Handling timeout for task ${task.id}`);
        
        if (task.status === 'running') {
            task.status = 'failed';
            task.endTime = Date.now();
            task.error = new Error(`Task timed out after ${this.taskTimeout}ms`);
            
            // Clear the timeout
            if (this.taskTimeouts.has(task.id)) {
                clearTimeout(this.taskTimeouts.get(task.id));
                this.taskTimeouts.delete(task.id);
            }
            
            // Release worker and clean up
            if (task.worker) {
                this._releaseWorker(task.worker);
            }
            
            this.runningTasks.delete(task.id);
            this.failedTasks.set(task.id, task);
            this.stats.tasksFailed++;
            
            if (task.callbacks.onError) {
                task.callbacks.onError(task.error, task);
            }
            this._emit('taskFailed', task);
            
            // Trigger immediate processing to handle queued tasks
            if (this.running) {
                this._processQueue();
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
        } else if (worker.type === 'wasm') {
            this.workerPools.wasm.assignWorker(worker, task);
        }
    }

    _releaseWorker(worker) {
        if (worker.type === 'cpu') {
            this.workerPools.cpu.releaseWorker(worker);
        } else if (worker.type === 'gpu') {
            this.workerPools.gpu.releaseWorker(worker);
        } else if (worker.type === 'webnn') {
            this.workerPools.webnn.releaseWorker(worker);
        } else if (worker.type === 'wasm') {
            this.workerPools.wasm.releaseWorker(worker);
        }
    }

    async _executeTaskOnRealWorker(task, worker, progressCallback, shouldStop) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Task execution timeout'));
            }, task.maxExecutionTime || 30000);

            // Set up worker message handlers
            const messageHandler = (event) => {
                const { type, taskId, result, error, progress, stats } = event.data;
                
                if (taskId !== task.id) return; // Ignore messages for other tasks

                console.log(`[TaskManager] Worker ${worker.id} message:`, event.data);
                this.logger(`[Worker ${worker.id}] ${JSON.stringify(event.data)}`, 'worker');
                
                switch (type) {
                    case 'completed':
                        clearTimeout(timeout);
                        worker.actualWorker.removeEventListener('message', messageHandler);
                        console.log(`[TaskManager] Task ${task.id} completed successfully`);
                        resolve(result);
                        break;
                    case 'error':
                        clearTimeout(timeout);
                        worker.actualWorker.removeEventListener('message', messageHandler);
                        console.error(`[TaskManager] Task ${task.id} failed:`, error);
                        reject(new Error(error));
                        break;
                    case 'progress':
                        console.log(`[TaskManager] Task ${task.id} progress: ${progress}%`);
                        if (progressCallback) {
                            progressCallback(progress, stats);
                        }
                        // Check if task should be stopped
                        if (shouldStop()) {
                            worker.actualWorker.postMessage({
                                type: 'cancel',
                                taskId: task.id
                            });
                        }
                        break;
                    case 'cancelled':
                        clearTimeout(timeout);
                        worker.actualWorker.removeEventListener('message', messageHandler);
                        console.log(`[TaskManager] Task ${task.id} was cancelled`);
                        resolve({ cancelled: true });
                        break;
                    default:
                        console.warn(`[TaskManager] Unknown message type from worker:`, type);
                }
            };

            worker.actualWorker.addEventListener('message', messageHandler);
            
            // Add error handlers for the worker
            const errorHandler = (error) => {
                console.error(`[TaskManager] Worker ${worker.id} error during task execution:`, error);
                clearTimeout(timeout);
                worker.actualWorker.removeEventListener('message', messageHandler);
                worker.actualWorker.removeEventListener('error', errorHandler);
                reject(new Error(`Worker error: ${error.message || 'Unknown worker error'}`));
            };
            
            worker.actualWorker.addEventListener('error', errorHandler);

            // Send task to worker
            console.log(`[TaskManager] Sending task ${task.id} to worker ${worker.id}`);
            console.log(`[TaskManager] Task data:`, {
                taskId: task.id,
                jobType: task.job.type,
                useRealInference: task.job.useRealInference,
                backend: task.job.backend,
                duration: task.job.duration,
                complexity: task.job.complexity
            });
            worker.actualWorker.postMessage({
                type: 'execute',
                data: {
                    taskId: task.id,
                    jobType: task.job.type,
                    duration: task.job.duration,
                    complexity: task.job.complexity,
                    resourceRequirements: task.job.resourceRequirements,
                    ...task.job
                }
            });
        });
    }

    // Event system
    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
        console.log(`DEBUG: Added event handler for '${event}', total handlers: ${this.eventHandlers[event].length}`);
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
        console.log(`DEBUG: Attempting to emit '${event}' event, handlers available: ${this.eventHandlers[event] ? this.eventHandlers[event].length : 0}`);
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach((handler, index) => {
                console.log(`DEBUG: Calling handler ${index + 1} for '${event}' event`);
                try {
                    handler(data);
                    console.log(`DEBUG: Handler ${index + 1} for '${event}' completed successfully`);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        } else {
            console.warn(`DEBUG: No handlers registered for event '${event}'`);
        }
    }

    /**
     * Get comprehensive statistics
     */
    getStats() {
        // Defensive stats reporting
        const queueSize = typeof this.heap.size === 'function' ? this.heap.size() : 0;
        const runningCount = typeof this.runningTasks.size === 'number' ? this.runningTasks.size : 0;
        const completedCount = typeof this.completedTasks.size === 'number' ? this.completedTasks.size : 0;
        const failedCount = typeof this.failedTasks.size === 'number' ? this.failedTasks.size : 0;
        
        return {
            queue: {
                size: queueSize,
                running: runningCount,
                completed: completedCount,
                failed: failedCount
            },
            workers: {
                cpu: this.workerPools.cpu.getStats(),
                gpu: this.workerPools.gpu.getStats(),
                webnn: this.workerPools.webnn.getStats(),
                wasm: this.workerPools.wasm.getStats()
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
