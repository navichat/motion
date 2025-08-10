/* eslint-disable */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(root);
  } else {
    if (!root.TaskScheduler) {
      var exp = factory(root);
      root.TaskScheduler = exp.TaskScheduler;
      try { root.TaskScheduler.TaskScheduler = exp.TaskScheduler; } catch (e) {}
      root.TaskSchedulerCtor = exp.TaskScheduler;
    }
  }
})(typeof self !== 'undefined' ? self : (typeof global !== 'undefined' ? global : this), function (root) {
/**
 * TaskScheduler built on FibonacciHeap with chunked execution and preemption hooks.
 */
let FibonacciHeapCtor = null;
try {
  if (typeof module === 'object' && module && module.exports) {
    const mod = require('./FibonacciHeap');
    FibonacciHeapCtor = mod.FibonacciHeap || mod;
  } else if (root && root.FibonacciHeap) {
    FibonacciHeapCtor = root.FibonacciHeap;
  }
} catch (e) {
  // fallthrough
}
if (!FibonacciHeapCtor) {
  throw new Error('[TaskScheduler] Missing FibonacciHeap dependency');
}
const FibonacciHeap = FibonacciHeapCtor;

class TaskScheduler {
  constructor(opts = {}) {
    this.heap = new FibonacciHeap();
    this.nodes = new Map(); // taskId -> heap node
    this.running = new Map(); // trackId -> taskId
    this.onChunk = opts.onChunk || (() => {});
    this.quantumMs = opts.quantumMs || 200; // default chunk quantum
    this.now = () => performance.now();
    this.taskById = new Map();
  this.iterators = new Map(); // taskId -> async iterator
  }

  submit(task) {
    const key = this._priorityKey(task);
    const node = this.heap.insert(key, task);
    this.nodes.set(task.id, node);
    this.taskById.set(task.id, task);
  }

  setPriority(taskId, newPriority) {
    const node = this.nodes.get(taskId);
    if (!node) return;
    this.heap.decreaseKey(node, newPriority);
    node.value.priority = newPriority;
  }

  preempt(target, opts = {}) {
    // target can be taskId or trackId
    const fadeOutMs = opts.fadeOutMs ?? 120;
    // mark matching running tasks for abort
    for (const [trackId, taskId] of this.running) {
      if (taskId === target || trackId === target) {
        const task = this.taskById.get(taskId);
        if (task && task.abort) task.abort.abort();
        this.running.delete(trackId);
        // caller will manage timeline fade outs/in via onChunk consumer
      }
    }
  }

  async runOnce(deadlineMs = 8) {
    const start = this.now();
    while (!this.heap.isEmpty() && (this.now() - start) < deadlineMs) {
      const node = this.heap.extractMin();
      if (!node) break;
      const task = node.value;
      const trackId = task.trackId;

      // Book running task in its track
      this.running.set(trackId, task.id);

      // Execute one quantum worth of work
      try {
        const endTime = this.now() + this.quantumMs;
        // Task.run is an async generator producing frames by chunk
        let iterator = this.iterators.get(task.id);
        if (!iterator) {
          iterator = task.run({ endTime, quantumMs: this.quantumMs });
          this.iterators.set(task.id, iterator);
        }
        const result = await iterator.next();
        if (!result.done) {
          // yielded a chunk
          const chunk = result.value;
          this.onChunk(task, chunk);
          // requeue with updated dynamic priority (if provided)
          const newKey = this._priorityKey(task);
          const newNode = this.heap.insert(newKey, task);
          this.nodes.set(task.id, newNode);
        } else {
          // task finished, clear running marker
          this.running.delete(trackId);
          this.iterators.delete(task.id);
        }
      } catch (e) {
        console.warn('[TaskScheduler] task error', task.id, e);
        this.running.delete(trackId);
        this.iterators.delete(task.id);
      }
    }
  }

  _priorityKey(task) {
    // Lower key = higher priority
    const base = task.priority ?? 10;
    const deadlineSlack = task.deadline ? Math.max(0, task.deadline - Date.now()) / 1000 : 999;
    const backendPenalty = task.backend === 'webgpu' ? 0 : task.backend === 'webnn' ? 0.5 : 1;
    return base + backendPenalty + (deadlineSlack * 0.001);
  }
}

// Return API for UMD
return { TaskScheduler };
});

/* eslint-enable */
