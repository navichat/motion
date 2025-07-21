# Task Management Engine

A sophisticated task scheduling system built for web browsers that uses a Fibonacci heap for efficient priority queue operations. This system coordinates CPU, GPU (WebGPU), and WebNN workers to execute tasks with support for preemption, priority scheduling, and real-time monitoring.

## 🌟 Features

### Core Components

- **🔧 Fibonacci Heap Implementation**: O(1) insert and decrease-key operations for efficient priority scheduling
- **🎮 Mock GPU Jobs**: Simulated WebGPU/WebNN workloads for testing (JobA, JobB, JobC)
- **⚙️ Task Manager**: Advanced scheduling engine with preemption and worker pool management
- **👥 Worker Pools**: Coordinated CPU, GPU, and WebNN worker threads
- **📊 Real-time Monitoring**: Live statistics and queue visualization

### Advanced Features

- **Priority-based Scheduling**: Lower numbers = higher priority with aging support
- **Task Preemption**: Interrupt lower-priority tasks for urgent work
- **Resource Allocation**: Memory and compute resource tracking
- **Interruptible Tasks**: Support for cancellation and resumption
- **Comprehensive Testing**: Full test suite with stress testing

## 🚀 Quick Start

### 1. Open the Demo

Open `task-manager-demo.html` in a modern web browser that supports:
- WebGPU (optional, fallback provided)
- Web Workers
- ES6+ JavaScript features

### 2. Run Tests

Click **"Run Full Test Suite"** to verify all components:
- ✅ Fibonacci Heap operations
- ✅ Mock GPU job execution
- ✅ Task manager basics
- ✅ Priority scheduling
- ✅ Queue fill/empty demo

### 3. Interactive Demo

Click **"Interactive Demo"** and use the browser console:

```javascript
// Add tasks with random priority
addTask();

// Add high priority task
addHighPriorityTask();

// Check system stats
showStats();

// Stop the interactive demo
stopInteractive();
```

## 📁 File Structure

```
js/
├── FibonacciHeap.js         # Fibonacci heap implementation
├── MockGPUJobs.js           # Simulated GPU workloads
├── TaskManager.js           # Main scheduling engine
├── TaskManagerTestSuite.js  # Comprehensive test suite
└── workers/
    ├── cpu-worker.js        # CPU worker implementation
    ├── gpu-worker.js        # WebGPU worker implementation
    └── webnn-worker.js      # WebNN worker implementation

task-manager-demo.html       # Interactive demo page
```

## 🔧 API Reference

### TaskManager

```javascript
const manager = new TaskManager({
    cpuWorkers: 2,           // Number of CPU workers
    gpuWorkers: 1,           // Number of GPU workers  
    webnnWorkers: 1,         // Number of WebNN workers
    maxConcurrentTasks: 4,   // Max parallel tasks
    preemptionEnabled: true, // Allow task preemption
    schedulingInterval: 100  // Scheduling loop interval (ms)
});

// Schedule a task
const taskId = manager.scheduleTask(job, priority, scheduledTime, options);

// Start/stop processing
manager.start();
manager.stop();

// Event handlers
manager.on('taskCompleted', (task) => console.log('Done:', task.id));
manager.on('taskStarted', (task) => console.log('Started:', task.id));
manager.on('taskPreempted', (task) => console.log('Preempted:', task.id));

// Get statistics
const stats = manager.getStats();
```

### MockGPUJobFactory

```javascript
// Create different job types
const jobA = MockGPUJobFactory.createJobA(complexity);  // Matrix operations
const jobB = MockGPUJobFactory.createJobB(complexity);  // Neural networks
const jobC = MockGPUJobFactory.createJobC(complexity);  // Media processing

// Create random job
const randomJob = MockGPUJobFactory.createRandomJob();

// Create batch of jobs
const jobs = MockGPUJobFactory.createJobBatch(10);
```

### FibonacciHeap

```javascript
const heap = new FibonacciHeap();

// Basic operations
const node = heap.insert(priority, value);
const min = heap.extractMin();
heap.decreaseKey(node, newPriority);
heap.delete(node);

// Utilities
const isEmpty = heap.isEmpty();
const size = heap.size();
const stats = heap.getStats();
```

## 🎯 Use Cases

This task management engine is designed for applications that need to coordinate multiple types of computational work:

### Real-world Applications
- **🎤 Voice Activity Detection**: Real-time audio processing
- **🗣️ Text-to-Speech (Kokoro)**: Audio synthesis workloads
- **👂 Speech-to-Text (Whisper)**: Audio transcription tasks
- **🤖 Small LLM Inference**: Language model operations
- **🎭 RSMT Motion**: Real-time stylized motion transition
- **🏃 DeepMimic**: Character animation and physics
- **😊 FaceFormer**: Facial expression synthesis
- **🎵 Audio2Gesture**: Audio-driven gesture generation

### Benefits
- **Predictable Performance**: Queue fills and empties in a controlled manner
- **Efficient Resource Use**: Smart allocation across CPU/GPU/WebNN
- **Responsive System**: High-priority tasks can preempt lower-priority work
- **Scalable Architecture**: Easy to add new worker types and job categories

## 🧪 Testing

The system includes comprehensive tests that verify:

1. **Fibonacci Heap Correctness**: Priority ordering and heap properties
2. **Mock Job Execution**: Progress tracking and completion
3. **Priority Scheduling**: High-priority tasks execute first
4. **Queue Management**: Predictable fill/empty behavior
5. **Worker Coordination**: Proper resource allocation
6. **Stress Testing**: High-volume task processing

### Running Tests

```javascript
// Browser console
const testSuite = new TaskManagerTestSuite();
await testSuite.runAllTests();

// Or use the demo page
window.runTaskManagerTests();
```

## 🔮 Next Steps

After validating the core scheduling system, you can proceed to implement:

1. **Real Worker Threads**: Replace mock workers with actual WebGPU/WebNN implementations
2. **Model Integration**: Add actual AI models (Whisper, Kokoro, etc.)
3. **Audio Pipeline**: Integrate with Web Audio API for real-time processing
4. **Performance Optimization**: Fine-tune scheduling parameters
5. **Production Deployment**: Add error handling and monitoring

## 🤝 Architecture Notes

### Why Fibonacci Heap?
- **O(1) insert**: Critical for real-time task scheduling
- **O(1) decrease-key**: Essential for priority adjustments and aging
- **O(log n) extract-min**: Acceptable for task dequeuing
- **Efficient merging**: Supports advanced scheduling algorithms

### Worker Pool Design
- **Type-specific pools**: CPU, GPU, and WebNN workers are separate
- **Resource awareness**: Tasks specify their computational requirements
- **Graceful fallback**: GPU/WebNN tasks can fall back to CPU simulation

### Preemption Strategy
- **Priority-based**: Higher priority tasks can interrupt lower priority ones
- **Voluntary cooperation**: Tasks check for interruption signals
- **State preservation**: Preempted tasks are re-queued with adjusted priority

## 📊 Performance Characteristics

- **Scheduling Overhead**: ~100ms intervals, configurable
- **Task Throughput**: Scales with worker pool size
- **Memory Usage**: O(n) where n = number of queued tasks
- **Priority Updates**: O(1) amortized via Fibonacci heap

## 🛠️ Browser Compatibility

- **Chrome**: Full WebGPU support (recommended)
- **Firefox**: CPU/WebNN fallback
- **Safari**: CPU fallback
- **Edge**: Full WebGPU support

## 📝 License

This task management engine is designed for research and development use. Adapt the components as needed for your specific application requirements.

---

**Ready to coordinate your computational workloads? Start with the demo and build your way up to production!** 🚀
