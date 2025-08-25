# Project Structure and Component Guide

Note: This document predates the latest reorganization. For the authoritative, up-to-date structure, see docs/STRUCTURE.md. This file is kept for historical reference.

## Overview

This document provides a comprehensive guide to the reorganized WebNN/WebGPU/WASM powered avatar system structure. The project has been systematically organized to enable component-based development, testing, and integration.

## Directory Structure

### Source Code Organization (`src/`)

#### Core System (`src/core/`)
**Purpose**: Fundamental system components that orchestrate the entire avatar platform.

```
src/core/
├── TaskManager.js              # Central task orchestration and job queue management
├── FibonacciHeap.js           # Priority queue implementation for task scheduling
└── SystemPerformanceAnalyzer.js # System performance monitoring and optimization
```

**Key Responsibilities:**
- **TaskManager**: Coordinates AI model execution, manages worker pools, handles job prioritization
- **FibonacciHeap**: Efficient priority queue for task scheduling and resource allocation
- **SystemPerformanceAnalyzer**: Monitors system health, memory usage, and performance metrics

#### AI Components (`src/ai/`)
**Purpose**: AI model implementations, job definitions, and worker management.

```
src/ai/
├── jobs/                       # AI model job definitions
│   ├── AIModelJobs.js         # Base AI job classes and factory patterns
│   ├── RealJobFactory.js      # Factory for creating real AI inference jobs
│   └── KNNJobs.js             # K-nearest neighbor and vector search jobs
└── workers/                   # AI model worker implementations
    ├── aiWorker.js            # Generic AI worker base class
    ├── conversationWorkerWorking.js # Conversation AI worker implementation
    ├── TinyLlamaWorker.js     # TinyLlama language model worker
    ├── WhisperWorker.js       # Whisper speech recognition worker
    └── RSMTWorker.js          # Real-time Stylized Motion Transition worker
```

**Model Categories:**
- **Language Models**: TinyLlama, DiabloGPT for conversational AI
- **Audio Processing**: Whisper STT, Kokoro TTS, VAD detection
- **Motion Generation**: RSMT, DeepMimic, Audio2Gesture, FaceFormer
- **Vector Search**: CloseVector, HNSW, UnifiedKNN for semantic matching

#### Components (`src/components/`)
**Purpose**: UI-free building blocks for animation, avatar control, and navigation.

```
src/components/
├── animation/
│   ├── AnimationBlender.js     # Multi-animation blending
│   ├── AnimationSync.js        # Timeline/frame sync helpers
│   ├── timeline/               # Animation timeline system
│   │   └── BVHTimeline.js
│   └── vrm/                    # VRM avatar system (25+ files)
│       ├── VRMBVHAdapter.js    # Core VRM + BVH integration
│       ├── conversation/       # Conversation/UX helpers for VRM
│       └── diagnostics/        # Debug and validation utilities
├── conversation/               # Conversation UI/logic helpers
└── pathfinding/                # Navigation/path utilities
```

**Capabilities:**
- **VRM Support**: Humanoid bone mapping and VRM integration
- **Motion Capture**: BVH parsing and timeline-driven playback
- **Animation Blending**: Smooth transitions and layered control
- **Diagnostics**: Built-in validation and debug utilities

#### Audio Processing (`src/audio/`)
**Purpose**: Speech synthesis, recognition, and audio manipulation.

```
src/audio/
├── tts/                      # Text-to-speech systems
│   ├── KokoroTTS.js         # Kokoro TTS implementation
│   └── SpeechT5TTS.js       # SpeechT5 TTS implementation
├── stt/                     # Speech-to-text systems
│   ├── WhisperSTT.js        # Whisper speech recognition
│   └── VADProcessor.js      # Voice activity detection
└── processors/              # Audio processing utilities
    ├── audioProcessor.js    # Base audio processing worklet
    └── AudioAnalyzer.js     # Audio feature analysis
```

**Features:**
- **High-Quality TTS**: Multiple synthesis engines for natural speech
- **Real-time STT**: Continuous speech recognition with voice activity detection
- **Audio Analysis**: Feature extraction for gesture generation and mood detection
- **WebAudio Integration**: Optimized for browser-based audio processing

#### Compute Backends (`src/compute/`)
**Purpose**: Hardware-accelerated computation across different platforms.

```
src/compute/
└── backends/                # Compute backend implementations
    ├── WebGPUBackend.js     # GPU-accelerated computation
    ├── WebNNBackend.js      # Neural network optimization
    └── WASMBackend.js       # WebAssembly fallback
```

**Backend Selection Strategy:**
- **WebGPU**: Preferred for large parallel workloads and GPU-available systems
- **WebNN**: Optimal for neural network inference with hardware acceleration
- **WASM**: Universal fallback ensuring compatibility across all browsers

#### Workers (`src/workers/`)
**Purpose**: Web worker implementations for background processing.

```
src/workers/
├── conversationWorkerWorking.js # Real-time conversation processing
├── audioWorkletProcessor.js     # Audio processing worklet
└── computeWorker.js             # General compute tasks
```

### Testing Infrastructure (`tests/`)

#### Unit Tests (`tests/unit/`)
**Purpose**: Component-specific validation and isolation testing.

```
tests/unit/
├── ai/                      # AI system component tests
│   └── ai-model-jobs.spec.js
├── avatar/                  # Avatar system component tests
│   └── avatar-animation.spec.js
├── audio/                   # Audio processing component tests
│   └── audio-processing.spec.js
├── compute/                 # Compute backend component tests
│   └── compute-backend.spec.js
├── motion/                  # Motion processing component tests
│   └── motion-processing.spec.js
└── system/                  # System integration component tests
    └── system-integration.spec.js
```

#### Integration Tests (`tests/integration/`)
**Purpose**: Cross-component validation and end-to-end workflows.

```
tests/integration/
├── e2e/                     # End-to-end workflow tests
│   └── capture-ai-results.spec.js
└── master-test-suite.spec.js # Comprehensive system validation
```

#### Manual Tests (`tests/manual/`)
**Purpose**: Interactive testing interfaces for human validation.

```
tests/manual/
├── test_conversation.html   # Real-time conversation testing
├── test_script.html        # Basic functionality validation
└── three_test_simple.html  # 3D rendering verification
```

### Demonstration Applications (`demos/`)

#### Feature-Specific Demos (`demos/`)
**Purpose**: Showcase individual system capabilities and provide development examples.

```
demos/
├── ai-inference/           # AI model demonstrations
│   └── task-manager-demo.html
├── avatar-animation/       # Avatar animation showcases
│   ├── vrm_test_animation_conversation.html
│   ├── vrm_ai_conversation_test.html
│   ├── bvh_timeline_demo.html
│   └── vrm_test_mesh_animation.html
├── audio-processing/       # Audio processing examples
│   ├── automated_kokoro_test.html
│   ├── verify_tts_models.html
│   └── test_tts_models.html
└── motion-capture/         # Motion capture demonstrations
    └── bvh_processing_demo.html
```

### Configuration and Tools

#### Configuration (`config/`)
```
config/
└── import-map.js           # Module path mapping and dependency management
```

#### Development Tools (`tools/`)
```
tools/
├── performance/            # Performance analysis utilities
└── debugging/              # Debugging and diagnostic tools
```

#### Static Assets (`assets/`)
```
assets/
├── models/                 # 3D models and VRM character files
├── audio/                  # Audio samples and test data
├── libraries/              # Third-party libraries (Three.js, etc.)
└── data/                   # Test datasets and reference files
```

## Component Integration Patterns

### Data Flow Architecture

```
Audio Input → VAD → Speech Recognition → Language Model → Response Generation
     ↓                                                           ↓
Audio Visualization                                        TTS Synthesis
     ↓                                                           ↓
Gesture Generation ← Motion Analysis ← BVH Processing ← Audio2Gesture
     ↓                                                           ↓
Avatar Animation → VRM Rendering → Display Output → Audio Output
```

### Compute Backend Selection

```javascript
// Example backend selection logic
class ComputeManager {
  async selectOptimalBackend(workload) {
    const capabilities = await this.detectCapabilities();
    
    // Neural network inference - prefer WebNN
    if (workload.type === 'neural-network' && capabilities.webnn) {
      return new WebNNBackend();
    }
    
    // Large parallel tasks - prefer WebGPU
    if (workload.parallel && workload.size > 1000 && capabilities.webgpu) {
      return new WebGPUBackend();
    }
    
    // Universal fallback - WASM
    return new WASMBackend();
  }
}
```

### Task Management Flow

```javascript
// Task lifecycle in TaskManager
const taskFlow = {
  1: 'Job Creation → Factory Pattern',
  2: 'Priority Assignment → Fibonacci Heap',
  3: 'Worker Selection → Capability Matching',
  4: 'Execution → Backend Selection',
  5: 'Result Processing → Data Pipeline',
  6: 'Cleanup → Resource Management'
};
```

## Development Workflow

### Adding New Components

#### 1. Create Component Files
```bash
# Create new AI model
touch src/ai/jobs/NewModelJob.js
touch src/ai/workers/NewModelWorker.js

# Create corresponding tests
touch tests/unit/ai/new-model-tests.spec.js
```

#### 2. Update Import Map
```javascript
// config/import-map.js
export const ImportMap = {
  ai: {
    NewModelJob: '/src/ai/jobs/NewModelJob.js',
    NewModelWorker: '/src/ai/workers/NewModelWorker.js'
  }
};
```

#### 3. Implement Component
```javascript
// src/ai/jobs/NewModelJob.js
export class NewModelJob {
  constructor(config) {
    this.type = 'new-model';
    this.config = config;
  }
  
  async execute(worker) {
    return await worker.processNewModel(this.config);
  }
}
```

#### 4. Create Tests
```javascript
// tests/unit/ai/new-model-tests.spec.js
test('should create new model job', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const job = new NewModelJob({ param: 'value' });
    return { created: !!job, type: job.type };
  });
  
  expect(result.created).toBe(true);
  expect(result.type).toBe('new-model');
});
```

### Testing Strategy

#### Component Testing
```bash
# Test individual components
npx playwright test tests/unit/ai/
npx playwright test tests/unit/avatar/
npx playwright test tests/unit/audio/

# Test component integration
npx playwright test tests/unit/system/
```

#### Integration Testing
```bash
# Full system validation
npx playwright test tests/integration/master-test-suite.spec.js

# End-to-end workflows
npx playwright test tests/integration/e2e/
```

### Performance Optimization

#### Component-Level Optimization
- **Lazy Loading**: Load components only when needed
- **Memory Management**: Implement proper cleanup and garbage collection
- **Caching**: Cache frequently used models and data
- **Worker Pools**: Reuse workers for similar tasks

#### System-Level Optimization
- **Backend Selection**: Choose optimal compute backend for each task
- **Load Balancing**: Distribute work across available resources
- **Priority Scheduling**: Use Fibonacci heap for efficient task prioritization
- **Resource Monitoring**: Track system performance and adjust accordingly

## Troubleshooting

### Common Issues

#### Import Path Errors
```javascript
// Check import map configuration
import { ImportMap } from '/config/import-map.js';
console.log('Available paths:', ImportMap);
```

#### Backend Compatibility
```javascript
// Test backend availability
const webgpuAvailable = 'gpu' in navigator;
const webnnAvailable = 'ml' in navigator;
const wasmAvailable = typeof WebAssembly !== 'undefined';
```

#### Performance Issues
```javascript
// Monitor system performance
const monitor = new SystemPerformanceAnalyzer();
monitor.startMonitoring();
```

### Debugging Tools

#### Component Inspector
```javascript
// Inspect component state
class ComponentInspector {
  static inspect(component) {
    return {
      type: component.constructor.name,
      properties: Object.keys(component),
      methods: Object.getOwnPropertyNames(Object.getPrototypeOf(component))
    };
  }
}
```

#### Performance Profiler
```javascript
// Profile component performance
class PerformanceProfiler {
  static async profile(operation, iterations = 10) {
    const times = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await operation();
      times.push(performance.now() - start);
    }
    return {
      average: times.reduce((a, b) => a + b) / times.length,
      min: Math.min(...times),
      max: Math.max(...times)
    };
  }
}
```

## Migration Guide

### From Old Structure
The reorganization maintains backward compatibility where possible. Key changes:

1. **File Locations**: Use import map to find new paths
2. **Module Loading**: Import from `src/` directories instead of `js/`
3. **Testing**: Use new test structure in `tests/` directory
4. **Demos**: Access demos from `demos/` subdirectories

### Updating Existing Code
```javascript
// Old import
import TaskManager from './js/TaskManager.js';

// New import
import { TaskManager } from './src/core/TaskManager.js';
```

For more information, see the main [REORGANIZATION_STATUS.md](../REORGANIZATION_STATUS.md) documentation.
