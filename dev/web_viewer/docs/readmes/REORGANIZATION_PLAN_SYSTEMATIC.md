# WebViewer Reorganization Plan - Component-Based Testing Architecture

## Overview
This plan reorganizes the dev/web_viewer folder to enable systematic testing of WebNN/WebGPU/WASM avatar components individually before integration.

## Current State Analysis
- **Mixed Organization**: Some files in organized `src/` and `tests/` structure, many legacy files in root
- **Scattered Tests**: Test files (.spec.js, .html) spread throughout root directory
- **Component Confusion**: Related files not grouped together
- **Import Dependencies**: Need to update import paths after reorganization

## Target Architecture

### Primary Structure
```
dev/web_viewer/
├── src/                          # Source code organized by component type
│   ├── core/                     # Core system components
│   │   ├── TaskManager.js
│   │   ├── SystemMonitor.js
│   │   └── PerformanceAnalyzer.js
│   ├── ai/                       # AI model components
│   │   ├── models/               # AI model implementations
│   │   ├── workers/              # AI worker threads
│   │   └── jobs/                 # Job definitions
│   ├── avatar/                   # Avatar system components
│   │   ├── vrm/                  # VRM character handling
│   │   ├── animation/            # Animation systems
│   │   └── motion/               # Motion processing (BVH, RSMT)
│   ├── audio/                    # Audio processing
│   │   ├── tts/                  # Text-to-speech
│   │   ├── stt/                  # Speech-to-text
│   │   └── processing/           # Audio analysis
│   ├── compute/                  # Compute backends
│   │   ├── webgpu/               # WebGPU implementations
│   │   ├── webnn/                # WebNN implementations
│   │   └── wasm/                 # WebAssembly implementations
│   └── utils/                    # Shared utilities
├── tests/                        # All testing infrastructure
│   ├── unit/                     # Component-specific unit tests
│   │   ├── core/
│   │   ├── ai/
│   │   ├── avatar/
│   │   ├── audio/
│   │   └── compute/
│   ├── integration/              # Cross-component integration tests
│   │   ├── ai-avatar/            # AI + Avatar integration
│   │   ├── audio-motion/         # Audio + Motion integration
│   │   └── full-pipeline/        # Complete system tests
│   ├── e2e/                      # End-to-end Playwright tests
│   ├── performance/              # Performance benchmarks
│   └── manual/                   # Interactive testing pages
├── demos/                        # Feature demonstrations
│   ├── ai-models/                # AI model showcases
│   ├── avatar-animation/         # Avatar animation demos
│   ├── audio-processing/         # Audio processing demos
│   └── compute-backends/         # Backend comparison demos
├── tools/                        # Development and analysis tools
│   ├── profiling/
│   ├── debugging/
│   └── validation/
├── docs/                         # Documentation
└── config/                       # Configuration files
```

## File Reorganization Strategy

### Phase 1: Core System Components
Move core system files to organized structure:
- TaskManager-related files → `src/core/`
- System monitoring → `src/core/`
- Performance analysis → `src/core/`

### Phase 2: AI Components
Organize AI-related files:
- AI model implementations → `src/ai/models/`
- Worker implementations → `src/ai/workers/`
- Job factories → `src/ai/jobs/`
- KNN implementations → `src/ai/vector/`

### Phase 3: Avatar System
Reorganize avatar components:
- VRM handling → `src/avatar/vrm/`
- BVH processing → `src/avatar/motion/`
- Animation blending → `src/avatar/animation/`
- RSMT transitions → `src/avatar/motion/`

### Phase 4: Audio Processing
Group audio components:
- TTS implementations → `src/audio/tts/`
- STT implementations → `src/audio/stt/`
- Audio analysis → `src/audio/processing/`

### Phase 5: Compute Backends
Organize compute implementations:
- WebGPU code → `src/compute/webgpu/`
- WebNN code → `src/compute/webnn/`
- WASM code → `src/compute/wasm/`

### Phase 6: Testing Infrastructure
Reorganize all tests:
- Unit tests by component → `tests/unit/[component]/`
- Integration tests → `tests/integration/`
- E2E tests → `tests/e2e/`
- Manual test pages → `tests/manual/`

### Phase 7: Demos and Documentation
Clean up demos and docs:
- Feature demos → `demos/[feature]/`
- Documentation → `docs/`
- Tools → `tools/`

## Import Path Updates Required

### Common Import Patterns to Update
```javascript
// Before
import TaskManager from './TaskManager.js';
import './js/aiWorker.js';

// After
import TaskManager from './src/core/TaskManager.js';
import './src/ai/workers/aiWorker.js';
```

### HTML File Updates
```html
<!-- Before -->
<script src="./js/TaskManager.js"></script>

<!-- After -->
<script src="./src/core/TaskManager.js"></script>
```

## Component Testing Strategy

### Individual Component Testing
Each component will have isolated tests:
1. **AI Models**: Test individual model inference without full system
2. **Avatar System**: Test VRM loading and animation separately
3. **Audio Processing**: Test TTS/STT independently
4. **Compute Backends**: Test WebGPU/WebNN/WASM separately

### Integration Testing
Progressive integration testing:
1. **Two-component**: AI + Avatar, Audio + Motion
2. **Three-component**: AI + Audio + Avatar
3. **Full System**: Complete pipeline testing

### Playwright Test Re-engineering
Reorganize Playwright tests to match new structure:
1. **Component Tests**: One test file per major component
2. **Integration Tests**: Cross-component workflow tests
3. **Performance Tests**: Backend comparison and optimization
4. **E2E Tests**: Complete user journey validation

## Benefits of This Organization

### Development Benefits
- **Clear Component Boundaries**: Easy to find and modify specific functionality
- **Independent Development**: Teams can work on components without conflicts
- **Easier Debugging**: Issues can be isolated to specific components
- **Better Code Reuse**: Components can be reused across different contexts

### Testing Benefits
- **Component Isolation**: Test individual parts without dependencies
- **Progressive Integration**: Build up complexity gradually
- **Performance Analysis**: Compare backends and implementations
- **Regression Prevention**: Component-specific tests catch issues early

### Maintenance Benefits
- **Organized Structure**: Easy navigation and understanding
- **Clear Dependencies**: Import paths show component relationships
- **Scalable Architecture**: Easy to add new components
- **Documentation Alignment**: Docs match code organization

## Implementation Steps

1. **Create new directory structure**
2. **Move files systematically by component type**
3. **Update import paths in moved files**
4. **Update HTML file script references**
5. **Reorganize and update Playwright tests**
6. **Validate all tests still pass**
7. **Update documentation**
8. **Clean up legacy files**

This reorganization will enable you to test individual WebNN/WebGPU/WASM avatar components systematically before integration, making development and debugging much more manageable.
