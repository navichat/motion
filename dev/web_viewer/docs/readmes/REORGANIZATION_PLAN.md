# Web Viewer Reorganization Plan

## Current Issues
- Over 200+ files in root directory
- Mixed HTML tests, JS modules, documentation, and assets
- Inconsistent naming conventions
- Difficult to locate specific components
- Hard to understand dependencies and imports

## Proposed New Structure

```
dev/web_viewer/
├── src/                          # Source code modules
│   ├── core/                     # Core system modules
│   │   ├── TaskManager.js
│   │   ├── FibonacciHeap.js
│   │   └── SystemPerformanceAnalyzer.js
│   ├── ai/                       # AI model related modules
│   │   ├── models/               # AI model implementations
│   │   ├── jobs/                 # Job factories and definitions
│   │   └── workers/              # AI inference workers
│   ├── avatar/                   # Avatar system modules
│   │   ├── vrm/                  # VRM character handling
│   │   ├── animation/            # Animation systems
│   │   └── motion/               # Motion capture and BVH
│   ├── audio/                    # Audio processing modules
│   │   ├── tts/                  # Text-to-speech
│   │   ├── processing/           # Audio analysis
│   │   └── worklets/             # Audio worklets
│   ├── compute/                  # Compute backend modules
│   │   ├── webgpu/               # WebGPU implementations
│   │   ├── webnn/                # WebNN implementations
│   │   ├── wasm/                 # WebAssembly modules
│   │   └── workers/              # Compute workers
│   └── utils/                    # Utility modules
│       ├── diagnostics/          # Debug and diagnostic tools
│       ├── performance/          # Performance monitoring
│       └── compatibility/        # Browser compatibility
├── tests/                        # All test files
│   ├── unit/                     # Unit tests
│   │   ├── ai/                   # AI model unit tests
│   │   ├── avatar/               # Avatar system unit tests
│   │   ├── audio/                # Audio system unit tests
│   │   └── compute/              # Compute backend unit tests
│   ├── integration/              # Integration tests
│   │   ├── e2e/                  # End-to-end tests (Playwright)
│   │   └── components/           # Component integration tests
│   └── performance/              # Performance and benchmark tests
├── demos/                        # Demo applications
│   ├── ai-inference/             # AI model demonstrations
│   ├── avatar-animation/         # Avatar animation demos
│   ├── audio-processing/         # Audio processing demos
│   └── complete-system/          # Full system demonstrations
├── docs/                         # Documentation
│   ├── api/                      # API documentation
│   ├── guides/                   # User guides
│   └── architecture/             # System architecture docs
├── assets/                       # Static assets
│   ├── models/                   # 3D models, ONNX models
│   ├── animations/               # BVH files, animation data
│   ├── audio/                    # Audio samples
│   └── textures/                 # Textures and materials
├── config/                       # Configuration files
│   ├── playwright.config.js      # Test configuration
│   ├── server.config.js          # Server configuration
│   └── build.config.js           # Build configuration
└── tools/                        # Development tools
    ├── serve_with_headers.py     # Development server
    ├── build-scripts/            # Build and deployment scripts
    └── dev-tools/                # Development utilities
```

## Migration Steps

### Phase 1: Create New Directory Structure
1. Create all new directories
2. Move existing files to appropriate locations
3. Update all import statements
4. Test basic functionality

### Phase 2: Organize Core Components
1. Move core system files (TaskManager, etc.)
2. Organize AI model files
3. Restructure avatar/animation systems
4. Update imports and dependencies

### Phase 3: Reorganize Tests
1. Move all test files to tests/ directory
2. Organize by type (unit, integration, e2e)
3. Update test configurations
4. Verify all tests pass

### Phase 4: Clean Up and Documentation
1. Remove duplicate files
2. Update documentation
3. Create new demo structure
4. Final testing and validation

## Files to Move

### Core System Files
- TaskManager.js → src/core/
- FibonacciHeap.js → src/core/
- SystemPerformanceAnalyzer.js → src/core/

### AI Model Files
- AIModelJobs.js → src/ai/jobs/
- RealJobFactory.js → src/ai/jobs/
- All AI worker files → src/ai/workers/

### Avatar System Files
- VRM*.js files → src/avatar/vrm/
- BVH*.js files → src/avatar/motion/
- Animation*.js files → src/avatar/animation/

### Test Files
- All *.spec.js files → tests/integration/e2e/
- test-*.html files → demos/ or tests/unit/
- *test.html files → demos/ or tests/unit/

### Assets
- *.bvh files → assets/animations/
- *.onnx files → assets/models/
- Audio files → assets/audio/

## Import Updates Required

Most files will need import path updates. For example:
```javascript
// Old
import { TaskManager } from './js/TaskManager.js';

// New  
import { TaskManager } from '../src/core/TaskManager.js';
```

## Benefits of New Structure
1. **Clear separation of concerns**: Each directory has a specific purpose
2. **Easier navigation**: Developers can quickly find relevant files
3. **Better testing**: Organized test structure
4. **Maintainable**: Clear dependencies and relationships
5. **Scalable**: Easy to add new components
6. **Professional**: Industry-standard project structure
