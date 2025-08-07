# Web Viewer Reorganization Plan

## Current Issues
1. **File Structure**: 300+ files mixed in root directory
2. **HTML Tests**: Scattered across root with no clear organization  
3. **JavaScript Modules**: Mixed organization, unclear dependencies
4. **Import Paths**: Inconsistent relative paths breaking modularity
5. **Test Files**: e2e tests mixed with unit tests and demo files
6. **Documentation**: Scattered across root, hard to find

## Proposed New Structure

```
dev/web_viewer/
├── src/                          # Main source code
│   ├── core/                     # Core framework files
│   │   ├── task-manager/         # TaskManager and related
│   │   ├── backends/            # Backend optimizers
│   │   └── utils/               # Utility functions
│   ├── models/                  # AI model integrations
│   │   ├── quantized/           # Quantized model optimizer
│   │   ├── audio/              # Audio models (Kokoro, SpeechT5, Whisper, VAD)
│   │   ├── motion/             # Motion models (RSMT, DeepMimic, FaceFormer)
│   │   └── language/           # Language models (TinyLlama, DiabloGPT)
│   ├── workers/                # Web Workers
│   │   ├── ai/                 # AI model workers
│   │   ├── compute/            # Compute workers (WebGPU, WASM, WebNN)
│   │   └── utils/              # Worker utilities
│   ├── components/             # Reusable UI components
│   │   ├── animation/          # Animation components
│   │   ├── bvh/               # BVH-related components
│   │   └── vrm/               # VRM-related components
│   └── assets/                 # Static assets
│       ├── animations/         # BVH files, reference motions
│       ├── models/            # ONNX models
│       └── data/              # Sample data
├── tests/                      # All test files
│   ├── e2e/                   # End-to-end tests
│   │   ├── playwright/        # Playwright test specs
│   │   └── html/             # HTML test pages for e2e
│   ├── unit/                  # Unit tests
│   │   ├── models/           # Model-specific tests
│   │   ├── workers/          # Worker tests
│   │   └── components/       # Component tests
│   ├── integration/           # Integration tests
│   │   ├── backend/          # Backend integration tests
│   │   └── pipeline/         # Full pipeline tests
│   └── manual/               # Manual testing pages
│       ├── demos/            # Demo pages
│       └── debug/            # Debug/diagnostic pages
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── guides/               # User guides
│   └── architecture/         # Architecture docs
├── config/                    # Configuration files
│   ├── playwright.config.js  # Test configuration
│   └── build.config.js       # Build configuration
└── tools/                     # Development tools
    ├── serve_with_headers.py # Development server
    └── build/                # Build scripts
```

## Migration Plan

### Phase 1: Create New Structure
1. Create new directory structure
2. Move core files to appropriate locations
3. Update import paths systematically

### Phase 2: Reorganize by Component Type
1. **Core Framework**: TaskManager, backend optimizers
2. **AI Models**: Quantized optimizer, model-specific code
3. **Workers**: Separate by function (AI, compute, utils)
4. **Components**: Reusable UI/logic components

### Phase 3: Test Reorganization
1. **e2e Tests**: Move all Playwright tests to tests/e2e/playwright/
2. **HTML Test Pages**: Move to tests/e2e/html/ or tests/manual/
3. **Unit Tests**: Create proper unit test structure
4. **Integration Tests**: Separate backend and pipeline tests

### Phase 4: Update Imports and Dependencies
1. Update all relative imports to use consistent paths
2. Create index files for clean imports
3. Update HTML script tags
4. Fix worker importScripts paths

### Phase 5: Update Documentation and Config
1. Update README files
2. Update Playwright configuration
3. Create component-specific documentation
4. Update build and deployment scripts

## Benefits After Reorganization

1. **Clear Separation**: Easy to find files by functionality
2. **Better Testing**: Organized test structure for different test types
3. **Maintainable Imports**: Consistent import paths
4. **Scalable**: Easy to add new components/models
5. **Developer Experience**: Clear where to put new files
6. **CI/CD Ready**: Organized structure for automated testing

## Files to Move (Examples)

### Core Framework
- `js/TaskManager.js` → `src/core/task-manager/TaskManager.js`
- `js/workers/compute-backend-optimizer.js` → `src/core/backends/ComputeBackendOptimizer.js`

### AI Models  
- `quantized-model-optimizer.js` → `src/models/quantized/QuantizedModelOptimizer.js`
- `js/kokoro-tts.js` → `src/models/audio/KokoroTTS.js`

### Workers
- `js/workers/cpu-worker-simple.js` → `src/workers/compute/CPUWorker.js`
- `js/workers/gpu-worker-real.js` → `src/workers/compute/GPUWorker.js`

### Tests
- `e2e-workload-test.spec.js` → `tests/e2e/playwright/workload-test.spec.js`
- `task-manager-demo.html` → `tests/e2e/html/task-manager-demo.html`

### Documentation
- `AVATAR_AI_SYSTEM.md` → `docs/guides/avatar-ai-system.md`
- `README.md` → `docs/README.md`
