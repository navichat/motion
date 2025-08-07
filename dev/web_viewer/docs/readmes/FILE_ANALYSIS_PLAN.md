# Comprehensive Web Viewer File Analysis and Reorganization Plan

## Current File Analysis (300+ files)

### Core JavaScript Modules (need reorganization)
```
js/
├── main.js (5803 lines) → src/core/main.js
├── TaskManager.js → src/core/task-manager/TaskManager.js ✅ MOVED
├── constants.js → src/core/constants.js
└── workers/
    ├── compute-backend-optimizer.js → src/core/backends/ ✅ MOVED
    ├── cpu-worker-simple.js → src/workers/compute/ ✅ MOVED  
    ├── gpu-worker-real.js → src/workers/compute/ ✅ MOVED
    ├── webnn-worker-simple.js → src/workers/ai/ ✅ MOVED
    └── wasm-worker-simple-real.js → src/workers/compute/
```

### AI Model Components (need organization)
```
Audio Models:
├── kokoro-tts.js → src/models/audio/kokoro/
├── phonemizer.js → src/models/audio/utils/
└── vad-processor.js → src/models/audio/vad/

Motion Models:
├── Audio2GestureBVHConverter.js → src/models/motion/audio2gesture/
├── DeepMimicPolicyLoader.js → src/models/motion/deepmimic/
├── RSMTBVHConverter.js → src/models/motion/rsmt/
└── FaceFormerBVHConverter.js → src/models/motion/faceformer/

Animation Components:
├── BVHTimeline.js → src/components/animation/timeline/
├── AnimationBlender.js → src/components/animation/blender/
└── VRMBVHAdapter.js → src/components/animation/vrm/
```

### Test Files (need systematic organization)
```
E2E Tests (Playwright):
├── e2e-workload-test.spec.js → tests/e2e/playwright/workload.spec.js ✅ MOVED
├── e2e-avatar-data-export.spec.js → tests/e2e/playwright/avatar-export.spec.js
├── e2e-webnn-inference-complete.spec.js → tests/e2e/playwright/webnn-inference.spec.js
├── knn-apple-to-apple-test.spec.js → tests/e2e/playwright/knn-benchmark.spec.js
└── worker-error-investigator.spec.js → tests/e2e/playwright/worker-errors.spec.js

HTML Test Pages:
├── task-manager-demo.html → tests/e2e/html/task-manager.html ✅ MOVED
├── simple_kokoro_test.html → tests/manual/debug/kokoro.html ✅ MOVED
├── debug_webgpu.html → tests/manual/debug/webgpu.html ✅ MOVED
├── bvh_test_suite_demo.html → tests/manual/demos/bvh-suite.html ✅ MOVED
└── animation_test.html → tests/manual/debug/animation.html

Unit Tests (need creation):
├── src/core/task-manager/TaskManager.test.js
├── src/models/quantized/QuantizedModelOptimizer.test.js
├── src/workers/compute/CPUWorker.test.js
└── src/components/animation/BVHTimeline.test.js
```

### Documentation (need organization)
```
Current:
├── AVATAR_AI_SYSTEM.md → docs/guides/avatar-ai-system.md
├── README.md → docs/README.md
├── KNN_PERFORMANCE_GUIDE.md → docs/guides/knn-performance.md
└── KOKORO_TTS_IMPROVEMENTS.md → docs/improvements/kokoro-tts.md

Needed:
├── docs/api/workers.md
├── docs/api/models.md
├── docs/guides/testing.md
└── docs/architecture/overview.md
```

## Implementation Plan

### Phase 1: Fix Current Test Infrastructure ✅ IN PROGRESS
1. Fix worker path resolution for existing tests
2. Validate reorganized structure works  
3. Create working test configuration

### Phase 2: Systematic Component Migration
1. **Core Framework**: Move main.js, constants, utilities
2. **AI Models**: Organize by type (audio, motion, language)
3. **Animation Components**: BVH, VRM, timeline components
4. **Worker System**: Ensure clean separation of concerns

### Phase 3: Test Suite Reorganization
1. **E2E Tests**: Component-specific test suites
2. **Unit Tests**: Individual component testing
3. **Integration Tests**: Cross-component validation
4. **Manual Tests**: Debug and demo pages

### Phase 4: Import System Modernization
1. Convert to ES6 modules where possible
2. Create clean import/export interfaces
3. Implement proper dependency injection
4. Update all relative paths

## Next Immediate Actions

1. **Quick Fix**: Create proxy HTML file that works with current paths
2. **Component Analysis**: Systematically analyze each remaining file
3. **Migration Strategy**: Move files in logical groups
4. **Test Validation**: Ensure each move preserves functionality
