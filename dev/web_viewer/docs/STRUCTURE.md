# Web Viewer Project Structure (Authoritative)

This document is the source of truth for the current dev/web_viewer layout. It supersedes older structure docs in this folder.

## Top-level layout

```
dev/web_viewer/
├─ README.md                    # Quickstart and pointers
├─ index.html                   # Entrypoint (when served at /dev/web_viewer/)
├─ src/                         # Source code (modern, maintained)
├─ tests/                       # Playwright tests (unit + integration + e2e containers)
├─ demos/                       # HTML demos used by tests and manual runs
├─ docs/                        # Documentation and reports
├─ data/                        # Models and test data
├─ assets/                      # 3D models, audio, scenes, etc.
├─ js/                          # Legacy scripts kept for reference/back-compat
├─ tools/                       # Dev and validation scripts
└─ MotionData/                  # Motion/BVH-related assets
```

## Source code (src/)

```
src/
├─ ai/                          # AI model orchestration & helpers
│  ├─ AIModelJobs.js
│  ├─ KNNJobs.js
│  ├─ ConversationNeuralNetwork.js
│  ├─ LlamaModule*.js           # Llama variants
│  ├─ DeepMimicPolicyLoader.js
│  ├─ jobs/                     # Job definitions
│  ├─ models/                   # Model configs/helpers
│  └─ workers/                  # AI workers
├─ audio/                       # Audio pipeline (VAD, TTS, STT, workers)
│  ├─ WhisperModule.js
│  ├─ KokoroModule.js
│  ├─ VoiceActivityDetector*.js
│  ├─ ConversationWorker*.js
│  ├─ audio-worklet.js
│  └─ processing/ tts/          # Submodules
├─ compute/                     # Compute backends
│  ├─ MockBackends.js
│  ├─ wasm/
│  ├─ webgpu/
│  └─ webnn/
├─ components/                  # UI/animation/conversation components
│  ├─ animation/
│  ├─ conversation/
│  ├─ pathfinding/
│  └─ index.js
├─ core/                        # Core runtime
│  ├─ TaskManager.js
│  ├─ FibonacciHeap.js
│  ├─ SystemPerformanceAnalyzer.js
│  ├─ backends/ constants.js index.js main.js
│  └─ task-manager/
├─ models/                      # Model wrappers and domain models
├─ testing/                     # Test harness helpers and suites
├─ utils/                       # Utilities (incl. servers)
│  ├─ serve_with_headers.py
│  └─ onnx/, quantization tools, scripts
└─ workers/                     # Shared workers (ai/compute)
```

## Tests (tests/)

```
tests/
├─ unit/
│  ├─ ai/                       # per-component AI tests
│  │  └─ ai-models.spec.js
│  ├─ avatar/
│  │  ├─ avatar-animation.spec.js
│  │  └─ avatar-motion.spec.js
│  ├─ audio/
│  ├─ compute/
│  │  └─ compute-backends.spec.js
│  ├─ motion/
│  └─ system/
├─ integration/
│  ├─ collect-all-outputs.spec.js
│  ├─ component-testing-demo.spec.js
│  └─ master-test-suite.spec.js
├─ e2e/                         # container for browser/e2e groupings
│  ├─ html/
│  └─ playwright/
└─ legacy-root-tests/            # Migrated root-level tests (if any)
```

## Demos (demos/)

```
demos/
├─ ai-inference/
│  └─ task-manager-demo.html    # Main orchestration demo
└─ html-tests/
   ├─ debug_onnx_runtime.html
   ├─ direct_ai_test.html
   ├─ test-model-fetch.html
   ├─ test-quantized-models.html
   └─ test-wasm-direct.html
```

## Docs (docs/)

- readmes/ … focused guides (AI models, compute backends, testing, etc.)
- reports/ … generated logs, summaries, outputs
- PROJECT_STRUCTURE_GUIDE.md … legacy; see note below
- TESTING_INFRASTRUCTURE.md … legacy; see note below

Note: The present file (STRUCTURE.md) is authoritative. Older docs remain for history and will link here.

## Legacy folders

- js/ … legacy scripts and workers. New development should prefer src/ counterparts.
- files under dev/web_viewer root with .spec.js … consider migrating into tests/ if encountered during maintenance.

## Conventions

- New code lives under src/ with clear domain subfolders.
- Tests live under tests/ with unit vs integration separation.
- Demos used by tests live under demos/.
- Tools that start servers or migrations live under src/utils or tools/.
