# Final Reorganization Cleanup Plan

## Current Status
The main reorganized structure is complete with:
- ✅ `src/` - Organized source code
- ✅ `tests/` - Comprehensive testing infrastructure  
- ✅ `demos/` - Feature demonstrations
- ✅ `docs/` - Complete documentation
- ✅ `config/` - Configuration files
- ✅ `tools/` - Development tools
- ✅ `assets/` - Static assets

## Remaining Cleanup Tasks

### 1. HTML Test Files to Move

#### Animation/Motion Testing
- `animation_test.html` → `tests/manual/animation/`
- `animation_backends_test_runner.html` → `tests/manual/animation/`
- `bvh_test.html` → `tests/manual/motion/`
- `bvh_test_suite_demo.html` → `tests/manual/motion/`
- `bvh_diagnostic.html` → `tests/manual/motion/`
- `motion_viewer.html` → `demos/avatar-animation/`
- `motion_transitions.html` → `demos/avatar-animation/`
- `skeleton_test.html` → `tests/manual/animation/`
- Various `*_viewer.html` files → `demos/avatar-animation/`

#### Audio/Speech Testing
- `test_audio_vrm_system.html` → `tests/manual/audio/`
- `test_phonemizer.html` → `tests/manual/audio/`
- `test_speecht5_direct.html` → `tests/manual/audio/`
- `test-audio-context.html` → `tests/manual/audio/`

#### AI/Worker Testing
- `test_conversation_worker.html` → `tests/manual/ai/`
- `test_enhanced_conversation_worker.html` → `tests/manual/ai/`
- `quick_worker_test.html` → `tests/manual/workers/`
- `simple_worker_test.html` → `tests/manual/workers/`
- `test_worker_syntax.html` → `tests/manual/workers/`

#### System Testing
- `basic_test.html` → `tests/manual/system/`
- `simple-test.html` → `tests/manual/system/`
- `test.html` → `tests/manual/system/`
- `diagnostic_test.html` → `tests/manual/system/`
- `integration_tests.html` → `tests/integration/manual/`

#### Performance Testing
- `hardware_performance_benchmark.html` → `tests/performance/`
- `realtime_performance_dashboard.html` → `tests/performance/`
- `taskmanager-perf-test.html` → `tests/performance/`

#### Specialized Testing
- `browser-compatibility-test.html` → `tests/compatibility/`
- `onnx-compatibility-test.html` → `tests/compatibility/`
- `test-error-collection.html` → `tests/debugging/`
- `debug_*.html` files → `tests/debugging/`

### 2. JavaScript Files to Organize

#### Component Implementations
- `Audio2GestureBVHConverter.js` → `src/audio/converters/`
- `DeepMimicPolicyLoader.js` → `src/ai/deepmimic/`
- `DeepMimicVRMBoneMapper.js` → `src/avatar/deepmimic/`
- `PathfindingBVHPlanner.js` → `src/motion/pathfinding/`
- `RSMTBVHConverter.js` → `src/motion/rsmt/`
- `VRMConversationInterface.js` → `src/avatar/conversation/`

#### Test Suites
- `AnimationBackendsTestSuite.js` → `tests/unit/animation/`
- `BVHAnimationTestSuite.js` → `tests/unit/motion/`
- `SampleDataGenerators.js` → `tests/utils/`
- `MockBackends.js` → `tests/mocks/`

#### Utilities
- `motion_analyzer.js` → `src/motion/analysis/`
- `motion_capture.js` → `src/motion/capture/`
- `phase_visualizer.js` → `src/visualization/`
- `style_controller.js` → `src/ui/`

### 3. Demo Files to Organize

#### Timeline Demos
- `audio2gesture_timeline_demo.html` → `demos/audio-processing/`
- `deepmimic_timeline_demo.html` → `demos/avatar-animation/`
- `faceformer_timeline_demo.html` → `demos/avatar-animation/`
- `pathfinding_timeline_demo.html` → `demos/motion-planning/`
- `rsmt_timeline_demo.html` → `demos/motion-transitions/`

#### Specialized Demos
- `ai_enhanced_platform.html` → `demos/complete-system/`
- `conversation-demo.html` → `demos/ai-inference/`
- `dependencyInjectionDemo.html` → `demos/architecture/`
- `multi_modal_animation_pipeline_demo.html` → `demos/complete-system/`
- `voiceChatDemo.html` → `demos/ai-inference/`

### 4. Playwright Tests to Organize

#### Move Remaining Test Files
- `worker-error-investigator.spec.js` → `tests/integration/e2e/`
- `e2e-workload-test.spec.js.backup` → `archive/`

### 5. Archive/Cleanup

#### Files to Archive
- Old README files → `archive/documentation/`
- Status/progress files → `archive/development/`
- Backup files → `archive/backups/`

#### Directories to Organize
- `audio2gesture/` → `src/audio/audio2gesture/`
- `deepmimic/` → `src/ai/deepmimic/`
- `faceformer/` → `src/ai/faceformer/`
- `rsmt/` → `src/motion/rsmt/`

## Implementation Priority

### Phase 1: Critical Test Organization
1. Move HTML test files to appropriate test directories
2. Update import paths in moved test files
3. Validate test functionality

### Phase 2: Component Code Organization  
1. Move JavaScript component files to src/
2. Update import statements and dependencies
3. Test component loading

### Phase 3: Demo Organization
1. Move demo files to appropriate demo categories
2. Update demo import paths
3. Validate demo functionality

### Phase 4: Archive and Cleanup
1. Move documentation to archive
2. Clean up root directory
3. Final validation of reorganized structure

## Success Criteria
- ✅ Clean root directory with only essential files
- ✅ All tests accessible and functional
- ✅ All demos working with new structure
- ✅ All import paths correctly updated
- ✅ Complete documentation of new structure
