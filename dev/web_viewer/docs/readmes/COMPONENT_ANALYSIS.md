# Web Viewer Component Analysis and Reorganization Plan

## Current Component Inventory

### Core Animation Components (Motion Models)
```
Audio2GestureBVHConverter.js → src/models/motion/audio2gesture/
- 481 lines, converts Audio2Gesture neural outputs to BVH
- Dependencies: ONNX runtime, audio processing
- Status: Ready to move

RSMTBVHConverter.js → src/models/motion/rsmt/
- 909 lines, realtime stylized motion transitions
- Dependencies: BVH processing, pose vectors, DeepPhase
- Status: Ready to move

DeepMimicPolicyLoader.js → src/models/motion/deepmimic/
- Currently empty, needs implementation
- Status: Placeholder file

Audio2GestureTimelineIntegration.js → src/components/animation/timeline/
- Integration with BVH timeline system
- Status: Ready to move

RSMTTimelineIntegration.js → src/components/animation/timeline/
- RSMT integration with timeline
- Status: Ready to move
```

### Animation System Components
```
BVHAnimationTestSuite.js → src/components/animation/testing/
- Animation testing framework
- Status: Ready to move

AnimationBackendsTestSuite.js → src/components/animation/testing/
- Backend testing for animation systems
- Status: Ready to move

DeepMimicVRMBoneMapper.js → src/components/animation/vrm/
- VRM bone mapping for DeepMimic
- Status: Ready to move

PathfindingBVHPlanner.js → src/components/pathfinding/
- BVH-based pathfinding planner
- Status: Ready to move

PathfindingTimelineIntegration.js → src/components/pathfinding/
- Timeline integration for pathfinding
- Status: Ready to move
```

### Voice/Audio Components
```
VRMConversationInterface.js → src/components/conversation/vrm/
- VRM conversation interface
- Status: Ready to move

voiceChatExample.js → src/examples/voice/
- Voice chat example implementation
- Status: Ready to move

workerVoiceChatExample.js → src/examples/voice/
- Worker-based voice chat example
- Status: Ready to move

test_phonemizer.js → src/audio/phonemizer/test/
- Phonemizer testing
- Status: Ready to move
```

### Motion Analysis and Utilities
```
motion_analyzer.js → src/utils/motion/
- Motion analysis utilities
- Status: Ready to move

motion_capture.js → src/utils/motion/
- Motion capture utilities
- Status: Ready to move

style_controller.js → src/utils/ui/
- UI style controller
- Status: Ready to move

phase_visualizer.js → src/utils/visualization/
- Phase visualization utilities
- Status: Ready to move
```

### Testing and Validation Components
```
MockBackends.js → src/testing/mocks/
- Mock backend implementations
- Status: Ready to move

SampleDataGenerators.js → src/testing/data/
- Sample data generation for testing
- Status: Ready to move

onnx-validator.js → src/utils/validation/
- ONNX model validation
- Status: Ready to move

onnx-version-manager.js → src/utils/onnx/
- ONNX version management
- Status: Ready to move
```

### Legacy/External Dependencies
```
three.min.js → assets/vendor/
- Three.js library (external dependency)
- Status: Move to assets

kokoro.web.js → assets/vendor/
- Kokoro TTS library
- Status: Move to assets
```

## Reorganization Strategy

### Phase 1: Core Motion Models
1. Move Audio2GestureBVHConverter.js to src/models/motion/audio2gesture/
2. Move RSMTBVHConverter.js to src/models/motion/rsmt/
3. Update imports and create index files

### Phase 2: Animation Components
1. Move timeline integration files
2. Move VRM components
3. Move animation testing suites

### Phase 3: Audio/Voice Components
1. Move conversation interfaces
2. Move voice chat examples
3. Organize audio utilities

### Phase 4: Utilities and Testing
1. Move motion utilities
2. Move testing frameworks
3. Move validation tools

### Phase 5: HTML Test Reorganization
1. Categorize HTML test files
2. Move to appropriate test directories
3. Update Playwright test configurations

## Import Dependencies Analysis

### Audio2GestureBVHConverter.js Dependencies:
- ONNX runtime (external)
- Audio processing APIs (browser)
- BVH timeline system (internal)

### RSMTBVHConverter.js Dependencies:
- BVH processing utilities (internal)
- Pose vector calculations (internal)
- Animation caching (internal)

### Next Steps:
1. Start with Phase 1 (Motion Models)
2. Test each move to ensure functionality
3. Update all import paths
4. Create comprehensive test suite
