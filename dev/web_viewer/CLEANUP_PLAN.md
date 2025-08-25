# Web Viewer Cleanup and Consolidation Plan

## Current Issue: Duplicate Organization Structures

We have two organizational structures that need to be consolidated:

1. **NEW (Correct)**: `src/models/motion/` and `src/components/`
2. **OLD (Duplicate)**: `src/avatar/motion/` and `src/avatar/animation/`

## Files to Consolidate and Remove

### Motion Model Files (already properly organized)
```
✅ src/models/motion/audio2gesture/Audio2GestureBVHConverter.js (KEEP - properly organized)
✅ src/models/motion/rsmt/RSMTBVHConverter.js (KEEP - properly organized)

❌ src/avatar/motion/Audio2GestureBVHConverter.js (REMOVE - duplicate)
❌ src/avatar/motion/RSMTBVHConverter.js (REMOVE - duplicate)
```

### Timeline Integration Files (need to move from avatar to components)
```
📦 src/avatar/motion/Audio2GestureTimelineIntegration.js → ALREADY in src/components/animation/timeline/
📦 src/avatar/motion/RSMTTimelineIntegration.js → ALREADY in src/components/animation/timeline/
📦 src/avatar/motion/PathfindingTimelineIntegration.js → ALREADY in src/components/pathfinding/

❌ Remove duplicates from src/avatar/motion/
```

### Additional Motion Files (need proper placement)
```
📦 src/avatar/motion/BVHTimeline.js → src/components/animation/timeline/
📦 src/avatar/motion/BVHTimelineCompositor.js → src/components/animation/timeline/
📦 src/avatar/motion/BVHTimelineVRMIntegration.js → src/components/animation/vrm/
📦 src/avatar/motion/motion_analyzer.js → src/utils/motion/
📦 src/avatar/motion/motion_capture.js → src/utils/motion/
```

### New Motion Models (need proper placement)
```
📦 src/avatar/motion/DeepMimicBVHConverter.js → src/models/motion/deepmimic/
📦 src/avatar/motion/FaceFormerBVHConverter.js → src/models/motion/faceformer/
📦 src/avatar/motion/DeepMimicTimelineIntegration.js → src/components/animation/timeline/
📦 src/avatar/motion/FaceFormerTimelineIntegration.js → src/components/animation/timeline/
```

## Cleanup Actions

1. Move missing components to proper locations
2. Remove duplicate files
3. Clean up empty directories
4. Update imports and exports
5. Validate final structure
