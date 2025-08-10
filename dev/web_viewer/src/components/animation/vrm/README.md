# VRM Avatar Integration

Purpose
- Bridge BVH/timeline data to VRM humanoid rigs and provide diagnostics.

Key Modules
- VRMBVHAdapter.js / EnhancedVRMBVHAdapter.js — BVH→VRM pose mapping
- DeepMimicVRMBoneMapper.js — motion model to VRM bone map helpers
- VRMConversationInterface*.js — conversation/UX integration variants
- VRMDiagnostics.js, VRMVisibilityFix.js, VRMLightingManager.js — utilities

Contracts
- Input: humanoid VRM model, pose/timeline frames, options
- Output: applied humanoid pose per frame, event hooks

Usage
```js
// import { VRMBVHAdapter } from './VRMBVHAdapter.js';
// const adapter = new VRMBVHAdapter(vrm);
// adapter.applyFrame(bvhFrame);
```

Dependencies
- three.js / VRM loader (provided by app layer)

Related Tests
- dev/web_viewer/src/testing/unit/avatar/
- dev/web_viewer/src/testing/integration/

Additional lightweight components
- AvatarBinder.js — minimal adapter used by tests and simple apps; supports stub mode (no VRM) to record updates.
- BVHTimelineVRMIntegration.js — converts BVHTimeline frames to binder updates; integrates smoothly with IchikaOrchestrator.bindAvatar.
- VRMLoader.js — deferred-import loader wrapper intended for demos (avoid in serverless tests).

Orchestrator Wiring
- IchikaOrchestrator.bindAvatar({ vrm }) will create an AvatarBinder and connect BVHTimelineVRMIntegration to its BVHTimeline.
