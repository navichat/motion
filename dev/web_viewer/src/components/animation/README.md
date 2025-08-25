# Animation Components

Purpose
- Compose animations, blend layers, and synchronize BVH/VRM timelines.

Key Modules
- AnimationBlender.js — cross-fade and layered blending utilities
- AnimationSync.js — timeline/frame sync helpers
- timeline/ — low-level BVH timeline abstraction
- vrm/ — VRM avatar adaptation and utilities

Inputs/Outputs
- Inputs: BVH frames, pose tracks, blend weights, timing options
- Outputs: blended pose frames, synchronized timelines

Usage
```js
import { blend } from './AnimationBlender.js';
// const result = blend(trackA, trackB, 0.5, { duration: 1.0 });
```

Related Tests
- dev/web_viewer/src/testing/unit/avatar/
- dev/web_viewer/src/testing/integration/
