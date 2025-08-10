# Timeline

Purpose
- Time-based sequencing for BVH/pose data, composition, and playback.

Key Modules
- BVHTimeline.js — core abstraction for frame-time mapping and composition
- *TimelineIntegration.js (per-model) — see src/components/animation/*

API (BVHTimeline.js)
- constructor({ frameTime, frames })
- getFrameAt(timeMs): PoseFrame
- compose(other, mode): Timeline

Related Tests
- dev/web_viewer/src/testing/unit/motion/
- dev/web_viewer/src/testing/integration/
