# Motion Models

This package contains motion-generation and conversion models used by the avatar system.

Included Models
- audio2gesture/ — audio→gesture BVH/timeline
- rsmt/ — real-time stylized motion transition (DeepPhase/StyleVAE/TransitionNet)
- deepmimic/ — humanoid motion conversions and timelines
- faceformer/ — audio→facial animation

Usage
- Each model folder documents its own usage and provides demos/tests
- Integrations with timelines are located under src/components/animation/*

Testing
- See dev/web_viewer/src/testing/integration and e2e workloads
- Per-model validators and demos live within each model folder
