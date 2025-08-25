# Components Overview

Purpose
- Reusable, UI-free building blocks for animation, avatar control, and navigation.

Structure
- animation/: blending, timeline, VRM avatar integration
- conversation/: interfaces for avatar conversation flows
- pathfinding/: navigation and BVH-aware planning

Conventions
- ES modules, no DOM/UI side effects
- Works in main thread and workers

Tests
- See dev/web_viewer/src/testing/unit and dev/web_viewer/src/testing/integration
