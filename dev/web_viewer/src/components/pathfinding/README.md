# Pathfinding

Purpose
- Navigation helpers and BVH-aware trajectory planning for timeline integration.

Key Modules
- PathfindingBVHPlanner.js — plan paths, emit pose/waypoint tracks
- PathfindingTimelineIntegration.js — wire into BVH timeline

Contracts
- Input: nav mesh/path constraints, speed, target(s)
- Output: composed pose/timeline suitable for animation playback

Related Tests
- dev/web_viewer/src/testing/unit/system/
- dev/web_viewer/src/testing/integration/
