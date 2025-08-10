# Ichika BVH scheduler – pinned context

Purpose: Keep this open while wiring scheduler → adapter → timelines.

What we’re building
- Preemptible TaskScheduler (Fibonacci heap) executes short tasks that emit BVH frame chunks.
- Each source writes to its own BVHTimeline; a TimelineMixer composes final pose to drive the VRM avatar.
- New speech/gestures can preempt current work; smooth crossfades via the adapter.

Key contracts
- Task: { id, type, priority, preemptible, timelineId, resources, signal, run(ctx): AsyncGenerator<{ t0, dt, frames }> }
- BVHTimeline: constructor({ framerate }); appendChunk(track, { t0, dt, frames }); clear(track, from?); snapshot(t0, t1)
- TimelineMixer: compose(timelines[], t, rules?) → FinalPose

Where code lives (this repo)
- Scheduler: src/utils/scheduler/{FibonacciHeap.js, TaskScheduler.js}
- BVH models: src/models/bvh/{BVHTimeline.js, TimelineMixer.js}
- Adapter: src/components/animation/timeline/TimelineChunkAdapter.js
- VRM integration: src/components/animation/vrm

Tests to run
- Serverless unit suite: dev/web_viewer/tests/unit/**/*\.serverless.spec.js
- Web-backed unit suite: dev/web_viewer/tests/unit (non-serverless)

Notes
- Constructors are exposed both on window.* and via CommonJS (module.exports). In tests, prefer a CommonJS-style shim to resolve constructors; keep window.* only as a fallback.
- Prefer the minimal BVHTimeline at src/models/bvh/BVHTimeline.js for unit/serverless tests; reserve the heavy component timeline for viewer features.
- Playwright web server runs on http://localhost:8080 with COOP/COEP headers. Shell/webServer timeouts are required by policy; we set webServer.timeout = 120_000.

See also
- Summary: docs/readmes/BVH_SCHEDULER_CONTEXT_SUMMARY.md
- Full plan: docs/readmes/BVH_ANIMATION_SCHEDULER_PLAN.md
- Orchestration: docs/ICHIKA_SCHEDULER_ORCHESTRATION_PLAN.md

## Known patterns

CommonJS-style shim in serverless tests to resolve constructors:

```js
// In page.evaluate
const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
const BVHTimelineCtor = mk(timelineCode, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
const TimelineChunkAdapterCtor = mk(adapterCode, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
```

Prefer injecting the minimal BVH timeline source in tests:

```js
await page.addScriptTag({ content: await (await fetch('/src/models/bvh/BVHTimeline.js')).text() });
```
