# Timeline Model Tasks

This folder contains chunk-yielding tasks and drivers that integrate ML-generated motion into the BVH timeline system.

## Files
- `Audio2GestureStubTask.js`
  - Deterministic, model-free generator for tests. Emits ~200ms chunks of upper-body sway.
- `Audio2GestureOrtTask.js`
  - ORT-backed task. Uses `window.ort` if available and `modelUrl` is provided. Falls back to deterministic frames.
- `ModelTaskRunner.js`
  - Drives any task that yields `{ t0, dt, frames }` into a `TimelineChunkAdapter`, with abort/preemption support.

## Tests
- Serverless:
  - `audio2gesture-stub-task.serverless.spec.js`
  - `audio2gesture-ort-task-fallback.serverless.spec.js`
  - `model-task-runner.serverless.spec.js`

## How to use (example)
```js
// Create timeline + adapter
const timeline = new BVHTimeline({ framerate: 30 });
const adapter = new TimelineChunkAdapter(timeline);

// Create a task and runner
const task = new Audio2GestureStubTask({ framerate: 30, chunkMs: 200 });
const runner = new ModelTaskRunner(adapter, { defaultTrack: 'gesture-upper', defaultFadeMs: 150, defaultWeight: 0.8 });

// Drive the task
const clock = { now: () => performance.now() / 1000 };
const handle = runner.start(task, { track: 'gesture-upper' }, { clock });

// Later, preempt
handle.abort();
```
