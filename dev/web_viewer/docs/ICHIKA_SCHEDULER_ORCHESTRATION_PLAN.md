## Ichika BVH Task Scheduling and Preemption Plan

This document outlines how we will schedule and preempt BVH-generating model tasks, layer independent BVHTimelines, and drive the Ichika VRM avatar.

### Goals
- Schedule BVH generation from FaceFormer, Audio2Gesture, RSMT, DeepMimic, and static BVH files.
- Each source writes to its own BVHTimeline (track) that can be composed into a final BVHTimeline.
- Drive Ichika (VRM) from the composed BVHTimeline.
- Support interrupts: preempt running tasks and replace their timelines when new conversation/speech arrives.

### Architecture
- Scheduler (Fibonacci-heap priority queue) + WorkerPool
- TaskFactory produces model/file-backed tasks (AsyncGenerators yielding frames in chunks)
- Timeline Orchestrator manages per-track BVHTimelines and a Compositor
- VRM Adapter maps composed BVH frames to Ichika

### Core contracts
- Task
  - id, kind, priority, deadline?, trackId, abort: AbortController
  - run(ctx): AsyncGenerator<FramesChunk, ResultMeta>
- FramesChunk
  - t0: number; dt: number; frames: PoseFrame[]; meta?: { confidence?: number; origin?: string }
- BVHTimeline helpers
  - append(trackId, chunk, opts?: { replaceFrom?: number; fadeInMs?: number })
  - clear(trackId, fromTime?: number)
- Scheduler
  - submit(task), preempt(trackId|taskId, opts?), setPriority(taskId, p), onChunk(cb)
- WorkerPool
  - acquire(kind), release(handle)

### Scheduling policy
- Priority bands: conversation > posture/gesture > transitions > background
- Priority function combines base, decay(timeSinceEvent), backendPenalty, deadlineSlack
- Preemption at chunk boundaries; hard-abort if new task priority outranks by ΔP
- Complexity: insert/decrease-key O(1) amortized; pop-min O(log n)

### TaskFactory
- createFaceFormerTask(phonemes|audio, opts)
- createAudio2GestureTask(audio, opts)
- createRSMTTask(prevSeq|bvh, style, opts)
- createDeepMimicTask(skill, params, opts)
- createBVHPlaybackTask(bvhFile|frames, opts)
- Shared opts: { startAt, duration?, mask?, blend: { inMs, outMs }, backend: 'webnn'|'webgpu'|'wasm'|'auto', qos }

### Timelines and composition
- Track-per-intent: face, gesture, locomotion/transition, skills, canned
- Compose with weights and bone masks; small cross-fades to avoid pops
- On preemption, schedule short RSMT transition when possible

### Preemption
- Soft: fade out old tracks (80–160 ms), fade in new tasks
- Hard: abort running task and clear from now onward (preserveTailMs optional)
- Priority mapping: speech 0, interrupt gestures 1, IK 2, locomotion 3, background 5

### Backends & workers
- Pool sizes: webgpu 1–2, webnn 1–2, wasm 2–4
- Prewarm common models; evict LRU on memory pressure
- Chunk size 0.2–0.5s; shrink under load for finer preemption

### Integration points (repo)
- Scheduler & Factory: `dev/web_viewer/src/utils/scheduler/`
- Model tasks: `dev/web_viewer/src/models/motion/*/tasks/`
- Timeline composition helpers: `dev/web_viewer/src/components/animation/timeline/`
- VRM mapping: existing VRM adapters

### Control flow (speech example)
1) ASR event → create FaceFormer + Audio2Gesture tasks
2) submit() with high priority, soft-preempt current face/gesture
3) Workers generate first chunks → append with fade-in
4) Compositor outputs final timeline → VRM animates Ichika

### Errors & fallbacks
- Deadline miss → reduce chunk size or fallback backend; lower weights
- GPU OOM → evict sessions and retry
- Worker crash → respawn and resubmit remaining work

### Metrics
- First-chunk latency ≤ 100 ms (goal ≤ 80 ms)
- Steady-state jitter < ±16 ms
- Memory gauges and utilization per backend

### Testing
- Unit: heap ops, scheduler preemption order, timeline append/clear/compose
- Integration: simulate locomotion then inject speech; assert fade-out/in and composition
- E2E: Ichika VRM scene with scripted interrupts, compare trajectories

### Milestones
- M1: Scheduler + WorkerPool + unit tests
- M2: Task wrappers (chunked AsyncGenerator + cancellation)
- M3: Track timelines + compositor + integration tests
- M4: Orchestrator + VRM hookup + demos
- M5: Perf tuning + metrics + E2E validations
