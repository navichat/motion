# BVH Animation Scheduler & Task Factory – Implementation Plan

This plan details how to use the Fibonacci heap task scheduler and a Task Factory to generate BVH animation frames from FaceFormer, Audio2Gesture, RSMT, DeepMimic, and BVH files, placing them onto independent BVHTimelines that can be composed and used to drive the VRM avatar "Ichika". It also specifies preemption/interrupt handling, so ongoing neural tasks/animations can be interrupted by higher-priority tasks (e.g., new speech/conversation).

## 1) Goals and non-goals

Goals
- Generate animation from multiple sources (FaceFormer, Audio2Gesture, RSMT, DeepMimic, BVH files) as BVH-compatible timeline segments.
- Maintain multiple independent BVHTimelines that can be composed into a final timeline to puppet the VRM avatar (Ichika).
- Use a Fibonacci heap-based scheduler (O(1) amortized insert/decrease-key) to manage all animation and NN inference work with priorities and deadlines.
- Support preemption/interrupt: stop or de-prioritize existing work and switch to new tasks/timelines quickly (e.g., on speech start).
- Ensure atomic timeline updates, predictable blending, and smooth transitions.

Non-goals
- Building training pipelines. We assume models are provided and callable.
- Replacing existing loaders or backends. We orchestrate them.

## 2) High-level architecture

Components
- TaskManager (existing: `src/core/TaskManager.js`)
  - Owns a FibonacciHeap (`src/core/FibonacciHeap.js`)
  - Manages a pool of workers (WebNN/WebGPU/WASM) via compute manager (`src/compute/*`).
  - Tracks running tasks, supports cancellation and priority changes.
- TaskFactory (new)
  - Creates typed tasks with well-defined contracts: FaceFormerTask, Audio2GestureTask, RSMTTransitionTask, DeepMimicTask, BVHDecodeTask, TimelineOps.
- BVHTimeline (new)
  - Per-source, append-only segments of channels/frames with metadata.
  - Supports atomic segment insertion, crossfade, layering, and versioning.
- TimelineMixer (new)
  - Composes N BVHTimelines into a final pose stream (per frame).
  - Applies weights, masks, and retargeting into VRM-compatible rig.
- VRMAnimator/AvatarController (existing hooks in `src/components/animation` or `src/avatar/*`)
  - Consumes final poses from TimelineMixer; drives Ichika in the render loop.

## 3) Core abstractions and contracts

Task (base)
- id: string
- type: 'faceformer' | 'audio2gesture' | 'rsmt' | 'deepmimic' | 'bvh-decode' | 'timeline-op'
- priority: number (lower is higher priority)
- deadline?: number (ms timestamp for EDF-like hints)
- preemptible: boolean
- timelineId?: string (target BVHTimeline)
- resources: { backend: 'webnn'|'webgpu'|'wasm'|'cpu', gpuMem?: number, cpuCost?: number }
- signal: AbortSignal (cancellation)
- run(ctx): Promise<TaskResult>
- onCancel?(reason): void

TaskResult
- ok: boolean
- output?: { frames?: BVHFrame[]; segment?: TimelineSegment; meta?: any }
- error?: Error

BVHFrame
- time: number (seconds) or frameIndex
- channels: Map<string /*bone.channel*/, number>

TimelineSegment
- startTime: number
- duration: number
- channels: string[]
- frames: BVHFrame[]
- track: 'body'|'face'|'hands'|'root' (for masking/weights)
- meta: { source: string, confidence?: number }

BVHTimeline
- id: string
- version: number
- tracks: Record<track, Array<TimelineSegment>>
- addSegment(segment, options): version++ (atomic)
- removeRange(range): version++
- snapshot(t0, t1): returns blended frames

TimelineMixer
- compose(timelines: BVHTimeline[], t: number, rules): FinalPose
- rules: layering, weights per track, crossfade windows, conflict policy

FinalPose
- map<string bone, { position, rotation, scale }>

## 4) Task types (TaskFactory)

TaskFactory API
- createFaceFormerTask({ audioFeatures, startTime, duration, timelineId, priority })
- createAudio2GestureTask({ audioFeatures, style, startTime, duration, timelineId, priority })
- createRSMTTransitionTask({ fromClip, toClip, startTime, timelineId, priority })
- createDeepMimicTask({ policy, context, duration, timelineId, priority })
- createBVHDecodeTask({ bvhBuffer|url, insertAt, timelineId, priority })
- createTimelineOpTask({ op: 'insert'|'remove'|'crossfade', args, timelineId, priority })

Notes
- FaceFormer → face track (blendshapes/face bones).
- Audio2Gesture → upper-body/hand tracks.
- RSMT → short transition segment between clips to avoid pops.
- DeepMimic → locomotion/whole-body where applicable.
- BVHDecode → imports external BVH as a segment.

## 5) Scheduler details (Fibonacci heap)

- Ready queue: FibonacciHeap keyed by (priority, optional deadline tie-breaker).
- Running set: track currently assigned tasks per worker.
- On event (e.g., speech start), new high-priority tasks get inserted; we also decrease-key existing related tasks to demote them.
- Preemption policy:
  - If a higher-priority task arrives and a lower-priority preemptible task is running on the needed resource, signal cancellation; resubmit remaining work as a new task if needed.
  - Non-preemptible tasks run to yield points (batch boundaries, frame-chunks) and then are demoted.
- Decrease-key usage: dynamic priority bumping, e.g., conversation tasks get higher priority while speaking.

Worker allocation
- Workers grouped by backend: webnn/webgpu/wasm/cpu.
- Per-task resource hints used for assignment.
- Backpressure: if GPU is saturated, fallback to WASM/CPU for short segments (depending on quality policy).

## 6) Preemption & interruption

Triggers
- VAD → speech start/stop
- New intent requiring new gestures/face motion
- User input (click/command) or system event

Mechanism
- Compute a new Job Group (ConversationGroup) with tasks: lip-sync (FaceFormer), gestures (Audio2Gesture), transition (RSMT), optional body motion (DeepMimic), and a timeline-op to crossfade from current.
- Assign group a higher priority (lower numeric value). Insert tasks into heap.
- Identify currently running tasks bound to same tracks/timeline; if preemptible, cancel via AbortController; else mark them to finish current chunk and not schedule further chunks.
- Use `decreaseKey` on scheduled-but-not-running tasks to demote them.

Graceful timeline switching
- Insert an RSMTTransitionTask to bridge current pose → new clip start.
- Apply crossfade windows at segment boundaries (e.g., 150–300 ms) per track.
- Ensure atomic `addSegment` with version bump.

## 7) Composition model (multi-timeline → final pose)

- Maintain separate BVHTimelines per source (faceformer, gestures, locomotion, imported BVH, etc.).
- Mixer rules example:
  - face track: FaceFormer overrides base with weight [0..1], priority to current conversation; fallback to idle face.
  - body/arms: Audio2Gesture blended over base idle locomotion; RSMT ensures continuity.
  - root motion: only one authoritative provider at a time (DeepMimic or BVH locomotion). If conflicts: latest group wins with short crossfade.
- At render time t: TimelineMixer samples each timeline’s segments, applies masks/weights, outputs FinalPose → VRM retargeter → Ichika.

## 8) Data flow examples

Conversation interrupt (speech starts)
1. VAD detects start → Event bus emits `conversation:start` with audio window.
2. TaskFactory builds tasks: FaceFormer(lip), Audio2Gesture(gestures), RSMT(transition), TimelineOps(insert/crossfade).
3. Scheduler inserts tasks at high priority; decreases priority of ongoing gesture tasks.
4. Workers run lip first (low latency), then gestures; timeline segments inserted atomically.
5. Mixer prefers conversation timelines for face/upper body while speaking.

Idle → walk → stop
1. DeepMimic locomotion generates root/body timeline.
2. Gesture/face timelines blend on top when necessary.
3. On stop, RSMT generates a stop transition.

## 9) Priorities & SLAs

Suggested base priorities (smaller is higher priority)
- 0–10: Critical, UI-critical (lip-sync frames, immediate transitions)
- 11–30: Conversation gesture generation (short horizon, e.g., 0.5–1.0s chunks)
- 31–60: Background locomotion synth (DeepMimic) and long-horizon planning
- 61+: Bulk BVH import/validation, non-urgent repairs

Deadlines
- Lip frames target <40–80 ms to first output.
- Gesture chunks 200–400 ms.
- Transitions target <150 ms lead time.

## 10) Resource & chunking strategy

- Generate small segments (e.g., 200–500 ms) to allow frequent preemption points.
- Use streaming outputs when supported (emit frames as they are ready).
- GPU budget: cap concurrent GPU tasks; avoid thrashing. Use WASM for tiny, latency-sensitive tasks if GPU queue is long.

## 11) Error handling & resilience

- Task errors return { ok: false, error } and emit telemetry; scheduler retries with exponential backoff for transient errors (configurable attempts).
- Timeline insertion is atomic; failed segments don’t modify timeline.
- On cancel, tasks must clean up allocations; partial results may be kept if coherent (e.g., partial frames segment).

## 12) Minimal APIs (proposed)

TaskFactory (TypeScript-like pseudo)
- interface CreateTaskOpts { priority?: number; preemptible?: boolean; timelineId: string; startTime?: number; duration?: number; }
- function createFaceFormerTask(opts & { audioFeatures: Float32Array }): Task
- function createAudio2GestureTask(opts & { audioFeatures: Float32Array, style?: string }): Task
- function createRSMTTransitionTask(opts & { fromPose: FinalPose, toHint?: any }): Task
- function createDeepMimicTask(opts & { policy: string, context?: any }): Task
- function createBVHDecodeTask(opts & { url?: string, buffer?: ArrayBuffer }): Task
- function createTimelineOpTask(opts & { op: 'insert'|'remove'|'crossfade', args: any }): Task

BVHTimeline
- addSegment(segment: TimelineSegment, options?: { crossfadeMs?: number, weight?: number }): number // returns new version
- snapshot(t0: number, t1: number): BVHFrame[]

Scheduler hooks (TaskManager)
- submit(task: Task)
- cancel(taskId: string, reason?: string)
- decreasePriority(taskId: string, newPriority: number)
- on(event, handler) // 'taskStarted'|'taskCompleted'|'taskCanceled' etc.

## 13) Testing plan

Unit
- FibonacciHeap: insert, extract-min, decrease-key, merge, stability under load.
- TaskManager: preemption logic, cancellation, retries, worker assignment.
- BVHTimeline: add/remove/crossfade; snapshot blending correctness.
- Mixer: track masking/weights; conflict resolution; deterministic composition.

Integration
- Conversation interrupt: verify ongoing animation is superseded within <200 ms; lip frames take precedence.
- Transition smoothness: no pose pops at boundaries (assert joint deltas < thresholds).
- Backend fallback: simulate GPU saturation and verify WASM fallback for short chunks.

E2E (Playwright)
- Scenario: Idle → speech → new gestures → stop. Assert console markers, frame counts, absence of long gaps; capture screenshots.
- Scenario: Import BVH while speaking; BVH tasks run but do not disrupt lip/gestures.

Artifacts
- Per-test reports in `dev/web_viewer/docs/reports/` with timings, preemption count, dropped/kept frames.

## 14) Telemetry & observability

- Emit scheduler events (queued, started, preempted, completed, canceled) with timestamps.
- Track per-task latency, chunk sizes, backend used, memory deltas.
- Timeline metrics: segments added/removed, active tracks, crossfade durations.

## 15) Rollout & milestones

M1 – Foundations (week 1–2)
- Implement BVHTimeline, TimelineSegment, TimelineMixer minimal.
- Wire TaskManager with FibonacciHeap preemption hooks.
- Add TaskFactory skeleton; implement BVHDecodeTask and TimelineOpTask.

M2 – Conversation path (week 3–4)
- Implement FaceFormerTask and Audio2GestureTask with small-chunk streaming.
- Add VAD event → ConversationGroup builder with priorities.
- Preemption policies + tests.

M3 – Transitions & locomotion (week 5–6)
- Implement RSMTTransitionTask for smooth clip changes.
- DeepMimicTask for base locomotion.
- Blend rules in Mixer per track (face/body/root).

M4 – Hardening (week 7+)
- Backpressure/fallback tuning, retries, metrics dashboards.
- E2E test coverage and performance budgets.

## 16) Edge cases to handle

- Rapid-fire interrupts (debounce conversation-start; keep a small hysteresis window).
- Long-running model inference: chunking, periodic yield points, cooperative cancel.
- Root motion conflicts: only one authoritative provider; short crossfade when switching.
- Clock drift: base all scheduling/mixing on a single monotonic clock.
- Memory pressure: enforce per-backend limits; drop low-importance tasks when necessary.

## 17) Mapping to existing code

- FibonacciHeap & TaskManager exist in `src/core/` – extend with preemption, cancel tokens, decrease-key API.
- Compute backends exist under `src/compute/` – use per-task resource hints to select device; support WASM fallback.
- Workers under `src/workers/` and `src/ai/workers/` – add cooperative cancel handling.
- Place new classes:
  - `src/models/bvh/BVHTimeline.js`, `src/models/bvh/TimelineMixer.js`
  - `src/core/task-factory/TaskFactory.js` (+ per-task files)
  - `src/core/scheduler/` (optional) for scheduler glue

## 18) Example flow (pseudo-code)

```js
// Build conversation job group
const group = buildConversationGroup(audioWindow, now());
for (const task of group.tasks) taskManager.submit(task);

// Preempt current low-priority gestures
taskManager.decreasePriority(currentGestureTaskId, 50);
taskManager.cancel(currentGestureTaskId, 'conversation-preempt');

// Mixer in render loop
function onFrame(t) {
  const pose = mixer.compose([faceTimeline, gestureTimeline, locomotionTimeline], t, rules);
  vrmAnimator.applyPose(pose);
}
```

This plan provides the technical blueprint to implement a preemptible, multi-source animation pipeline using a Fibonacci heap scheduler and a task factory, producing composable BVHTimelines to drive Ichika.
