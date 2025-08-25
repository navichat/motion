# Ichika BVH scheduler – quick context (for your editor side pane)

Purpose: One-page summary you can keep open while implementing the BVH animation pipeline with preemptible scheduling.

Core idea
- Use a Fibonacci-heap scheduler and worker pools (WebNN/WebGPU/WASM/CPU) to run short, chunked tasks that produce BVH frames.
- Each source writes to its own BVHTimeline (track). A TimelineMixer composes them into the final pose that drives the VRM avatar (Ichika).
- New, higher-priority events (e.g., speech) can preempt ongoing work and crossfade timelines smoothly.

Essential contracts
- Task (produced by TaskFactory)
	- fields: id, type ('faceformer'|'audio2gesture'|'rsmt'|'deepmimic'|'bvh-decode'|'timeline-op'), priority (lower is higher), preemptible, timelineId, resources { backend }, signal (AbortSignal)
	- run(ctx): Promise<{ ok, output?: { frames?: BVHFrame[]; segment?: TimelineSegment } }>
- BVHFrame: { time: number, channels: Map<string, number> }
- TimelineSegment: { startTime, duration, frames: BVHFrame[], track: string, meta?: any }
- BVHTimeline API (this repo):
	- constructor({ framerate = 30 })
	- addSegment(segment, opts?): number // returns new version
	- appendChunk(trackId, { t0, dt, frames }, opts?): number
	- clear(trackId, fromTime?): number
	- snapshot(t0, t1): BVHFrame[]
- TimelineMixer API (this repo):
	- compose(timelines: BVHTimeline[], t: number, rules?): FinalPose
	- FinalPose: Map<string bone, { position?, rotation?, scale? }>

Scheduling & preemption
- Queue: FibonacciHeap keyed by effective priority (insert/decrease-key O(1) amortized, pop-min O(log n)).
- Priority bands: speech/face (0–10) > gestures (11–30) > locomotion/transition (31–60) > background (61+).
- Preemption: cancel or let finish current chunk; requeue with small priority boost; insert short RSMT transition where possible; crossfade 80–160 ms.

Where things live (paths in this repo)
- Scheduler & pools: `src/utils/scheduler/{FibonacciHeap.js, TaskScheduler.js, WorkerPool.js}`
- BVH models (this change): `src/models/bvh/{BVHTimeline.js, TimelineMixer.js}`
- Timeline adapters & integrations: `src/components/animation/timeline/*`
- VRM hookup: `src/components/animation/vrm/*`

Minimal step-by-step
1) Implement BVHTimeline + TimelineMixer (lite versions) – done here.
2) Wire TaskScheduler on chunk outputs → use Timeline adapters to append to timelines.
3) In render loop, call TimelineMixer.compose(...) → send FinalPose to VRM.
4) Add preemption hooks: on speech start, cancel/soft-preempt prior face/gesture tasks and crossfade.

Testing checklist
- Heap ops and scheduler preemption (exists): `tests/unit/scheduler/*`
- New: BVHTimeline ops (append/clear/snapshot): `tests/unit/motion/bvh-timeline.spec.js`
- New: TimelineMixer compose returns a pose: `tests/unit/motion/timeline-mixer.spec.js`

Performance targets
- First lip/gesture chunk ≤ 100 ms; jitter < ±16 ms; chunk size 0.2–0.5 s.

See also
- Full plan: `docs/readmes/BVH_ANIMATION_SCHEDULER_PLAN.md`
- Orchestration notes: `docs/ICHIKA_SCHEDULER_ORCHESTRATION_PLAN.md`

