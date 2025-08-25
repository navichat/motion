IchikaOrchestrator quickstart

- Purpose: Wire TaskScheduler -> TimelineChunkAdapter -> BVHTimeline to feed chunked frames into tracks with preemption and fades.

Usage (minimal):
- Construct with optional injected pieces to keep tests serverless:
  - `new IchikaOrchestrator({ timeline, adapter, scheduler, onEvent })`
  - If not provided, it will fallback to minimal-safe implementations.

Key APIs:
- `submitTask(task)`: task is an async generator (run) that yields { t0, dt, frames } chunks; include `id`, `trackId`, `priority`.
- `preempt(target, opts)`: target is a taskId or a trackId. If a taskId is provided, the orchestrator resolves its `trackId` via the scheduler and clears future clips only on that track (using the adapter). The cutoff time is the timeline's `currentTime`, so clips ending before `currentTime` are preserved.
- `runSlice(deadlineMs)`: time-sliced run loop step; call repeatedly.
- `bindAvatar(avatarApi)`: reserve for real Ichika VRM hookups.

Extended APIs in this branch:
- `startBaseClip(name)`: Looks up a registry entry and schedules it on its track. If the entry `data` is a prebuilt `BVHClip`, it passes it through to the adapter via `opts.clip`.
- `scheduleSpeechFromTts(tts, { fps, preempt, faceFadeInMs, gestureFadeInMs })`: Converts a simple TTS object into face/audio chunks, preempts those tracks, and appends with fades. The composed timeline metadata propagates `faceViseme` and `gestureEnergy` which the VRM integration maps to blendshapes.
- Bone masks: When provided via clip/frame metadata (`metadata.boneMask`), only those bones are affected during blending.

Demo & tests:
- Demo: `dev/web_viewer/demos/ichika_classroom_demo.html` wires ClipRegistry, StageController, and the orchestrator with Start/Point/Wave/Speak buttons.
- Unit-web checks (require dev server): speech preemption and action button logs.
- Serverless checks: viseme→blendshape, bone-mask blending, and prebuilt-clip passthrough.

Testing notes:
- Serverless specs inject the modules into a blank page, resolve constructors across window/commonjs shapes, and assert clips/events.
- Fade-in envelopes are applied in the adapter via frame metadata.weightEnvelope.
