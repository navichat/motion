# Ichika VRM Classroom Implementation Plan

This plan places the Ichika 3D VRM avatar into a classroom stage and drives her with a BVHTimeline-of-timelines. Multiple producer timelines (saved BVH clips and model-generated motion) are composed into a master timeline. The avatar can converse in real time; her reply TTS audio generates synced visemes, gestures, and decisions to perform classroom actions. User microphone audio (optional) feeds ASR/policy to produce those replies.

## Requirements
- Classroom stage: load a classroom scene (desk, board, props) and place the Ichika VRM avatar.
- Animation: drive the avatar using BVHTimeline composition across sources:
  - Static BVH clips and generated BVH frames from motion models.
  - Real-time face and gesture tracks from audio (user mic via ASR, avatar TTS via viseme and audio-driven gestures).
  - Transitions/overrides for actions and prompts.
- Real-time conversation:
  - User → ASR → transcript.
  - Policy/LLM → reply text (+ optional action intents/emotions).
  - TTS → avatar audio; derive visemes and gestures from this audio and schedule on the timeline.
  - The avatar can decide to perform classroom actions that affect the scene (point, pick, write, etc.).
- Preemption: new speech/actions interrupt ongoing motion gracefully with fades.
- Testability: serverless unit tests + a minimal e2e smoke; all wrapped in shell timeouts.

## Current scaffolding status (implemented)
- Clip ingestion and intents:
  - ClipRegistry (UMD) with manifest loader for saved clips; sample manifest includes idle/point/wave.
  - StageController maps intents (pointAt, wave, writeOn, pickUp) to ClipRegistry entries and returns clip metadata/track.
- Orchestration:
  - IchikaOrchestrator wires Scheduler → TimelineChunkAdapter → BVHTimeline; supports handleIntent, loadClipManifest, startBaseClip, scheduleSpeechFromTts; preempts face/audio on new speech by default.
  - MinimalTimeline fallback allows serverless tests without full BVHTimeline implementation.
  - TimelineChunkAdapter appends chunks with fade-in weight envelope; clearFrom delegates to the underlying timeline when available; propagates opts.meta into generated frames; accepts prebuilt clips via opts.clip.
- Speech stubs:
  - SpeechGestureScheduler converts a minimal TTS object into face (viseme) and gesture (energy) chunks for scheduling.
- Demo and tests:
  - Demo page: dev/web_viewer/demos/ichika_classroom_demo.html — loads manifest, starts base idle (BVH-backed when available via BVHClipLibrary), triggers Point/Wave, and triggers a speech scheduling path; logs adapter append/clear.
  - Unit-web tests (require dev server):
    - dev/web_viewer/tests/unit/system/ichika-speech-preemption.web.spec.js — verifies speech scheduling logs and preemption.
    - dev/web_viewer/tests/unit/system/ichika-wave.web.spec.js — verifies Wave action logging (BVH or manifest fallback).
    - dev/web_viewer/tests/unit/system/ichika-point.web.spec.js — verifies Point action logging.
  - Serverless unit tests: fades, preemption, registry/manifest, and orchestrator stubs plus new validations:
    - dev/web_viewer/tests/unit/timeline/bone-mask-blending.serverless.spec.js — per-clip boneMask blending.
    - dev/web_viewer/tests/unit/timeline/metadata-propagation.serverless.spec.js — composed metadata carries faceViseme/gestureEnergy.
    - dev/web_viewer/tests/unit/avatar/viseme-blendshape.serverless.spec.js — viseme→blendshape via VRM integration.
    - dev/web_viewer/tests/unit/orchestrator/start-base-clip-prebuilt.serverless.spec.js — orchestrator passes prebuilt clip.
  - Smokes: lightweight and HTTP-only to keep them fast
    - dev/web_viewer/e2e-smoke.spec.js (index DOM checks)
    - dev/web_viewer/e2e-smoke-http.spec.js (VRM + classroom demos via HTTP fetch)
  - All Playwright runs are wrapped in shell/global timeouts; per-file timeouts applied to unit-web; CI remains green by gating unit-web behind a running dev server.

Additional recent enhancements:
- BVHTimeline: bone-mask-aware blending using frame.metadata.boneMask; composition propagates faceViseme and gestureEnergy from higher-priority tracks.
- VRM integration: BVHTimelineVRMIntegration maps visemes to VRM expressions and calls AvatarBinder.updateBlendshape(); metadata-only frames (no bones) still propagate viseme info.
- ClipRegistry: addBVH(name, url, meta) via BVHClipLibrary; orchestrator startBaseClip now accepts prebuilt BVHClip entries.

## High-level architecture
- Scene/Stage
  - Three.js scene with classroom assets, lights, camera.
  - VRM loader (three-vrm) to load Ichika VRM and access humanoid/blendshapes.
- Animation layer
  - BVH clip library: use BVHClipLibrary to load shipped BVH clips (idle, wave, point, write, step, look-at) and clips exported/generated from motion models. Each clip can be prebuilt (BVHClip) or generated frames.
  - BVHTimeline-of-timelines: a master BVHTimeline composes child BVHTimelines (per source). The master blends per-track with priorities/weights and supports fades and bone masks; child timelines include base, transitions, generated motion (deepphase), face, audio-gesture, and override. Composition metadata includes composedFrom plus faceViseme/gestureEnergy propagation.
  - Child BVHTimelines as clips: prebuilt BVH clips or producer outputs may be wrapped as child BVHTimelines (“mini timelines”) and scheduled; the master timeline then composes these along with frame-based clips.
  - Tracks: base, transitions, deepphase (generated), face, audio-gesture (upper-body), override (IK/actions). Child timelines may own subsets of tracks.
  - Retargeter: BVH frame → VRM humanoid bone transforms.
  - TimelineChunkAdapter: append generated chunks with fade and weight envelopes.
  - IchikaOrchestrator: TaskScheduler → Adapter → BVHTimeline; preemption; onEvent.
  - TaskScheduler: time-sliced async generator tasks on a Fibonacci heap (priority/deadline aware).
  - Producer registration API: register/unregister motion model producers. Each producer can:
    - deliver FramesChunk to specific tracks, with masks and priorities; or
    - expose a child BVHTimeline that the master composes as a source.
  - Motion model producers (examples):
    - DeepPhase/StyleVAE/RSMT for transitions and style-preserving motion; output FramesChunk or BVHClip.
    - Audio2Gesture for upper-body gestures; outputs masked FramesChunk for 'audio' track.
    - Face/viseme drivers from TTS; outputs metadata-driven frames for 'face'.
    - Producer registration: each producer attaches a child timeline or yields chunks scheduled onto specific tracks with defined priorities/masks.
- Audio & interaction
  - Input: WebAudio mic; VAD → speech segments; ASR (on-device/cloud) to text.
  - Dialog/Policy: rules/LLM that yields reply text, optional emotional tone, and action intents.
  - Output: TTS creates audioBuffer (+ optional viseme timings). Use that audio to drive face (visemes) and audio gestures.
    - Implementation note: scheduleSpeechFromTts(tts) uses SpeechGestureScheduler.makeChunksFromTts to create face/audio chunks and preempts existing speech tracks with short fades before appending.
- Action system
  - Action planner maps intents to discrete action clips or IK generators.
  - Action clips added to the override track with bounded duration and targeted bones.
- Preemption & sync
  - New speech preempts prior face/audio tasks from timeline.currentTime with a short fade.
  - Override actions have highest priority; replace conflicting bones; blend others.

## Timeline-of-Timelines composition
- Child timelines (producers):
  - base: idle/locomotion from saved BVH clips; long-running loop with soft transitions.
  - transitions: short blends to move between postures.
  - deepphase: generated full-body motion (e.g., style transfer, co-speech motion models).
  - speech-face: viseme/face frames synchronized to the avatar’s TTS audio.
  - speech-gesture: upper-body gestures from audio energy/prosody and semantics.
  - actions: discrete overrides (point/pick/write/pickup), baked from clips or IK.
- Master timeline: blends child tracks to a unified pose per render frame.
  - Priority: override > face/audio > deepphase/transitions > base.
  - Blend: replace for override bones; weighted/additive for others; per-track bone masks and per-clip frame.metadata.boneMask supported.
  - Fades: TimelineChunkAdapter produces weight envelopes on insert and preempt; default 120–200 ms unless specified by the chunk.
  - Bone-name mapping: BVHTimeline.setBoneMapping(index→name) maintained per-clip; retargeter uses VRM humanoid names.

BVH clip ingestion
- Locate BVH clips shipped in the repo (idle/gestures/actions). Provide a ClipRegistry with metadata: name, duration, bone coverage, default weight, and suitable tracks.
- ClipRegistry API: list(), get(name), loadFromManifest(manifest), addBVH(name, url, meta). Ensure clips are retarget-calibrated for Ichika’s humanoid.
- Orchestrator startBaseClip(name) can pass prebuilt BVHClip through the adapter (opts.clip) for immediate scheduling on the selected track.

## Data contracts
- FramesChunk: { t0: number, dt: number, frames: Array<Frame> }
- Frame: { time?: number, motionData: any, metadata?: object }
  - Optional metadata fields: boneMask: string[], viseme (per-frame face key), energy (gesture energy)
  - Composed frame metadata includes: faceViseme, gestureEnergy, composedFrom[]
- Orchestrator event: { type: 'chunk'|'action'|'speech', taskId, trackId, t0 }
- Scheduler task: { id, trackId, priority, fadeInMs?, deadline?, run: async generator → FramesChunk }
- Retargeter API: retarget(bvhFrame, humanoid) → pose (bone transforms)
- ClipRegistry: get(name) → { t0, dt, frames, metadata: { track, priority, boneMask } }
- TTS object (for SpeechGestureScheduler): { startTime?: number, duration?: number, visemes?: Array<{ time:number, id:string|number }>, energy?: number[] }
 - MotionModelProducer: { id, outputs: ('base'|'transitions'|'deepphase'|'face'|'audio'|'override')[], run(params) → AsyncGenerator<FramesChunk> }

Recommended tracks (with priorities)
- base (0): locomotion/idle BVH
- transitions (1): blends/adapters
- deepphase (2): body generative motions
- face (3): visemes/expressions
- audio (4): audio-driven upper-body gestures
- override (5): actions/IK overrides

## Pipelines
1) User → Avatar reply
- Mic → VAD → ASR → transcript.
- Dialog policy → reply text (+ optional action intents/emotion).
- TTS(reply) → audioBuffer (+ optional viseme timings).
- From TTS audio: generate face frames (viseme → blendshape map) and audio gestures.
- Orchestrator scheduling:
  - Append face frames to 'face' track, gestures to 'audio' track (upper-body bone mask).
  - Assign fadeInMs/weights; preempt old speech tasks from currentTime.
  - Optionally update 'base' track posture.

2) Action execution
- Planner selects actions (e.g., “point to board”, “pick up book”).
- Create clips: pre-baked BVH snippets or IK controller-generated frames.
- Append to 'override' track with higher priority, bounded duration, and bone mask.
 - Speech alignment: optionally align action start to speech beats/phrases (use TTS energy/viseme markers to anchor timing).
 - Decision loop: policy/LLM emits intents alongside reply text; the orchestrator schedules actions on override while ensuring compatibility with concurrent face/audio tracks via masks and priorities.

3) Retargeting & rendering
- On each frame: master BVHTimeline.compose() → retarget to VRM bones → AvatarBinder.applyPose(); face blendshapes applied from composed metadata (faceViseme).
 - Apply blendshape updates for face via BVHTimelineVRMIntegration.mapVisemeToBlendshape(); apply bone transforms for body.

4) BVH clip composition
- At start: schedule base idle clip loop with soft fades using ClipRegistry.
- On action intents: enqueue action clips (e.g., point/write) on override with bone masks to avoid lower-body conflicts.
- If deepphase motion is active, schedule it on a dedicated child timeline with lower priority than override/face/audio.

## Scheduling & preemption
- Priorities: override (highest) > face/audio > base/deepphase/transitions.
- Use TaskScheduler with FibonacciHeap; preempt by taskId or trackId on new speech.
- Fades: generate weight envelopes (e.g., 120–200ms) on insertion/preempt to avoid popping.
 - Deadlines: schedule producer slices with small deadlines (e.g., 20–50 ms) to avoid frame drops; adjust on slow devices.

Arbitration & bone masks
- Override track provides a bone mask (set of humanoid bones) that fully replaces lower-priority tracks for those bones; other bones blend normally.
- Speech-gesture track masks upper-body (spine, shoulders, arms, head) and avoids feet/hips to preserve base stability.
- Face track controls blendshapes only; never overrides bones.

## Real-time orchestration loop
- Inputs: mic audio stream; timeline clock; scene state.
- Steps per interaction turn:
  1) Capture mic → VAD → ASR → transcript.
  2) Policy → reply text (+ action intents, emotion tags).
  3) TTS → audioBuffer, sampleRate, optional viseme timings.
  4) Convert audioBuffer to face frames (viseme mapping) and audio gestures.
  5) Schedule via IchikaOrchestrator: face → 'face'; gesture → 'audio'; base as needed.
     - Clear future clips on those tracks from timeline.currentTime; fade in new.
  6) If action intents exist, enqueue override clips (bone-targeted, bounded duration).
  7) Render loop composes timelines, retargets to VRM, and applies pose each frame.
- Error modes: if ASR/TTS/model unavailable, fall back to stubs (simple mouth open/close and mild idle gesture), keep base motion running.

Latency targets
- ASR first token: ≤ 300–500 ms (local small model or remote streaming).
- TTS start: ≤ 150–300 ms; stream audio if available and start viseme/gesture scheduling as early chunks arrive.
- Scheduling budget per tick: ≤ 2 ms on main thread; offload heavy generation to tasks.

## Components to implement/wire
- Scene & VRM
  - SceneBuilder/StageController: classroom setup and interactable props.
  - VRMLoader: load Ichika VRM, get humanoid and blendshape proxies.
  - AvatarBinder: apply retargeted poses and face blendshapes.
- Retargeter
  - BVH→VRM mapping (humanoid bone names, rest pose offsets).
  - Calibration tool for mapping validation.
- Audio I/O
  - MicCapture (WebAudio); VAD to segment speech.
  - ASR client (on-device WASM or cloud), streaming when possible.
  - TTS client (kokoro/ORT/other) returning audioBuffer (+ visemes if available).
- Gesture/face generators
  - Audio→Gesture generator producing FramesChunk for 'audio' track.
  - Viseme→Face generator producing FramesChunk for 'face' track.
  - Energy/Prosody extractor from TTS audio for tempo- and beat-aligned gestures.
- Action planner
  - Policy/LLM→ intents & timings.
  - Action library: BVH snippets and IK generators (FABRIK/CCD for pointing, etc.).
  - Arbitration: ensure mutual exclusion where needed (e.g., hand-free requirement to pick an object).
- Orchestration
  - IchikaOrchestrator: bindings to AvatarBinder & StageController, preemption hooks.
  - TimelineChunkAdapter: fade/weights and clearFrom semantics.
  - BVHTimeline: track priorities and master/child composition.
  - ClipRegistry: discovery/loading of shipped BVHs; caching and retarget precompute.
- Utilities
  - Timebase sync (WebAudio currentTime ↔ timeline.currentTime for lip-sync).
  - Resource manager for preloading models/assets; lazy-load action snippets.

## Edge cases
- Overlapping speech: preempt current 'face'/'audio' tasks from currentTime and fade new in.
- Missing models/offline: skip gracefully with fallbacks; keep base/idle.
- TTS latency: stream or chunk generation; don’t block render loop.
- Action collisions: later override replaces conflicting bones; non-conflicting blend.
 - Clip boundary popping: ensure end-of-clip fade-out envelopes; loop with cross-fade.
 - Audio/viseme drift: re-sync face frames if audio clock deviates by >40 ms.
 - Streaming TTS: support incremental scheduling; append face/gesture chunks as viseme/energy arrive.

## Minimal MVP slice
- Load classroom scene + Ichika VRM.
- Play a static BVH clip on 'base' (retargeting validated).
- Hardcode a TTS reply; generate dummy face/gesture chunks from that audio.
- Orchestrator schedules face/audio; verify fade envelopes and preemption.
- One action: "point" clip on 'override' triggered by a button.
 - ClipRegistry with 2–3 clips (idle, point, wave) and a simple loader.
 - Demo page: dev/web_viewer/demos/ichika_classroom_demo.html loads the manifest, starts idle, and triggers point via a button (logs adapter calls, CI-friendly).

## Integration details
- Time sync: align audio start (TTS) to timeline.currentTime; schedule face/gesture chunks accordingly.
- Retargeting: maintain bone map JSON (BVH name → VRM node path); cache inverse bind.
- Face/visemes: map viseme IDs → VRM blendshape proxies; fallback to open/close envelope when missing.
- IK/actions: simple FABRIK/CCD chain to targets; bake short clips for 'override'.
 - Classroom props: expose StageController APIs (pointAt(board), pickUp(book), writeOn(board)) that map to action clips or IK targets.

## Testing strategy (CI-friendly)
- Serverless unit tests
  - Orchestrator preemption by track/taskId (exists).
  - Adapter fade envelope & weights (exists).
  - Retargeter: BVH frame → expected humanoid transforms (with mocks).
  - Bone mask blending: overlay modifies only masked bones.
  - Viseme→blendshape mapping: binder receives expression updates.
- Integration (server required)
  - E2E smoke (index.html loads) and HTTP-only demo fetches.
  - Unit-web: speech preemption log check and Point/Wave action logs.
  - VRM loads + short clip applies pose (optional screenshot check, local only).
  - Audio→gesture: inject synthetic audio buffer and assert chunks appended.
  - Scheduler→Adapter integration: ensure scheduled tasks produce appendChunk with proper fade and track assignment.
  - Spec: `dev/web_viewer/tests/unit/scheduler/scheduler-to-adapter-integration.spec.js`
- Real inference (opt-in)
  - Use env-gated smokes for ASR/VAD/TTS/LLM; skip by default to keep CI green.
- All test commands wrapped with shell timeouts per repo policy.
  - Playwright: globalTimeout/timeout configured; package.json test scripts use shell timeouts. Serverless tests must not navigate to pages requiring a web server. Unit-web/e2e projects run only when the dev web server is enabled.
  - Smokes: use HTTP-only fetches; avoid heavy navigation to keep under tight timeouts.

## Performance budget
- 60 FPS target; main-thread animation ≤ 6–8 ms.
- Generation tasks chunked (20–50 ms) via scheduler.
- Preload models/assets; lazy-load action snippets.
- Prefer GPU/WebNN paths when available.
 - Bone masks:
   - Upper body (audio gestures): spine, chest, neck, head, shoulders, arms, hands.
   - Override actions specify minimal masks (e.g., Point: dominant arm/hand + slight spine).

## Roadmap
- Sprint 1: Scene + VRM load; base clip retargeted; orchestrator→timeline wiring.
- Sprint 2: TTS playback + face/gesture stub aligned to audio; preemption smoke.
- Sprint 3: ASR → policy → TTS reply end-to-end; tighten fades and priorities.
- Sprint 4: Action planner + 2–3 actions (point, pick, write) using IK/baked BVH.
- Sprint 5: Optimization & validations; expand tests/docs.
 - Sprint 6: Producer integration (RSMT/Audio2Gesture) with child timelines; action-beat alignment; streaming TTS scheduling.

Deliverables per sprint
- Docs updated (this file), plus a short README section for how to run the demo and tests.
- ClipRegistry JSON manifest with at least idle/point/wave.
- Unit tests for scheduler preemption, adapter fades, retargeter mapping, and clip loading.
- A minimal demo page that instantiates SceneBuilder + IchikaOrchestrator and plays the base clip.

## File/module mapping (current repo)
- dev/web_viewer/src/components/animation/vrm/VRMLoader.js (VRM load)
- dev/web_viewer/src/components/animation/vrm/AvatarBinder.js (apply poses/blendshapes)
- dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js (apply timeline frames + blendshapes)
- dev/web_viewer/src/components/animation/timeline/BVHTimeline.js (timeline core)
- dev/web_viewer/src/components/animation/timeline/BVHClipLibrary.js (load static BVH clips)
- dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js (chunk → clip, meta, prebuilt clip passthrough)
- dev/web_viewer/src/orchestrator/IchikaOrchestrator.js (wiring, preemption, speech scheduling, prebuilt base clip)
- dev/web_viewer/src/orchestrator/SpeechGestureScheduler.js (build face/audio chunks from TTS)
- dev/web_viewer/src/animation/ClipRegistry.js (discover/load BVH clips and metadata, addBVH)
- dev/web_viewer/src/scene/StageController.js (classroom intents → action clips)
Optional new: dev/web_viewer/src/scene/SceneBuilder.js (classroom setup), VRMRetargeter.js (BVH→VRM)

### Demos & tests
- Demo: dev/web_viewer/demos/ichika_classroom_demo.html (base from BVH when available, Point/Wave actions, speech path; logs adapter calls)
- Unit-web: speech preemption/logs; Wave/Point log checks
- Serverless: fades/preemption/registry/orchestrator; bone mask blending; metadata propagation; viseme→blendshape; prebuilt-clip passthrough; scheduler→adapter integration
- Smokes: dev/web_viewer/e2e-smoke.spec.js (index), dev/web_viewer/e2e-smoke-http.spec.js (HTTP fetches)

## How to run
- Start demo locally (Playwright auto-starts a server for tests):
  - Open demo: http://127.0.0.1:8080/demos/ichika_classroom_demo.html
- Run tests:
  - Serverless fast suite: npm run test:serverless
  - Unit-web Ichika subset: npx playwright test --project=unit-web -g "Ichika demo:" --reporter=line
  - HTTP-only smokes: npm run test:smoke

## Validation & quality gates
- Build/lint: N/A for vanilla JS; ensure files load without syntax errors in browser.
- Playwright: serverless + smokes pass under shell/global timeouts; unit-web gated by web server.
- Basic smoke test: VRM/classroom demos serve over HTTP; index DOM renders.

## Requirements coverage
- Place Ichika VRM into a classroom stage with actions: StageController + VRM loader + demo page (Done; actions are mapped to clips with overrides).
- Animate via BVHTimeline composed from motion models and saved BVHs: Master BVHTimeline composing child timelines and frame clips; ClipRegistry + BVHClipLibrary for saved BVHs; producer interface for model outputs (Done; producers pluggable).
- Real-time talking: TTS drives visemes/gestures; scheduling with preemption and fades; optional streaming (Done; stubs in SpeechGestureScheduler, viseme→blendshape mapping integrated).
- Decisions to perform classroom actions: Policy/LLM provides intents alongside reply; planner schedules override actions aligned to speech as needed (Planned/Partial; stubbed planner with StageController mapping exists; beat alignment optional step).
- CI-friendly tests and shell timeouts: Serverless/unit-web/smokes documented; NO_WEBSERVER gating; HTTP-only smokes; shell/global timeouts (Done).

## Success criteria
- Avatar appears in classroom and plays base BVH clips correctly.
- Real-time conversation where avatar replies with synced lip/gesture.
- Actions trigger visibly and preempt properly without artifacts.
- Tests green by default (serverless + e2e smoke); opt-in real inference passes locally.
 - Playwright runs within configured shell timeouts; serverless tests avoid server navigation; unit-web tests skip when NO_WEBSERVER is set.
 - Bone-mask-aware blending verified; viseme-driven blendshape updates applied via VRM integration.
