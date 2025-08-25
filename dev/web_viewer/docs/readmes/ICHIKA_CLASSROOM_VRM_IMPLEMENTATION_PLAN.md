# Ichika VRM in Classroom — Implementation Plan

Purpose: Integrate the Ichika 3D VRM avatar into the classroom stage, animate it via composited BVH timelines from our motion models and BVH frames, and enable real-time conversation where the avatar speaks, gestures, and performs classroom actions based on audio and decisions.

## Status update (August 2025)
- Core animation stack in place and validated with unit-web tests:
  - BVHTimeline composition, priority order, seeking, fade-weight semantics, metadata propagation, buffer invalidation, and prebuffer behaviors are covered by deterministic tests.
  - VRM integration smokes pass (adapter receives bone updates; metadata propagates). A viseme driver smoke validates viseme→blendshape mapping.
  - TimelineMixer smokes pass standalone and with two real BVHTimelines.
- Playwright config enforces shell timeouts and IPv4-only baseURL; PW_GREP enabled for CI targeting of VRM tests.
- BVH asset library + chunk adapter produce BVHClip consistently across generated and static sources, ensuring getFrameAtTime works during composition.

## Requirements (checklist)
- Place Ichika (VRM humanoid) into the classroom stage and render it.
- Animate Ichika by compositing BVHTimelines from both model-generated motion and saved BVH files that already live in the repo (e.g., under `dev/web_viewer/MotionData` and similar asset folders).
- Drive animation using BVHTimeline per track with a TimelineMixer to produce the final pose (gesture, emotes, actions, locomotion, visemes).
- Real-time conversation loop (avatar-centric):
  - User talks → ASR → intent/response → TTS audio for avatar.
  - Use the avatar’s own TTS PCM/clock as the master sync to generate gestures (Audio2Gesture) and visemes (Faceformer or blendshape driver) in real time.
  - Preempt/adjust running motions when new speech or decisions arrive.
- Classroom actions: avatar can decide and perform stage actions (point, walk, interact) in the classroom based on dialogue/policy outputs.

Deliverable clarification:
- Final composition is explicitly the mix of multiple BVHTimelines: some fed by motion models (Audio2Gesture, RSMT/DeepMimic, Faceformer/visemes) and others by saved BVH files from the repo via BVHClipLibrary; the composed pose drives the VRM avatar in the classroom stage.

## High-level architecture
- Scene/Stage
  - Classroom stage loader with object registry and semantic anchors (e.g., board, desk, door).
- Avatar subsystem (VRM)
  - VRM loader (GLTF/VRM), humanoid bone map, runtime pose applier.
  - BVH→VRM retargeter (Humanoid mapping + scale + root motion policy).
- Motion/Animation pipeline
  - TaskScheduler (Fibonacci heap) orchestrates short jobs that yield BVH chunks.
  - TimelineChunkAdapter converts chunks to timeline clips with crossfades.
  - BVHTimeline (per source/track) + TimelineMixer to compose final pose.
- BVH Asset Library
  - BVH clip loader to ingest curated/saved BVH files from the repo.
  - BVHClipLibrary index with searchable metadata (id, category, duration, joint set).
  - Motion primitives map to library clips (e.g., wave, point, sit/stand, walk steps).
- Audio/Conversational pipeline
  - ASR frontend (browser mic or server) → text + timestamps.
  - Policy/LLM for dialogue + action decisions.
  - TTS to audio stream for avatar voice with word/phoneme timings when available.
  - Audio2Gesture model (onnx) to generate gesture BVH chunks from audio features.
- Actions/Planning
  - Simple behavior planner mapping intents to classroom actions (point, move, show).
  - Motion primitive library feeding BVH clips per action.

## Concrete modules and paths (current wiring)
- Timeline core
  - `dev/web_viewer/src/components/animation/timeline/BVHTimeline.js`
  - `dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js`
  - `dev/web_viewer/src/components/animation/timeline/BVHClipLibrary.js`
  - `dev/web_viewer/src/models/bvh/TimelineMixer.js`
- VRM integration
  - `dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js` (retargeter entrypoint, viseme→blendshape mapping)
- Tests (unit-web)
  - `dev/web_viewer/tests/unit/animation/*.spec.js` including VRM smokes and viseme driver
- Playwright config and policies
  - `playwright.config.js` (IPv4 baseURL, webServer timeout, PW_GREP support)
  - `.github/instructions/playwright.instructions.md` (shell timeout policy)

## Data flow (realtime loop)
1) User speech captured → ASR → transcript with timestamps.
2) Dialogue policy produces response text + optional actions (and priorities).
3) TTS starts streaming avatar audio; expose audio playback clock and phoneme/word timings (when available).
4) Derive audio features from the avatar’s TTS PCM → Audio2Gesture model → emit BVH gesture chunks on the `gesture-upper` track.
5) Visemes from phonemes (Faceformer or blendshape mapping) → drive `speech-mouth` track or blendshapes directly.
6) Actions selected → enqueue motion primitive generators (BVH chunks) per action track; may combine model-generated chunks and BVHClipLibrary clips.
7) TaskScheduler schedules/merges chunks; TimelineChunkAdapter appends to BVHTimelines with fades; TimelineMixer composes → retarget to VRM → render.
8) New user input or policy changes preempt currently running tasks per priority (clearFrom + small crossfade).

Notes:
- The master clock is the avatar TTS playback time; gesture chunks (Audio2Gesture) and visemes align to this clock for natural sync.
- Generated chunks and library clips are both wrapped as BVHClip so clip sampling and composedFrom metadata are consistent across sources.

## Key contracts (APIs)
- Task (generator)
  - Input: context { clock, framerate, abortSignal, params }
  - Output: yields { t0, dt, frames } BVH chunks; optional { weight, meta }
  - Error modes: abort on signal; yield partial chunks; backoff exception.
- TimelineChunkAdapter
  - appendChunk(trackName, { t0, dt, frames }, { fadeInMs?, weight? }) → clipId
  - clearFrom(trackName, tFrom) for preemption/overwrite.
- BVHTimeline
  - constructor({ framerate, lookaheadFrames? }); appendChunk; clear; snapshot.
- TimelineMixer
  - compose(t, rules?) → FinalPose (humanoid local transforms).
- Retargeter
  - apply(finalPose, avatar) with humanoid mapping & root policy.
- Stage actions
  - action(id, params) → task generator yielding BVH chunks on dedicated tracks (e.g., locomotion, upper-body point).
- BVHClipLibrary
  - load(paths[]) → index; get(id|tag) → { framerate, frames, meta }
  - clipToChunks(clip, { t0, dt, loop?, window? }) → iterable of { t0, dt, frames }
  - normalize(clip, { scale, boneMap }) to align with VRM bone set.
- AudioFeatureExtractor
  - fromAudioNode(audioContext, sourceNode, { hopMs, winMs }) → async iterator of feature tensors.
- VisemeDriver
  - fromPhonemes(phonemes, timings) → timeline events for speech-mouth track or direct VRM blendshape updates.

Semantics notes (tested):
- composedFrom.weight equals the clip’s configured weight; the fade-in envelope is tracked on frame metadata (e.g., `weightEnvelope`) but is not multiplied into composedFrom.
- Bone mapping may be an index→name map for minimal clips; VRM integration includes a default name map that can be overridden.

## Tracks and priorities
- Tracks: speech-mouth, gesture-upper, emote-face, locomotion, attention/aim, action-upper.
- Priorities (highest first): emergency/interrupt > speech-mouth sync > locomotion/action > gesture/emote.
- Crossfade defaults: speech-mouth none; gesture 150–250ms; locomotion 300–600ms; emote 100–200ms.
 - Source types per track:
   - gesture-upper: Audio2Gesture chunks (live) + BVHClipLibrary overlays (e.g., stylistic beats).
   - locomotion: DeepMimic/RSMT chunks + BVH walk cycles from library.
   - action-upper: pointed actions (point, wave) sourced from library or procedural.
   - speech-mouth: viseme driver aligned to TTS timeline.

Mixer behavior:
- Last-writer-wins for overlapping channels unless a track-specific blend mode dictates additive behavior. Replace vs additive is configured per-clip.

## Synchronization and latency
- Master clock: avatar audio playback time for speech-aligned tracks.
- Latency budgets: end-to-end < 150–250ms for perceived sync; chunk size ~100–200ms.
- Lookahead: 0.5–1.0s where available; preempt with immediate clearFrom + short crossfade.
- Audio2Gesture: use sliding window (e.g., 20–40ms hop) and aggregate to BVH chunks.
 - BVH library clips: prefetch next seconds of clip and align to current mixer time; trim/loop with clean boundaries.

## Retargeting (BVH → VRM Humanoid)
- Bone mapping table (BVH joint names → VRM humanoid bones) with calibration step.
- Root motion policy: options for stationary upper-body vs full-body locomotion.
- Scale normalization: bio-scale to avatar height.
- Jaw/viseme mapping: phoneme timings to VRM blendshapes.

Implementation detail:
- `BVHTimelineVRMIntegration.mapVisemeToBlendshape` maps common visemes/IDs to VRM expression names (e.g., 'A' → 'aa'); when available, we can switch to Faceformer for richer mouth motion.

## Classroom integration
- Placement: avatar anchor transform in classroom coordinates; configurable seat/stance.
- Interaction: action primitives to point at anchors, walk to location, look-at objects.
- Object registry: semantic IDs and transforms; expose to planner.
 - Stage policy: planner reads object registry and current dialogue intent to schedule actions on appropriate tracks with priorities.

Action examples:
- Point at board (upper-body action track) while speaking; walk to desk (locomotion track) between utterances; look-at student (attention/aim track) during Q&A.

## Components to implement/wire
- VRM loader & pose applier under src/components/animation/vrm/ (or equivalent).
- Retargeter utility: BVH joint map → VRM humanoid; blendshape driver for visemes.
- Audio pipeline adapters: ASR client, TTS client, audio clock provider, audio feature extractor.
- Gesture generator task using audio2gesture onnx (already present) emitting BVH chunks.
- Motion primitives library (BVH snippets) and task wrappers (walk, point, wave).
- Behavior planner: maps intents to tasks with priorities; publishes to scheduler.
- Scheduler wiring: one TaskScheduler instance with onChunk → adapter.appendChunk.
- Timelines: per-track BVHTimeline instances; TimelineMixer into final pose; retarget to VRM.
- UI hooks: mic toggle, speak button, debug overlays (timelines, priorities), latency meters.
- BVH asset ingestion: BVHClipLibrary (scan `dev/web_viewer/MotionData` and other BVH folders), clip normalizer, and clip→chunk adapter.
- Saved BVH playback: action primitives can reference library clip IDs; locomotion can stitch step cycles with RSMT/DeepMimic transitions.

Operational notes:
- Configure model URLs in centralized config and enable dynamic loader flags to prefer legacy modules for smokes when needed.
- Ensure IPv4-only baseURL (127.0.0.1) in tests; keep shell timeouts on all Playwright invocations.

## Machine Learning Models (in scope)

This plan integrates several ML models. For each, we list expected inputs/outputs, runtime, and how it plugs into timelines.

### Audio2Gesture (ONNX)
- Purpose: Generate upper-body gesture motion from speech audio.
- Model artifact(s):
  - Primary: `audio2gesture_step_fixed.onnx` (repo root)
  - Test data: optional JSON fixtures for local validation when available
- Runtime: onnxruntime-web (WebGPU preferred; WASM fallback)
- Inputs/Outputs (contract):
  - Input: windowed audio features from speech audio (e.g., log-mel or MFCCs); produced by a Web Audio-based feature extractor. Sliding window hop 20–40 ms.
  - Output: per-frame joint transforms suitable for BVH frames (positions/rotations for upper-body VRM bones). We wrap outputs into BVH frames: `{ time, motionData, metadata }`.
- Scheduler integration:
  - A gesture task consumes an audio clock and feature stream and yields BVH chunks: `{ t0, dt, frames }` with dt ≈ 0.2–0.4s and small crossfades (150–250 ms).
- Notes: If model input feature shape differs, adapt in the AudioFeatureExtractor adapter and validate shapes at runtime; log-sample a first inference during init.

Implementation note: prefer driving Audio2Gesture from the avatar’s TTS PCM so gesture timing reflects what Ichika actually says (not just what the user said). Maintain a small buffer to absorb playback jitter and align chunk boundaries to the TTS clock.

### RSMT (DeepPhase → StyleVAE → TransitionNet)
- Purpose: Real-time stylized motion transitions between motion states.
- Model artifacts (see RSMT docs): `deepphase.onnx`, `stylevae.onnx`, `transitionnet.onnx`.
- Runtime: onnxruntime-web (WASM/WebGPU). Batch-friendly; supports caching.
- Inputs/Outputs (from RSMT completion summary):
  - DeepPhase: input 132-d skeleton channels → 32-d phase
  - StyleVAE: 32-d phase → 8-d manifold
  - TransitionNet: 48-d manifold/params → 132-d skeleton
- Scheduler integration:
  - An RSMT transition task yields a BVH clip bridging source→target motions with style metadata. Typical transition length 20–60 frames.
- Placement: ensure models are hosted or packaged; configure URLs in the RSMT inference module.
 - Library tie-in: when transitions are long or CPU-bound, fall back to library bridge clips and crossfade with shorter model-generated transitions.

### DeepMimic (motion synthesis to BVH)
- Purpose: Synthesize locomotion/acrobatics and other motions; stream into timelines.
- Model artifacts: ONNX models as configured by `DeepMimicInference` (see DeepMimic BVH Integration Guide). Exact filenames configurable.
- Runtime: onnxruntime-web (WASM/WebGPU) or server-side when needed.
- Inputs/Outputs (contract):
  - Input: target motion params (speed, direction, turn rate, etc.) and/or prior state.
  - Output: per-frame BVH-compatible skeleton frames for VRM bone set.
- Scheduler integration:
  - Real-time DeepMimic task produces rolling BVH chunks for locomotion/action tracks; longer crossfades (300–600 ms).
 - Library tie-in: seed/terminate locomotion with library clips (start/stop/turn), and use DeepMimic for in-between synthesis where feasible.

### ASR/TTS and Visemes
- ASR: Adapter interface for capturing transcript + word timings; model may be external (cloud) or local (when available). Not required for offline demos.
- TTS: Adapter that returns audio stream and phoneme/word timings; we use the audio playback clock as the master sync for speech-aligned tracks.
- Visemes: If phoneme timings are available or FaceFormer-like mouth models are present, drive VRM blendshapes via a `VisemeDriver` that yields a `speech-mouth` track or directly manipulates blendshapes.
 - Clock contract: TTS adapter exposes a high-resolution playback clock and onBoundary events (word/phoneme) to align visemes and gesture chunk boundaries.

### Runtime providers and model loading
- Default provider order: WebGPU → WASM. Detect capabilities and fall back robustly.
- Model path configuration: keep URLs centralized (e.g., `dev/web_viewer/models/*.onnx` or CDN). Use `dev/web_viewer/config/models.config.js` to set per-model URLs. For `audio2gesture_step_fixed.onnx`, reference via a stable path or copy into the viewer’s served assets.
- Health checks: during initialization, run a tiny shape-probe inference and record provider, latency, and any warnings in diagnostics UI/logs.

### Common scheduler contract for model tasks
- Input: context with `{ clock, framerate, abortSignal, featureProviders }`.
- Output: yields `{ t0, dt, frames }` with metadata `{ model, provider, latencyMs }`.
- Error modes: abort promptly; yield partial chunks when possible; expose error in metadata for UI.
 - BVH library contract: library-backed tasks should annotate metadata with `{ source: 'library', clipId }` for debugging and provenance.

## Testing strategy
- Unit
  - Adapter fade/crossfade and clearFrom behavior across tracks.
  - Mixer composition rules and weight handling.
  - Retargeter bone mapping for sample poses; viseme mapping from phonemes.
  - Gesture generator: deterministic output on known audio fixtures.
- Integration (Playwright)
  - Serverless: scheduler → adapter → minimal BVHTimeline composing mock chunks.
  - Web-backed: classroom loads with VRM; apply sample composite pose without errors.
  - Audio sync: simulated audio clock drives gesture chunks; verify timestamp alignment.
  - BVH assets: load and play saved BVH clip on avatar; ensure correct scaling and bone mapping.
- E2E
  - Talk to avatar (mock ASR/TTS): mouth movement + gesture present within tolerance.
  - Trigger a classroom action (e.g., point at board) while speaking; ensure priority blending works.
  - Mixed-source composition: blend a saved BVH clip (e.g., wave) with model-generated gesture while speaking; verify crossfade metrics.
- Infra
  - Keep webServer timeout configured (120s) and explicit action/navigation/expect timeouts.

Verification artifacts (present):
- Unit-web animation tests (all green):
  - Composition, priority order, seek around overlaps.
  - Fade-weight behavior (composedFrom equals clip weight).
  - Metadata propagation (viseme/energy), buffer invalidation, prebuffer behavior.
  - VRM integration smokes and viseme driver mapping to blendshapes.
  - TimelineMixer smokes (standalone and with two BVHTimelines).
- How to run:
  ```bash
  # All animation unit-web tests (starts server automatically)
  npm run test:unit:animation

  # Only VRM-related tests
  PW_GREP="VRM|viseme" npm run test:unit:web

  # Single spec
  npx playwright test --project=unit-web dev/web_viewer/tests/unit/animation/vrm-viseme-driver-web.spec.js --reporter=line
  ```

## Performance targets
- 60 FPS render on mid-tier laptop GPU; <4ms CPU per frame for animation apply.
- Gesture/model inference <20ms per 200ms chunk on CPU; cache features when possible.
- Memory: avoid unbounded timelines; prune old clips; reuse buffers.

## Milestones
1) Scene + VRM
  - Classroom scene loads; VRM avatar loads and idles.
2) BVH retargeter + static playback
  - Play a BVH clip on the avatar through BVHTimeline → Mixer → Retargeter.
3) Scheduler + adapter wiring
  - Push scheduled mock chunks on multiple tracks; validate fades/preemption.
4) Audio loop + gesture
  - TTS playback with audio clock; audio2gesture produces synced gestures.
5) Behavior planner + actions
  - Map intents to actions; point/walk primitives feeding timelines with priorities.
6) Realtime conversation demo
  - Full loop with mic input → reply speech + gestures + action in classroom.
7) Hardening + tests
  - Add unit/integration/e2e tests; polish UX and instrumentation.
8) BVH asset library integration
  - Index saved BVH files; normalize to VRM bone set; demonstrate playback and blending with model-generated motion.

Next milestone deltas (from current status):
- Wire the full real-time conversation loop with live ASR and TTS adapters, using the TTS clock as master and the Audio2Gesture stream plugged into the gesture track.
- Add planner-driven classroom actions (point, walk) with priorities and verify blending while speaking.
- Add a minimal E2E demo page with Ichika in the classroom executing the full pipeline; include a Playwright smoke to validate render + motion within a short window.

## Risks and mitigations
- Retargeting mismatch → add calibration tool and fallback bone sets.
- Audio sync drift → use audio hardware clock, periodic resync, and short fades.
- Preemption artifacts → enforce clearFrom + minimal crossfade; keep chunk sizes small.
- Performance bottlenecks → profile, reduce allocations, and downgrade detail when needed.
 - BVH asset variability → normalize frame rates, joint naming, and scale; precompute metadata and test representative clips.

## Deliverables
- Scene integration (classroom) with Ichika VRM avatar.
- VRM retargeter and blendshape driver utilities.
- Scheduler/adapter/timeline wiring with per-track timelines and mixer.
- Audio pipeline (ASR/TTS adapters) and gesture generator (onnx) hooked to scheduler.
- Behavior planner for stage actions.
- Test suites (unit, serverless, web-backed, e2e) with shell timeouts configured.

Quality gates
- Build/lint/typecheck clean for web viewer modules.
- Unit-web animation + VRM tests green in CI (workers=1 with retries).
- Optional E2E smoke passes under `RUN_FULL_E2E=1`.
