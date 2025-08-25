/**
 * IchikaOrchestrator: wires TaskScheduler -> TimelineChunkAdapter -> BVHTimeline.
 * Minimal, framework-free orchestrator that is safe for serverless tests.
 */
(function(factory){
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory(require);
  } else {
  // In browsers, provide a null-returning require stub so optional requires don't appear truthy.
  const req = () => null;
    const api = factory(req);
    if (typeof window !== 'undefined') {
      // Provide both constructor and namespace styles for broad test compatibility
      const ctor = api.IchikaOrchestrator;
      // Primary: constructor at window.IchikaOrchestrator
      window.IchikaOrchestrator = ctor;
      // Also provide a namespace object for destructuring
      window.IchikaOrchestratorNS = api;
      // Allow destructuring from the function object itself
      if (ctor && typeof ctor === 'function') {
        ctor.IchikaOrchestrator = ctor;
      }
    }
  }
})(function(require){
  const g = (typeof window !== 'undefined') ? window : {};
  const TaskScheduler = (g.TaskScheduler) || (require && require('../utils/scheduler/TaskScheduler.js')?.TaskScheduler);
  const BVHTimeline = (g.BVHTimeline && g.BVHTimeline.BVHTimeline) || g.BVHTimeline || (require && require('../components/animation/timeline/BVHTimeline.js')?.BVHTimeline);
  const _TCA = g.TimelineChunkAdapter || (require && require('../components/animation/timeline/TimelineChunkAdapter.js')?.TimelineChunkAdapter);
  const TimelineChunkAdapter = (typeof _TCA === 'function') ? _TCA : (_TCA && _TCA.TimelineChunkAdapter);
  // Optional VRM pieces
  const _AB = g.AvatarBinder || (require && require('../components/animation/vrm/AvatarBinder.js')?.AvatarBinder);
  const AvatarBinder = (typeof _AB === 'function') ? _AB : (_AB && _AB.AvatarBinder);
  const _VRI = g.BVHTimelineVRMIntegration || (require && require('../components/animation/vrm/BVHTimelineVRMIntegration.js'));
  const BVHTimelineVRMIntegration = (typeof _VRI === 'function') ? _VRI : (_VRI && (_VRI.default || _VRI.BVHTimelineVRMIntegration || _VRI));
  const SGSns = (g.SpeechGestureScheduler) || (require && require('./SpeechGestureScheduler.js')) || null;
  const SpeechGestureScheduler = (SGSns && (SGSns.SpeechGestureScheduler || SGSns)) || null;

  // Minimal timeline fallback for serverless environments
  class MinimalTimeline {
    constructor() {
      this.currentTime = 0;
      this._tracks = new Map(); // trackName -> [{ startTime, duration, weight, blendMode, generator }]
    }
    addClip(trackName, clip) {
      const arr = this._tracks.get(trackName) || [];
      arr.push(clip);
      this._tracks.set(trackName, arr);
      return { track: trackName, index: arr.length - 1 };
    }
    clearFrom(trackName, fromTime = 0) {
      const arr = this._tracks.get(trackName) || [];
      const kept = arr.filter(c => (c.startTime || 0) < fromTime);
      this._tracks.set(trackName, kept);
    }
    dispose() { this._tracks.clear(); }
  }

  // Minimal scheduler fallback to keep orchestrator usable in serverless/unit environments
  class MinimalScheduler {
    constructor(opts = {}) {
      this.onChunk = opts.onChunk || (() => {});
      this.quantumMs = opts.quantumMs || 50;
      this.entries = new Map(); // id -> { task, it }
    }
    submit(task) {
      this.entries.set(task.id, { task, it: task.run() });
    }
    preempt(target) {
      for (const [id, { task }] of this.entries) {
        if (id === target || task.trackId === target) this.entries.delete(id);
      }
    }
    async runOnce(deadlineMs = 12) {
      const start = performance.now();
      for (const [id, entry] of Array.from(this.entries)) {
        if (performance.now() - start > deadlineMs) break;
        try {
          const res = await entry.it.next();
          if (!res.done) this.onChunk(entry.task, res.value);
          else this.entries.delete(id);
        } catch (e) {
          this.entries.delete(id);
        }
      }
    }
  }

  class IchikaOrchestrator {
    constructor(opts = {}) {
      // Core pieces (allow injection for testing)
      let TimelineCtor = BVHTimeline;
      if (typeof TimelineCtor !== 'function') {
        console.warn('[IchikaOrchestrator] Falling back to MinimalTimeline');
        TimelineCtor = MinimalTimeline;
      }
      this.timeline = opts.timeline || new TimelineCtor({ lookaheadFrames: 0 });
      // Optional: StageController and ClipRegistry
      this.stageController = opts.stageController || null;
      this.clipRegistry = opts.clipRegistry || null;
      // Create adapter with safe fallback when export shape differs
      if (opts.adapter) {
        this.adapter = opts.adapter;
      } else {
        try {
          this.adapter = new TimelineChunkAdapter(this.timeline);
        } catch (e) {
          console.warn('[IchikaOrchestrator] Using minimal adapter fallback:', e?.message || e);
          const tl = this.timeline;
          this.adapter = {
            appendChunk(trackName, chunk, _opts = {}) {
              const clip = {
                type: 'generated',
                startTime: chunk.t0,
                duration: chunk.dt,
                weight: 1,
                blendMode: 'replace',
                generator: async () => (chunk.frames?.[0] || { motionData: [], metadata: {} })
              };
              return tl.addClip(trackName || 'base', clip);
            }
          };
        }
      }
  this.onEvent = opts.onEvent || (() => {});

      const onChunk = (task, chunk) => {
        // Append to the task's target track
        try {
          this.adapter.appendChunk(task.trackId || 'base', chunk, { fadeInMs: task.fadeInMs ?? 80 });
          this.onEvent({ type: 'chunk', taskId: task.id, trackId: task.trackId, t0: chunk.t0 });
        } catch (e) {
          console.warn('[IchikaOrchestrator] appendChunk failed', e);
        }
      };

      let SchedulerCtor = TaskScheduler;
      if (typeof SchedulerCtor !== 'function') {
        console.warn('[IchikaOrchestrator] Falling back to MinimalScheduler');
        SchedulerCtor = MinimalScheduler;
      }
      this.scheduler = opts.scheduler || new SchedulerCtor({ quantumMs: opts.quantumMs || 50, onChunk });
    }

    // Registry helpers to ingest clip manifests and start base idle
    setClipRegistry(registry) { this.clipRegistry = registry; }
    loadClipManifest(manifest) {
      if (!this.clipRegistry) {
        const CRns = (typeof window !== 'undefined') ? (window.ClipRegistry || null) : null;
        const CRCtor = CRns && (CRns.ClipRegistry || CRns);
        if (CRCtor && typeof CRCtor === 'function') this.clipRegistry = new CRCtor();
      }
      if (!this.clipRegistry || typeof this.clipRegistry.loadFromManifest !== 'function') return 0;
      this.clipRegistry.loadFromManifest(manifest);
      return (this.clipRegistry.list && this.clipRegistry.list().length) || 0;
    }
    startBaseClip(name = 'idle') {
      if (!this.clipRegistry) return null;
      const entry = this.clipRegistry.get(name);
      if (!entry) return null;
      const isClipObject = entry.data && typeof entry.data === 'object' && (entry.data.bvhData || entry.data.generator || entry.data.type);
      const effectiveChunk = isClipObject
        ? { t0: 0, dt: (entry.meta && entry.meta.duration) || 0, frames: [] }
        : (entry.data || { t0: 0, dt: (entry.meta && entry.meta.duration) || 2.0, frames: [] });
      const track = (entry.meta && entry.meta.track) || 'base';
      this.adapter.appendChunk(track, effectiveChunk, {
        clip: isClipObject ? entry.data : undefined,
        fadeInMs: (entry.meta && entry.meta.fadeInMs) || 150,
        boneMask: (entry.meta && entry.meta.boneMask) || [],
        priority: (entry.meta && entry.meta.priority) || 0
      });
      this.onEvent({ type: 'start-base', clip: entry.name, track, t0: effectiveChunk.t0 });
      return entry;
    }

    // From a TTS object (duration, visemes, energy), schedule face/audio tracks with fades and preemption
    scheduleSpeechFromTts(tts, opts = {}) {
      if (!SpeechGestureScheduler) return null;
  // Ensure chunks start at current timeline time by default, so late scheduling doesn't land in the past.
  const now = (this.timeline && typeof this.timeline.currentTime === 'number') ? this.timeline.currentTime : 0;
  const ttsWithStart = (typeof tts?.startTime === 'number') ? tts : { ...tts, startTime: now };
  const { faceChunk, gestureChunk } = SpeechGestureScheduler.makeChunksFromTts(ttsWithStart, { fps: opts.fps || 30 });
      // Preempt existing speech tracks if requested
      if (opts.preempt !== false) {
        try { this.preempt('face'); } catch {}
        try { this.preempt('audio'); } catch {}
      }
      this.adapter.appendChunk('face', faceChunk, { fadeInMs: opts.faceFadeInMs || 120, meta: { source: 'tts' } });
      this.adapter.appendChunk('audio', gestureChunk, { fadeInMs: opts.gestureFadeInMs || 120, boneMask: ['spine','neck','leftArm','rightArm','leftForeArm','rightForeArm','leftShoulder','rightShoulder'], meta: { source: 'tts' } });
      this.onEvent({ type: 'speech', tracks: ['face','audio'], t0: faceChunk.t0 });
      return { faceChunk, gestureChunk };
    }

    // Handle a high-level intent (e.g., 'pointAt') and schedule the corresponding clip on override track
    handleIntent(intent, params = {}) {
      const stage = this.stageController;
      if (!stage || typeof stage.mapIntentToClip !== 'function') return null;
      const entry = stage.mapIntentToClip(intent, params);
      if (!entry || !entry.meta) return null;
      // Compose a minimal FramesChunk for scheduling
      const chunk = entry.data || { t0: 0, dt: entry.meta.duration || 1.0, frames: [] };
      // Schedule on override track with fade and boneMask
      this.adapter.appendChunk(entry.meta.track || 'override', chunk, {
        fadeInMs: entry.meta.fadeInMs || 120,
        boneMask: entry.meta.boneMask || [],
        priority: entry.meta.priority || 5
      });
      this.onEvent({ type: 'intent', intent, clip: entry.name, track: entry.meta.track, t0: chunk.t0 });
      return entry;
    }

    submitTask(task) { this.scheduler.submit(task); }
    preempt(target, opts = {}) {
      this.scheduler.preempt(target, opts);
      // Best-effort: clear future clips on the target track after preemption
      try {
        if (this.adapter && typeof this.adapter.clearFrom === 'function') {
          // Resolve track: if target is a taskId, look up its trackId.
          let trackId = target;
          if (this.scheduler && this.scheduler.taskById && this.scheduler.taskById.has && this.scheduler.taskById.has(target)) {
            const t = this.scheduler.taskById.get(target);
            if (t && t.trackId) trackId = t.trackId;
          }
          const fromTime = (this.timeline && typeof this.timeline.currentTime === 'number') ? this.timeline.currentTime : 0;
          this.adapter.clearFrom(trackId, fromTime);
        }
      } catch (e) {
        // non-fatal
      }
    }
    async runSlice(deadlineMs = 12) { return this.scheduler.runOnce(deadlineMs); }

    bindAvatar(avatarApi) {
      // Provide convenient wiring for VRM avatars.
      // Accept either:
      //  - an AvatarBinder-like object (has updateBone)
      //  - a VRM instance (has humanoid)
      //  - an object { vrm }
      this.avatar = avatarApi;
      try {
        // Resolve binder
        let binder = null;
        if (avatarApi && typeof avatarApi.updateBone === 'function') {
          binder = avatarApi;
        } else if (AvatarBinder) {
          const vrm = (avatarApi && (avatarApi.vrm || (avatarApi.humanoid ? avatarApi : null))) || null;
          binder = new AvatarBinder(vrm);
        }

        // Resolve integration
        if (binder && BVHTimelineVRMIntegration) {
          this.vrmIntegration = new BVHTimelineVRMIntegration(binder, { smoothing: false, framerate: 30 });
          if (typeof this.vrmIntegration.connectTimeline === 'function') {
            this.vrmIntegration.connectTimeline(this.timeline);
          }
          this.binder = binder;
          return { binder, integration: this.vrmIntegration };
        }
      } catch (e) {
        // Non-fatal; fall back to storing avatar only
      }
      return { binder: null, integration: null };
    }
    dispose() { this.timeline?.dispose?.(); this.avatar = null; }
  }

  return { IchikaOrchestrator };
});
