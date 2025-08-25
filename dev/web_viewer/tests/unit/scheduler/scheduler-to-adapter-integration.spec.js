import { test, expect } from '@playwright/test';
test.skip(!!process.env.NO_WEBSERVER, 'Requires web server');

// Validates TaskScheduler -> TimelineChunkAdapter -> BVHTimeline wiring

test('scheduler appends chunks to BVHTimeline via TimelineChunkAdapter', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: '/src/utils/scheduler/FibonacciHeap.js' });
  await page.addScriptTag({ url: '/src/utils/scheduler/TaskScheduler.js' });

  const result = await page.evaluate(async () => {
    // Resolve constructors via CommonJS shims to avoid window shape issues
    const [tsRes, tlRes, adRes] = await Promise.all([
      fetch('/src/utils/scheduler/TaskScheduler.js'),
      fetch('/src/models/bvh/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tsCode, tlCode, adCode] = await Promise.all([tsRes.text(), tlRes.text(), adRes.text()]);
    const tsMod = { exports: {} };
    const tlMod = { exports: {} };
    const adMod = { exports: {} };
    const TaskSchedulerCtor = (new Function('window','module','exports', tsCode + '; return module.exports && module.exports.TaskScheduler;'))(window, tsMod, tsMod.exports);
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const TimelineChunkAdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    if (typeof TaskSchedulerCtor !== 'function') throw new Error('TaskScheduler ctor missing');
    if (typeof BVHTimelineCtor !== 'function') throw new Error('BVHTimeline ctor missing');
    if (typeof TimelineChunkAdapterCtor !== 'function') throw new Error('TimelineChunkAdapter ctor missing');
    const scheduler = new TaskSchedulerCtor({ quantumMs: 50 });
    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);

    scheduler.onChunk = (task, chunk) => {
      adapter.appendChunk(task.trackId || 'audio', chunk, { fadeInMs: 50 });
    };

  // Test task yielding two chunks across scheduler cycles
    function makeTask(id, trackId, priority) {
      return {
        id,
        trackId,
        priority,
        abort: new AbortController(),
        run: async function* () {
      // Generate 6 frames for 0.2s chunk at 30fps (two sequential chunks)
      const frames1 = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { id, k: 1 } }));
      const frames2 = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { id, k: 2 } }));
      yield { t0: 0.0, dt: 0.2, frames: frames1 };
      yield { t0: 0.25, dt: 0.2, frames: frames2 };
        }
      };
    }

    const t = makeTask('T1', 'audio', 1);
    scheduler.submit(t);

    // Run a few cycles to process
    for (let i = 0; i < 3; i++) {
      await scheduler.runOnce(100);
    }

    return {
      clipCount: timeline.tracks.audio ? timeline.tracks.audio.clips.length : 0,
      hasTrack: !!timeline.tracks.audio,
    };
  });

  expect(result.hasTrack).toBe(true);
  expect(result.clipCount).toBeGreaterThanOrEqual(2);
});

test('scheduler preempt triggers clearFrom to drop future chunks on track', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: '/src/utils/scheduler/FibonacciHeap.js' });
  await page.addScriptTag({ url: '/src/utils/scheduler/TaskScheduler.js' });

  const result = await page.evaluate(async () => {
    const [tsRes, tlRes, adRes] = await Promise.all([
      fetch('/src/utils/scheduler/TaskScheduler.js'),
      fetch('/src/models/bvh/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tsCode, tlCode, adCode] = await Promise.all([tsRes.text(), tlRes.text(), adRes.text()]);
    const tsMod = { exports: {} }, tlMod = { exports: {} }, adMod = { exports: {} };
    const TaskSchedulerCtor = (new Function('window','module','exports', tsCode + '; return module.exports && module.exports.TaskScheduler;'))(window, tsMod, tsMod.exports);
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const TimelineChunkAdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const scheduler = new TaskSchedulerCtor({ quantumMs: 10 });
    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);

    scheduler.onChunk = (task, chunk) => {
      adapter.appendChunk(task.trackId || 'audio', chunk, { fadeInMs: 50, meta: { taskId: task.id } });
    };

    function makeTask(id, trackId) {
      return {
        id, trackId, priority: 1, abort: new AbortController(),
        run: async function* () {
          const frames1 = Array.from({ length: 15 }, (_, i) => ({ time: i/30, motionData: [] }));
          const frames2 = Array.from({ length: 15 }, (_, i) => ({ time: i/30, motionData: [] }));
          yield { t0: 0.0, dt: 0.5, frames: frames1 };
          yield { t0: 0.5, dt: 0.5, frames: frames2 };
        }
      };
    }

    const t = makeTask('P1', 'audio');
    scheduler.submit(t);

    // Process only the first chunk
    await scheduler.runOnce(50);
    const before = timeline.tracks.audio ? timeline.tracks.audio.clips.length : 0;

    // Preempt the track and clear future clips from cutoff 0.25s
    scheduler.preempt('audio');
    adapter.clearFrom('audio', 0.25);

    const after = timeline.tracks.audio ? timeline.tracks.audio.clips.length : 0;
    return { before, after };
  });

  expect(result.before).toBeGreaterThanOrEqual(1);
  expect(result.after).toBe(0);
});

test('adapter propagates meta to created clip entries', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/models/bvh/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const TimelineChunkAdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);
    const chunk = { t0: 0, dt: 0.2, frames: Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { tag: 'inner' } })) };
    adapter.appendChunk('gesture', chunk, { meta: { source: 'unit', label: 'test-chunk' } });
    const clip = timeline.tracks.gesture?.clips?.[0];
    return { hasClip: !!clip, meta: clip?.meta || clip?.metadata || {} };
  });

  expect(result.hasClip).toBe(true);
  expect(result.meta).toMatchObject({ source: 'unit', label: 'test-chunk' });
});

test('adapter fade-in injects weightEnvelope on frames via BVHClip generator', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    // Map index 5 to 'head' so track influence masks match this bone
    if (typeof timeline.setBoneMapping === 'function') {
      timeline.setBoneMapping({ 5: 'head' });
    }
    const adapter = new AdapterCtor(timeline);
    const frames = Array.from({ length: 12 }, (_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    adapter.appendChunk('base', { t0: 0, dt: 0.4, frames }, { fadeInMs: 100, meta: { label: 'fade-test' } });

    // Sample mid-fade at ~50ms
    const t = 0.05;
    let frame;
    if (typeof timeline.getFrameAtTime === 'function') {
      frame = await timeline.getFrameAtTime(t);
    } else {
      // Fallback: manually access first clip's generator
      const clip = timeline.tracks.base?.clips?.[0];
      frame = clip && clip.generator ? await clip.generator(t, Math.floor(t * 30)) : null;
    }
    const env = frame?.metadata?.weightEnvelope;
    return { env };
  });

  expect(result.env).toBeGreaterThan(0);
  expect(result.env).toBeLessThanOrEqual(1);
});

test('scheduler prioritizes lower priority value and honors setPriority changes', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: '/src/utils/scheduler/FibonacciHeap.js' });
  await page.addScriptTag({ url: '/src/utils/scheduler/TaskScheduler.js' });

  const result = await page.evaluate(async () => {
    const [tsRes] = await Promise.all([
      fetch('/src/utils/scheduler/TaskScheduler.js')
    ]);
    const tsCode = await tsRes.text();
    const tsMod = { exports: {} };
    const TaskSchedulerCtor = (new Function('window','module','exports', tsCode + '; return module.exports && module.exports.TaskScheduler;'))(window, tsMod, tsMod.exports);

    const scheduler = new TaskSchedulerCtor({ quantumMs: 5 });
    const order = [];
    scheduler.onChunk = (task, chunk) => { order.push(task.id); };

    function makeOneShotTask(id, priority) {
      return {
        id,
        trackId: id, // unique track per task to avoid preemption interference
        priority,
        abort: new AbortController(),
        run: async function* () { yield { t0: 0, dt: 0.05, frames: [{ time: 0 }] }; }
      };
    }

    // Submit two tasks: A lower priority (higher number) and B higher priority (lower number)
    const tA = makeOneShotTask('A', 5);
    const tB = makeOneShotTask('B', 1);
    scheduler.submit(tA);
    scheduler.submit(tB);

    // Run once; expect B to run before A
    await scheduler.runOnce(50);

    // Now change priorities and resubmit more work for each by re-queueing finitely-producers
    const tC = makeOneShotTask('C', 10);
    const tD = makeOneShotTask('D', 10);
    scheduler.submit(tC);
    scheduler.submit(tD);
    scheduler.setPriority('C', 0); // make C the most urgent
    await scheduler.runOnce(50);

    return { order };
  });

  // First chunk should be from B before A; later C should appear before D after reprioritization
  expect(result.order[0]).toBe('B');
  expect(result.order.includes('A')).toBeTruthy();
  const later = result.order.slice(1);
  expect(later.indexOf('C')).toBeLessThan(later.indexOf('D'));
});

test('scheduler.preempt(taskId) aborts task and prevents future chunks', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: '/src/utils/scheduler/FibonacciHeap.js' });
  await page.addScriptTag({ url: '/src/utils/scheduler/TaskScheduler.js' });

  const result = await page.evaluate(async () => {
    const tsCode = await (await fetch('/src/utils/scheduler/TaskScheduler.js')).text();
    const tsMod = { exports: {} };
    const TaskSchedulerCtor = (new Function('window','module','exports', tsCode + '; return module.exports && module.exports.TaskScheduler;'))(window, tsMod, tsMod.exports);
    const scheduler = new TaskSchedulerCtor({ quantumMs: 5 });
    const seen = [];
    scheduler.onChunk = (task, chunk) => { seen.push({ id: task.id, t0: chunk.t0 }); };

    const abortCtl = new AbortController();
    const state = { yielded: 0 };
    const task = {
      id: 'X', trackId: 'audio', priority: 1, abort: abortCtl,
      run: async function* () {
        // Observe abort before emitting any work in this quantum
        if (abortCtl.signal.aborted) {
          throw new Error('aborted');
        }
        if (state.yielded === 0) {
          state.yielded++;
          yield { t0: 0.0, dt: 0.1, frames: [{ time: 0 }] };
        } else if (state.yielded === 1) {
          if (abortCtl.signal.aborted) throw new Error('aborted');
          state.yielded++;
          yield { t0: 0.1, dt: 0.1, frames: [{ time: 0.1 }] };
        }
      }
    };

    scheduler.submit(task);
    // Process first chunk
    await scheduler.runOnce(20);
    // Preempt by taskId and attempt another cycle
    scheduler.preempt('X');
    await scheduler.runOnce(20);

    return { count: seen.length, first: seen[0] };
  });

  expect(result.count).toBe(1);
  expect(result.first.id).toBe('X');
});

test('adapter.clearFrom keeps clips that fully end before cutoff and removes overlapping/future', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/models/bvh/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
    // Two clips: one fully before cutoff (0..0.2), one overlapping/after (0.25..0.45)
    const framesShort = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [] }));
    adapter.appendChunk('audio', { t0: 0.0, dt: 0.2, frames: framesShort }, { meta: { id: 'first' } });
    adapter.appendChunk('audio', { t0: 0.25, dt: 0.2, frames: framesShort }, { meta: { id: 'second' } });

    const before = timeline.tracks.audio.clips.length;
    adapter.clearFrom('audio', 0.25);
    const after = timeline.tracks.audio.clips.length;
    const remaining = timeline.tracks.audio.clips[0];
    return { before, after, remainingStart: remaining?.startTime, remainingEnd: remaining ? remaining.startTime + remaining.duration : -1 };
  });

  expect(result.before).toBe(2);
  expect(result.after).toBe(1);
  expect(result.remainingStart).toBe(0);
  expect(result.remainingEnd).toBeLessThan(0.25);
});

test('fade-in weightEnvelope increases over time within fade window', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
    const frames = Array.from({ length: 12 }, (_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    adapter.appendChunk('base', { t0: 0, dt: 0.4, frames }, { fadeInMs: 120 });

    const t1 = 0.02; // 20ms
    const t2 = 0.08; // 80ms
    const f1 = await timeline.getFrameAtTime(t1);
    const f2 = await timeline.getFrameAtTime(t2);
    return { e1: f1?.metadata?.weightEnvelope ?? -1, e2: f2?.metadata?.weightEnvelope ?? -1 };
  });

  expect(result.e1).toBeGreaterThanOrEqual(0);
  expect(result.e1).toBeLessThan(result.e2);
  expect(result.e2).toBeLessThanOrEqual(1);
});

test('track priority ordering overlays higher-priority track (audio > face) on shared bones', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
  const makeHeadYaw = (deg, meta = {}) => { const md = new Array(20); md[5] = [0,0,0, 0, deg, 0]; return { time: 0.0, motionData: md, metadata: meta }; };

    // face (priority 3) writes head yaw = 5; audio (priority 4) writes head yaw = 12
  adapter.appendChunk('face', { t0: 0.0, dt: 0.2, frames: [makeHeadYaw(5)] }, { blendMode: 'replace' });
  // Add boneMask to explicitly target head bone index name used by default mapping
  adapter.appendChunk('audio', { t0: 0.0, dt: 0.2, frames: [makeHeadYaw(12, { boneMask: ['bone_5'] }), makeHeadYaw(12, { boneMask: ['bone_5'] })] }, { blendMode: 'replace' });
    const f = await timeline.getFrameAtTime(0.0);
    const yaw = (f.motionData?.[5]?.[4]) || 0;
    return { yaw };
  });

  // Audio overlays face on shared bones due to higher priority → yaw ~12
  expect(result.yaw).toBeCloseTo(12, 1);
});
test('composition propagates face viseme and audio energy in composed frame', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);

    // Overlapping face and audio frames at t ≈ 0.15
    const faceFrames = Array.from({ length: 12 }, (_, i) => ({ time: i/30, motionData: [], metadata: { viseme: 'A' } }));
    const audioFrames = Array.from({ length: 12 }, (_, i) => ({ time: i/30, motionData: [], metadata: { energy: 0.75 } }));
    adapter.appendChunk('face', { t0: 0.0, dt: 0.4, frames: faceFrames }, { fadeInMs: 0 });
    adapter.appendChunk('audio', { t0: 0.0, dt: 0.4, frames: audioFrames }, { fadeInMs: 0 });

    const t = 0.15;
    const frame = await timeline.getFrameAtTime(t);
    return { faceViseme: frame?.metadata?.faceViseme, gestureEnergy: frame?.metadata?.gestureEnergy };
  });

  expect(result.faceViseme).toBeDefined();
  expect(String(result.faceViseme).toLowerCase()).toBe('a');
  expect(result.gestureEnergy).toBeCloseTo(0.75, 2);
});

test('blendMode additive vs replace affects composed rotation values', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    // Helper to make a sparse frame with a head yaw
    const makeFrameWithHeadYaw = (deg, meta = {}) => {
      const md = new Array(20);
      md[5] = [0,0,0, 0, deg, 0]; // head yaw
      return { time: 0.0, motionData: md, metadata: meta };
    };

    // Replace: overlay should end up with the overlay value (~10 deg)
    const timelineR = new BVHTimelineCtor({ framerate: 30 });
    const adapterR = new AdapterCtor(timelineR);
    adapterR.appendChunk('base', { t0: 0.0, dt: 0.2, frames: [makeFrameWithHeadYaw(5)] }, { blendMode: 'replace' });
    // Mask head bone explicitly to avoid bone name mapping mismatch
    adapterR.appendChunk('audio', { t0: 0.0, dt: 0.2, frames: [makeFrameWithHeadYaw(10, { boneMask: ['bone_5'] })] }, { blendMode: 'replace' });
    const fRep = await timelineR.getFrameAtTime(0.0);
    const yawRep = (fRep.motionData?.[5]?.[4]) || 0;

    // Additive: overlay adds to base (~15 deg)
    const timelineA = new BVHTimelineCtor({ framerate: 30 });
    const adapterA = new AdapterCtor(timelineA);
    adapterA.appendChunk('base', { t0: 0.0, dt: 0.2, frames: [makeFrameWithHeadYaw(5)] }, { blendMode: 'replace' });
    adapterA.appendChunk('audio', { t0: 0.0, dt: 0.2, frames: [makeFrameWithHeadYaw(10, { boneMask: ['bone_5'] })] }, { blendMode: 'additive' });
    const fAdd = await timelineA.getFrameAtTime(0.0);
    const yawAdd = (fAdd.motionData?.[5]?.[4]) || 0;

    return { yawRep, yawAdd };
  });

  expect(result.yawRep).toBeCloseTo(10, 1);
  expect(result.yawAdd).toBeCloseTo(15, 1);
});
