import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Integration smoke: Use two real BVHTimelines to generate frames, adapt to TimelineMixer input, and verify last-writer-wins

test('TimelineMixer integrates with BVHTimelines via adapted frames (last-writer-wins)', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const [tlRes, adRes, libRes, mixRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js'),
      fetch('/src/components/animation/timeline/BVHClipLibrary.js'),
      fetch('/src/models/bvh/TimelineMixer.js')
    ]);
    const [tlCode, adCode, libCode, mixCode] = await Promise.all([tlRes.text(), adRes.text(), libRes.text(), mixRes.text()]);

    const tlMod = { exports: {} };
    const adMod = { exports: {} };
    const libMod = { exports: {} };
    const mixMod = { exports: {} };

    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const LibNs = (new Function('window','module','exports', libCode + '; return module.exports || window.BVHClipLibrary;'))(window, libMod, libMod.exports);
    const MixerCtor = (new Function('window','module','exports', mixCode + '; return (module.exports && module.exports.TimelineMixer) || window.TimelineMixer;'))(window, mixMod, mixMod.exports);

    if (typeof BVHTimelineCtor !== 'function' || typeof AdapterCtor !== 'function' || !LibNs || typeof MixerCtor !== 'function') {
      throw new Error('Missing constructors');
    }

    const { BVHClipLibrary } = LibNs;

    // Create two real timelines and adapters
  const tlA = new BVHTimelineCtor({ framerate: 30 });
  const tlB = new BVHTimelineCtor({ framerate: 30 });
  // Ensure indices map to names used by track influence sets
  tlA.setBoneMapping(new Map([[5, 'head'], [7, 'leftArm']]));
  tlB.setBoneMapping(new Map([[5, 'head'], [7, 'leftArm']]));
    const adA = new AdapterCtor(tlA);
    const adB = new AdapterCtor(tlB);
    const lib = new BVHClipLibrary();

    // Seed a base static clip on A (not strictly required)
    await lib.addStaticClip(tlA, 'base', '/assets/bvh/minimal_idle.bvh', 0.0, { weight: 1.0, blendMode: 'replace' });

    // Helper: make generated frame with specific bone rotations in motionData indices
    const makeFrame = (t, rotations = {}) => {
      const md = [];
      if (rotations.headZ !== undefined) {
        md[5] = [0,0,0,0,0,rotations.headZ]; // head at index 5
      }
      if (rotations.leftArmX !== undefined) {
        md[7] = [0,0,0,rotations.leftArmX,0,0]; // leftArm at index 7
      }
      return { time: t, motionData: md, metadata: {} };
    };

    // Timeline A: later head frame at 0.05 (rotZ 5)
    adA.appendChunk('audio', { t0: 0.05, dt: 0.01, frames: [makeFrame(0, { headZ: 5 })] }, { fadeInMs: 0, weight: 1.0, blendMode: 'replace' });

    // Timeline B: earlier leftArm frame at 0.04 and later head frame at 0.06 (rotZ 15)
    adB.appendChunk('audio', { t0: 0.04, dt: 0.01, frames: [makeFrame(0, { leftArmX: 10 })] }, { fadeInMs: 0, weight: 1.0, blendMode: 'replace' });
    adB.appendChunk('audio', { t0: 0.06, dt: 0.01, frames: [makeFrame(0, { headZ: 15, leftArmX: 10 })] }, { fadeInMs: 0, weight: 1.0, blendMode: 'replace' });

    // Extract frames at relevant timestamps and adapt to Mixer input format with channels maps
    const sampleTimes = [0.04, 0.05, 0.06];
  async function toMixerTimeline(tl) {
      const frames = [];
      for (const t of sampleTimes) {
    // Avoid buffer quantization collisions across close timestamps
    tl.frameBuffer.clear();
        const f = await tl.getFrameAtTime(t);
        // Build channels from motionData indices we used
        const channels = new Map();
        const md = f.motionData || [];
        const head = md[5];
        const larm = md[7];
        if (head && head.length >= 6) channels.set('head', { rotZ: head[5] });
        if (larm && larm.length >= 6) channels.set('leftArm', { rotX: larm[3] });
        frames.push({ time: t, channels });
      }
      return { tracks: { composed: { clips: [{ frames }] } } };
    }

    const mtA = await toMixerTimeline(tlA);
    const mtB = await toMixerTimeline(tlB);

    const mixer = new MixerCtor();
    const pose = mixer.compose([mtA, mtB], 0.06);
    const head = pose.get('head');
    const leftArm = pose.get('leftArm');

    return { head, leftArm };
  });

  expect(result.head && result.head.rotZ).toBe(15);
  expect(result.leftArm && result.leftArm.rotX).toBe(10);
});
