import { test, expect } from '@playwright/test';

// Group: VRM
// Checks that BVHTimelineVRMIntegration maps timeline viseme metadata to VRM blendshape updates.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('VRM viseme driver maps viseme metadata to blendshape updates [VRM][animation]', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const [tlRes, adRes, libRes, vrmRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js'),
      fetch('/src/components/animation/timeline/BVHClipLibrary.js'),
      fetch('/src/components/animation/vrm/BVHTimelineVRMIntegration.js')
    ]);
    const [tlCode, adCode, libCode, vrmCode] = await Promise.all([tlRes.text(), adRes.text(), libRes.text(), vrmRes.text()]);

    const tlMod = { exports: {} };
    const adMod = { exports: {} };
    const libMod = { exports: {} };
    const vrmMod = { exports: {} };

    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const LibNs = (new Function('window','module','exports', libCode + '; return module.exports || window.BVHClipLibrary;'))(window, libMod, libMod.exports);
    const VRMIntegrationCtor = (new Function('window','module','exports', vrmCode + '; return module.exports || window.BVHTimelineVRMIntegration;'))(window, vrmMod, vrmMod.exports);

    if (typeof BVHTimelineCtor !== 'function' || typeof AdapterCtor !== 'function' || !LibNs || !VRMIntegrationCtor) {
      throw new Error('Missing constructors');
    }

    const { BVHClipLibrary } = LibNs;
    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
    const lib = new BVHClipLibrary();

    // Minimal base clip to initialize track
    await lib.addStaticClip(timeline, 'base', '/assets/bvh/minimal_idle.bvh', 0.0, { weight: 1.0, blendMode: 'replace' });

    // Create a face chunk with viseme metadata that should map to a non-neutral expression
    const viseme = 'A'; // maps to 'aa'
    const faceFrames = Array.from({ length: 3 }, (_, i) => ({ time: i/30, motionData: [], metadata: { viseme } }));
    adapter.appendChunk('face', { t0: 0.0, dt: 0.1, frames: faceFrames }, { fadeInMs: 0, weight: 0.8, blendMode: 'additive' });

    const blendshapeCalls = [];
    const mockVRMAdapter = {
      updateBone: () => {},
      updateBlendshape: (name, weight) => blendshapeCalls.push({ name, weight }),
      update: () => {}
    };

    const integration = new VRMIntegrationCtor(mockVRMAdapter, { framerate: 30, smoothing: false });
    integration.connectTimeline(timeline);

    // Trigger composition
    timeline.seek(0.02);
    await new Promise(r => setTimeout(r, 0));

    return { calls: blendshapeCalls };
  });

  expect(result.calls.length).toBeGreaterThan(0);
  const first = result.calls[0];
  expect(typeof first.name).toBe('string');
  expect(first.name).toBe('aa'); // 'A' -> 'aa'
  expect(first.weight).toBeGreaterThan(0);
});
