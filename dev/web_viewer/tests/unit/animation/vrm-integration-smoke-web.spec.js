import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Smoke: BVHTimelineVRMIntegration connects, processes a composed frame, and calls adapter.updateBone

test('BVHTimelineVRMIntegration processes frames and updates VRM adapter', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const [libRes, tlRes, adRes, vrmRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHClipLibrary.js'),
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js'),
      fetch('/src/components/animation/vrm/BVHTimelineVRMIntegration.js')
    ]);
    const [libCode, tlCode, adCode, vrmCode] = await Promise.all([libRes.text(), tlRes.text(), adRes.text(), vrmRes.text()]);

    const libMod = { exports: {} };
    const tlMod = { exports: {} };
    const adMod = { exports: {} };
    const vrmMod = { exports: {} };

    const LibNs = (new Function('window','module','exports', libCode + '; return module.exports || window.BVHClipLibrary;'))(window, libMod, libMod.exports);
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const VRMIntegrationCtor = (new Function('window','module','exports', vrmCode + '; return module.exports || window.BVHTimelineVRMIntegration;'))(window, vrmMod, vrmMod.exports);

    if (!LibNs || typeof BVHTimelineCtor !== 'function' || typeof AdapterCtor !== 'function' || !VRMIntegrationCtor) {
      throw new Error('Missing constructors');
    }

  const { BVHClipLibrary } = LibNs;
  const timeline = new BVHTimelineCtor({ framerate: 30 });
  // Map indices used in generated frames to names expected by audio track influence
  timeline.setBoneMapping(new Map([[5, 'head'], [7, 'leftArm']]));
    const adapter = new AdapterCtor(timeline);
    const lib = new BVHClipLibrary();

    // Mock VRM adapter capturing updateBone calls
    const calls = [];
    const mockVRMAdapter = {
      updateBone: (name, pos, rot) => { calls.push({ name, pos, rot }); },
      update: () => { /* no-op */ }
    };

    const integration = new VRMIntegrationCtor(mockVRMAdapter, { framerate: 30, smoothing: false });
    integration.connectTimeline(timeline);

    // Add a base clip (static) and a generated chunk with real motionData; use replace to ensure overlay writes
    await lib.addStaticClip(timeline, 'base', '/assets/bvh/minimal_idle.bvh', 0.0, { weight: 1.0, blendMode: 'replace' });

    // Create generated frames with bone indices present in VRM integration mapping
    const makeFrame = (t, zdeg) => {
      const md = [];
      // Bone 5 = head, with 6 values [px,py,pz, rx,ry,rz]
      md[5] = [0, 0, 0, 0, 0, zdeg];
      // Bone 7 = leftArm
      md[7] = [0, 0, 0, 10, 0, 0];
      return { time: t, motionData: md, metadata: { tag: 'gen' } };
    };
    const frames = Array.from({ length: 6 }, (_, i) => makeFrame(i/30, i * 5));
    // Replace blendMode so result gets populated even if base is empty at indices
    adapter.appendChunk('audio', { t0: 0.1, dt: 0.2, frames }, { fadeInMs: 0, weight: 0.5, blendMode: 'replace' });

    // Trigger onFrameUpdate via seek inside overlap
    timeline.seek(0.12);

    // Give the promise microtasks a tick to resolve onFrameUpdate processing
    await new Promise(r => setTimeout(r, 0));

    const stats = integration.getStats();
    return { framesProcessed: stats.framesProcessed, callCount: calls.length, firstCall: calls[0] || null };
  });

  expect(result.framesProcessed).toBeGreaterThan(0);
  expect(result.callCount).toBeGreaterThan(0);
  // Basic sanity on the first call structure
  if (result.firstCall) {
    expect(typeof result.firstCall.name).toBe('string');
    expect(result.firstCall.pos && typeof result.firstCall.pos.x).toBe('number');
    expect(result.firstCall.rot && typeof result.firstCall.rot.z).toBe('number');
  }
});
