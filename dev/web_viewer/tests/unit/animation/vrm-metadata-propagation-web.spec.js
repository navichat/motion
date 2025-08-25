import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates that BVHTimelineVRMIntegration carries composed metadata (viseme/energy) into the VRM frame it applies

test('BVHTimelineVRMIntegration propagates faceViseme and gestureEnergy in vrmFrame.metadata', async ({ page }) => {
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

    // Ensure bone mapping for indices used if needed
    timeline.setBoneMapping(new Map([[5, 'head'], [7, 'leftArm']]));

    // Prepare face + audio chunks with metadata
    const viseme = 'I';
    const energy = 0.33;
    const faceFrames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { viseme } }));
    const audioFrames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { energy } }));

    await lib.addStaticClip(timeline, 'base', '/assets/bvh/minimal_idle.bvh', 0.0, { weight: 1.0, blendMode: 'replace' });
    adapter.appendChunk('face', { t0: 0.0, dt: 0.2, frames: faceFrames }, { fadeInMs: 0, weight: 0.7, blendMode: 'additive' });
    adapter.appendChunk('audio', { t0: 0.0, dt: 0.2, frames: audioFrames }, { fadeInMs: 0, weight: 0.6, blendMode: 'additive' });

    const mockVRMAdapter = { updateBone: () => {}, update: () => {} };
    const integration = new VRMIntegrationCtor(mockVRMAdapter, { framerate: 30, smoothing: false });

    // Intercept applyFrameToVRM to capture vrmFrame metadata
    let capturedMeta = null;
    const originalApply = integration.applyFrameToVRM.bind(integration);
    integration.applyFrameToVRM = (vrmFrame, t) => {
      capturedMeta = vrmFrame && vrmFrame.metadata;
      return originalApply(vrmFrame, t);
    };

    integration.connectTimeline(timeline);

    // Trigger composition at overlap time
    timeline.seek(0.05);
    await new Promise(r => setTimeout(r, 0));

    return { viseme: capturedMeta && capturedMeta.faceViseme, energy: capturedMeta && capturedMeta.gestureEnergy };
  });

  expect(result.viseme).toBe('I');
  expect(result.energy).toBeCloseTo(0.33, 6);
});
