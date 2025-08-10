import { test, expect } from '@playwright/test';

// Multi-frame VRM application smoke: bind a mock VRM adapter and verify multiple updateBone calls over time.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Ichika classroom: VRM binder receives multiple frame updates during speech [VRM][classroom]', async ({ page }) => {
  await page.goto('/demos/ichika_classroom_demo.html');

  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));

  const result = await page.evaluate(async () => {
    let { orch } = window.__ichikaDemo || {};
    const [tlRes, adRes, vrmRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js'),
      fetch('/src/components/animation/vrm/BVHTimelineVRMIntegration.js')
    ]);
    const [tlCode, adCode, vrmCode] = await Promise.all([tlRes.text(), adRes.text(), vrmRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} }, vrmMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const VRMIntegrationCtor = (new Function('window','module','exports', vrmCode + '; return module.exports || window.BVHTimelineVRMIntegration;'))(window, vrmMod, vrmMod.exports);

    // Rebuild fresh timeline+adapter to avoid prior test state
    const BVHTimelineCtorValid = typeof BVHTimelineCtor === 'function' ? BVHTimelineCtor : (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
    const NewOrch = window.IchikaOrchestrator || (window.IchikaOrchestratorNS && window.IchikaOrchestratorNS.IchikaOrchestrator);
    const timeline = new BVHTimelineCtorValid({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
    orch = new NewOrch({ timeline, adapter });
    window.__ichikaDemo.orch = orch;

    // Mock VRM binder
    const boneCalls = [];
    const blendCalls = [];
    const mockBinder = {
      updateBone: (name, pos, rot) => boneCalls.push({ name, pos, rot }),
      updateBlendshape: (name, weight) => blendCalls.push({ name, weight }),
      update: () => {}
    };
    const integration = VRMIntegrationCtor ? new VRMIntegrationCtor(mockBinder, { smoothing: false, framerate: 30 }) : null;
    if (integration) {
      integration.connectTimeline(orch.timeline);
      orch.vrmIntegration = integration;
    }

    // Schedule a short speech sequence
    orch.scheduleSpeechFromTts({
      duration: 0.36,
      visemes: [ { time: 0.0, id: 'A' }, { time: 0.12, id: 'E' }, { time: 0.24, id: 'O' } ],
      energy: [0.3, 0.6, 0.5]
    }, { faceFadeInMs: 0, gestureFadeInMs: 0 });

    // Apply multiple frames over time
    const sampleTimes = [0.05, 0.12, 0.18, 0.24, 0.30, 0.35];
    for (const t of sampleTimes) {
      try { if (orch.timeline && typeof orch.timeline.seek === 'function') orch.timeline.seek(t); } catch {}
      if (orch.timeline && orch.vrmIntegration && typeof orch.timeline.getFrameAtTime === 'function') {
        const tf = await orch.timeline.getFrameAtTime(t);
        const vf = orch.vrmIntegration.convertTimelineFrameToVRM(tf, t);
        orch.vrmIntegration.applyFrameToVRM(vf, t);
      }
    }

    return { boneCalls: boneCalls.length, blendCalls: blendCalls.length };
  });

  expect(Math.max(result.boneCalls, result.blendCalls)).toBeGreaterThan(1);
});
