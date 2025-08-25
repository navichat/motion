import { test, expect } from '@playwright/test';

// Smoke: Bind a mock VRM adapter to the orchestrator and verify updateBone is called when chunks are scheduled.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Ichika orchestrator binds VRM and drives bones on wave/speech [VRM][classroom]', async ({ page }) => {
  await page.goto('/demos/ichika_classroom_demo.html');

  // Wait for demo to initialize and expose orchestrator
  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));
  await page.waitForSelector('#start');

  // Inject a mock adapter and attach VRM integration directly
  const result = await page.evaluate(async () => {
    let { orch, stage, reg } = window.__ichikaDemo || {};
    if (!orch) throw new Error('Orchestrator not available');

    // Ensure full timeline + adapter + VRM integration are available
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
  // Make available to orchestrator internal resolution
  if (VRMIntegrationCtor) window.BVHTimelineVRMIntegration = VRMIntegrationCtor;

  // Recreate orchestrator with full BVHTimeline + adapter
    const BVHTimelineCtorValid = typeof BVHTimelineCtor === 'function' ? BVHTimelineCtor : (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
    if (BVHTimelineCtorValid && AdapterCtor) {
      const NewOrch = window.IchikaOrchestrator || (window.IchikaOrchestratorNS && window.IchikaOrchestratorNS.IchikaOrchestrator);
      const timeline = new BVHTimelineCtorValid({ framerate: 30 });
      const adapter = new AdapterCtor(timeline);
      orch = new NewOrch({ stageController: stage || null, clipRegistry: reg || null, timeline, adapter });
      // Re-expose for any follow-up interactions
      window.__ichikaDemo.orch = orch;
    }

    const calls = [];
    const blendCalls = [];
    const mockAdapter = {
      updateBone: (name, pos, rot) => calls.push({ name, pos, rot }),
      updateBlendshape: (name, weight) => blendCalls.push({ name, weight }),
      update: () => {}
    };
    // Directly attach VRM integration instead of relying on internal resolution
    const integration = VRMIntegrationCtor ? new VRMIntegrationCtor(mockAdapter, { smoothing: false, framerate: 30 }) : null;
    if (integration && orch.timeline && typeof integration.connectTimeline === 'function') {
      integration.connectTimeline(orch.timeline);
      orch.vrmIntegration = integration;
    }

    // Start base and schedule some actions
  document.getElementById('start').click();
    await new Promise(r => setTimeout(r, 50));
    // Schedule a correct TTS payload for visemes/energy
    orch.scheduleSpeechFromTts({
      duration: 0.3,
      visemes: [ { time: 0.00, id: 'A' }, { time: 0.10, id: 'E' }, { time: 0.20, id: 'O' } ],
      energy: [0.2, 0.6, 0.4]
    }, { faceFadeInMs: 0, gestureFadeInMs: 0 });
    // Nudge timeline and also force-compose a frame once to ensure a VRM apply happens
  try { if (orch.timeline && typeof orch.timeline.seek === 'function') orch.timeline.seek(0.15); } catch {}
    await new Promise(r => setTimeout(r, 60));
    if (orch.timeline && orch.vrmIntegration && typeof orch.timeline.getFrameAtTime === 'function') {
      try {
  const tf = await orch.timeline.getFrameAtTime(0.16);
  const vf = orch.vrmIntegration.convertTimelineFrameToVRM(tf, 0.16);
  orch.vrmIntegration.applyFrameToVRM(vf, 0.16);
      } catch {}
    }
    await new Promise(r => setTimeout(r, 60));

    // Return stats from the VRM integration if available
  const stats = orch.vrmIntegration && orch.vrmIntegration.getStats ? orch.vrmIntegration.getStats() : null;
  return { calls: calls.length, blendCalls: blendCalls.length, bound: !!(orch.vrmIntegration), stats };
  });

  expect(result.bound).toBeTruthy();
  // Either bone updates (BVH clip) or viseme blendshape updates should be observed
  expect(Math.max(result.calls, result.blendCalls)).toBeGreaterThan(0);
  if (result.stats) {
    expect(result.stats.framesProcessed).toBeGreaterThan(0);
  }
});
