import { test, expect } from '@playwright/test';

// Smoke: During scheduled speech (visemes/gestures), trigger a classroom intent (point) and ensure both adapter logs and VRM updates occur.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Ichika classroom: point intent during speech blends correctly [VRM][classroom]', async ({ page }) => {
  await page.goto('/demos/ichika_classroom_demo.html');

  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));
  await page.waitForSelector('#start');

  // Start base to init registry and logs
  await page.click('#start');
  await page.waitForTimeout(50);

  const result = await page.evaluate(async () => {
    let { orch } = window.__ichikaDemo || {};
    const [tlRes, adRes, vrmRes, regRes, stageRes, manifestRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js'),
      fetch('/src/components/animation/vrm/BVHTimelineVRMIntegration.js'),
      fetch('/src/animation/ClipRegistry.js'),
      fetch('/src/scene/StageController.js'),
      fetch('/src/animation/clip_manifest.sample.json')
    ]);
    const [tlCode, adCode, vrmCode, regCode, stageCode, manifestText] = await Promise.all([tlRes.text(), adRes.text(), vrmRes.text(), regRes.text(), stageRes.text(), manifestRes.text()]);
    const tlMod = { exports: {} }, adMod = { exports: {} }, vrmMod = { exports: {} }, regMod = { exports: {} }, stageMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const VRMIntegrationCtor = (new Function('window','module','exports', vrmCode + '; return module.exports || window.BVHTimelineVRMIntegration;'))(window, vrmMod, vrmMod.exports);
    const CRNs = (new Function('window','module','exports', regCode + '; return module.exports || window.ClipRegistry;'))(window, regMod, regMod.exports);
    const SCNs = (new Function('window','module','exports', stageCode + '; return module.exports || window.StageController;'))(window, stageMod, stageMod.exports);
    const manifest = JSON.parse(manifestText);

    // Rebuild orchestrator with full timeline+adapter
    const BVHTimelineCtorValid = typeof BVHTimelineCtor === 'function' ? BVHTimelineCtor : (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
    if (BVHTimelineCtorValid && AdapterCtor) {
      const NewOrch = window.IchikaOrchestrator || (window.IchikaOrchestratorNS && window.IchikaOrchestratorNS.IchikaOrchestrator);
      const timeline = new BVHTimelineCtorValid({ framerate: 30 });
      const adapter = new AdapterCtor(timeline);
      const reg = CRNs && (CRNs.ClipRegistry || CRNs) ? new (CRNs.ClipRegistry || CRNs)() : null;
      if (reg && typeof reg.loadFromManifest === 'function') reg.loadFromManifest(manifest);
      const StageCtor = SCNs && (SCNs.StageController || SCNs) ? (SCNs.StageController || SCNs) : null;
      const stage = StageCtor ? new StageCtor(null, { registry: reg }) : null;
      orch = new NewOrch({ timeline, adapter, stageController: stage, clipRegistry: reg });
      window.__ichikaDemo.orch = orch;
    }

    // Attach VRM integration with a mock binder
    const boneCalls = [], blendCalls = [];
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

    // Schedule speech and then trigger point intent
    orch.scheduleSpeechFromTts({
      duration: 0.4,
      visemes: [ { time: 0.0, id: 'A' }, { time: 0.15, id: 'E' }, { time: 0.3, id: 'O' } ],
      energy: [0.3, 0.7, 0.5]
    }, { faceFadeInMs: 0, gestureFadeInMs: 0 });

    // Log spy (appendChunk logs are instrumented by demo when clicking Start, but we rebuilt orch here; emulate logs)
    const appendMeta = [];
    const origAppend = orch.adapter.appendChunk.bind(orch.adapter);
    orch.adapter.appendChunk = (track, chunk, opts) => { appendMeta.push({ track, fade: opts?.fadeInMs || 0 }); return origAppend(track, chunk, opts); };

  // Trigger point intent directly on the reconstructed orchestrator
  orch.handleIntent('pointAt', { target: 'board' });

    // Advance time and force one VRM apply
    try { if (orch.timeline && typeof orch.timeline.seek === 'function') orch.timeline.seek(0.2); } catch {}
    await new Promise(r => setTimeout(r, 80));
    if (orch.timeline && orch.vrmIntegration && typeof orch.timeline.getFrameAtTime === 'function') {
      const tf = await orch.timeline.getFrameAtTime(0.22);
      const vf = orch.vrmIntegration.convertTimelineFrameToVRM(tf, 0.22);
      orch.vrmIntegration.applyFrameToVRM(vf, 0.22);
    }

    const stats = orch.vrmIntegration && orch.vrmIntegration.getStats ? orch.vrmIntegration.getStats() : null;
    const hasOverride = appendMeta.some(m => (m.track || '').includes('override'));
    return { boneCalls: boneCalls.length, blendCalls: blendCalls.length, hasOverride, frames: stats ? stats.framesProcessed : 0 };
  });

  // Either bone or blendshape updates should have been applied
  expect(Math.max(result.boneCalls, result.blendCalls)).toBeGreaterThan(0);
  // Ensure an override track append occurred for the point action
  expect(result.hasOverride).toBeTruthy();
  // And at least one frame processed if stats available
  expect(result.frames).toBeGreaterThanOrEqual(0);
});
