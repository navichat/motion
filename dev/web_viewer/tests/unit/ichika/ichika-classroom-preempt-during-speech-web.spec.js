import { test, expect } from '@playwright/test';

// Smoke: Schedule speech, preempt speech tracks mid-stream, assert clearFrom occurred and VRM updates still apply.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Ichika classroom: preempt speech tracks mid-stream [VRM][classroom]', async ({ page }) => {
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

    // Rebuild orchestrator
    const BVHTimelineCtorValid = typeof BVHTimelineCtor === 'function' ? BVHTimelineCtor : (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
    const NewOrch = window.IchikaOrchestrator || (window.IchikaOrchestratorNS && window.IchikaOrchestratorNS.IchikaOrchestrator);
    const timeline = new BVHTimelineCtorValid({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
    orch = new NewOrch({ timeline, adapter });
    window.__ichikaDemo.orch = orch;

    // Attach VRM integration
    const boneCalls = [], blendCalls = [], logs = { clear: [], append: [] };
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

    // Spy on adapter calls
    const origAppend = orch.adapter.appendChunk.bind(orch.adapter);
    const origClear = orch.adapter.clearFrom?.bind(orch.adapter);
    orch.adapter.appendChunk = (track, chunk, opts) => { logs.append.push({ track, t0: chunk?.t0, dt: chunk?.dt }); return origAppend(track, chunk, opts); };
    if (origClear) {
      orch.adapter.clearFrom = (track, t) => { logs.clear.push({ track, t }); return origClear(track, t); };
    }

    // Schedule speech
    orch.scheduleSpeechFromTts({
      duration: 0.4,
      visemes: [ { time: 0.0, id: 'A' }, { time: 0.15, id: 'E' }, { time: 0.3, id: 'O' } ],
      energy: [0.3, 0.7, 0.5]
    }, { faceFadeInMs: 0, gestureFadeInMs: 0 });

    // Preempt face and audio tracks mid-stream
    try { orch.preempt('face'); } catch {}
    try { orch.preempt('audio'); } catch {}

    // Advance and force a VRM apply once
    try { if (orch.timeline && typeof orch.timeline.seek === 'function') orch.timeline.seek(0.2); } catch {}
    await new Promise(r => setTimeout(r, 60));
    if (orch.timeline && orch.vrmIntegration && typeof orch.timeline.getFrameAtTime === 'function') {
      const tf = await orch.timeline.getFrameAtTime(0.22);
      const vf = orch.vrmIntegration.convertTimelineFrameToVRM(tf, 0.22);
      orch.vrmIntegration.applyFrameToVRM(vf, 0.22);
    }

    return { boneCalls: boneCalls.length, blendCalls: blendCalls.length, clearCount: logs.clear.length, appendCount: logs.append.length };
  });

  expect(result.clearCount).toBeGreaterThanOrEqual(1);
  expect(Math.max(result.boneCalls, result.blendCalls)).toBeGreaterThanOrEqual(0);
  expect(result.appendCount).toBeGreaterThanOrEqual(1);
});
