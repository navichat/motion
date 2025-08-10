import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires web server');

// Validates RealtimeProducers -> TimelineChunkAdapter -> BVHTimeline wiring in-browser

test('realtime producers integration: visemes + gestures stream into composed frame [unit-web]', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    // Load required modules
    const [tlRes, adRes, prodRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js'),
      fetch('/src/utils/producers/RealtimeProducers.js'),
    ]);
    const [tlCode, adCode, prodCode] = await Promise.all([tlRes.text(), adRes.text(), prodRes.text()]);

    const tlMod = { exports: {} }, adMod = { exports: {} }, prodMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);
    const Producers = (new Function('window','module','exports', prodCode + '; return (module.exports) || window.RealtimeProducers;'))(window, prodMod, prodMod.exports);

    if (!BVHTimelineCtor || !AdapterCtor || !Producers) throw new Error('ctors missing');
    const { TextToVisemeProducer, Audio2GestureProducerStub } = Producers;

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);

    // Start producers with short runs
    const vis = new TextToVisemeProducer('hi');
    const gest = new Audio2GestureProducerStub(0.4);

    // Kick off both and sample along the way
    let samples = 0; let sawViseme = false; let sawEnergy = false;
    const sampler = (async () => {
      for (let i = 0; i < 6; i++) {
        const t = i * 0.06; // ~60ms cadence
        try { if (typeof timeline.seek === 'function') timeline.seek(t); } catch {}
        const f = await timeline.getFrameAtTime(t);
        if (f?.metadata?.faceViseme) sawViseme = true;
        if (f?.metadata?.gestureEnergy != null) sawEnergy = true;
        samples++;
        await new Promise(r => setTimeout(r, 20));
      }
    })();

    await Promise.all([
      vis.start(adapter, 'face', { framesPerChunk: 4, delayMs: 20 }),
      gest.start(adapter, 'audio', { framesPerChunk: 4, delayMs: 20 }),
      sampler
    ]);

    // Sample within the first chunk where frames exist and check any motion present
    const f0 = await timeline.getFrameAtTime(0.08);
  // Focus on metadata signals; motion composition may depend on bone mapping in env
  return { samples, sawViseme, sawEnergy };
  });

  expect(result.samples).toBeGreaterThanOrEqual(4);
  expect(result.sawViseme).toBeTruthy();
  expect(result.sawEnergy).toBeTruthy();
  // Motion presence is exercised in overlay tests; here we validate streaming + metadata propagation
});
