import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Ensures adapter preserves weight and blendMode when appending a generated chunk

test('TimelineChunkAdapter preserves weight and blendMode on added clip', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/models/bvh/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);

    const tlMod = { exports: {} };
    const adMod = { exports: {} };
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    if (typeof BVHTimelineCtor !== 'function' || typeof AdapterCtor !== 'function') throw new Error('Constructors missing');

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);

    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    const opts = { fadeInMs: 60, weight: 0.42, blendMode: 'additive' };
    adapter.appendChunk('check', { t0: 0.0, dt: 0.2, frames }, opts);

    const track = timeline.tracks.check;
    const clip = track?.clips?.[0] || null;
    const weight = clip?.weight;
    const blendMode = clip?.blendMode;

    return { weight, blendMode };
  });

  expect(result.weight).toBeCloseTo(0.42, 2);
  expect(result.blendMode).toBe('additive');
});
