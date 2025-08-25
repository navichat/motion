import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Verifies fade-in weightEnvelope metadata applied by TimelineChunkAdapter's generator

test('TimelineChunkAdapter applies fade-in envelope on generated frames', async ({ page }) => {
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

    // Append a generated chunk with fadeInMs=120ms
    const frames = Array.from({ length: 12 }, (_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    adapter.appendChunk('test', { t0: 0.0, dt: 0.4, frames }, { fadeInMs: 120, weight: 1.0 });

    const track = timeline.tracks.test;
    if (!track || track.clips.length === 0) throw new Error('No clip added');
    const clip = track.clips[0];

    // Probe the generator at different local times
    const early = await clip.generator(0.03, 1);  // 30ms -> ~0.25 envelope
    const mid = await clip.generator(0.12, 4);    // 120ms -> ~1.0 envelope
    const late = await clip.generator(0.30, 9);   // after fade -> 1.0 envelope

    const e = early?.metadata?.weightEnvelope ?? null;
    const m = mid?.metadata?.weightEnvelope ?? null;
    const l = late?.metadata?.weightEnvelope ?? null;

    return { e, m, l };
  });

  expect(result.e).toBeGreaterThan(0);
  expect(result.e).toBeLessThan(1);
  expect(result.m).toBeGreaterThanOrEqual(1);
  expect(result.l).toBeGreaterThanOrEqual(1);
});
