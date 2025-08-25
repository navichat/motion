import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Minimal timeline path uses fallback branch (no removeClip)
test('TimelineChunkAdapter.clearFrom removes overlapping/future clips on minimal BVHTimeline', async ({ page }) => {
  await page.goto('/index.html');
  const result = await page.evaluate(async () => {
    // Load modules inside page context
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

    // Two clips on 'test': c1 ends before cutoff; c2 overlaps/after cutoff
    adapter.appendChunk('test', { t0: 0.0, dt: 0.3, frames: Array.from({length:9}).map((_,i)=>({time:i/30})) });
    adapter.appendChunk('test', { t0: 0.4, dt: 0.5, frames: Array.from({length:15}).map((_,i)=>({time:i/30})) });

    const before = (timeline.tracks.test?.clips.length) || 0;
    adapter.clearFrom('test', 0.5);
    const after = (timeline.tracks.test?.clips.length) || 0;
    return { before, after };
  });

  expect(result.before).toBe(2);
  expect(result.after).toBe(1);
});

// Full timeline path uses removeClip branch
test('TimelineChunkAdapter.clearFrom uses removeClip on full BVHTimeline', async ({ page }) => {
  await page.goto('/index.html');
  const stats = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
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

    adapter.appendChunk('audio', { t0: 0.0, dt: 0.3, frames: Array.from({length:9}).map((_,i)=>({time:i/30})) });
    adapter.appendChunk('audio', { t0: 0.25, dt: 0.3, frames: Array.from({length:9}).map((_,i)=>({time:i/30})) });
    adapter.appendChunk('audio', { t0: 1.0, dt: 0.2, frames: Array.from({length:6}).map((_,i)=>({time:i/30})) });

    const before = timeline.tracks.audio?.clips.length || 0;
    adapter.clearFrom('audio', 0.5);
    const after = timeline.tracks.audio?.clips.length || 0;
    return { before, after };
  });

  expect(stats.before).toBe(3);
  expect(stats.after).toBe(1);
});
