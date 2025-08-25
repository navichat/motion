import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates composing a static BVH clip with a generated chunk into BVHTimeline

test('compose static BVH clip + generated chunk via TimelineChunkAdapter', async ({ page }) => {
  await page.goto('/index.html');

  // Load required scripts (robust CommonJS-style eval)
  const result = await page.evaluate(async () => {
    const [libRes, tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHClipLibrary.js'),
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [libCode, tlCode, adCode] = await Promise.all([libRes.text(), tlRes.text(), adRes.text()]);
    const libMod = { exports: {} };
    const tlMod = { exports: {} };
    const adMod = { exports: {} };
    const LibNs = (new Function('window','module','exports', libCode + '; return module.exports || window.BVHClipLibrary;'))(window, libMod, libMod.exports);
    const TimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    const { BVHClipLibrary } = LibNs || {};

    if (!BVHClipLibrary) throw new Error('BVHClipLibrary missing');
    if (typeof TimelineCtor !== 'function') throw new Error('BVHTimeline ctor missing');
    if (typeof AdapterCtor !== 'function') throw new Error('TimelineChunkAdapter ctor missing');

  const timeline = new TimelineCtor({ framerate: 30 });
  const adapter = new AdapterCtor(timeline);
  const lib = new BVHClipLibrary();

    // Add a static clip from BVH
    const clipId = await lib.addStaticClip(
      timeline,
      'base',
      '/assets/bvh/minimal_idle.bvh',
      0.0,
      { weight: 1.0, blendMode: 'replace' }
    );

    // Add a generated chunk on 'audio' to overlay
    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { tag: 'gen' } }));
    adapter.appendChunk('audio', { t0: 0.1, dt: 0.2, frames }, { fadeInMs: 50, weight: 0.7, blendMode: 'additive' });

    // Query stats
    const stats = {
      baseCount: timeline.tracks.base ? timeline.tracks.base.clips.length : 0,
      audioCount: timeline.tracks.audio ? timeline.tracks.audio.clips.length : 0,
    };

    return stats;
  });

  expect(result.baseCount).toBeGreaterThan(0);
  expect(result.audioCount).toBeGreaterThan(0);
});
