import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates that BVHClipLibrary parses Frames/Frame Time into a proper duration

test('BVHClipLibrary parses duration from BVH header', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const [libRes, tlRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHClipLibrary.js'),
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
    ]);
    const [libCode, tlCode] = await Promise.all([libRes.text(), tlRes.text()]);

    const libMod = { exports: {} };
    const tlMod = { exports: {} };
    const LibNs = (new Function('window','module','exports', libCode + '; return module.exports || window.BVHClipLibrary;'))(window, libMod, libMod.exports);
    const TimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);

    if (!LibNs || typeof TimelineCtor !== 'function') throw new Error('Missing constructors');
    const { BVHClipLibrary } = LibNs;
    const lib = new BVHClipLibrary();
    const timeline = new TimelineCtor({ framerate: 30 });

    const clipId = await lib.addStaticClip(timeline, 'base', '/assets/bvh/minimal_idle.bvh', 0.0);

    const track = timeline.tracks.base;
    const clip = track?.clips?.find(c => c.id === clipId) || track?.clips?.[0];
    const duration = clip?.duration;

    return { duration };
  });

  // 10 frames * 0.0333333s = ~0.333333s with tiny header rounding
  expect(result.duration).toBeGreaterThan(0.32);
  expect(result.duration).toBeLessThan(0.34);
});
