import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates that prebuffering adds frames when content exists and avoids buffering when none exists

test('BVHTimeline prebufferFrames buffers only when active clips exist', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const [tlRes, adRes] = await Promise.all([
      fetch('/src/components/animation/timeline/BVHTimeline.js'),
      fetch('/src/components/animation/timeline/TimelineChunkAdapter.js')
    ]);
    const [tlCode, adCode] = await Promise.all([tlRes.text(), adRes.text()]);

    const tlMod = { exports: {} };
    const adMod = { exports: {} };

    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    if (typeof BVHTimelineCtor !== 'function' || typeof AdapterCtor !== 'function') {
      throw new Error('Missing constructors');
    }

    const timeline = new BVHTimelineCtor({ framerate: 30, lookaheadFrames: 10 });
    const adapter = new AdapterCtor(timeline);

    // Case 1: No content, prebuffer should add nothing
    timeline.currentTime = 0.0;
    await timeline.prebufferFrames();
    const countEmpty = timeline.frameBuffer.getFramesInRange(0.0, 0.5).length;

    // Case 2: Add content in [0.0, 0.2), prebuffer should add frames within lookahead
    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    adapter.appendChunk('audio', { t0: 0.0, dt: 0.2, frames }, { fadeInMs: 0, weight: 1.0, blendMode: 'replace' });

    // Generate one frame to seed
    await timeline.getFrameAtTime(0.0);

    // Now prebuffer
    await timeline.prebufferFrames();
    const rangeEnd = 0.0 + (timeline.frameBuffer.lookaheadFrames * timeline.frameBuffer.frameTime);
    const countWithContent = timeline.frameBuffer.getFramesInRange(0.0, rangeEnd + 1e-6).length;

    return { countEmpty, countWithContent };
  });

  expect(result.countEmpty).toBe(0);
  expect(result.countWithContent).toBeGreaterThan(0);
});
