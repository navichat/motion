import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates that removing a clip invalidates buffered frames in its time range

test('BVHTimeline removes buffered frames on clip removal (invalidateRange)', async ({ page }) => {
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

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);

    // Append a generated chunk at [0.0, 0.2)
    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    const clipId = adapter.appendChunk('audio', { t0: 0.0, dt: 0.2, frames }, { fadeInMs: 0, weight: 1.0, blendMode: 'replace' });

    // Generate and buffer a frame inside the range
    const frame = await timeline.getFrameAtTime(0.05);
    if (!frame) throw new Error('No frame generated');

    // Ensure buffer contains frames in [0, 0.2]
    const before = timeline.frameBuffer.getFramesInRange(0.0, 0.2).length;

    // Remove clip (should invalidate [0.0, 0.2])
    const removed = timeline.removeClip('audio', clipId);

    // Inspect buffer again
    const after = timeline.frameBuffer.getFramesInRange(0.0, 0.2).length;

    return { before, after, removed };
  });

  expect(result.removed).toBe(true);
  expect(result.before).toBeGreaterThan(0);
  expect(result.after).toBe(0);
});
