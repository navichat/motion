import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates composedFrom ordering follows track priority (base before audio)

test('Full BVHTimeline composedFrom order respects track priorities', async ({ page }) => {
  await page.goto('/index.html');

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
    const BVHTimelineCtor = (new Function('window','module','exports', tlCode + '; return (module.exports && module.exports.BVHTimeline) || window.BVHTimeline;'))(window, tlMod, tlMod.exports);
    const AdapterCtor = (new Function('window','module','exports', adCode + '; return (module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;'))(window, adMod, adMod.exports);

    if (!LibNs || typeof BVHTimelineCtor !== 'function' || typeof AdapterCtor !== 'function') {
      throw new Error('Missing constructors');
    }

    const { BVHClipLibrary } = LibNs;
    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new AdapterCtor(timeline);
    const lib = new BVHClipLibrary();

    // Base BVH on 'base'
    await lib.addStaticClip(timeline, 'base', '/assets/bvh/minimal_idle.bvh', 0.0, { weight: 1.0, blendMode: 'replace' });

    // Generated chunk on 'audio' overlapping [0.1, 0.3)
    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { tag: 'gen' } }));
    adapter.appendChunk('audio', { t0: 0.1, dt: 0.2, frames }, { fadeInMs: 50, weight: 0.5, blendMode: 'additive' });

    // Frame at overlap time
    const frame = await timeline.getFrameAtTime(0.15);
    const composed = frame?.metadata?.composedFrom || [];
    const order = composed.map(e => e.track);

    return { order };
  });

  // Expect increasing priority order -> base (0) before audio (4)
  expect(result.order[0]).toBe('base');
  expect(result.order[1]).toBe('audio');
});
