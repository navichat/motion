import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates that composedFrom weights incorporate clip weight and fade-in envelope over time

test('Full BVHTimeline composedFrom weights reflect fade-in envelope', async ({ page }) => {
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

    // Generated chunk on 'audio' overlapping [0.1, 0.3) with weight=0.5 and fadeIn 50ms
    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { tag: 'gen' } }));
    adapter.appendChunk('audio', { t0: 0.1, dt: 0.2, frames }, { fadeInMs: 50, weight: 0.5, blendMode: 'additive' });

    // Sample early within fade (0.11s) and later after fade completes (0.15s)
    const frameEarly = await timeline.getFrameAtTime(0.11);
    const frameLate = await timeline.getFrameAtTime(0.15);
    const compEarly = (frameEarly?.metadata?.composedFrom || []).map(c => ({ track: c.track, weight: c.weight, blendMode: c.blendMode }));
    const compLate = (frameLate?.metadata?.composedFrom || []).map(c => ({ track: c.track, weight: c.weight, blendMode: c.blendMode }));

    return { compEarly, compLate };
  });

  // Expect base present with ~1 weight at both times
  const baseEarly = result.compEarly.find(e => e.track === 'base');
  const baseLate = result.compLate.find(e => e.track === 'base');
  expect(baseEarly && baseEarly.blendMode === 'replace' && baseEarly.weight >= 0.99).toBe(true);
  expect(baseLate && baseLate.blendMode === 'replace' && baseLate.weight >= 0.99).toBe(true);

  // Audio at both times should reflect the clip's configured weight in composedFrom
  const audioEarly = result.compEarly.find(e => e.track === 'audio');
  expect(audioEarly && audioEarly.blendMode === 'additive').toBe(true);
  if (audioEarly) {
    expect(Math.abs(audioEarly.weight - 0.5)).toBeLessThan(1e-6);
  }

  // Audio at 0.15s remains ~0.5
  const audioLate = result.compLate.find(e => e.track === 'audio');
  expect(audioLate && audioLate.blendMode === 'additive').toBe(true);
  if (audioLate) {
    expect(Math.abs(audioLate.weight - 0.5)).toBeLessThan(1e-6);
  }
});
