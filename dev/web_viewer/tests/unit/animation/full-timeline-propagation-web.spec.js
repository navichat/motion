import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Validates that BVHTimeline propagates face viseme and audio energy from tracks into composed frame metadata

test('Full BVHTimeline propagates faceViseme and gestureEnergy metadata', async ({ page }) => {
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

    // Prepare frames for face and audio tracks with metadata to be propagated
    const viseme = 'AA';
    const energy = 0.42;

    const faceFrames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { viseme } }));
    const audioFrames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, motionData: [], metadata: { energy } }));

    // Append overlapping chunks at t0=0 for 0.2s
    adapter.appendChunk('face', { t0: 0.0, dt: 0.2, frames: faceFrames }, { fadeInMs: 0, weight: 0.7, blendMode: 'additive' });
    adapter.appendChunk('audio', { t0: 0.0, dt: 0.2, frames: audioFrames }, { fadeInMs: 0, weight: 0.6, blendMode: 'additive' });

    // Get a frame within overlap
    const frame = await timeline.getFrameAtTime(0.05);
    const md = frame?.metadata || {};
    const composed = md.composedFrom || [];

    return {
      viseme: md.faceViseme,
      energy: md.gestureEnergy,
      hasFace: composed.some(e => e.track === 'face'),
      hasAudio: composed.some(e => e.track === 'audio')
    };
  });

  expect(result.hasFace).toBe(true);
  expect(result.hasAudio).toBe(true);
  expect(result.viseme).toBe('AA');
  expect(result.energy).toBeCloseTo(0.42, 6);
});
