import { test, expect } from '@playwright/test';

test('TimelineChunkAdapter appends generated clip', async ({ page }) => {
  await page.goto('/index.html');
  // Only load the adapter; use a fake minimal timeline to keep this unit test isolated
  await page.addScriptTag({ url: '/src/components/animation/timeline/TimelineChunkAdapter.js' });
  await page.waitForFunction(() => typeof window.TimelineChunkAdapter === 'function');

  const count = await page.evaluate(() => {
    const TimelineChunkAdapterCtor = (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;
    if (typeof TimelineChunkAdapterCtor !== 'function') throw new Error('constructor missing');
    // Minimal fake timeline with addClip API
    const tl = { tracks: {}, addClip(trackName, clip) { (this.tracks[trackName] ||= { clips: [] }).clips.push(clip); return clip.id || (clip.id = `${trackName}-${Date.now()}`); } };
    const adapter = new TimelineChunkAdapterCtor(tl);
    const frames = new Array(6).fill(0).map((_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    adapter.appendChunk('audio', { t0: 0, dt: 0.2, frames }, { fadeInMs: 50 });
    return tl.tracks.audio.clips.length;
  });

  expect(count).toBe(1);
});
