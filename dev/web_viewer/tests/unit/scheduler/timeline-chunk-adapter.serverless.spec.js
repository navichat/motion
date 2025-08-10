import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const TL_PATH = path.join(ROOT, 'dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER_PATH = path.join(ROOT, 'dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const TL_CODE = fs.readFileSync(TL_PATH, 'utf8');
const ADAPTER_CODE = fs.readFileSync(ADAPTER_PATH, 'utf8');

test('TimelineChunkAdapter appends generated clip (serverless)', async ({ page }) => {
  await page.addScriptTag({ content: TL_CODE });
  await page.addScriptTag({ content: ADAPTER_CODE });

  const count = await page.evaluate(({ TL_CODE, ADAPTER_CODE }) => {
    let BVHTimelineCtor = (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
    if (typeof BVHTimelineCtor !== 'function') {
      const mod = { exports: {} };
      try { (new Function('window','module','exports', TL_CODE + '; return;'))(window, mod, mod.exports); } catch {}
      BVHTimelineCtor = mod.exports.BVHTimeline || (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
    }
    let TimelineChunkAdapterCtor = (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;
    if (typeof TimelineChunkAdapterCtor !== 'function') {
      const mod2 = { exports: {} };
      try { (new Function('window','module','exports', ADAPTER_CODE + '; return;'))(window, mod2, mod2.exports); } catch {}
      TimelineChunkAdapterCtor = mod2.exports.TimelineChunkAdapter || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter;
    }
  if (typeof TimelineChunkAdapterCtor !== 'function') throw new Error('constructors missing');
  // If minimal BVHTimeline isn't available, use a fake timeline with addClip
  const tl = (typeof BVHTimelineCtor === 'function') ? new BVHTimelineCtor({ framerate: 30 }) : { tracks: {}, addClip(trackName, clip) { (this.tracks[trackName] ||= { clips: [] }).clips.push(clip); return clip.id || (clip.id = `${trackName}-${Date.now()}`); } };
    const adapter = new TimelineChunkAdapterCtor(tl);
    const frames = new Array(6).fill(0).map((_, i) => ({ time: i/30, motionData: [], metadata: {} }));
    adapter.appendChunk('audio', { t0: 0, dt: 0.2, frames }, { fadeInMs: 50 });
    return tl.tracks.audio.clips.length;
  }, { TL_CODE, ADAPTER_CODE });

  expect(count).toBe(1);
});
