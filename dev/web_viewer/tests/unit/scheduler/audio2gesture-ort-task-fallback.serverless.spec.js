import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const ORT_TASK = read('dev/web_viewer/src/components/animation/timeline/tasks/Audio2GestureOrtTask.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate ORT task fallback yields BVH chunks without ORT/model configured

test('Audio2GestureOrtTask yields fallback chunks when ORT/model missing (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, ADAPTER);
  await inject(page, ORT_TASK);

  const res = await page.evaluate(async () => {
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    const BVHTimelineCtor = mk(document.scripts[document.scripts.length-3].text, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const TimelineChunkAdapterCtor = mk(document.scripts[document.scripts.length-2].text, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
    const OrtMod = mk(document.scripts[document.scripts.length-1].text, '(module.exports) || window.Audio2GestureOrtTask');

    const { Audio2GestureOrtTask } = OrtMod;

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);

    let now = 0;
    const clock = { now: () => now };
    const ctrl = new AbortController();

    const task = new Audio2GestureOrtTask({ framerate: 30, chunkMs: 200, modelUrl: undefined });
    const iter = task.run({ clock, abortSignal: ctrl.signal });

    const first = await iter.next();
    now += 0.2;
    const second = await iter.next();
    ctrl.abort();

    const chunk1 = first.value; const chunk2 = second.value;
    adapter.appendChunk('gesture-upper', chunk1, { fadeInMs: 120 });
    adapter.appendChunk('gesture-upper', chunk2, { fadeInMs: 120 });

    const track = timeline.tracks['gesture-upper'];

    return { chunks: [chunk1, chunk2].every(Boolean), frames: [chunk1.frames.length, chunk2.frames.length], clips: track?.clips?.length || 0 };
  });

  expect(res.chunks).toBeTruthy();
  expect(res.frames[0]).toBeGreaterThan(0);
  expect(res.frames[1]).toBeGreaterThan(0);
  expect(res.clips).toBe(2);
});
