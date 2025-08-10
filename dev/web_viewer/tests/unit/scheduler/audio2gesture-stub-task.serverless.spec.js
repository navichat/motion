import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const STUB = read('dev/web_viewer/src/components/animation/timeline/tasks/Audio2GestureStubTask.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate stub task yields BVH chunks and appends via adapter

test('Audio2GestureStubTask yields chunks and appends to timeline (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, ADAPTER);
  await inject(page, STUB);

  const res = await page.evaluate(async () => {
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    const BVHTimelineCtor = mk(document.scripts[document.scripts.length-3].text, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const TimelineChunkAdapterCtor = mk(document.scripts[document.scripts.length-2].text, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
    const StubMod = mk(document.scripts[document.scripts.length-1].text, '(module.exports) || window.Audio2GestureStubTask');

    const { Audio2GestureStubTask } = StubMod;

    if (typeof BVHTimelineCtor !== 'function') throw new Error('BVHTimeline ctor missing');
    if (typeof TimelineChunkAdapterCtor !== 'function') throw new Error('TimelineChunkAdapter ctor missing');
    if (typeof Audio2GestureStubTask !== 'function' && !Audio2GestureStubTask) throw new Error('Stub task missing');

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);

    // Fake clock advancing by dt on each yield
    let now = 0;
    const clock = { now: () => now };

    // Abort after two chunks
    const ctrl = new AbortController();

    const task = new Audio2GestureStubTask({ framerate: 30, chunkMs: 200 });
    const iter = task.run({ clock, abortSignal: ctrl.signal });

    const first = await iter.next();
    now += 0.2;
    const second = await iter.next();
    ctrl.abort();

    const chunk1 = first.value; const chunk2 = second.value;
    adapter.appendChunk('gesture-upper', chunk1, { fadeInMs: 150, weight: 0.8 });
    adapter.appendChunk('gesture-upper', chunk2, { fadeInMs: 150, weight: 0.8 });

    const track = timeline.tracks['gesture-upper'];

    return {
      yielded: !!chunk1 && !!chunk2,
      frames1: chunk1.frames.length,
      frames2: chunk2.frames.length,
      clips: track?.clips?.length || 0
    };
  });

  expect(res.yielded).toBeTruthy();
  expect(res.frames1).toBeGreaterThan(0);
  expect(res.frames2).toBeGreaterThan(0);
  expect(res.clips).toBe(2);
});
