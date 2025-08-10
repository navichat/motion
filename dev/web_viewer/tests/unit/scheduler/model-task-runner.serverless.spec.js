import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const STUB = read('dev/web_viewer/src/components/animation/timeline/tasks/Audio2GestureStubTask.js');
const RUNNER = read('dev/web_viewer/src/components/animation/timeline/tasks/ModelTaskRunner.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate ModelTaskRunner drives a chunk-yielding task into the adapter and supports abort.

test('ModelTaskRunner drives stub task and appends multiple chunks (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, ADAPTER);
  await inject(page, STUB);
  await inject(page, RUNNER);

  const res = await page.evaluate(async () => {
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    const BVHTimelineCtor = mk(document.scripts[document.scripts.length-4].text, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const TimelineChunkAdapterCtor = mk(document.scripts[document.scripts.length-3].text, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
    const StubMod = mk(document.scripts[document.scripts.length-2].text, '(module.exports) || window.Audio2GestureStubTask');
    const RunnerMod = mk(document.scripts[document.scripts.length-1].text, '(module.exports) || window.ModelTaskRunner');

    const { Audio2GestureStubTask } = StubMod;
    const { ModelTaskRunner } = RunnerMod;

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);

    // Fake clock
    let now = 0;
    const clock = { now: () => now };

    const ctrl = new AbortController();
    const task = new Audio2GestureStubTask({ framerate: 30, chunkMs: 150 });
    const runner = new ModelTaskRunner(adapter, { defaultTrack: 'gesture-upper', defaultFadeMs: 120, defaultWeight: 0.7 });

    const handle = runner.start(task, { track: 'gesture-upper', abortSignal: ctrl.signal }, { clock });

    // Advance and let a few chunks flow
    await new Promise(r => setTimeout(r, 0));
    now += 0.15;
    await new Promise(r => setTimeout(r, 0));
    now += 0.15;
    await new Promise(r => setTimeout(r, 0));

    ctrl.abort();
    await handle.done();

    const track = timeline.tracks['gesture-upper'];
    return { clips: track?.clips?.length || 0 };
  });

  expect(res.clips).toBeGreaterThanOrEqual(2);
});
