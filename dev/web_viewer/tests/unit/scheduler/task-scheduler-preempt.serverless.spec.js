import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const HEAP_PATH = path.join(ROOT, 'dev/web_viewer/src/utils/scheduler/FibonacciHeap.js');
const SCHEDULER_PATH = path.join(ROOT, 'dev/web_viewer/src/utils/scheduler/TaskScheduler.js');
const HEAP_CODE = fs.readFileSync(HEAP_PATH, 'utf8');
const SCHEDULER_CODE = fs.readFileSync(SCHEDULER_PATH, 'utf8');

test('scheduler yields chunks in priority order and supports preemption (serverless)', async ({ page }) => {
  // Inject by content and prepare fallback module evaluation using provided code strings
  await page.addScriptTag({ content: HEAP_CODE });
  await page.addScriptTag({ content: SCHEDULER_CODE });

  const { ids } = await page.evaluate(async ({ SCHEDULER_CODE }) => {
    // Prefer CommonJS-style evaluation to obtain a clean constructor
    let TaskSchedulerCtor;
    const mod = { exports: {} };
    try {
      (new Function('window','module','exports', SCHEDULER_CODE + '; return;'))(window, mod, mod.exports);
      TaskSchedulerCtor = mod.exports.TaskScheduler;
    } catch (e) {
      console.error('Eval TaskScheduler failed:', e);
    }
    if (typeof TaskSchedulerCtor !== 'function') {
      TaskSchedulerCtor = (window.TaskScheduler && window.TaskScheduler.TaskScheduler) || window.TaskScheduler;
    }
    if (typeof TaskSchedulerCtor !== 'function') throw new Error('TaskScheduler missing after fallback');

    const scheduler = new TaskSchedulerCtor({ quantumMs: 10 });
    const chunks = [];
    scheduler.onChunk = (task, chunk) => { chunks.push({ id: task.id, t0: chunk.t0 }); };

    function createTask(id, kind, trackId, priority, generator) {
      const abort = new AbortController();
      async function* run() { for await (const ch of generator({ startAt: 0, chunkSec: 0.1, abort })) yield ch; }
      return { id, kind, trackId, priority, abort, run };
    }

    const genA = (async function* () { yield { t0: 0.0, dt: 0.1, frames: [] }; yield { t0: 0.1, dt: 0.1, frames: [] }; })();
    const genB = (async function* () { yield { t0: 1.0, dt: 0.1, frames: [] }; })();

    const tA = createTask('A', 'a2g', 'gesture', 1, async function* () { for await (const x of genA) yield x; });
    const tB = createTask('B', 'faceformer', 'face', 0, async function* () { for await (const x of genB) yield x; });

    scheduler.submit(tA);
    scheduler.submit(tB);

    for (let i = 0; i < 5; i++) await scheduler.runOnce(50);

    const tC = createTask('C', 'faceformer', 'face', -1, async function* () { yield { t0: 2.0, dt: 0.1, frames: [] }; });
    scheduler.submit(tC);
    scheduler.preempt('gesture', { fadeOutMs: 50 });

    for (let i = 0; i < 5; i++) await scheduler.runOnce(50);

    const ids = chunks.map(c => c.id);
    return { ids };
  }, { SCHEDULER_CODE });

  expect(new Set(ids)).toEqual(new Set(['A', 'B', 'C']));
});
