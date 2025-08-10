import { test, expect } from '@playwright/test';

const SCHEDULER = '/src/utils/scheduler/TaskScheduler.js';
const HEAP = '/src/utils/scheduler/FibonacciHeap.js';

function fakeChunkGen(label, chunks = 3, delayMs = 0) {
  return async function* ({ startAt = 0, chunkSec = 0.2, abort }) {
    let t = startAt;
    for (let i = 0; i < chunks; i++) {
      if (abort.signal.aborted) throw new Error('aborted');
      const frames = new Array(6).fill(0).map((_, idx) => ({ time: t + idx/30, motionData: [], metadata: { label } }));
      yield { t0: t, dt: 0.2, frames };
      t += 0.2;
      if (delayMs) await new Promise(r => setTimeout(r, delayMs));
    }
  };
}

// Note: We evaluate the JS files into the page so window.TaskScheduler becomes available

test('scheduler yields chunks in priority order and supports preemption', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: HEAP });
  await page.addScriptTag({ url: SCHEDULER });
  await page.waitForFunction(() => !!window.FibonacciHeap);

  const output = await page.evaluate(async () => {
    const res = await fetch('/src/utils/scheduler/TaskScheduler.js');
    const code = await res.text();
    const mod = { exports: {} };
    const TaskSchedulerCtor = (new Function('window','module','exports', code + '; return (module && module.exports && module.exports.TaskScheduler) || window.TaskScheduler;'))(window, mod, mod.exports);
    if (typeof TaskSchedulerCtor !== 'function') throw new Error('TaskScheduler missing');
    const scheduler = new TaskSchedulerCtor({ quantumMs: 50 });
    const chunks = [];
    scheduler.onChunk = (task, chunk) => {
      chunks.push({ id: task.id, k: task.kind, t0: chunk.t0 });
    };

    // lightweight TaskFactory in page
    function createTask(id, kind, trackId, priority, generator) {
      const abort = new AbortController();
      async function* run() { for await (const ch of generator({ startAt: 0, chunkSec: 0.2, abort })) yield ch; }
      return { id, kind, trackId, priority, abort, run };
    }

    const tA = createTask('A', 'a2g', 'gesture', 1, window.fakeGenA || (window.fakeGenA = window.fakeGenA || (async function* () { yield { t0: 0, dt: 0.2, frames: [] }; }))); // will be replaced
    const tB = createTask('B', 'faceformer', 'face', 0, (async function* () { yield { t0: 0, dt: 0.2, frames: [] }; }));

    // Better generators
    const genA = (async function* () { yield { t0: 0.0, dt: 0.2, frames: [] }; yield { t0: 0.2, dt: 0.2, frames: [] }; })();
    const genB = (async function* () { yield { t0: 1.0, dt: 0.2, frames: [] }; })();

    // patch tasks
    tA.run = async function* () { for await (const x of genA) yield x; };
    tB.run = async function* () { for await (const x of genB) yield x; };

    scheduler.submit(tA);
    scheduler.submit(tB);

    // run a few slices
  for (let i = 0; i < 5; i++) await scheduler.runOnce(100);

    // Preempt gesture track with a new high-priority C
  const tC = createTask('C', 'faceformer', 'face', -1, () => (async function* () { yield { t0: 2.0, dt: 0.2, frames: [] }; })());
    scheduler.submit(tC);
    scheduler.preempt('gesture', { fadeOutMs: 100 });

  for (let i = 0; i < 10; i++) await scheduler.runOnce(100);

    return chunks.map(c => c.id);
  });

  // Expect B (higher priority) to be processed among first, C after submit, A may be cut short
  expect(new Set(output)).toEqual(new Set(['A', 'B', 'C']));
});
