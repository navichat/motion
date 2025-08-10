import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
function read(p) { return fs.readFileSync(path.join(ROOT, p), 'utf8'); }

const FIB = read('dev/web_viewer/src/utils/scheduler/FibonacciHeap.js');
const SCHED = read('dev/web_viewer/src/utils/scheduler/TaskScheduler.js');
const TL_MIN = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Verify preemption by taskId halts further chunk events for that task

test('IchikaOrchestrator preempt by taskId stops events (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, FIB);
  await inject(page, SCHED);
  await inject(page, TL_MIN);
  await inject(page, ADAPTER);
  await inject(page, ORCH);

  const res = await page.evaluate(async ({ TL_CODE, ADAPTER_CODE }) => {
    const Orch = window.IchikaOrchestrator;
    if (!Orch) throw new Error('Orchestrator missing');

    let BVHTimelineCtor = window.BVHTimeline || (window.BVHTimeline && window.BVHTimeline.BVHTimeline);
    if (typeof BVHTimelineCtor !== 'function') {
      const mod = { exports: {} };
      try { (new Function('window','module','exports', TL_CODE + '; return;'))(window, mod, mod.exports); } catch {}
      BVHTimelineCtor = mod.exports.BVHTimeline || window.BVHTimeline || (window.BVHTimeline && window.BVHTimeline.BVHTimeline);
    }
    let TimelineChunkAdapterCtor = window.TimelineChunkAdapter || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter);
    if (typeof TimelineChunkAdapterCtor !== 'function') {
      const mod2 = { exports: {} };
      try { (new Function('window','module','exports', ADAPTER_CODE + '; return;'))(window, mod2, mod2.exports); } catch {}
      TimelineChunkAdapterCtor = mod2.exports.TimelineChunkAdapter || window.TimelineChunkAdapter || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter);
    }

    const events = [];
    const tl = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(tl);
    const orch = new Orch({ onEvent: (e) => events.push(e), adapter, timeline: tl });

    async function* longGen() {
      for (let i = 0; i < 20; i++) {
        yield { t0: i * 0.1, dt: 0.1, frames: [{}] };
      }
    }
    const task = { id: 'TaskX', trackId: 'audio', priority: 1, run: async function* () { for await (const ch of longGen()) yield ch; } };

    orch.submitTask(task);
    for (let i = 0; i < 4; i++) await orch.runSlice(40);
    const before = events.filter(e => e.taskId === 'TaskX').length;

    orch.preempt('TaskX'); // preempt by task id

    for (let i = 0; i < 6; i++) await orch.runSlice(40);
    const after = events.filter(e => e.taskId === 'TaskX').length;

    return { before, after };
  }, { TL_CODE: TL_MIN, ADAPTER_CODE: ADAPTER });

  expect(res.before).toBeGreaterThan(0);
  expect(res.after).toBe(res.before); // should not increase post-preempt
});
