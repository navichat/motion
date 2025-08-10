import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
function read(p) { return fs.readFileSync(path.join(ROOT, p), 'utf8'); }

const FIB = read('dev/web_viewer/src/utils/scheduler/FibonacciHeap.js');
const SCHED = read('dev/web_viewer/src/utils/scheduler/TaskScheduler.js');
const TL = read('dev/web_viewer/src/components/animation/timeline/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Smoke orchestration: submit a short task and expect chunk events + timeline clip

test('IchikaOrchestrator end-to-end chunk append (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, FIB);
  await inject(page, SCHED);
  await inject(page, TL);
  await inject(page, ADAPTER);
  await inject(page, ORCH);

  const res = await page.evaluate(async () => {
    const Orch = window.IchikaOrchestrator;
    if (!Orch) throw new Error('Orchestrator missing');
    const events = [];
    const orch = new Orch({ onEvent: (e) => events.push(e) });

    async function* gen() {
      yield { t0: 0.0, dt: 0.2, frames: [{}, {}, {}, {}, {}, {}] };
      yield { t0: 0.2, dt: 0.2, frames: [{}, {}, {}, {}, {}, {}] };
    }
    const task = { id: 'T1', trackId: 'audio', priority: 0, abort: new AbortController(), run: async function* () { for await (const ch of gen()) yield ch; } };

    orch.submitTask(task);
    for (let i = 0; i < 6; i++) await orch.runSlice(60);

    const clipCount = orch.timeline.tracks.audio?.clips.length || 0;
    return { events, clipCount };
  });

  expect(res.clipCount).toBeGreaterThanOrEqual(1);
  expect(res.events.some(e => e.type === 'chunk' && e.taskId === 'T1')).toBeTruthy();
});
