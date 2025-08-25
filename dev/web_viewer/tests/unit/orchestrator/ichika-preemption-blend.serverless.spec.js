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

// Validate that a new speech event preempts the current audio track task and appends a new clip
test('IchikaOrchestrator preempts on new speech and appends new clip (serverless)', async ({ page }) => {
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

    // Long-running producer yielding many chunks
    async function* longGen() {
      for (let i = 0; i < 50; i++) {
        yield { t0: i * 0.1, dt: 0.1, frames: [{}] };
      }
    }
    // Short producer
    async function* shortGen() {
      for (let i = 0; i < 3; i++) {
        yield { t0: i * 0.2, dt: 0.2, frames: [{ meta: 'speech' }] };
      }
    }

    const task1 = { id: 'T_audio_1', trackId: 'audio', priority: 1, run: async function*(){ for await (const ch of longGen()) yield ch; } };
    orch.submitTask(task1);

    // Run a few slices to accumulate some chunks from task1
    for (let i = 0; i < 5; i++) await orch.runSlice(30);
    const preemptEvents = events.filter(e => e.taskId === 'T_audio_1').length;

    // New speech arrives; explicitly preempt the 'audio' track before submitting the new task
    orch.preempt('audio');

    const task2 = { id: 'T_audio_2', trackId: 'audio', priority: 0, run: async function*(){ for await (const ch of shortGen()) yield ch; } };
    orch.submitTask(task2);

    // Run slices to process the new task
    for (let i = 0; i < 10; i++) await orch.runSlice(40);

    const postEventsT1 = events.filter(e => e.taskId === 'T_audio_1').length;
    const postEventsT2 = events.filter(e => e.taskId === 'T_audio_2').length;

    const audioClips = orch.timeline.tracks.audio?.clips || [];
    return {
      preemptEvents,
      postEventsT1,
      postEventsT2,
      clipCount: audioClips.length,
      hasSpeechClip: audioClips.some(c => c.type === 'generated'),
    };
  });

  // We should have produced some events before preemption
  expect(res.preemptEvents).toBeGreaterThan(0);
  // After preemption, no new events should be attributed to the first task
  expect(res.postEventsT1).toBe(res.preemptEvents);
  // The second task should have produced at least one chunk event
  expect(res.postEventsT2).toBeGreaterThan(0);
  // And the timeline should have at least one clip (from either task), typically 1-2
  expect(res.clipCount).toBeGreaterThanOrEqual(1);
  expect(res.hasSpeechClip).toBeTruthy();
});
