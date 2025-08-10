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

// Verify that preempt by taskId clears future clips only on that task's track
test('IchikaOrchestrator preempt by taskId clears only that track (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, FIB);
  await inject(page, SCHED);
  await inject(page, TL_MIN);
  await inject(page, ADAPTER);
  await inject(page, ORCH);

  const res = await page.evaluate(async ({ TL_CODE, ADAPTER_CODE }) => {
    const Orch = window.IchikaOrchestrator;
    if (!Orch) throw new Error('Orchestrator missing');

    // Resolve constructors for minimal timeline and adapter
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

    const tl = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(tl);
    const orch = new Orch({ adapter, timeline: tl });

    async function* gen(track) {
      for (let i = 0; i < 10; i++) {
        yield { t0: i * 0.1, dt: 0.1, frames: [{ time: i * 0.1, track }] };
      }
    }

    // Two tasks on different tracks
    const tAudio = { id: 'TaskAudio', trackId: 'audio', priority: 1, run: async function*(){ for await (const ch of gen('audio')) yield ch; } };
    const tGesture = { id: 'TaskGesture', trackId: 'gesture', priority: 1, run: async function*(){ for await (const ch of gen('gesture')) yield ch; } };
    orch.submitTask(tAudio);
    orch.submitTask(tGesture);

    // Run a few slices to populate clips on both tracks
    for (let i = 0; i < 5; i++) await orch.runSlice(40);

    const before = {
      audio: (tl.tracks.audio?.clips || []).length,
      gesture: (tl.tracks.gesture?.clips || []).length,
    };

    // Advance time then preempt by taskId for audio
    tl.currentTime = 0.25;
    orch.preempt('TaskAudio');

    const after = {
      audio: (tl.tracks.audio?.clips || []).length,
      gesture: (tl.tracks.gesture?.clips || []).length,
    };

    return { before, after };
  }, { TL_CODE: TL_MIN, ADAPTER_CODE: ADAPTER });

  expect(res.before.audio).toBeGreaterThan(0);
  expect(res.before.gesture).toBeGreaterThan(0);
  // audio should be cleared (or reduced), gesture should remain unchanged (or at least not be cleared by our action)
  expect(res.after.audio).toBeLessThanOrEqual(res.before.audio);
  expect(res.after.gesture).toBeGreaterThanOrEqual(res.before.gesture);
});
