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

// Validate that fadeInMs on tasks produces a weightEnvelope ramp in frames via the adapter

test('IchikaOrchestrator applies fade envelope via adapter (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, FIB);
  await inject(page, SCHED);
  await inject(page, TL_MIN);
  await inject(page, ADAPTER);
  await inject(page, ORCH);

  const res = await page.evaluate(async ({ TL_CODE, ADAPTER_CODE }) => {
    const Orch = window.IchikaOrchestrator;
    if (!Orch) throw new Error('Orchestrator missing');

    // Resolve constructors robustly
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

    async function* gen() {
      for (let i = 0; i < 10; i++) {
        yield { t0: i * 0.05, dt: 0.05, frames: Array.from({length: 4}, () => ({ motionData: [], metadata: {} })) };
      }
    }
    const task = { id: 'fadeTask', trackId: 'audio', fadeInMs: 100, priority: 0, run: async function* () { for await (const ch of gen()) yield ch; } };

    orch.submitTask(task);
    for (let i = 0; i < 8; i++) await orch.runSlice(40);

  const track = tl.tracks.audio;
    const clip = track && track.clips && track.clips[0];
    if (!clip || typeof clip.generator !== 'function') return { ok: false };

    const f0 = await clip.generator(0.0, 0);
    const f1 = await clip.generator(0.05, 1);
    const f2 = await clip.generator(0.2, 4);

    const e0 = (f0.metadata && f0.metadata.weightEnvelope) ?? 0;
    const e1 = (f1.metadata && f1.metadata.weightEnvelope) ?? 0;
    const e2 = (f2.metadata && f2.metadata.weightEnvelope) ?? 0;

    return { ok: true, e0, e1, e2 };
  }, { TL_CODE: TL_MIN, ADAPTER_CODE: ADAPTER });

  expect(res.ok).toBeTruthy();
  expect(res.e0).toBeGreaterThanOrEqual(0);
  expect(res.e0).toBeLessThanOrEqual(0.05);
  expect(res.e1).toBeGreaterThan(0.15);
  expect(res.e1).toBeLessThan(0.6);
  expect(res.e2).toBeGreaterThanOrEqual(0.99);
});
