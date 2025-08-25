import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Serverless integration: inject scheduler + adapter into page, fake minimal timeline

test('scheduler pushes chunks to TimelineChunkAdapter', async ({ page }) => {
  await page.goto('about:blank');

  // Load sources and evaluate via CommonJS-like shims to avoid window globals
  const repo = process.cwd();
  const files = {
    heap: 'dev/web_viewer/src/utils/scheduler/FibonacciHeap.js',
    scheduler: 'dev/web_viewer/src/utils/scheduler/TaskScheduler.js',
    // Prefer lightweight BVHTimeline for tests
    bvhTimeline: 'dev/web_viewer/src/models/bvh/BVHTimeline.js',
    adapter: 'dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js',
  };
  const sources = Object.fromEntries(Object.entries(files).map(([k, rel]) => [k, fs.readFileSync(path.join(repo, rel), 'utf8')]));

  const result = await page.evaluate(async ({ sources }) => {
    // Evaluate each module and obtain constructors from module.exports
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    mk(sources.heap, 'true');
    const TaskSchedulerCtor = mk(sources.scheduler, 'module.exports && module.exports.TaskScheduler');
    const BVHTimelineCtor = mk(sources.bvhTimeline, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const TimelineChunkAdapterCtor = mk(sources.adapter, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
    if (typeof TaskSchedulerCtor !== 'function') throw new Error('TaskScheduler ctor missing');
    if (typeof BVHTimelineCtor !== 'function') throw new Error('BVHTimeline ctor missing');
    if (typeof TimelineChunkAdapterCtor !== 'function') throw new Error('TimelineChunkAdapter ctor missing');

    const tl = new BVHTimelineCtor({ lookaheadFrames: 0, framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(tl);
    const scheduler = new TaskSchedulerCtor({ quantumMs: 30, onChunk: (task, chunk) => {
      adapter.appendChunk(task.trackId, chunk, { fadeInMs: 0 });
    }});

    // Simple generator that yields two chunks
    async function* gen() {
      yield { t0: 0.0, dt: 0.2, frames: [{}, {}, {}, {}, {}, {}] };
      yield { t0: 0.2, dt: 0.2, frames: [{}, {}, {}, {}, {}, {}] };
    }

    const task = {
      id: 'T1', kind: 'test', trackId: 'audio', priority: 0,
      abort: new AbortController(),
      run: async function* () { for await (const ch of gen()) yield ch; }
    };

    scheduler.submit(task);
    for (let i = 0; i < 5; i++) await scheduler.runOnce(50);

    return {
      tracks: Object.keys(tl.tracks),
      audioClips: tl.tracks.audio?.clips.length || 0
    };
  }, { sources });

  expect(result.tracks.length).toBeGreaterThan(0);
  expect(result.audioClips).toBeGreaterThanOrEqual(1);
});
