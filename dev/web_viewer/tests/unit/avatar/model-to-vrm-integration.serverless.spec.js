import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const RUNNER = read('dev/web_viewer/src/components/animation/timeline/tasks/ModelTaskRunner.js');
const STUB = read('dev/web_viewer/src/components/animation/timeline/tasks/Audio2GestureStubTask.js');
const INTEGRATION = read('dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// End-to-end (serverless): model stub → runner → adapter → (clip.generator) → VRM integration

test('Stub model to VRM integration through adapter pipeline (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, ADAPTER);
  await inject(page, RUNNER);
  await inject(page, STUB);
  await inject(page, INTEGRATION);

  const res = await page.evaluate(async () => {
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    const BVHTimelineCtor = mk(document.scripts[document.scripts.length-5].text, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const TimelineChunkAdapterCtor = mk(document.scripts[document.scripts.length-4].text, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
    const RunnerMod = mk(document.scripts[document.scripts.length-3].text, '(module.exports) || window.ModelTaskRunner');
    const StubMod = mk(document.scripts[document.scripts.length-2].text, '(module.exports) || window.Audio2GestureStubTask');
    const IntegrationCtor = mk(document.scripts[document.scripts.length-1].text, '(module && module.exports) ? module.exports : window.BVHTimelineVRMIntegration');

    const { ModelTaskRunner } = RunnerMod;
    const { Audio2GestureStubTask } = StubMod;

    const timeline = new BVHTimelineCtor({ framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);
    const runner = new ModelTaskRunner(adapter, { defaultTrack: 'gesture-upper' });

    // Produce a couple of chunks via stub
    let now = 0; const clock = { now: () => now };
    const ctrl = new AbortController();
    const task = new Audio2GestureStubTask({ framerate: 30, chunkMs: 200 });
    const handle = runner.start(task, { track: 'gesture-upper', abortSignal: ctrl.signal }, { clock });
    await new Promise(r => setTimeout(r, 0)); now += 0.2; await new Promise(r => setTimeout(r, 0)); ctrl.abort(); await handle.done();

    // Pull a generated frame from the first clip via its generator
    const track = timeline.tracks['gesture-upper'];
    if (!track || !track.clips || track.clips.length === 0) throw new Error('No clips appended');
    const clip = track.clips[0];
    const frame = await clip.generator(0.1, 3);

    // VRM adapter stub that records updates
    const calls = [];
    const vrmAdapter = { updateBone: (bone, pos, rot) => calls.push({ bone, pos, rot }), update: () => {} };
    const integration = new IntegrationCtor(vrmAdapter, { smoothing: false });

    // Directly apply the frame to VRM via integration
    integration.handleTimelineFrame(frame, 0.1);

    return { clipCount: track.clips.length, calls: calls.length, hasAny: calls.length > 0 };
  });

  expect(res.clipCount).toBeGreaterThan(0);
  expect(res.hasAny).toBeTruthy();
  expect(res.calls).toBeGreaterThan(0);
});
