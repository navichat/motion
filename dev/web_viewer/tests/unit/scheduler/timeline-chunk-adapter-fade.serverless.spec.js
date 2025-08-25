import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
function read(p) { return fs.readFileSync(path.join(ROOT, p), 'utf8'); }

// Prefer minimal BVHTimeline model for stability
const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Ensure fadeInMs option produces a weightEnvelope ramp in generated frames
// and that clip.weight respects the provided weight option

test('TimelineChunkAdapter fade-in envelope and weight propagation (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  // Inject code; we will resolve constructors via module.exports in page context
  await inject(page, TL);
  await inject(page, ADAPTER);

  const res = await page.evaluate(async () => {
    // Resolve via CommonJS-like shim
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    const BVHTimelineCtor = mk(document.scripts[document.scripts.length-2].text, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const TimelineChunkAdapterCtor = mk(document.scripts[document.scripts.length-1].text, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
    if (typeof BVHTimelineCtor !== 'function') throw new Error('BVHTimeline ctor missing');
    if (typeof TimelineChunkAdapterCtor !== 'function') throw new Error('TimelineChunkAdapter ctor missing');
    const timeline = new BVHTimelineCtor({ lookaheadFrames: 0, framerate: 30 });
    const adapter = new TimelineChunkAdapterCtor(timeline);

    // Prepare a simple chunk with multiple frames
    const chunk = { t0: 0, dt: 0.5, frames: Array.from({ length: 30 }, () => ({ motionData: [], metadata: {} })) };

    const clipId = adapter.appendChunk('audio', chunk, { fadeInMs: 200, weight: 0.5, blendMode: 'replace' });
    const track = timeline.tracks.audio;
    const clip = track.clips.find(c => c.id === clipId);

    // Validate clip properties
    const weight = clip.weight;
    const blend = clip.blendMode;

    // Sample frames from generator at various local times to see the envelope
    const f0 = await clip.generator(0.0, 0);          // at 0ms => envelope ~0
    const f1 = await clip.generator(0.05, 2);         // at 50ms => envelope ~0.25
    const f2 = await clip.generator(0.2, 6);          // at 200ms => envelope clamps to 1

    const e0 = (f0.metadata && f0.metadata.weightEnvelope) ?? 0;
    const e1 = (f1.metadata && f1.metadata.weightEnvelope) ?? 0;
    const e2 = (f2.metadata && f2.metadata.weightEnvelope) ?? 0;

    return { weight, blend, e0, e1, e2 };
  });

  expect(res.weight).toBeCloseTo(0.5, 5);
  expect(res.blend).toBe('replace');

  // Envelope checks: starts near 0, rises by ~0.25 at 50ms (within tolerance), reaches 1 by 200ms
  expect(res.e0).toBeGreaterThanOrEqual(0);
  expect(res.e0).toBeLessThanOrEqual(0.05);
  expect(res.e1).toBeGreaterThan(0.15);
  expect(res.e1).toBeLessThan(0.35);
  expect(res.e2).toBeGreaterThanOrEqual(0.99);
  expect(res.e2).toBeLessThanOrEqual(1.01);
});
