import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');
const SGS = read('dev/web_viewer/src/orchestrator/SpeechGestureScheduler.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate scheduleSpeechFromTts creates and appends face/audio chunks with fades and preemption

test('[orchestrator] scheduleSpeechFromTts schedules face and audio tracks (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, SGS);
  await inject(page, ORCH);

  const res = await page.evaluate(() => {
    const Orch = window.IchikaOrchestrator;
    const orch = new Orch({});
    let calls = [];
    orch.adapter.appendChunk = (track, chunk, opts) => calls.push({ track, dt: chunk.dt, fade: opts?.fadeInMs, mask: opts?.boneMask });
    const tts = { duration: 0.8, visemes: [{ time: 0.0, id: 'A' }, { time: 0.4, id: 'O' }], energy: [0.1,0.2,0.7,0.5,0.2] };
    const out = orch.scheduleSpeechFromTts(tts, { faceFadeInMs: 100, gestureFadeInMs: 140 });
    return { n: calls.length, tracks: calls.map(c=>c.track), fades: calls.map(c=>c.fade), hasMask: calls[1]?.mask?.length>0, dtFace: out.faceChunk.dt };
  });

  expect(res.n).toBe(2);
  expect(res.tracks).toEqual(expect.arrayContaining(['face','audio']));
  expect(res.fades).toEqual(expect.arrayContaining([100,140]));
  expect(res.hasMask).toBeTruthy();
  expect(res.dtFace).toBeGreaterThan(0);
});
