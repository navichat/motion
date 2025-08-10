import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');
const MANIFEST = read('dev/web_viewer/src/animation/clip_manifest.sample.json');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate orchestrator loads a clip manifest and starts base idle clip

test('[orchestrator] loadClipManifest + startBaseClip schedules base (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, REG);
  await inject(page, ORCH);

  const res = await page.evaluate((mjson) => {
    const { IchikaOrchestrator } = window; // constructor
    const manifest = JSON.parse(mjson);
    const orch = new IchikaOrchestrator({});
    let appended = null;
    orch.adapter.appendChunk = (track, chunk, opts) => { appended = { track, chunk, opts }; };
    const count = orch.loadClipManifest(manifest);
    const entry = orch.startBaseClip('idle');
    return { count, appendedTrack: appended?.track, name: entry?.name, prio: appended?.opts?.priority };
  }, MANIFEST);

  expect(res.count).toBeGreaterThanOrEqual(3);
  expect(res.appendedTrack).toBe('base');
  expect(res.name).toBe('idle');
  expect(res.prio).toBe(0);
});
