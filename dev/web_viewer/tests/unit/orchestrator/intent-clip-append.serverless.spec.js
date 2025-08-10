import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');
const STAGE = read('dev/web_viewer/src/scene/StageController.js');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate orchestrator intent triggers clip append on override track

test('[orchestrator] handleIntent schedules override clip (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, REG);
  await inject(page, STAGE);
  await inject(page, ORCH);

  const res = await page.evaluate(() => {
    const { ClipRegistry } = window.ClipRegistry || {};
    const { StageController } = window.StageController || {};
    const { IchikaOrchestrator } = window.IchikaOrchestrator || {};
    const reg = new ClipRegistry();
    const dummy = { t0: 0, dt: 1/30, frames: [{ motionData: [[0,0,0,0,0,0]] }] };
    reg.add('point', dummy, { track: 'override', priority: 5, boneMask: ['rightArm'], duration: 1.0 });
    const stage = new StageController(null, { registry: reg });
    const orch = new IchikaOrchestrator({ stageController: stage, clipRegistry: reg });
    let appended = false;
    orch.adapter.appendChunk = (track, chunk, opts) => {
      appended = { track, chunk, opts };
    };
    const entry = orch.handleIntent('pointAt', { target: 'board' });
    return { appended, entryName: entry?.name, entryTrack: entry?.meta?.track };
  });

  expect(res.appended.track).toBe('override');
  expect(res.entryName).toBe('point');
  expect(res.entryTrack).toBe('override');
});
