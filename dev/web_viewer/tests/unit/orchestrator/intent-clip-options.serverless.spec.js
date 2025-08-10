import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');
const STAGE = read('dev/web_viewer/src/scene/StageController.js');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Ensure handleIntent forwards fade/boneMask/priority into adapter options

test('[orchestrator] handleIntent forwards clip options (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, REG);
  await inject(page, STAGE);
  await inject(page, ORCH);

  const res = await page.evaluate(() => {
    const { ClipRegistry } = window.ClipRegistry || {};
    const { StageController } = window.StageController || {};
    const { IchikaOrchestrator } = window; // constructor
    const reg = new ClipRegistry();
    const dummy = { t0: 0, dt: 1/30, frames: [] };
    reg.add('point', dummy, { track: 'override', priority: 7, boneMask: ['rightArm','rightHand'], fadeInMs: 180, duration: 0.9 });
    const stage = new StageController(null, { registry: reg });
    const orch = new IchikaOrchestrator({ stageController: stage, clipRegistry: reg });
    let optsSeen = null;
    orch.adapter.appendChunk = (_track, _chunk, opts) => { optsSeen = opts; };
    orch.handleIntent('pointAt', { target: 'board' });
    return { fade: optsSeen?.fadeInMs, prio: optsSeen?.priority, maskLen: optsSeen?.boneMask?.length };
  });

  expect(res.fade).toBe(180);
  expect(res.prio).toBe(7);
  expect(res.maskLen).toBe(2);
});
