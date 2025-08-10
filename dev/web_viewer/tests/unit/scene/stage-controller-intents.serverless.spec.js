import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const STAGE = read('dev/web_viewer/src/scene/StageController.js');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Ensure StageController maps intents to ClipRegistry entries with override track metadata

test('[stage] StageController maps intent to clip (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, REG);
  await inject(page, STAGE);

  const res = await page.evaluate(() => {
    const { ClipRegistry } = window.ClipRegistry || {};
    const { StageController } = window.StageController || {};
    const reg = new ClipRegistry();
    const dummy = { t0: 0, dt: 1/30, frames: [] };
    reg.add('point', dummy, { track: 'override', priority: 5, boneMask: ['rightArm'] });
    const stage = new StageController(null, { registry: reg });
    const evt = stage.perform('pointAt', { target: 'board' });
    return { name: evt.clip?.name, track: evt.clip?.meta?.track, prio: evt.clip?.meta?.priority };
  });

  expect(res.name).toBe('point');
  expect(res.track).toBe('override');
  expect(res.prio).toBe(5);
});
