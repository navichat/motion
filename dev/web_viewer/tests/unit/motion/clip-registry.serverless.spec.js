import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Fast serverless: registry accepts clips with metadata and lists/gets them

test('[clips] ClipRegistry add/list/get/remove (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, REG);

  const res = await page.evaluate(() => {
    const { ClipRegistry } = window.ClipRegistry || {};
    if (!ClipRegistry) throw new Error('ClipRegistry missing');
    const reg = new ClipRegistry();
    const chunk = { t0: 0, dt: 1/30, frames: [{ motionData: [[0,0,0,0,0,0]] }] };
    reg.add('idle', chunk, { track: 'base', priority: 0, boneMask: ['hips','spine'], duration: 2.0 });
    reg.add('point', chunk, { track: 'override', priority: 5, boneMask: ['rightArm','rightForeArm','rightHand'], duration: 1.0 });
    const list = reg.list();
    const got = reg.get('point');
    reg.remove('idle');
    return { count: list.length, hasPoint: !!got, removedIdle: !reg.has('idle'), maskLen: got?.meta?.boneMask?.length };
  });

  expect(res.count).toBeGreaterThanOrEqual(2);
  expect(res.hasPoint).toBeTruthy();
  expect(res.removedIdle).toBeTruthy();
  expect(res.maskLen).toBeGreaterThan(0);
});
