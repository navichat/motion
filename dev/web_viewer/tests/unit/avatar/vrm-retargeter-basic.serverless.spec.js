import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const RETARGETER = read('dev/web_viewer/src/retarget/VRMRetargeter.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Fast serverless check: BVH joints map to VRM bones via mapping

test('[retarget] VRMRetargeter maps BVH joints to VRM bones (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, RETARGETER);

  const res = await page.evaluate(() => {
    const VRMRetargeter = window.VRMRetargeter || (function(){ const m={exports:{}}; return m; })();
    const mapping = { 'Hips': 'hips', 'Spine': 'spine', 'Head': 'head' };
    const r = new VRMRetargeter(mapping);
    const bvhFrame = {
      joints: {
        Hips: { pos: [0,1,0], rot: [0,0,0,1] },
        Spine: { pos: [0,1.2,0], rot: [0.1,0.2,0.3,0.9] },
        Head: { pos: [0,1.6,0], rot: [0,0.5,0,0.86] },
      }
    };
    const out = r.retarget(bvhFrame, /*humanoid*/null);
    return {
      keys: Object.keys(out),
      headPosY: out.head?.pos?.[1],
      spineRotY: out.spine?.rot?.[1]
    };
  });

  expect(res.keys).toEqual(expect.arrayContaining(['hips','spine','head']));
  expect(res.headPosY).toBeCloseTo(1.6, 5);
  expect(res.spineRotY).toBeCloseTo(0.2, 5);
});
