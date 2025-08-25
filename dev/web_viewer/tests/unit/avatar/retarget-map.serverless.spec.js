import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const INTEGRATION = read('dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js');
const BINDER = read('dev/web_viewer/src/components/animation/vrm/AvatarBinder.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate bone mapping produces expected VRM bone names

test('BVH→VRM mapping covers core humanoid bones (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, INTEGRATION);
  await inject(page, BINDER);

  const res = await page.evaluate(() => {
    const IntegrationCtor = window.BVHTimelineVRMIntegration || (function(){ const m={exports:{}}; return m; })();
    const { AvatarBinder } = window;
    const binder = new AvatarBinder(); // stub
    const integ = new IntegrationCtor(binder, { smoothing: false });

    // Create motionData for indices 0..19 with distinct rotations
    const motionData = Array.from({ length: 20 }, (_, i) => [0, 0, 0, i, i*2, i*3]);
    const frame = { motionData, metadata: {} };
    integ.handleTimelineFrame(frame, 0.0);

    const names = new Set(integ.frameQueue[integ.frameQueue.length-1]?.vrmFrame ? Object.keys(integ.frameQueue[integ.frameQueue.length-1].vrmFrame.bones) : binder.records.map(r=>r.boneName));
    const expected = ['hips','spine','neck','head','leftArm','rightArm','leftUpLeg','rightUpLeg','leftFoot','rightFoot'];
    const present = expected.filter(n => names.has(n));
    return { count: names.size, present, expected };
  });

  expect(res.count).toBeGreaterThan(8);
  for (const n of res.expected) expect(res.present).toContain(n);
});
