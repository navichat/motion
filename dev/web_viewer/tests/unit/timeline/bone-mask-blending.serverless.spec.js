import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const TL = read('dev/web_viewer/src/components/animation/timeline/BVHTimeline.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Serverless validation: per-clip boneMask should restrict overlay to targeted bones only.

test('Bone mask blending replaces only masked bones (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);

  const res = await page.evaluate(async () => {
    const { BVHTimeline } = window;
    const tl = new BVHTimeline({ lookaheadFrames: 0, framerate: 30 });

    // Base frame with hips/spine values
    const baseClip = new window.BVHClip({
      type: 'generated', startTime: 0, duration: 1, blendMode: 'replace', weight: 1,
      generator: async () => ({ motionData: [ [0,0,0, 0,0,0], [0,0,0, 0,0,0] ], metadata: {} })
    });

    // Overlay targeting only 'spine' via boneMask metadata on frame
    const overlayClip = new window.BVHClip({
      type: 'generated', startTime: 0, duration: 1, blendMode: 'replace', weight: 1,
      generator: async () => ({ motionData: [ [1,1,1, 10,10,10], [2,2,2, 20,20,20] ], metadata: { boneMask: ['spine'] } })
    });

    // Setup bone mapping: 0 -> hips, 1 -> spine
    tl.setBoneMapping({ 0: 'hips', 1: 'spine' });
    tl.addClip('base', baseClip);
    tl.addClip('override', overlayClip);

    const frame = await tl.getFrameAtTime(0.0);
    const hips = frame.motionData[0];
    const spine = frame.motionData[1];
    return { hips, spine };
  });

  // Hips should remain base (zeros), spine should be overlay values
  expect(res.hips).toEqual([0,0,0, 0,0,0]);
  expect(res.spine).toEqual([2,2,2, 20,20,20]);
});
