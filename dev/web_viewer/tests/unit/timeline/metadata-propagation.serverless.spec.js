import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const TL = read('dev/web_viewer/src/components/animation/timeline/BVHTimeline.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Ensure face viseme and audio gesture energy propagate to composed frame metadata.

test('Composed metadata carries faceViseme and gestureEnergy (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);

  const res = await page.evaluate(async () => {
    const { BVHTimeline, BVHClip } = window;
    const tl = new BVHTimeline({ framerate: 30 });

    const base = new BVHClip({ type: 'generated', startTime: 0, duration: 1, weight: 1, blendMode: 'replace', generator: async () => ({ motionData: [], metadata: {} }) });
    const face = new BVHClip({ type: 'generated', startTime: 0, duration: 1, weight: 1, blendMode: 'replace', generator: async () => ({ motionData: [], metadata: { viseme: 'A' } }) });
    const audio = new BVHClip({ type: 'generated', startTime: 0, duration: 1, weight: 1, blendMode: 'additive', generator: async () => ({ motionData: [], metadata: { energy: 0.7 } }) });

    tl.addClip('base', base);
    tl.addClip('face', face);
    tl.addClip('audio', audio);

    const frame = await tl.getFrameAtTime(0.0);
    return { faceViseme: frame.metadata?.faceViseme, gestureEnergy: frame.metadata?.gestureEnergy };
  });

  expect(res.faceViseme).toBeDefined();
  expect(typeof res.gestureEnergy).toBe('number');
});
