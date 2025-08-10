import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');
const TL = read('dev/web_viewer/src/components/animation/timeline/BVHTimeline.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate orchestrator.startBaseClip passes prebuilt BVHClip via adapter opts.clip

test('[orchestrator] startBaseClip uses prebuilt clip when available (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, REG);
  await inject(page, ORCH);

  const res = await page.evaluate(() => {
    const { BVHTimeline, BVHClip } = window;
    const { ClipRegistry } = window.ClipRegistry?.ClipRegistry ? window.ClipRegistry : { ClipRegistry: window.ClipRegistry };
    const Orch = window.IchikaOrchestrator;

    const reg = new ClipRegistry();
    const clip = new BVHClip({ type: 'static', startTime: 0, duration: 1.5, weight: 1, blendMode: 'replace', bvhData: 'HIERARCHY\nFrames: 45\nFrame Time: 0.0333' });
    reg.add('idle_bvh', clip, { track: 'base', duration: 1.5, fadeInMs: 120 });

    let sawClipObj = false;
    const orch = new Orch({ clipRegistry: reg, adapter: {
      appendChunk(track, chunk, opts) {
        sawClipObj = !!opts?.clip && typeof opts.clip === 'object' && !!(opts.clip.bvhData || opts.clip.generator || opts.clip.type);
      }
    }});

    orch.startBaseClip('idle_bvh');

    return { sawClipObj };
  });

  expect(res.sawClipObj).toBeTruthy();
});
