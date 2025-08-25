import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Serverless test to ensure config is importable and getters/setters work

test('ModelUrlConfig get/set works (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, CONFIG);

  const res = await page.evaluate(() => {
    const mod = (typeof module !== 'undefined' && module.exports) ? module.exports : window.ModelUrlConfig;
    if (!mod) throw new Error('ModelUrlConfig missing');

    const { getModelUrl, setModelUrl } = mod;

    const before = getModelUrl('audio2gesture');
    setModelUrl('audio2gesture', '/models/audio2gesture_step_fixed.onnx');
    const after = getModelUrl('audio2gesture');

    setModelUrl('rsmt.deepPhase', '/models/deepphase.onnx');
    const deepPhase = getModelUrl('rsmt.deepPhase');

    return { before, after, deepPhase };
  });

  expect(res.before === undefined || res.before === null).toBeTruthy();
  expect(res.after).toBe('/models/audio2gesture_step_fixed.onnx');
  expect(res.deepPhase).toBe('/models/deepphase.onnx');
});
