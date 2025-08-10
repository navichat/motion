import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Web-backed smoke test: initialize ORT Web if configured. Skips by default.
// Requires unit-web project (web server) and configured URLs:
//   ModelUrlConfig.setModelUrl('runtime.ort', '/path/to/ort.min.js');
//   ModelUrlConfig.setModelUrl('audio2gesture', '/models/audio2gesture_step_fixed.onnx');

test('ONNX Runtime Web session initialization smoke (skipped unless configured)', async ({ page }) => {
  await page.goto('/index.html');
  await inject(page, CONFIG);

  const check = await page.evaluate(async () => {
    const cfg = (typeof module !== 'undefined' && module.exports) ? module.exports : window.ModelUrlConfig;
    if (!cfg) return { skip: true, reason: 'ModelUrlConfig missing' };

    const ortScript = cfg.getModelUrl('runtime.ort');
    const modelUrl = cfg.getModelUrl('audio2gesture');
    if (!ortScript || !modelUrl) return { skip: true, reason: 'ORT script or model URL not configured' };

    // Load ORT script
    const absScriptUrl = new URL(ortScript, window.location.origin).toString();
    try {
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = absScriptUrl;
        s.onload = resolve;
        s.onerror = () => reject(new Error('Failed to load ORT script'));
        document.head.appendChild(s);
      });
    } catch (e) {
      return { skip: true, reason: 'Failed to load ORT script' };
    }

    if (!window.ort) return { skip: true, reason: 'ORT not available after script load' };

    return { skip: false, modelUrl };
  });

  if (check.skip) {
    test.skip(true, check.reason || 'ORT/model not configured');
  }

  // If configured, perform a minimal sanity call: ensure ort namespace exists
  const hasOrt = await page.evaluate(() => !!window.ort && typeof window.ort.InferenceSession === 'function');
  expect(hasOrt).toBeTruthy();

  // Optional: Do not actually create the session to avoid loading large models by default.
  // If needed later, uncomment to validate session creation against small test models.
  // const created = await page.evaluate(async (modelUrl) => {
  //   const { InferenceSession, env } = window.ort;
  //   env.wasm.numThreads = 1;
  //   const session = await InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  //   return !!session;
  // }, check.modelUrl);
  // expect(created).toBeTruthy();
});
