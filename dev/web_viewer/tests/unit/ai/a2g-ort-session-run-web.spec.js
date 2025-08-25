import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');
const A2G_TASK = read('dev/web_viewer/src/components/animation/timeline/tasks/Audio2GestureOrtTask.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Web-backed smoke: create an ORT session and run one a2g chunk when configured.
// Skips by default unless ModelUrlConfig has runtime.ort and audio2gesture URLs set.
test('Audio2Gesture ORT session run smoke (skipped unless configured)', async ({ page }) => {
  await page.goto('/index.html');
  await inject(page, CONFIG);
  await inject(page, A2G_TASK);

  // Auto-configure ModelUrlConfig from env or local served model if present
  const ORT_URL = process.env.RUNTIME_ORT || 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.min.js';
  const LOCAL_A2G = path.join(ROOT, 'dev/web_viewer/models/audio2gesture_step_fixed.onnx');
  const A2G_URL = process.env.AUDIO2GESTURE_URL || (fs.existsSync(LOCAL_A2G) ? '/models/audio2gesture_step_fixed.onnx' : undefined);
  if (ORT_URL || A2G_URL) {
    await page.evaluate(({ ORT_URL, A2G_URL }) => {
      const cfg = window.ModelUrlConfig || (typeof module !== 'undefined' && module.exports);
      if (!cfg || typeof cfg.setModelUrl !== 'function') return;
      if (ORT_URL) cfg.setModelUrl('runtime.ort', ORT_URL);
      if (A2G_URL) cfg.setModelUrl('audio2gesture', A2G_URL);
    }, { ORT_URL, A2G_URL });
  }

  const check = await page.evaluate(async () => {
    const cfg = window.ModelUrlConfig || (typeof module !== 'undefined' && module.exports);
    if (!cfg) return { skip: true, reason: 'ModelUrlConfig missing' };
    const ortScript = cfg.getModelUrl('runtime.ort');
    const modelUrl = cfg.getModelUrl('audio2gesture');
    if (!ortScript || !modelUrl) return { skip: true, reason: 'ORT script or model URL not configured' };

    // Load ORT script
    const absScriptUrl = new URL(ortScript, window.location.origin).toString();
    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = absScriptUrl; s.onload = resolve; s.onerror = () => reject(new Error('Failed to load ORT script'));
      document.head.appendChild(s);
    }).catch(() => ({ failed: true }));
    if (!window.ort) return { skip: true, reason: 'ORT not available after script load' };

    return { skip: false, modelUrl };
  });

  if (check.skip) {
    test.skip(true, check.reason || 'Not configured');
  }

  const ok = await page.evaluate(async (modelUrl) => {
    const ctor = window.Audio2GestureOrtTask && window.Audio2GestureOrtTask.Audio2GestureOrtTask;
    if (typeof ctor !== 'function') return false;
    const task = new ctor({ framerate: 10, chunkMs: 100, provider: 'wasm', modelUrl });
    const inited = await task.initialize(window);
    if (!inited) return false;
    const featureProvider = { async next() { return { mfcc: new Float32Array(13) }; } };
    const it = task.run({ featureProvider });
    const first = await it.next();
    if (!first || !first.value || !first.value.frames || first.value.frames.length === 0) return false;
    return first.value.frames[0].metadata && first.value.frames[0].metadata.model === 'a2g_ort';
  }, check.modelUrl);

  expect(ok).toBe(true);
});
