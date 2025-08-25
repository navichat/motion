import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');
const BOOT = read('dev/web_viewer/src/testing/real_inference_bootstrap.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Verifies ModelUrlConfig presence and basic setModelUrl/getModelUrl round-trip (serverless, no network).

test('ModelUrlConfig exists and set/get works (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, CONFIG);
  // Load bootstrap (no-op without query params, but safe)
  await inject(page, BOOT);

  const out = await page.evaluate(() => {
    const cfg = window.ModelUrlConfig;
    if (!cfg) return { hasCfg: false };
    // Round-trip set/get for simple key and nested key
    cfg.setModelUrl('whisper', '/models/whisper.onnx');
    cfg.setModelUrl('runtime.ort', 'https://example.com/ort-wasm.min.js');
    return {
      hasCfg: true,
      whisper: cfg.getModelUrl('whisper'),
      ort: cfg.getModelUrl('runtime.ort') || (cfg.MODELS?.runtime?.ort?.url ?? cfg.MODELS?.runtime?.ort),
    };
  });

  expect(out.hasCfg).toBeTruthy();
  expect(out.whisper).toBe('/models/whisper.onnx');
  if (out.ort !== undefined) expect(out.ort).toBe('https://example.com/ort-wasm.min.js');
});
