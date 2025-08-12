import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Requires web server for asset loading; CDN-based quick checks.
const SHOULD_SKIP = !process.env.RUN_REAL_INFERENCE || !!process.env.CI || !!process.env.SKIP_REAL_INFERENCE;

test.describe('real inference (web)', () => {
  if (SHOULD_SKIP) test.skip(true, 'Skip real inference in CI/automated runs');

  test('kokoro-js TTS synth quick check', async ({ page }) => {
    test.slow();
  if (!process.env.RUN_KOKORO) test.skip(true, 'Set RUN_KOKORO=1 to enable kokoro-js synth check');
    await page.goto('/index.html');
  const KOKORO_URL = process.env.KOKORO_JS_URL || 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.min.js';
  await page.addScriptTag({ url: KOKORO_URL });
    const ok = await page.evaluate(async () => {
      if (!window.kokoro) return false;
      try {
        const tts = await window.kokoro.create();
        const audio = await tts.speak('Hello Ichika');
        return !!audio?.getAudioBuffer || !!audio?.buffer || !!audio;
      } catch { return false; }
    });
    expect(ok).toBeTruthy();
  });

  test('transformers VAD pipeline exists (silero)', async ({ page }) => {
    test.slow();
    await page.goto('/index.html');
  const TRANSFORMERS_URL = process.env.TRANSFORMERS_URL || 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
    const VAD_LOCAL = process.env.TRANSFORMERS_MODEL_VAD;
    if (!VAD_LOCAL) test.skip(true, 'Provide TRANSFORMERS_MODEL_VAD served folder (e.g., /vendor/models/silero-vad)');
    // Verify the local folder exists and has minimal metadata
    const servedRoot = path.join(process.cwd(), 'dev/web_viewer');
    const rel = VAD_LOCAL.replace(/^\//, '');
    const abs = path.join(servedRoot, rel);
    const hasConfig = fs.existsSync(path.join(abs, 'config.json'));
    const hasModelJson = fs.existsSync(path.join(abs, 'model.json'));
    const weightCandidates = [
      'model.onnx',
      path.join('onnx', 'model.onnx'),
      'pytorch_model.bin',
      'model.safetensors'
    ];
    const hasWeights = weightCandidates.some(p => fs.existsSync(path.join(abs, p)));
    if (!fs.existsSync(abs) || !hasConfig || !hasModelJson || !hasWeights) {
      test.skip(true, `Local VAD weights not found under ${abs}; provide real model files or let this skip`);
    }
  const ok = await page.evaluate(async ({ url, modelOverride }) => {
      try {
  const mod = await import(url);
  // Prefer local folder when provided; make base '/' so absolute served paths work
  mod.env.allowLocalModels = !!modelOverride;
  mod.env.allowRemoteModels = !modelOverride; // avoid network if we have local
  mod.env.localModelPath = modelOverride ? '/' : null;
  const { pipeline } = mod;
  const modelId = modelOverride ? modelOverride.replace(/^\/+/, '') : 'Xenova/silero-vad';
  let vad;
  try {
    vad = await pipeline('voice-activity-detection', modelId);
  } catch (e) {
    // Fallback: older versions expose VAD via audio-classification task
    vad = await pipeline('audio-classification', modelId);
  }
        return typeof vad?.__proto__ === 'object';
      } catch { return false; }
  }, { url: TRANSFORMERS_URL, modelOverride: VAD_LOCAL || null });
    expect(ok).toBeTruthy();
  });
});
