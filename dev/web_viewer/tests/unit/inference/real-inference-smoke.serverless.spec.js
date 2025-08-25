import { test, expect } from '@playwright/test';

// Real inference smoke: load lightweight pipelines and run 1 inference each.
// Guarded to remain fast and skip if offline/CI limited.

test('real inference: transformers whisper-tiny.en transcribes 1s of silence', async ({ page }) => {
  test.slow();
  await page.goto('about:blank');
  const run = !!process.env.RUN_REAL_INFERENCE && !process.env.CI;
  if (!run) test.skip(true, 'Set RUN_REAL_INFERENCE=1 to enable');

  const res = await page.evaluate(async () => {
    // Ensure transformers is available
    if (!window.transformers) {
      try {
        const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
        // Some builds export under default
        window.transformers = mod?.default || mod;
      } catch (e) { return false; }
    }
    const { pipeline } = window.transformers || {};
    if (!pipeline) return false;
    const asr = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
    // 1s of silence (16k mono)
    const sampleRate = 16000;
    const audio = new Float32Array(sampleRate);
    const out = await asr(audio, { chunk_length_s: 1, return_timestamps: false });
    return typeof out?.text === 'string';
  });
  expect(res).toBeTruthy();
});

test('real inference: transformers text generation (gpt2) prompt', async ({ page }) => {
  test.slow();
  await page.goto('about:blank');
  const run = !!process.env.RUN_REAL_INFERENCE && !process.env.CI;
  if (!run) test.skip(true, 'Set RUN_REAL_INFERENCE=1 to enable');

  const res = await page.evaluate(async () => {
    if (!window.transformers) {
      try {
        const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
        window.transformers = mod?.default || mod;
      } catch (e) { return false; }
    }
    const { pipeline } = window.transformers || {};
    if (!pipeline) return false;
  const gen = await pipeline('text-generation', 'Xenova/gpt2');
    const out = await gen('Hello Ichika:', { max_new_tokens: 8 });
    return typeof out?.[0]?.generated_text === 'string';
  });
  expect(res).toBeTruthy();
});
