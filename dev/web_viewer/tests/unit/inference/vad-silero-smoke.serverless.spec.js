import { test, expect } from '@playwright/test';

// Real VAD smoke using transformers Silero VAD; gated to avoid CI/network.

test('real inference: silero-vad returns segments array (silence)', async ({ page }) => {
  const run = !!process.env.RUN_REAL_INFERENCE && !process.env.CI;
  if (!run) test.skip(true, 'Set RUN_REAL_INFERENCE=1 to enable');
  await page.goto('about:blank');

  const ok = await page.evaluate(async () => {
    try {
      let T = window.transformers;
      if (!T) {
        const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
        T = mod?.default || mod;
        window.transformers = T;
      }
      const { pipeline } = T || {};
      if (!pipeline) return false;
      const vad = await pipeline('voice-activity-detection', 'Xenova/silero-vad');
      const sr = 16000;
      const audio = new Float32Array(sr); // 1s silence
      const out = await vad(audio, { return_timestamps: true });
      return Array.isArray(out?.[0]?.chunks || out?.chunks || out);
    } catch (e) { return false; }
  });

  expect(ok).toBeTruthy();
});
