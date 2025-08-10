import { test, expect } from '@playwright/test';

// Opt-in: SpeechT5 TTS via transformers.js (may be heavy; keep short and gated)

test('real inference: SpeechT5 loads pipeline', async ({ page }) => {
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
      // Load speech-synthesis pipeline (SpeechT5) – may fallback if unavailable
      const synth = await pipeline('text-to-speech', 'Xenova/speecht5_tts', { quantized: true }).catch(() => null);
      return !!synth;
    } catch (e) { return false; }
  });

  expect(ok).toBeTruthy();
});
