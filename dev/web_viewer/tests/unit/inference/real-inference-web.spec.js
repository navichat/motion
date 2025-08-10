import { test, expect } from '@playwright/test';

// Requires web server for asset loading; CDN-based quick checks. Skipped on CI by default.

test('real inference (web): kokoro-js TTS synth quick check', async ({ page }) => {
  test.slow();
  if (process.env.CI && process.env.SKIP_REAL_INFERENCE) test.skip(true, 'Skip on CI');
  await page.goto('/index.html');
  await page.addScriptTag({ url: 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.min.js' });
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

test('real inference (web): transformers VAD pipeline exists (silero)', async ({ page }) => {
  test.slow();
  if (process.env.CI && process.env.SKIP_REAL_INFERENCE) test.skip(true, 'Skip on CI');
  await page.goto('/index.html');
  await page.addScriptTag({ url: 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js' });
  const ok = await page.evaluate(async () => {
    try {
      const { pipeline } = window.transformers;
      const vad = await pipeline('voice-activity-detection', 'Xenova/silero-vad');
      return typeof vad?.__proto__ === 'object';
    } catch { return false; }
  });
  expect(ok).toBeTruthy();
});
