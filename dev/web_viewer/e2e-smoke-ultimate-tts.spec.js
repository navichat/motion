// Lightweight smoke for ultimate conversation demo on-device TTS UI.
// Includes optional gated real runs using transformers.js engines.

import { test, expect } from '@playwright/test';

test.describe('Ultimate demo on-device TTS', () => {
  test('UI smoke: essential controls exist', async ({ page, baseURL }) => {
    await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?backend=none');
    await expect(page.locator('#mic')).toBeVisible();
    await expect(page.locator('#say')).toBeVisible();
  });

  const RUN_REAL = (process.env.RUN_REAL_INFERENCE || '').toLowerCase() === 'transformers';
  const slow = { timeout: 120_000 }; // allow model load

  if (RUN_REAL) {
    test('REAL: SpeechT5 on-device generates PCM and schedules', slow, async ({ page, baseURL }) => {
      await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?engine=speecht5&backend=none&useTts=0&playAudio=0&autoPreload=1');
      await page.getByRole('button', { name: /Say/i }).click();
  await expect(page.locator('#log')).toContainText(/Using on-device SpeechT5|Scheduled TTS animation|Scheduled on-device TTS|Scheduled TTS for text/i, { timeout: 90_000 });
    });

    test('REAL: Kokoro on-device generates PCM and schedules', slow, async ({ page, baseURL }) => {
      await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?engine=kokoro&backend=none&useTts=0&playAudio=0&autoPreload=1');
      await page.getByRole('button', { name: /Say/i }).click();
  await expect(page.locator('#log')).toContainText(/Using on-device Kokoro|Scheduled TTS animation|Scheduled on-device TTS|Scheduled TTS for text/i, { timeout: 90_000 });
    });
  } else {
    test.skip('REAL: transformers gated', async () => {});
  }
});
