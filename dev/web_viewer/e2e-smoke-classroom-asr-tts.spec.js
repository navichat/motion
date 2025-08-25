import { test, expect } from '@playwright/test';

// Lightweight smoke for classroom on-device controls; REAL path gated.

test.describe('Classroom ASR and TTS wiring', () => {
  test('UI exists: Say, Preload, Listen & Reply', async ({ page, baseURL }) => {
    await page.goto(baseURL + '/demos/ichika_classroom_demo.html');
    await expect(page.locator('#ttsSay')).toBeVisible();
    await expect(page.locator('#ttsPreload')).toBeVisible();
    await expect(page.locator('#listenReply')).toBeVisible();
  });

  const RUN_REAL = (process.env.RUN_REAL_INFERENCE || '').toLowerCase() === 'transformers';
  const slow = { timeout: 120_000 };

  if (RUN_REAL) {
  test('REAL: Whisper + Kokoro loop schedules animation [E2E][REAL]', slow, async ({ page, baseURL }) => {
      await page.goto(baseURL + '/demos/ichika_classroom_demo.html?engine=kokoro&asr=whisper&asrModel=Xenova/whisper-tiny.en&listenSec=1&playAudio=0&autoPreload=1');
      await page.getByRole('button', { name: /Listen & Reply/i }).click();
      await expect(page.locator('#log')).toContainText(/ASR: Whisper model|📝 Heard:/i, { timeout: 90_000 });
      await expect(page.locator('#log')).toContainText(/Scheduled TTS animation/i, { timeout: 90_000 });
    });
  } else {
    test.skip('REAL gated', async () => {});
  }
});
