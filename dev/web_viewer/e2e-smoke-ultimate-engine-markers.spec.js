import { test, expect } from '@playwright/test';

test.describe('Ultimate engine markers smoke', () => {
  test('beeps backend logs engine marker [E2E][smoke]', async ({ page, baseURL }) => {
    await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?backend=beeps&playAudio=0');
    await page.fill('#text', 'marker check');
    await page.getByRole('button', { name: /Say/i }).click();
    const log = page.locator('#log');
  await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 15000 });
    await expect(log).toContainText(/[PLAYWRIGHT] TTS engine=beeps useTts=0|🎛️ Using backend:\s*beeps/);
  });

  test('speech backend logs engine marker [E2E][smoke]', async ({ page, baseURL }) => {
    await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?backend=speech&playAudio=0');
    await page.fill('#text', 'marker check');
    await page.getByRole('button', { name: /Say/i }).click();
    const log = page.locator('#log');
  await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 15000 });
    await expect(log).toContainText(/[PLAYWRIGHT] TTS engine=speech useTts=0|🎛️ Using backend:\s*speech/);
  });

  test('speecht5 backend logs engine marker [E2E][smoke]', async ({ page, baseURL }) => {
    await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?backend=speecht5&playAudio=0');
    await page.fill('#text', 'marker check');
    await page.getByRole('button', { name: /Say/i }).click();
    const log = page.locator('#log');
  await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 15000 });
    await expect(log).toContainText(/[PLAYWRIGHT] TTS engine=speecht5 useTts=0|🎛️ Using backend:\s*speecht5/);
  });
});
