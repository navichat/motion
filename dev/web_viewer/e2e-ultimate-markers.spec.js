import { test, expect } from '@playwright/test';

const DEMO_URL = '/demos/ichika_voice_conversation_demo.html';

test.describe('[E2E][Ultimate][Markers]', () => {
  test('Beeps backend emits identifiable markers', async ({ page }) => {
    const url = `${DEMO_URL}?backend=beeps&playAudio=0&autoListen=0`;
    await page.goto(url);
    await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });

    await page.fill('#text', 'Marker check beeps.');
    await page.getByRole('button', { name: /Say/i }).click();

    // Expressions update
    await expect.poll(async () => (
      await page.evaluate(() => window.__ultimateDemo?.getStats?.().expressions || 0)
    ), { timeout: 15000, intervals: [200, 400, 800] }).toBeGreaterThan(0);

    // Any of these is sufficient for diagnostics
    await page.waitForFunction(() => {
      const t = document.getElementById('log')?.textContent || '';
      return /\[PLAYWRIGHT\] TTS engine=beeps /.test(t)
        || /\[PLAYWRIGHT\] TTS done engine=beeps /.test(t)
        || /Using backend: beeps/.test(t);
    }, null, { timeout: 15000 });
  });

  test('SpeechT5 emits identifiable markers (stub/ORT/on-device)', async ({ page }) => {
    const url = `${DEMO_URL}?backend=speecht5&playAudio=0&autoListen=0`;
    await page.goto(url);
    await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });

    await page.fill('#text', 'Marker check speechT5.');
    await page.getByRole('button', { name: /Say/i }).click();

    // Expressions update
    await expect.poll(async () => (
      await page.evaluate(() => window.__ultimateDemo?.getStats?.().expressions || 0)
    ), { timeout: 20000, intervals: [200, 400, 800, 1600] }).toBeGreaterThan(0);

    // Accept any of the path markers or generic done marker
    await page.waitForFunction(() => {
      const t = document.getElementById('log')?.textContent || '';
      return /\[PLAYWRIGHT\] TTS done engine=speecht5 /.test(t)
        || /SpeechT5 stub path/.test(t)
        || /SpeechT5 ORT path initialized/.test(t)
        || /SpeechT5 on-device audio ok/.test(t)
        || /\[PLAYWRIGHT\] TTS start engine=speecht5 /.test(t);
    }, null, { timeout: 25000 });
  });
});
