import { test, expect } from '@playwright/test';

// REAL: Kokoro on-device generates PCM and schedules animation (gated)
// Requires a kokoro-js runtime URL via env KOKORO_JS (or use ModelUrlConfig in the server).

const RUN_REAL = (process.env.RUN_REAL_INFERENCE === 'true' || process.env.RUN_REAL_INFERENCE === '1');
const KOKORO_JS = process.env.KOKORO_JS || '';

// Skip unless explicitly enabled and runtime provided
const describeFn = (RUN_REAL && KOKORO_JS) ? test.describe : test.describe.skip;
describeFn('Ultimate Kokoro TTS smoke', () => {
  test('REAL: Kokoro on-device generates PCM and schedules', async ({ page, baseURL }) => {
    const params = new URLSearchParams({ backend: 'kokoro', playAudio: '0', kokoroJs: KOKORO_JS });
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?' + params.toString();
    await page.goto(url);

    // Drive via UI to force a TTS utterance
    await page.fill('#text', 'This is a Kokoro on-device speech test.');
    await page.getByRole('button', { name: /Say/i }).click();

    const log = page.locator('#log');
    // Expect on-device path success and scheduling markers
    await expect(log).toContainText('kokoro audio ok', { timeout: 120000 });
    await expect(log).toContainText(/\[PLAYWRIGHT\] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 30000 });

    // No fallbacks
    await expect(log).not.toContainText(/kokoro error|falling back to beeps/i);

    // Expressions > 0
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
    }, { timeout: 15000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);
  });
});
