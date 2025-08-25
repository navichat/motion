import { test, expect } from '@playwright/test';

// REAL: SpeechT5 on-device generates PCM and schedules animation (gated)
// Uses transformers.js (Xenova) in the page. Requires models to be resolvable by the server.

const RUN_REAL = (process.env.RUN_REAL_INFERENCE === 'true' || process.env.RUN_REAL_INFERENCE === '1');
const SPEECHT5_MODEL = process.env.SPEECHT5_MODEL || 'Xenova/speecht5_tts';

// Skip unless explicitly enabled
(RUN_REAL ? test.describe : test.describe.skip)('Ultimate SpeechT5 TTS smoke (on-device)', () => {
  test('REAL: SpeechT5 on-device generates PCM and schedules', async ({ page, baseURL }) => {
    const params = new URLSearchParams({
      backend: 'speecht5',
      speecht5OnDevice: '1',
      speecht5Model: SPEECHT5_MODEL,
      speecht5Spk: 'random',
      playAudio: '0',
    });
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?' + params.toString();
    await page.goto(url);

    // Drive via UI to force a TTS utterance
    await page.fill('#text', 'This is a SpeechT5 on-device speech test.');
    await page.getByRole('button', { name: /Say/i }).click();

    const log = page.locator('#log');
    // Expect on-device path success and scheduling markers
    await expect(log).toContainText('SpeechT5 on-device audio ok', { timeout: 120000 });
    await expect(log).toContainText(/\[PLAYWRIGHT\] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 30000 });

    // No beeps fallback in this real path
    await expect(log).not.toContainText(/falling back to beeps|beeps fallback/i);

    // Expressions > 0
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
    }, { timeout: 15000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);
  });
});
