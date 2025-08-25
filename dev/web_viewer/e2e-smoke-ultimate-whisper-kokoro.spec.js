import { test, expect } from '@playwright/test';

// REAL: Whisper (transformers.js) + Kokoro (kokoro-js) end-to-end
// Gated by RUN_REAL_INFERENCE and KOKORO_JS

const RUN_REAL = (process.env.RUN_REAL_INFERENCE === 'true' || process.env.RUN_REAL_INFERENCE === '1');
const KOKORO_JS = process.env.KOKORO_JS || '';

(RUN_REAL && KOKORO_JS ? test.describe : test.describe.skip)('Ultimate REAL: Whisper + Kokoro', () => {
  test('Mic→Whisper(on-device)→Kokoro(on-device) schedules animation', async ({ page, baseURL }) => {
    const params = new URLSearchParams({
      backend: 'kokoro',
      asr: 'whisper',
      asrModel: 'Xenova/whisper-tiny.en',
      listenSec: '1',
      playAudio: '0',
      kokoroJs: KOKORO_JS,
    });
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?' + params.toString();
    await page.goto(url);

    // Trigger listen & reply
    await page.getByRole('button', { name: /Listen & Reply/i }).click();

    const log = page.locator('#log');
    // Expect ASR heard text and Kokoro success marker
    await expect(log).toContainText(/\u{1F4DD}\s*Heard:/u, { timeout: 120000 }); // 📝 Heard:
    await expect(log).toContainText('kokoro audio ok', { timeout: 120000 });

    // Also expect a scheduling marker
    await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 60000 });

    // Expressions > 0 confirm animation applied
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
    }, { timeout: 20000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);
  });
});
