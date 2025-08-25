import { test, expect } from '@playwright/test';

// REAL: Whisper (transformers.js) ASR-only smoke, no Kokoro needed
// Gated by RUN_REAL_INFERENCE

const RUN_REAL = (process.env.RUN_REAL_INFERENCE === 'true' || process.env.RUN_REAL_INFERENCE === '1');
const describeFn = RUN_REAL ? test.describe : test.describe.skip;

describeFn('Ultimate REAL: Whisper ASR only', () => {
  test('Mic→Whisper(on-device) hears text and schedules beeps', async ({ page, baseURL }) => {
    const params = new URLSearchParams({
      backend: 'beeps',
      playAudio: '0',
      asr: 'whisper',
      asrModel: 'Xenova/whisper-tiny.en',
      listenSec: '1',
    });
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?' + params.toString();
    await page.goto(url);

    // Trigger listen & reply
    await page.getByRole('button', { name: /Listen & Reply/i }).click();

    const log = page.locator('#log');
    // Expect ASR heard text and scheduling marker (beeps backend)
    await expect(log).toContainText(/\u{1F4DD}\s*Heard:/u, { timeout: 120000 }); // 📝 Heard:
    await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 60000 });
    await expect(log).toContainText(/[PLAYWRIGHT] TTS engine=beeps useTts=0|🎛️ Using backend:\s*beeps/);

    // Expressions > 0 confirm animation applied
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
    }, { timeout: 20000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);
  });
});
