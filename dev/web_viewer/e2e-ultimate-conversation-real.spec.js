// Ultimate conversation (real inference): Whisper ASR + SpeechT5 on-device TTS
// Skips unless RUN_REAL_INFERENCE=1 is set to avoid heavy downloads in routine CI.

const { test, expect } = require('@playwright/test');

const RUN_REAL = process.env.RUN_REAL_INFERENCE === '1';

async function waitForLog(page, needle, timeout = 60000) {
  await page.waitForFunction(
    (n) => (document.getElementById('log')?.textContent || '').includes(n),
    needle,
    { timeout }
  );
}

test.describe('[Ultimate][Real][Whisper+SpeechT5]', () => {
  test.skip(!RUN_REAL, 'Set RUN_REAL_INFERENCE=1 to run real Whisper+SpeechT5 test');

  test('Mic Whisper -> SpeechT5 on-device reply schedules animation and increases expressions', async ({ page }) => {
    // Request whisper ASR and on-device SpeechT5; allow audio playback
    const url = '/demos/ichika_voice_conversation_demo.html?backend=speecht5&asr=whisper&playAudio=1&speecht5OnDevice=1&listenSec=2';
    await page.goto(url);

    // Wait for demo API
    await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 20000 });

    // Start mic and run a listen & reply
    await page.evaluate(() => window.__ultimateDemo?.startMic?.());
    await page.evaluate(() => window.__ultimateDemo?.listenAndReply?.());

    // ASR marker (whisper)
    await waitForLog(page, '[PLAYWRIGHT] ASR done engine=whisper status=ok', 90000);

    // TTS markers (SpeechT5 on-device)
    await waitForLog(page, '[PLAYWRIGHT] TTS start engine=speecht5 path=on-device', 90000);
    await waitForLog(page, '[PLAYWRIGHT] TTS done engine=speecht5 path=on-device status=ok', 120000);
    await waitForLog(page, '[PLAYWRIGHT] Scheduled TTS animation', 60000);

    // Expressions should increase
    await page.waitForFunction(() => {
      const st = window.__ultimateDemo?.getStats?.();
      return st && typeof st.expressions === 'number' && st.expressions > 0;
    }, null, { timeout: 20000 });

    await page.evaluate(() => window.__ultimateDemo?.stopAll?.());
  });
});
