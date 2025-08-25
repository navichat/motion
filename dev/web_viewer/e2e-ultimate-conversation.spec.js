// Ultimate conversation test: mic -> fake ASR -> TTS reply; audio drives gestures
// This uses the demo's exposed window.__ultimateDemo helpers and stable [PLAYWRIGHT] log markers.

const { test, expect } = require('@playwright/test');

async function waitForLog(page, needle, timeout = 15000) {
  await page.waitForFunction(
    (n) => (document.getElementById('log')?.textContent || '').includes(n),
    needle,
    { timeout }
  );
}

test.describe('Ultimate conversation', () => {
  test('Mic fake ASR -> reply schedules animation and increases expressions', async ({ page }) => {
  // Deterministic intermediate path: fake ASR + beeps backend + no audio playback (gestures still scheduled)
  await page.goto('/demos/ichika_voice_conversation_demo.html?backend=beeps&asr=fake&playAudio=0&listenSec=1');

  // Wait for demo API
  await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });

    // Start mic (uses fake device in CI via launch args)
    await page.evaluate(() => window.__ultimateDemo?.startMic?.());

    // Trigger a listen & reply cycle using fake ASR
    await page.evaluate(() => window.__ultimateDemo?.listenAndReply?.());

  // Assert generic markers appear
  await waitForLog(page, '👂 Listening (fake ASR)…');
  await waitForLog(page, '📝 Heard: ');
  await waitForLog(page, '🗣️ Scheduled TTS for text:');

    // Verify expressions increased (> 0)
    await page.waitForFunction(() => {
      const st = window.__ultimateDemo?.getStats?.();
      return st && typeof st.expressions === 'number' && st.expressions > 0;
    }, null, { timeout: 8000 });

    // Cleanup
    await page.evaluate(() => window.__ultimateDemo?.stopAll?.());
  });
});
