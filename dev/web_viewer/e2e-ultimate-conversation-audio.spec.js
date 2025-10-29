// Ultimate audio-driven conversation test: mic -> fake ASR -> TTS beeps
// Assertions: audio energy monitoring markers appear and expression counters
// increase DURING playback (playAudio=1).

const { test, expect } = require('@playwright/test');

async function waitForLog(page, needle, timeout = 15000) {
  await page.waitForFunction(
    (n) => (document.getElementById('log')?.textContent || '').includes(n),
    needle,
    { timeout }
  );
}

test.describe('Ultimate audio-driven conversation', () => {
  test.setTimeout(60_000);
  test('Energy-driven expressions increase during playback', async ({ page }) => {
    await page.goto('/demos/ichika_voice_conversation_demo.html?backend=beeps&asr=fake&playAudio=1&listenSec=1');

    await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });
    await waitForLog(page, '[PLAYWRIGHT] Demo ready');

    await page.evaluate(() => window.__ultimateDemo?.startMic?.());

    // Kick off one listen+reply
    await page.evaluate(() => window.__ultimateDemo?.listenAndReply?.());

    // Wait for audio energy monitor to start, then snapshot counters
    await waitForLog(page, '[PLAYWRIGHT] Audio energy monitoring start');
    const before = await page.evaluate(() => window.__ultimateDemo.getStats());

    // Wait until energy monitoring ends, then snapshot again
    await waitForLog(page, '[PLAYWRIGHT] Audio energy monitoring end');
    const after = await page.evaluate(() => window.__ultimateDemo.getStats());

    // Validate that audio playback drove animation updates
    expect(after.applied).toBeGreaterThan(before.applied);
    expect(after.expressions).toBeGreaterThan(before.expressions);

    // Stronger bound: ensure more than a single deterministic tick
    expect(after.expressions - before.expressions).toBeGreaterThanOrEqual(2);

    await page.evaluate(() => window.__ultimateDemo?.stopAll?.());
  });
});
