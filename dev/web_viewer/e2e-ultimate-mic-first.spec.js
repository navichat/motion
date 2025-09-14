// Shell-timeout compliant deterministic Playwright test
// Verifies mic can start, a single listen+reply runs, and audio-driven expressions increase
const { test, expect } = require('@playwright/test');

test.describe('Ultimate conversation - mic-first path', () => {
  test.setTimeout(45_000);

  test('Mic starts, listen+reply completes, expressions increase', async ({ page, context }) => {
    const params = new URLSearchParams({
      backend: 'beeps', // deterministic TTS beeps
      asr: 'fake',      // deterministic ASR
      listenSec: '1',
      playAudio: '1'
    });
    const url = `/demos/ichika_voice_conversation_demo.html?${params.toString()}`;

    await page.goto(url);

    await page.waitForFunction(() => typeof window.__ultimateDemo === 'object');
    await page.waitForFunction(() => /\[PLAYWRIGHT\]/.test(document.getElementById('log')?.textContent || ''));

    const before = await page.evaluate(() => {
      const s = window.__ultimateDemo.getStats();
      return { applied: s.applied || 0, expressions: s.expressions || 0 };
    });

    // Start mic (fake UI/device flags are set in playwright.config.js)
    const micStarted = await page.evaluate(async () => {
      try {
        await window.__ultimateDemo.startMic();
        return true;
      } catch (e) {
        return false;
      }
    });
    expect(micStarted).toBeTruthy();

    // Run one listen+reply and wait for it to settle
    await page.evaluate(() => window.__ultimateDemo.listenAndReply());

    // Wait for some scheduling to apply animations
    await page.waitForTimeout(1500);

    const after = await page.evaluate(() => {
      const s = window.__ultimateDemo.getStats();
      return { applied: s.applied || 0, expressions: s.expressions || 0 };
    });

    // Expect counters to increase indicating audio-driven updates applied
    expect(after.applied).toBeGreaterThan(before.applied);
    expect(after.expressions).toBeGreaterThan(before.expressions);

    // Optional sanity: ensure at least one marker was printed
    const logText = await page.textContent('#log');
    expect(/\[PLAYWRIGHT\]/.test(logText || '')).toBeTruthy();
  });
});
