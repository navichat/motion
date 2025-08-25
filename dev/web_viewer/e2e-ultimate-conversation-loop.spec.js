import { test, expect } from '@playwright/test';

const RUN_LOOP = process.env.RUN_CONV_LOOP === '1';

const DEMO_URL = '/demos/ichika_voice_conversation_demo.html';

async function waitForLog(page, needle, timeout = 15000) {
  await page.waitForFunction(
    (n) => (document.getElementById('log')?.textContent || '').includes(n),
    needle,
    { timeout }
  );
}

test.describe('[E2E][Ultimate][ConversationLoop]', () => {
  test.skip(!RUN_LOOP, 'Set RUN_CONV_LOOP=1 to run the conversation loop test');
  test('Two fake cycles schedule animations and increase expressions', async ({ page }) => {
    await page.goto(`${DEMO_URL}?backend=beeps&asr=fake&playAudio=0&listenSec=1`);
    await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });

  // Run two cycles
  await page.evaluate(() => window.__ultimateDemo?.conversationLoop?.(2));

  // Wait for loop completion marker and scheduled animation
  await waitForLog(page, '[PLAYWRIGHT] Conv loop done n=2');
  await waitForLog(page, '🗣️ Scheduled TTS for text:');

    // Expressions should be > 0
    await page.waitForFunction(() => {
      const st = window.__ultimateDemo?.getStats?.();
      return st && typeof st.expressions === 'number' && st.expressions > 0;
    }, null, { timeout: 8000 });

    const stats = await page.evaluate(() => window.__ultimateDemo?.getStats?.());
    expect(stats.expressions).toBeGreaterThan(0);
  });
});
