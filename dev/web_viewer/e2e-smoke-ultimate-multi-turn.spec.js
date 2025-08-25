// [E2E] Ultimate multi-turn conversation loop (fake ASR) with automatic continuation
const { test, expect } = require('@playwright/test');

test.describe('[E2E][Ultimate][Conversation][MultiTurn][FakeASR]', () => {
  test('Multi-turn auto conversation proceeds for requested turns', async ({ page }) => {
  const turns = 3; // desired additional chained listens after first kickoff
  await page.goto(`/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&autoListen=0&listenSec=1&turns=${turns}`);
    await page.waitForFunction(() => typeof window.__ultimateDemo?.getConversationStats === 'function');
    // Kick off first listen manually
    await page.evaluate(() => window.__ultimateDemo.listenAndReply());
    await page.waitForFunction((t) => {
      const s = window.__ultimateDemo.getConversationStats();
      return s.listens >= (t+1) && s.replies >= (t+1); // initial + turns
    }, turns, { timeout: 45000 });
    const stats = await page.evaluate(() => window.__ultimateDemo.getConversationStats());
  expect(stats.listens).toBeGreaterThanOrEqual(turns+1);
  expect(stats.replies).toBeGreaterThanOrEqual(turns+1);
    const expr = await page.evaluate(() => window.__ultimateDemo.getStats().expressions);
    expect(expr).toBeGreaterThan(0);
  });
});
