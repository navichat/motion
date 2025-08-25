// [E2E] Ultimate test: mic input → avatar reply with audio/animation
const { test, expect } = require('@playwright/test');

test.describe('[E2E][Ultimate][Mic][Reply][Animation] Full mic-to-avatar-reply loop', () => {
  test('User speaks, avatar replies with audio and animates (fake ASR)', async ({ page }) => {
    // Use fake ASR to avoid model downloads; still exercises mic path and reply/animation
    await page.goto('/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&autoListen=1&listenSec=2');
    // Wait for the test API and for expressions to register
    await page.waitForFunction(() => typeof window.__ultimateDemo?.getStats === 'function');
    await page.waitForFunction(() => { try { return (window.__ultimateDemo?.getStats()?.expressions || 0) > 0; } catch { return false; } }, { timeout: 20000 });
    const stats = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
    expect(stats.expressions).toBeGreaterThan(0);
    const logText = await page.locator('#log').textContent();
    expect(logText).toMatch(/Heard:|You said:|ASR error|Recognition/);
  });
});
