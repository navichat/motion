// [E2E] On-device transformers.js SpeechT5 smoke
const { test, expect } = require('@playwright/test');

test.describe('[E2E][Voice][SpeechT5][OnDevice] Conversation demo on-device SpeechT5 smoke', () => {
  test('[E2E][Voice][SpeechT5][OnDevice] schedules visemes and produces expressions', async ({ page }) => {
  const url = `/demos/ichika_voice_conversation_demo.html?backend=speecht5&speecht5OnDevice=1&speecht5Spk=random`;
    await page.goto(url);
    // Ensure page API exists
    await page.waitForFunction(() => typeof window.__ultimateDemo?.sayText === 'function');
    // Trigger a reply using on-device SpeechT5 (no mic)
    await page.evaluate(async () => { await window.__ultimateDemo.sayText('hello on-device speech t five', false); });
    // Wait for expressions to be applied
    await page.waitForFunction(() => { try { return (window.__ultimateDemo?.getStats()?.expressions || 0) > 0; } catch { return false; } }, { timeout: 12000 });
    const stats = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
    expect(stats.expressions).toBeGreaterThan(0);
  });
});
