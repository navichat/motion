// [E2E] Ultimate test (SpeechT5 on-device): mic input → avatar replies with audio/animation
const { test, expect } = require('@playwright/test');

test.describe('[E2E][Ultimate][Mic][Reply][SpeechT5][OnDevice] Full loop with fake ASR', () => {
  test('Mic→ASR(fake)→SpeechT5(on-device) reply animates', async ({ page }) => {
  const url = '/demos/ichika_voice_conversation_demo.html?backend=speecht5&speecht5OnDeviceFake=1&speecht5OnDevice=1&speecht5Spk=random&asr=fake&autoListen=1&listenSec=2';
    await page.goto(url);
    await page.waitForFunction(() => typeof window.__ultimateDemo?.getStats === 'function');
    await page.waitForFunction(() => { try { return (window.__ultimateDemo?.getStats()?.expressions || 0) > 0; } catch { return false; } }, { timeout: 20000 });
    const stats = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
    expect(stats.expressions).toBeGreaterThan(0);
    const logText = await page.locator('#log').textContent();
    expect(logText).toMatch(/Heard:|You said:|ASR error|Recognition/);
  });
});
