import { test, expect } from '@playwright/test';

const DEMO_URL = '/demos/ichika_voice_conversation_demo.html';

test.describe('[E2E][Ultimate][Markers][ASR]', () => {
  test('Scheduling marker appears when text is scheduled directly', async ({ page }) => {
    const url = `${DEMO_URL}?backend=beeps&playAudio=0&autoListen=0`;
    await page.goto(url);

    // Wait for demo surface
    await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });

    // Directly schedule text to avoid ASR dependency
    await page.evaluate(() => window.__ultimateDemo?.sayText?.('hello world', false));

    await page.waitForFunction(() => /🗣️ Scheduled TTS for text:/.test(document.getElementById('log')?.textContent||''), null, { timeout: 15000 });
  });

  test('Fake ASR via Listen & Reply emits heard text and schedules TTS', async ({ page }) => {
    const url = `${DEMO_URL}?backend=beeps&playAudio=0&autoListen=0&asr=fake&listenSec=1`;
    await page.goto(url);
  await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 10000 });

    // Trigger listen+reply
    await page.getByRole('button', { name: /Listen & Reply/i }).click();
    
    // Assert generic, stable markers for fake ASR path
    await page.waitForFunction(() => /👂 Listening \(fake ASR\)…/.test(document.getElementById('log')?.textContent||''), null, { timeout: 15000 });
    await page.waitForFunction(() => /📝 Heard: /.test(document.getElementById('log')?.textContent||''), null, { timeout: 20000 });
    await page.waitForFunction(() => /🗣️ Scheduled TTS for text:/.test(document.getElementById('log')?.textContent||''), null, { timeout: 20000 });
  });
});
