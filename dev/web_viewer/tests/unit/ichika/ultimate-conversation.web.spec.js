import { test, expect } from '@playwright/test';

// Ultimate conversation test: schedule TTS, see viseme/energy metadata propagate, and binder expressions update (stub).

test.describe.configure({ mode: 'serial' });

const DEMO_URL = '/demos/ichika_voice_conversation_demo.html';

// CI-safe: relies on fake mic flags set in playwright.config when server is enabled.

test('ultimate conversation: say text produces chunks and expressions', async ({ page, baseURL }) => {
  await page.goto(baseURL + DEMO_URL);
  await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 15000 });

  // Trigger a short utterance with audio playback disabled (faster/CI-safe)
  await page.evaluate(() => window.__ultimateDemo.sayText('hello world', false));
  // Wait for speech scheduling log first
  await expect.poll(async () => await page.locator('#log').textContent() || '').toContain('speech@');
  // Then poll until expressions are observed (viseme mapping via integration)
  await expect.poll(async () => (await page.evaluate(() => window.__ultimateDemo.getStats())).expressions)
    .toBeGreaterThan(0);
});

// Optional mic path smoke (auto-faked in CI)

test('ultimate conversation: mic path starts and yields chunks', async ({ page, baseURL }) => {
  await page.goto(baseURL + DEMO_URL);
  await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 15000 });
  await page.click('#mic');
  // Give it a moment to start and pump
  await expect.poll(async () => await page.locator('#log').textContent() || '').toContain('Mic started');
  // Expect chunk events within a short time window
  await expect.poll(async () => await page.locator('#log').textContent() || '').toContain('chunk@');
});
