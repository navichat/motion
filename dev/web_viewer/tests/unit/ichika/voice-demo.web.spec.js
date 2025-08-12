import { test, expect } from '@playwright/test';

// Requires web server; interacts with the voice_chat_demo.html

test.describe.configure({ mode: 'serial' });

test('voice demo loads and exposes controls', async ({ page, baseURL }) => {
  const url = baseURL + '/demos/voice_chat_demo.html';
  await page.goto(url);
  await expect(page.locator('#mic')).toBeVisible();
  await expect(page.locator('#say')).toBeVisible();
  await page.evaluate(() => !!window.__voiceDemo);
});

test('voice demo mock speak triggers scheduler loop', async ({ page, baseURL }) => {
  const url = baseURL + '/demos/voice_chat_demo.html';
  await page.goto(url);
  await page.click('#say');
  // Wait for at least one orchestrator chunk event to be logged
  const log = page.locator('#log');
  await expect.poll(async () => (await log.textContent()) || '').toContain('chunk@');
});
