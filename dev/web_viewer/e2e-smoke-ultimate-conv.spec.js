import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// E2E Smoke: load the ultimate conversation demo and ensure the page boots and exposes controls.

test('Conversation smoke: ultimate demo is reachable and functional [E2E][Conversation]', async ({ page, baseURL }) => {
  await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html');
  await expect(page.locator('#mic')).toBeVisible();
  await expect(page.locator('#say')).toBeVisible();
  await page.evaluate(() => !!window.__ultimateDemo);
});
