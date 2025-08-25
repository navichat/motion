import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Conversation smoke: idle → speak → action produces updates and flags [E2E][VRM][Conversation]', async ({ page }) => {
  await page.goto('/tests/e2e/html/conversation-smoke.html');
  await page.waitForFunction(() => !!window.__convSmoke, null, { timeout: 15000 });
  const stats = await page.evaluate(() => window.__convSmoke);
  if (stats && stats.error) {
    throw new Error('conversation smoke error: ' + stats.error);
  }
  expect(Math.max(stats.boneCalls, stats.blendCalls)).toBeGreaterThan(1);
  expect(stats.sawViseme).toBeTruthy();
  expect(stats.sawEnergy).toBeTruthy();
  expect(stats.sawAction).toBeTruthy();
  expect(stats.samples).toBeGreaterThan(3);
});
