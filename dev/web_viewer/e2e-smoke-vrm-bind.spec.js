import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('VRM bind smoke page produces updates [E2E][VRM]', async ({ page }) => {
  // Server root is dev/web_viewer/
  await page.goto('/tests/e2e/html/vrm-bind-smoke.html');
  await page.waitForFunction(() => !!window.__vrmSmoke, null, { timeout: 5000 });
  const stats = await page.evaluate(() => window.__vrmSmoke);
  expect(Math.max(stats.boneCalls, stats.blendCalls)).toBeGreaterThan(1);
});
