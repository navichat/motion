import { test, expect } from '@playwright/test';

// Root-level E2E: visits a minimal VRM bind smoke page and asserts updates occurred.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('VRM bind smoke page produces updates [E2E][VRM]', async ({ page }) => {
  await page.goto('/dev/web_viewer/tests/e2e/html/vrm-bind-smoke.html');
  await page.waitForFunction(() => !!window.__vrmSmoke);
  const stats = await page.evaluate(() => window.__vrmSmoke);
  expect(Math.max(stats.boneCalls, stats.blendCalls)).toBeGreaterThan(1);
});
