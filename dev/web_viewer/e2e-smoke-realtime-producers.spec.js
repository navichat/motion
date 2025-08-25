import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// E2E smoke: real-time viseme + gesture producers stream frames; VRM adapter applies updates
// This mirrors the conversation/classroom smokes but focuses on streaming cadence.

test('Realtime producers smoke: streaming visemes + gestures produce VRM updates [E2E][VRM][Realtime]', async ({ page }) => {
  await page.goto('/tests/e2e/html/realtime-producers-smoke.html');
  await page.waitForFunction(() => !!window.__rtSmoke, null, { timeout: 20000 });
  const stats = await page.evaluate(() => window.__rtSmoke);
  if (stats && stats.error) {
    throw new Error('realtime-producers smoke error: ' + stats.error);
  }
  expect(stats.samples).toBeGreaterThanOrEqual(5);
  expect(Math.max(stats.boneCalls, stats.blendCalls)).toBeGreaterThan(1);
  expect(stats.sawViseme).toBeTruthy();
  expect(stats.sawEnergy).toBeTruthy();
});
