import { test, expect } from '@playwright/test';

// Skip when no web server is running
test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Keep per-file timeout modest
test.setTimeout(30_000);

test('Ichika demo: point intent schedules override clip (unit-web)', async ({ page }) => {
  const url = '/demos/ichika_classroom_demo.html';
  await page.goto(url);

  // Wait for demo to initialize
  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));
  await page.waitForFunction(() => !!document.getElementById('start') && typeof document.getElementById('start').onclick === 'function');

  const log = page.locator('#log');

  await page.getByRole('button', { name: 'Start Idle' }).click();
  await expect(log).toContainText('Manifest entries:', { timeout: 5000 });

  await page.getByRole('button', { name: 'Point at Board' }).click();
  await expect(log).toContainText('Intent scheduled:', { timeout: 5000 });
});
