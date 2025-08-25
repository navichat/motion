import { test, expect } from '@playwright/test';

// Skip when no web server is running
test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Keep per-file timeout modest
test.setTimeout(30_000);

test('Ichika demo: wave schedules a clip and logs (unit-web)', async ({ page }) => {
  const url = '/demos/ichika_classroom_demo.html';
  await page.goto(url);

  // Wait for demo to initialize
  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));

  const log = page.locator('#log');

  await page.getByRole('button', { name: 'Start Idle' }).click();
  await expect(log).toContainText('Manifest entries:', { timeout: 5000 });

  await page.getByRole('button', { name: 'Wave' }).click();

  // It may log either Wave scheduled (BVH) or Wave (manifest) scheduled; wait briefly for log update
  await expect.poll(async () => (await log.textContent()) || '').toMatch(/Wave scheduled:|Wave \(manifest\) scheduled:/);
});
