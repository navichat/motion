import { test, expect } from '@playwright/test';

test.describe('Ichika demo: wave action', () => {
  test.skip(!!process.env.NO_WEBSERVER, 'Requires web server');
  test.setTimeout(120_000);

  test('Wave schedules via BVH or manifest', async ({ page }) => {
    await page.goto('/demos/ichika_classroom_demo.html');
    await page.waitForSelector('#start');
  await page.waitForFunction(() => !!window.__ichikaDemo, { timeout: 10000 });

    await page.click('#start');
  const log = page.locator('#log');

    await page.click('#wave');

    // Accept either direct BVH schedule or manifest fallback
  await expect.poll(async () => (await log.textContent()) || '', { timeout: 10000 }).toMatch(/Wave scheduled:|Wave \(manifest\) scheduled:/);
  });
});
