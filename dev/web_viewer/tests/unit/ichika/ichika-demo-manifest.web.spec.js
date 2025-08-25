import { test, expect } from '@playwright/test';

test.describe('Ichika demo: manifest load + idle start', () => {
  test.skip(!!process.env.NO_WEBSERVER, 'Requires web server');
  test.setTimeout(120_000);

  test('loads manifest and starts base idle', async ({ page }) => {
    await page.goto('/demos/ichika_classroom_demo.html');
    await page.waitForSelector('#start');
  // Wait for the demo to finish script injection and expose __ichikaDemo
  await page.waitForFunction(() => !!window.__ichikaDemo, { timeout: 10000 });

    await page.click('#start');

    // Poll the log for expected lines
    const log = page.locator('#log');
  await expect.poll(async () => (await log.textContent()) || '', { timeout: 10000 }).toMatch(/Manifest entries:|Started base clip:/);
  await expect.poll(async () => (await log.textContent()) || '', { timeout: 10000 }).toContain('Started base clip:');
  });
});
