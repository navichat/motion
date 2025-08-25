import { test, expect } from '@playwright/test';

test.describe('Ichika demo: speech scheduling', () => {
  test.skip(!!process.env.NO_WEBSERVER, 'Requires web server');
  test.setTimeout(120_000);

  test('Speak schedules face/audio tracks and logs', async ({ page }) => {
    await page.goto('/demos/ichika_classroom_demo.html');
    await page.waitForSelector('#start');
  await page.waitForFunction(() => !!window.__ichikaDemo, { timeout: 10000 });
    await page.click('#start');
    const log = page.locator('#log');
    await page.click('#speech');
    // Verify speech scheduled log
  await expect.poll(async () => (await log.textContent()) || '', { timeout: 10000 }).toContain('Speech scheduled:');
  });
});
