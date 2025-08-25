import { test, expect } from '@playwright/test';

test('debug workload generation', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => console.log(`[BROWSER] ${msg.text()}`));
  
  await page.goto('http://localhost:8080/dev/web_viewer/debug-workload.html');
  await page.click('#testButton');
  await page.waitForTimeout(10000);
  
  const output = await page.textContent('#output');
  console.log('OUTPUT:', output);
});
