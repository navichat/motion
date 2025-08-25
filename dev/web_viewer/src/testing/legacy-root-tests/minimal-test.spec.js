import { test, expect } from '@playwright/test';

test.describe('Minimal Page Load Test', () => {
  test('should load the page and wait for scripts', async ({ page }) => {
    test.setTimeout(30000);
    
    console.log('🔍 Testing minimal page load...');
    
    // Listen for console messages to see script loading
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error(`BROWSER ERROR: ${msg.text()}`);
      } else {
        console.log(`BROWSER: ${msg.text()}`);
      }
    });
    
    // Listen for page errors
    page.on('pageerror', error => {
      console.error(`PAGE ERROR: ${error.message}`);
    });
    
    // Listen for response events to check for 404s
    page.on('response', response => {
      if (response.status() >= 400) {
        console.error(`HTTP ERROR: ${response.status()} for ${response.url()}`);
      }
    });
    
    // Navigate to the page
    await page.goto('http://localhost:8082/dev/web_viewer/task-manager-demo.html');
    
    // Wait for the main.js script to load by checking if our function exists
    await page.waitForFunction(() => {
      return typeof window.runRealWorkloadTest === 'function';
    }, { timeout: 15000 });
    
    console.log('✅ runRealWorkloadTest function is available!');
    
    // Now check for the button
    const button = page.locator('button[onclick="runRealWorkloadTest()"]');
    await expect(button).toBeVisible({ timeout: 5000 });
    
    console.log('✅ Button is visible!');
    
    // Click the button
    await button.click();
    console.log('✅ Button clicked successfully!');
  });
});
