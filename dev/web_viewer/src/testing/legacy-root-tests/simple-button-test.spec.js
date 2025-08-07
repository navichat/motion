import { test, expect } from '@playwright/test';

test.describe('Simple Button Test', () => {
  test('should find and click the workload button', async ({ page }) => {
    // Set a reasonable timeout
    test.setTimeout(30000); // 30 seconds
    
    console.log('🤖 Starting Simple Button Test...');
    console.log('🌐 Navigating to task-manager-demo.html...');
    
    // Navigate to the demo page
    await page.goto('/dev/web_viewer/task-manager-demo.html');
    
    // Wait for the page to load
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait 2 seconds for any dynamic content
    
    // Check if the button exists
    const button = page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' });
    
    // Wait for the button to be visible
    await expect(button).toBeVisible({ timeout: 10000 });
    
    console.log('✅ Button found and is visible!');
    
    // Click the button
    await button.click();
    
    console.log('✅ Button clicked successfully!');
    
    // Wait a few seconds to see if any console messages appear
    await page.waitForTimeout(5000);
    
    console.log('✅ Test completed successfully!');
  });
});
