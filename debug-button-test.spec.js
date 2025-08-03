import { test, expect } from '@playwright/test';

test.describe('Debug Button Test', () => {
  test('should debug the page loading and button visibility', async ({ page }) => {
    test.setTimeout(60000); // 1 minute
    
    console.log('🔍 Starting Debug Test...');
    
    // Listen for all console messages
    page.on('console', msg => {
      console.log(`BROWSER: ${msg.text()}`);
    });
    
    // Listen for errors
    page.on('pageerror', error => {
      console.error(`PAGE ERROR: ${error.message}`);
    });
    
    // Navigate to the demo page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('/dev/web_viewer/task-manager-demo.html');
    
    // Wait for all network requests to complete
    await page.waitForLoadState('networkidle');
    console.log('✅ Network idle reached');
    
    // Wait a bit more for any dynamic content
    await page.waitForTimeout(5000);
    console.log('✅ Waited 5 seconds for dynamic content');
    
    // Take a screenshot
    await page.screenshot({ path: 'debug-page.png', fullPage: true });
    console.log('✅ Screenshot saved');
    
    // Try to find buttons by different methods
    console.log('🔍 Looking for buttons...');
    
    // Method 1: By text content
    const buttonByText = page.locator('text=🚀 Real WASM/GPU/WebNN Workload');
    const isVisibleByText = await buttonByText.isVisible();
    console.log(`Button by text visible: ${isVisibleByText}`);
    
    // Method 2: By onclick attribute
    const buttonByOnClick = page.locator('button[onclick="runRealWorkloadTest()"]');
    const isVisibleByOnClick = await buttonByOnClick.isVisible();
    console.log(`Button by onclick visible: ${isVisibleByOnClick}`);
    
    // Method 3: Count all buttons
    const allButtons = page.locator('button');
    const buttonCount = await allButtons.count();
    console.log(`Total buttons found: ${buttonCount}`);
    
    // Method 4: List all button texts
    for (let i = 0; i < buttonCount; i++) {
      const buttonText = await allButtons.nth(i).textContent();
      console.log(`Button ${i}: "${buttonText}"`);
    }
    
    // Method 5: Check page title
    const title = await page.title();
    console.log(`Page title: ${title}`);
    
    // Method 6: Check if specific element exists
    const workloadButton = page.locator('button:has-text("🚀 Real WASM/GPU/WebNN Workload")');
    const workloadButtonExists = await workloadButton.count();
    console.log(`Workload button count: ${workloadButtonExists}`);
    
    if (workloadButtonExists > 0) {
      console.log('✅ Found workload button, trying to click...');
      await workloadButton.click();
      console.log('✅ Button clicked!');
    } else {
      console.log('❌ Workload button not found');
    }
  });
});
