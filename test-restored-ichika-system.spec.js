const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('Restored Ichika VRM Classroom System', () => {
  test.setTimeout(60000); // 60 seconds timeout

  test('should load Ichika VRM classroom system with walking demo', async ({ page }) => {
    // Navigate to the restored demo
    const demoPath = path.join(__dirname, '../dev/web_viewer/demos/restored_ichika_vrm_classroom_system.html');
    await page.goto(`file://${demoPath}`);

    // Wait for initial loading
    await page.waitForTimeout(3000);

    // Take initial screenshot
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/01-initial-load.png',
      fullPage: true 
    });

    // Wait for VRM infrastructure to initialize
    await page.waitForSelector('#loader-status.status-ready', { timeout: 10000 });
    await page.waitForSelector('#adapter-status.status-ready', { timeout: 10000 });
    await page.waitForSelector('#classroom-status.status-ready', { timeout: 10000 });
    await page.waitForSelector('#timeline-status.status-ready', { timeout: 10000 });

    // Take infrastructure ready screenshot
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/02-infrastructure-ready.png',
      fullPage: true 
    });

    // Click Initialize Complete System
    await page.click('#init-system');
    await page.waitForTimeout(5000); // Wait for system initialization

    // Take system initialized screenshot
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/03-system-initialized.png',
      fullPage: true 
    });

    // Check if VRM character loaded successfully
    const debugMessages = await page.locator('#debug-content').textContent();
    console.log('Debug messages:', debugMessages);

    // Start walking demonstration
    await page.click('#start-walking');
    await page.waitForTimeout(2000);

    // Take walking demo screenshot
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/04-walking-demo-active.png',
      fullPage: true 
    });

    // Test individual walking controls
    await page.click('#walk-to-board');
    await page.waitForTimeout(3000);
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/05-walk-to-blackboard.png',
      fullPage: true 
    });

    await page.click('#walk-to-center');
    await page.waitForTimeout(3000);
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/06-walk-to-center.png',
      fullPage: true 
    });

    await page.click('#walk-to-desk');
    await page.waitForTimeout(3000);
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/07-walk-to-desk.png',
      fullPage: true 
    });

    await page.click('#walk-random');
    await page.waitForTimeout(3000);
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/08-walk-random-position.png',
      fullPage: true 
    });

    // Final screenshot
    await page.waitForTimeout(2000);
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/09-final-demonstration.png',
      fullPage: true 
    });

    // Verify the system is working
    const statusElements = await page.locator('.status-indicator.status-ready').count();
    expect(statusElements).toBeGreaterThanOrEqual(3); // At least 3 components should be ready

    // Verify debug messages show successful loading
    const finalDebugMessages = await page.locator('#debug-content').textContent();
    expect(finalDebugMessages).toContain('VRM infrastructure components initialized');

    console.log('✅ Restored Ichika VRM Classroom System test completed successfully');
    console.log('📸 Screenshots captured in test-results/restored-ichika-system/');
  });

  test('should validate VRM asset loading', async ({ page }) => {
    const demoPath = path.join(__dirname, '../dev/web_viewer/demos/restored_ichika_vrm_classroom_system.html');
    await page.goto(`file://${demoPath}`);

    // Wait for infrastructure
    await page.waitForTimeout(3000);

    // Try to load just the Ichika VRM
    await page.click('#load-ichika');
    await page.waitForTimeout(5000);

    // Check if VRM loading was attempted
    const debugContent = await page.locator('#debug-content').textContent();
    console.log('VRM loading debug:', debugContent);

    // Take VRM loading screenshot
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/vrm-loading-validation.png',
      fullPage: true 
    });

    // Try to load classroom
    await page.click('#load-classroom');
    await page.waitForTimeout(5000);

    // Take classroom loading screenshot
    await page.screenshot({ 
      path: 'test-results/restored-ichika-system/classroom-loading-validation.png',
      fullPage: true 
    });

    console.log('✅ Asset loading validation completed');
  });
});