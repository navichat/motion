const { test, expect } = require('@playwright/test');

test.describe('Ichika VRM Classroom Walking Demo', () => {
  test.setTimeout(120000); // 2 minutes timeout for shell compliance

  test('Capture working Ichika VRM walking in classroom', async ({ page }) => {
    // Navigate to demo
    await page.goto('http://localhost:8080/demos/restored_ichika_classroom_walking_demo_v2.html');
    
    // Wait for page to load
    await page.waitForTimeout(3000);
    
    // Take initial screenshot
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/01-initial-page.png',
      fullPage: true
    });
    
    // Initialize system
    await page.click('button:has-text("Initialize System")');
    await page.waitForTimeout(2000);
    
    // Take screenshot after initialization  
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/02-system-initialized.png',
      fullPage: true
    });
    
    // Load assets
    await page.click('button:has-text("Load Classroom & Avatar")');
    await page.waitForTimeout(5000); // Give time for assets to load
    
    // Take screenshot after asset loading
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/03-assets-loaded.png', 
      fullPage: true
    });
    
    // Start walking demo
    await page.click('button:has-text("Start Walking Demo")');
    await page.waitForTimeout(2000);
    
    // Take screenshot of walking demo
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/04-walking-demo-started.png',
      fullPage: true
    });
    
    // Test individual walking controls
    await page.click('button:has-text("Walk to Blackboard")');
    await page.waitForTimeout(3000);
    
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/05-walk-to-blackboard.png',
      fullPage: true
    });
    
    // Test animation controls
    await page.click('button:has-text("Wave Hello")');
    await page.waitForTimeout(2000);
    
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/06-wave-animation.png',
      fullPage: true
    });
    
    // Test camera controls
    await page.click('button:has-text("Follow Avatar")');
    await page.waitForTimeout(1000);
    
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/07-follow-camera.png',
      fullPage: true
    });
    
    // Final screenshot showing the working system
    await page.screenshot({
      path: '/home/runner/work/motion/motion/test-results/restored-ichika-demo/08-final-working-demo.png',
      fullPage: true
    });
    
    // Verify system status indicators
    const sceneStatus = await page.locator('#scene-status').getAttribute('class');
    const classroomStatus = await page.locator('#classroom-status').getAttribute('class'); 
    const avatarStatus = await page.locator('#avatar-status').getAttribute('class');
    const walkingStatus = await page.locator('#walking-status').getAttribute('class');
    
    console.log('System Status:', {
      scene: sceneStatus,
      classroom: classroomStatus, 
      avatar: avatarStatus,
      walking: walkingStatus
    });
    
    // Check that systems are ready (should have 'ready' class)
    expect(sceneStatus).toContain('ready');
    
    console.log('✅ Ichika VRM Classroom Walking Demo screenshots captured successfully');
  });
});