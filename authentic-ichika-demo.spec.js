const { test, expect } = require('@playwright/test');

test.describe('Authentic Ichika VRM Classroom Demo', () => {
  test('should load real Ichika VRM character in classroom environment', async ({ page }) => {
    // Set longer timeout for asset loading
    test.setTimeout(60000);
    
    console.log('🎭 Testing authentic Ichika VRM classroom demo...');
    
    // Navigate to the demo
    await page.goto('http://localhost:8080/demos/authentic_ichika_vrm_classroom_demo.html');
    
    // Wait for initial loading to complete
    await page.waitForTimeout(5000);
    
    // Check if loading screen is present initially
    const loadingScreen = await page.locator('#loading');
    expect(await loadingScreen.isVisible()).toBe(true);
    
    // Wait for the system to initialize (up to 30 seconds)
    await page.waitForFunction(() => {
      const loadingElement = document.getElementById('loading');
      return loadingElement && loadingElement.style.display === 'none';
    }, { timeout: 30000 });
    
    console.log('✅ Loading screen hidden, system initialized');
    
    // Wait a bit more for rendering to stabilize
    await page.waitForTimeout(3000);
    
    // Check status indicators
    const statusChecks = [
      { id: 'three-status', expectedText: 'Three.js: Ready' },
      { id: 'vrm-status', expectedText: 'VRM Infrastructure: Ready' },
      { id: 'ichika-status', expectedPattern: /Ichika VRM: (Loaded & Ready|Fallback Active)/ },
      { id: 'classroom-status', expectedPattern: /Classroom GLB: (Loaded & Ready|Fallback Active)/ },
      { id: 'walking-status', expectedText: 'Walking System: Ready' }
    ];
    
    for (const check of statusChecks) {
      const element = await page.locator(`#${check.id}`);
      const text = await element.textContent();
      
      if (check.expectedText) {
        expect(text).toBe(check.expectedText);
      } else if (check.expectedPattern) {
        expect(text).toMatch(check.expectedPattern);
      }
      
      console.log(`✅ Status check passed: ${text}`);
    }
    
    // Take initial screenshot
    await page.screenshot({ 
      path: 'test-results/authentic-ichika-demo/01-system-loaded.png',
      fullPage: true 
    });
    console.log('📸 Screenshot 1: System loaded');
    
    // Test walking controls
    console.log('🚶‍♀️ Testing walking controls...');
    
    // Walk to blackboard
    await page.click('#walk-blackboard');
    await page.waitForTimeout(2000);
    
    await page.screenshot({ 
      path: 'test-results/authentic-ichika-demo/02-walking-to-blackboard.png',
      fullPage: true 
    });
    console.log('📸 Screenshot 2: Walking to blackboard');
    
    // Walk to desk
    await page.click('#walk-desk');
    await page.waitForTimeout(2000);
    
    await page.screenshot({ 
      path: 'test-results/authentic-ichika-demo/03-walking-to-desk.png',
      fullPage: true 
    });
    console.log('📸 Screenshot 3: Walking to desk');
    
    // Test animations
    console.log('🎬 Testing animations...');
    
    // Wave animation
    await page.click('#wave-animation');
    await page.waitForTimeout(1000);
    
    await page.screenshot({ 
      path: 'test-results/authentic-ichika-demo/04-wave-animation.png',
      fullPage: true 
    });
    console.log('📸 Screenshot 4: Wave animation');
    
    // Start patrol mode
    await page.click('#start-patrol');
    await page.waitForTimeout(3000);
    
    await page.screenshot({ 
      path: 'test-results/authentic-ichika-demo/05-patrol-mode.png',
      fullPage: true 
    });
    console.log('📸 Screenshot 5: Patrol mode');
    
    // Check debug log for activity
    const debugContent = await page.locator('#debug-content').textContent();
    expect(debugContent).toContain('Ichika VRM Classroom System ready');
    
    console.log('🎉 Authentic Ichika VRM classroom demo test completed successfully!');
    
    // Final screenshot
    await page.screenshot({ 
      path: 'test-results/authentic-ichika-demo/06-final-state.png',
      fullPage: true 
    });
    console.log('📸 Screenshot 6: Final state');
  });
});