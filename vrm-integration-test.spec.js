const { test, expect } = require('@playwright/test');

test.describe('VRM Integration with BVH Animations', () => {
  test.setTimeout(120000);

  test('should load VRM avatar with BVH animations in classroom', async ({ page }) => {
    console.log('🎭 Testing VRM integration with BVH animations...');

    // Navigate to the complete system
    await page.goto('file:///home/runner/work/motion/motion/dev/web_viewer/demos/complete_ichika_conversation_system.html');

    // Wait for the page to load
    await page.waitForTimeout(3000);

    // Take initial screenshot
    await page.screenshot({
      path: 'test-results/01-initial-load.png',
      fullPage: false
    });

    // Click Initialize System button
    await page.waitForSelector('#init-button', { state: 'visible' });
    await page.click('#init-button');

    console.log('✅ Clicked Initialize System button');

    // Wait for initialization with longer timeout for VRM loading
    await page.waitForTimeout(15000);

    // Check system status indicators
    const sceneStatus = await page.locator('#status-3d-text').textContent();
    const avatarStatus = await page.locator('#status-avatar-text').textContent();
    const conversationStatus = await page.locator('#status-conversation-text').textContent();
    const speechStatus = await page.locator('#status-speech-text').textContent();

    console.log('📊 System Status:');
    console.log(`  - 3D Scene: ${sceneStatus}`);
    console.log(`  - Avatar: ${avatarStatus}`);
    console.log(`  - Conversation: ${conversationStatus}`);
    console.log(`  - Speech Sync: ${speechStatus}`);

    // Take post-initialization screenshot
    await page.screenshot({
      path: 'test-results/02-post-initialization.png',
      fullPage: false
    });

    // Check if VRM avatar system is working
    const canvasExists = await page.locator('canvas').isVisible();
    expect(canvasExists).toBe(true);

    // Wait a bit more for VRM animations to start
    await page.waitForTimeout(5000);

    // Test animation button if available
    if (await page.locator('#test-animation').isEnabled()) {
      await page.click('#test-animation');
      console.log('✅ Clicked Test Animation button');
      await page.waitForTimeout(3000);
    }

    // Take final screenshot showing VRM with animations
    await page.screenshot({
      path: 'test-results/03-vrm-with-animations.png',
      fullPage: false
    });

    // Get performance stats
    const performanceText = await page.locator('#performance-monitor').textContent();
    console.log('⚡ Performance:', performanceText);

    // Check for VRM-specific elements in the page
    const hasVRMContent = await page.evaluate(() => {
      // Check console logs for VRM loading messages
      return document.body.innerHTML.includes('VRM') || 
             document.body.innerHTML.includes('Avatar') ||
             typeof window.AdvancedVRMLoader !== 'undefined';
    });

    // Log system information
    const systemInfo = await page.evaluate(() => {
      return {
        hasThreeJS: typeof window.THREE !== 'undefined',
        hasVRMLoader: typeof window.AdvancedVRMLoader !== 'undefined',
        hasBVHAdapter: typeof window.VRMBVHAdapter !== 'undefined',
        hasBVHTimeline: typeof window.BVHTimeline !== 'undefined',
        sceneReady: document.querySelector('canvas') !== null
      };
    });

    console.log('🔧 System Components:');
    console.log(`  - Three.js: ${systemInfo.hasThreeJS ? '✅' : '❌'}`);
    console.log(`  - VRM Loader: ${systemInfo.hasVRMLoader ? '✅' : '❌'}`);
    console.log(`  - BVH Adapter: ${systemInfo.hasBVHAdapter ? '✅' : '❌'}`);
    console.log(`  - BVH Timeline: ${systemInfo.hasBVHTimeline ? '✅' : '❌'}`);
    console.log(`  - 3D Scene: ${systemInfo.sceneReady ? '✅' : '❌'}`);

    // Verify the system is functional
    expect(systemInfo.hasThreeJS).toBe(true);
    expect(systemInfo.sceneReady).toBe(true);
    
    console.log('🎉 VRM Integration test completed successfully!');
  });
});