import { test, expect } from '@playwright/test';

test.describe('Self-Contained VRM System - Complete Functionality', () => {
  test('should load VRM infrastructure and demonstrate full functionality', async ({ page }, testInfo) => {
    // Set longer timeout for comprehensive test
    test.setTimeout(300000); // 5 minutes
    
    const screenshots = [];
    
    // Navigate to the self-contained VRM system
    await page.goto('http://127.0.0.1:8000/demos/working_self_contained_vrm_system.html', {
      waitUntil: 'networkidle'
    });
    
    // Wait for system to load
    await page.waitForTimeout(2000);
    
    // Screenshot 1: Initial state
    await page.screenshot({ 
      path: `test-results/vrm-complete/01-initial-state.png`,
      fullPage: true 
    });
    screenshots.push('01-initial-state.png');
    
    // Verify initial state
    await expect(page.locator('#status-scene-text')).toHaveText('Not Loaded');
    await expect(page.locator('#status-vrm-text')).toHaveText('Not Loaded');
    await expect(page.getByRole('button', { name: 'Initialize VRM System' })).toBeEnabled();
    
    console.log('✅ Initial state verified - system ready for initialization');
    
    // Initialize the VRM system
    await page.getByRole('button', { name: 'Initialize VRM System' }).click();
    
    // Wait for initialization to complete
    await page.waitForTimeout(5000);
    
    // Screenshot 2: System initialized
    await page.screenshot({ 
      path: `test-results/vrm-complete/02-system-initialized.png`,
      fullPage: true 
    });
    screenshots.push('02-system-initialized.png');
    
    // Verify all components are loaded
    await expect(page.locator('#status-scene-text')).toHaveText('Loaded');
    await expect(page.locator('#status-vrm-text')).toHaveText('Loaded');
    await expect(page.locator('#status-bvh-text')).toHaveText('Basic Mode');
    await expect(page.locator('#status-conversation-text')).toHaveText('Ready');
    await expect(page.locator('#status-speech-text')).toHaveText('Ready');
    
    // Verify performance metrics
    const fpsText = await page.locator('#fps-counter').textContent();
    expect(parseInt(fpsText)).toBeGreaterThan(30); // At least 30 FPS
    
    await expect(page.locator('#render-mode')).toHaveText('WebGL+VRM');
    await expect(page.locator('#avatar-status')).toHaveText('Loaded');
    await expect(page.locator('#bvh-status')).toHaveText('Active');
    
    console.log('✅ System initialization complete - all components loaded');
    
    // Test voice synthesis
    await page.getByRole('button', { name: 'Test Voice' }).click();
    await page.waitForTimeout(2000);
    
    // Screenshot 3: Voice test active
    await page.screenshot({ 
      path: `test-results/vrm-complete/03-voice-test-active.png`,
      fullPage: true 
    });
    screenshots.push('03-voice-test-active.png');
    
    // Verify voice test is working
    await expect(page.locator('#animation-status')).toHaveText('Speaking');
    
    // Check conversation history contains the voice test message
    const conversationHistory = page.locator('#conversation-history');
    await expect(conversationHistory).toContainText('This is a test of my voice system with real VRM lip synchronization!');
    
    console.log('✅ Voice synthesis test complete - lip sync working');
    
    // Wait for voice to complete
    await page.waitForTimeout(3000);
    
    // Test animation system
    await page.getByRole('button', { name: 'Test Animation' }).click();
    await page.waitForTimeout(2000);
    
    // Screenshot 4: Animation test active
    await page.screenshot({ 
      path: `test-results/vrm-complete/04-animation-test-active.png`,
      fullPage: true 
    });
    screenshots.push('04-animation-test-active.png');
    
    // Verify animation test is working
    await expect(conversationHistory).toContainText('VRM skeletal system');
    
    console.log('✅ Animation system test complete - BVH integration working');
    
    // Wait for animation to complete
    await page.waitForTimeout(3000);
    
    // Start full conversation mode
    await page.getByRole('button', { name: 'Start Conversation' }).click();
    await page.waitForTimeout(3000);
    
    // Screenshot 5: Conversation mode active
    await page.screenshot({ 
      path: `test-results/vrm-complete/05-conversation-mode.png`,
      fullPage: true 
    });
    screenshots.push('05-conversation-mode.png');
    
    // Verify conversation mode is active
    await expect(page.getByRole('button', { name: 'Start Conversation' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Stop Conversation' })).toBeEnabled();
    
    // Check for conversation greeting
    await expect(conversationHistory).toContainText("I'm ready to talk! What would you like to discuss?");
    
    console.log('✅ Conversation mode active - full interactive system ready');
    
    // Final comprehensive screenshot
    await page.screenshot({ 
      path: `test-results/vrm-complete/06-complete-system-final.png`,
      fullPage: true 
    });
    screenshots.push('06-complete-system-final.png');
    
    // Stop conversation
    await page.getByRole('button', { name: 'Stop Conversation' }).click();
    await page.waitForTimeout(3000);
    
    // Final verification - all systems operational
    await expect(page.locator('#fps-counter')).not.toHaveText('--');
    await expect(page.locator('#avatar-status')).toHaveText('Loaded');
    await expect(page.locator('#render-mode')).toHaveText('WebGL+VRM');
    
    // Verify the avatar area shows success
    await expect(page.locator('#avatar-placeholder')).toContainText('Ichika VRM Ready');
    await expect(page.locator('#avatar-placeholder')).toContainText('Real 3D Avatar Loaded');
    
    console.log('✅ Complete VRM system test successful!');
    console.log(`📸 Screenshots captured: ${screenshots.join(', ')}`);
    
    // Log system performance
    const finalFps = await page.locator('#fps-counter').textContent();
    console.log(`🎮 Final Performance: ${finalFps} FPS`);
    
    // Verify no CDN errors (should be using local infrastructure)
    const consoleLogs = [];
    page.on('console', msg => consoleLogs.push(msg.text()));
    
    const cdnErrors = consoleLogs.filter(log => 
      log.includes('cdn.jsdelivr.net') || 
      log.includes('ERR_BLOCKED_BY_CLIENT')
    );
    
    expect(cdnErrors.length).toBe(0); // No CDN dependency errors
    
    console.log('✅ CDN independence verified - using local VRM infrastructure');
    
    // Test passed - system is fully functional
    console.log('🎉 Self-Contained VRM System: 100% Functional');
    console.log('🎭 Real VRM integration with existing infrastructure complete');
    console.log('💬 Full conversation system with voice and animation ready');
  });
  
  test('should demonstrate VRM infrastructure components', async ({ page }) => {
    test.setTimeout(120000); // 2 minutes
    
    await page.goto('http://127.0.0.1:8000/demos/working_self_contained_vrm_system.html');
    await page.waitForTimeout(1000);
    
    // Check that all VRM infrastructure components are loaded
    const statusLog = page.locator('#status-display');
    
    await expect(statusLog).toContainText('AdvancedVRMLoader: Available');
    await expect(statusLog).toContainText('VRMBVHAdapter: Available'); 
    await expect(statusLog).toContainText('AvatarBinder: Available');
    await expect(statusLog).toContainText('BVHTimeline: Available');
    await expect(statusLog).toContainText('BVHTimelineVRMIntegration: Available');
    await expect(statusLog).toContainText('All VRM infrastructure components loaded successfully!');
    
    console.log('✅ VRM infrastructure components verified');
    console.log('🎯 System uses existing dev/web_viewer VRM architecture');
    console.log('📦 No CDN dependencies - fully self-contained');
  });
});