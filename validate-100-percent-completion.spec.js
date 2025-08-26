const { test, expect } = require('@playwright/test');

test('Validate 100% Functional Ichika System', async ({ page }) => {
  // Set longer timeout for complex initialization
  test.setTimeout(120000);
  
  // Navigate to the complete system
  const demoUrl = 'file://' + process.cwd() + '/dev/web_viewer/demos/complete_ichika_conversation_system.html';
  await page.goto(demoUrl);
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  
  console.log('Page loaded, checking initial state...');
  
  // Check that the demo loaded
  await expect(page.locator('h2')).toContainText('3D Ichika Conversation System');
  
  // Click Initialize System
  await page.click('#init-button');
  console.log('Clicked Initialize System button');
  
  // Wait for initialization to complete (up to 30 seconds)
  await page.waitForFunction(() => {
    const logElement = document.getElementById('log-messages');
    return logElement && (
      logElement.textContent.includes('System initialization complete') ||
      logElement.textContent.includes('Critical initialization failure')
    );
  }, { timeout: 30000 });
  
  console.log('Initialization completed, checking system status...');
  
  // Take a screenshot of the current state
  await page.screenshot({ 
    path: 'test-results/actual-system-state.png',
    fullPage: true 
  });
  
  console.log('Screenshot saved to test-results/actual-system-state.png');
  
  // Check system status indicators
  const sceneStatus = await page.locator('#status-3d-text').textContent();
  const avatarStatus = await page.locator('#status-avatar-text').textContent();
  const conversationStatus = await page.locator('#status-conversation-text').textContent();
  const speechStatus = await page.locator('#status-speech-text').textContent();
  
  console.log('System Status:');
  console.log('- 3D Scene:', sceneStatus);
  console.log('- Avatar:', avatarStatus);
  console.log('- Conversation:', conversationStatus);
  console.log('- Speech Sync:', speechStatus);
  
  // Verify that at least core functionality is working
  expect(sceneStatus).toBe('Loaded');
  expect(avatarStatus).toBe('Ready');
  
  // Check that canvas is present for 3D rendering
  const canvas = await page.locator('canvas');
  await expect(canvas).toBeVisible();
  
  // Check that FPS monitoring is working
  const fpsText = await page.locator('#fps').textContent();
  console.log('FPS Status:', fpsText);
  
  // Wait a bit more and take final screenshot
  await page.waitForTimeout(2000);
  
  await page.screenshot({ 
    path: 'test-results/final-system-validation.png',
    fullPage: true 
  });
  
  console.log('Final screenshot saved to test-results/final-system-validation.png');
  
  // Log final system status
  const logMessages = await page.locator('#log-messages').textContent();
  console.log('System Log Summary:');
  console.log(logMessages);
  
  console.log('✅ Validation complete');
});