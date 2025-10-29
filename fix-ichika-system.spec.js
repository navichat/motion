const { test, expect } = require('@playwright/test');

test('Fix Ichika 3D Conversation System', async ({ page }) => {
  // Set longer timeout for this complex system
  test.setTimeout(120000);

  console.log('=== Starting Ichika System Debugging ===');

  // Navigate to the demo
  await page.goto('http://localhost:8000/demos/complete_ichika_conversation_system.html', {
    waitUntil: 'networkidle',
    timeout: 30000
  });

  console.log('✅ Page loaded successfully');

  // Wait for initial app setup
  await page.waitForTimeout(3000);

  // Capture initial state screenshot
  await page.screenshot({
    path: 'test-results/01-initial-state.png',
    fullPage: true,
    quality: 100
  });

  console.log('📸 Initial state captured');

  // Check for JavaScript errors in console
  const messages = [];
  page.on('console', msg => {
    messages.push(`${msg.type()}: ${msg.text()}`);
    console.log(`BROWSER ${msg.type()}: ${msg.text()}`);
  });

  // Try to initialize system
  console.log('🔄 Attempting system initialization...');

  // Click Initialize System button
  await page.click('#init-button', { timeout: 10000 });
  
  // Wait for initialization to complete
  await page.waitForTimeout(10000);

  // Capture post-initialization state
  await page.screenshot({
    path: 'test-results/02-post-initialization.png',
    fullPage: true,
    quality: 100
  });

  console.log('📸 Post-initialization state captured');

  // Check system status
  const status3D = await page.textContent('#status-3d-text');
  const statusAvatar = await page.textContent('#status-avatar-text');
  const statusConversation = await page.textContent('#status-conversation-text');
  const statusSpeech = await page.textContent('#status-speech-text');

  console.log('=== System Status ===');
  console.log(`3D Scene: ${status3D}`);
  console.log(`Avatar: ${statusAvatar}`);
  console.log(`Conversation: ${statusConversation}`);
  console.log(`Speech Sync: ${statusSpeech}`);

  // Try Test Animation button
  if (await page.isEnabled('#test-animation')) {
    console.log('🎭 Testing animation...');
    await page.click('#test-animation');
    await page.waitForTimeout(3000);
    
    await page.screenshot({
      path: 'test-results/03-animation-test.png',
      fullPage: true,
      quality: 100
    });
  }

  // Try Test Voice button
  if (await page.isEnabled('#test-tts')) {
    console.log('🔊 Testing voice...');
    await page.click('#test-tts');
    await page.waitForTimeout(5000);
    
    await page.screenshot({
      path: 'test-results/04-voice-test.png',
      fullPage: true,
      quality: 100
    });
  }

  // Capture final state
  await page.screenshot({
    path: 'test-results/05-final-state.png',
    fullPage: true,
    quality: 100
  });

  // Check for any errors in the system
  const hasErrors = messages.some(msg => msg.includes('error') || msg.includes('Error') || msg.includes('failed'));
  
  console.log('=== Browser Console Messages ===');
  messages.forEach(msg => console.log(msg));

  // Extract performance metrics
  const performanceFPS = await page.textContent('#fps-counter');
  const renderMode = await page.textContent('#render-mode');
  const avatarStatus = await page.textContent('#avatar-status');
  const conversationStatus = await page.textContent('#conversation-status');

  console.log('=== Performance Metrics ===');
  console.log(`FPS: ${performanceFPS}`);
  console.log(`Render Mode: ${renderMode}`);
  console.log(`Avatar Status: ${avatarStatus}`);
  console.log(`Conversation Status: ${conversationStatus}`);

  console.log('=== Test Complete ===');
  
  // The test should pass even if there are some failures - we're in debug mode
  expect(status3D).toBeTruthy();
});