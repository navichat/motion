const { chromium } = require('playwright-core');

async function validateSystem() {
  console.log('Starting Ichika system validation...');
  
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  try {
    // Navigate to the complete system
    const demoUrl = 'file://' + process.cwd() + '/dev/web_viewer/demos/complete_ichika_conversation_system.html';
    console.log('Loading demo from:', demoUrl);
    
    await page.goto(demoUrl);
    await page.waitForLoadState('networkidle');
    
    console.log('Page loaded, checking initial state...');
    
    // Check that the demo loaded
    const title = await page.locator('h2').textContent();
    console.log('Demo title:', title);
    
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
    
    // Check system status indicators
    const sceneStatus = await page.locator('#status-3d-text').textContent();
    const avatarStatus = await page.locator('#status-avatar-text').textContent();
    const conversationStatus = await page.locator('#status-conversation-text').textContent();
    const speechStatus = await page.locator('#status-speech-text').textContent();
    
    console.log('\n🎯 SYSTEM STATUS REPORT:');
    console.log('========================');
    console.log('🌐 3D Scene:', sceneStatus);
    console.log('👤 Avatar:', avatarStatus);  
    console.log('💬 Conversation:', conversationStatus);
    console.log('🎤 Speech Sync:', speechStatus);
    
    // Take a screenshot of the current state
    await page.screenshot({ 
      path: 'test-results/actual-system-state.png',
      fullPage: true 
    });
    
    console.log('\n📸 Screenshot saved to test-results/actual-system-state.png');
    
    // Check that canvas is present for 3D rendering
    const canvas = await page.locator('canvas').count();
    console.log('🖼️  Canvas elements found:', canvas);
    
    // Check FPS monitoring
    try {
      const fpsText = await page.locator('#fps').textContent();
      console.log('⚡ FPS Status:', fpsText);
    } catch (e) {
      console.log('⚡ FPS monitor not available');
    }
    
    // Get the system log
    const logMessages = await page.locator('#log-messages').textContent();
    console.log('\n📋 SYSTEM LOG SUMMARY:');
    console.log('=======================');
    console.log(logMessages);
    
    // Wait a bit more and take final screenshot
    await page.waitForTimeout(2000);
    
    await page.screenshot({ 
      path: 'test-results/final-system-validation.png',
      fullPage: true 
    });
    
    console.log('\n📸 Final screenshot saved to test-results/final-system-validation.png');
    
    // Calculate completion percentage
    const statusMap = { 'Loaded': 1, 'Ready': 1, 'Failed': 0, 'Not Loaded': 0, 'Not Ready': 0 };
    const sceneScore = statusMap[sceneStatus] || 0;
    const avatarScore = statusMap[avatarStatus] || 0; 
    const conversationScore = statusMap[conversationStatus] || 0;
    const speechScore = statusMap[speechStatus] || 0;
    
    const completionPercent = ((sceneScore + avatarScore + conversationScore + speechScore) / 4) * 100;
    
    console.log('\n🏆 COMPLETION ANALYSIS:');
    console.log('=======================');
    console.log(`📊 Overall Completion: ${completionPercent}%`);
    
    if (completionPercent === 100) {
      console.log('🎉 SUCCESS: System is 100% functional!');
    } else if (completionPercent >= 50) {
      console.log('⚠️  PARTIAL: System has basic functionality');  
    } else {
      console.log('❌ FAILED: System has critical issues');
    }
    
    console.log('\n✅ Validation complete');
    
    return {
      completionPercent,
      sceneStatus,
      avatarStatus, 
      conversationStatus,
      speechStatus,
      screenshotPath: 'test-results/final-system-validation.png'
    };
    
  } catch (error) {
    console.error('❌ Validation failed:', error);
    return null;
  } finally {
    await browser.close();
  }
}

// Run validation
validateSystem().then(result => {
  if (result) {
    console.log(`\n🎯 Final Result: ${result.completionPercent}% completion`);
    process.exit(0);
  } else {
    console.log('\n❌ Validation failed');
    process.exit(1);
  }
});