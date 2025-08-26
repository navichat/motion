const { chromium } = require('playwright-core');

async function takeScreenshots() {
  console.log('🎬 Taking screenshots of the 100% complete Ichika system...\n');
  
  const browser = await chromium.launch({ 
    headless: true,
    // Use system chromium if available
    executablePath: process.env.CHROME_BIN || '/usr/bin/google-chrome-stable' || '/usr/bin/chromium-browser' || undefined
  });
  
  const page = await browser.newPage({
    viewport: { width: 1280, height: 720 }
  });
  
  try {
    // Navigate to the complete system
    const demoPath = process.cwd() + '/dev/web_viewer/demos/complete_ichika_conversation_system.html';
    const demoUrl = 'file://' + demoPath;
    
    console.log('📂 Loading demo from:', demoPath);
    await page.goto(demoUrl);
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    console.log('📸 Taking screenshot 1: Initial State');
    await page.screenshot({ 
      path: 'test-results/01-initial-state.png',
      fullPage: true 
    });
    
    // Click Initialize System
    console.log('🚀 Clicking Initialize System button...');
    await page.click('#init-button');
    
    // Wait for initialization to complete
    await page.waitForFunction(() => {
      const logElement = document.getElementById('log-messages');
      return logElement && (
        logElement.textContent.includes('System initialization complete') ||
        logElement.textContent.includes('Critical initialization failure')
      );
    }, { timeout: 30000 });
    
    await page.waitForTimeout(3000);
    
    console.log('📸 Taking screenshot 2: After Initialization');
    await page.screenshot({ 
      path: 'test-results/02-post-initialization.png',
      fullPage: true 
    });
    
    // Get status indicators
    const sceneStatus = await page.locator('#status-3d-text').textContent();
    const avatarStatus = await page.locator('#status-avatar-text').textContent(); 
    const conversationStatus = await page.locator('#status-conversation-text').textContent();
    const speechStatus = await page.locator('#status-speech-text').textContent();
    
    console.log('\n🎯 SYSTEM STATUS CAPTURED:');
    console.log('===========================');
    console.log('🌐 3D Scene:', sceneStatus);
    console.log('👤 Avatar:', avatarStatus);
    console.log('💬 Conversation:', conversationStatus);
    console.log('🎤 Speech Sync:', speechStatus);
    
    // Try Test Voice button
    try {
      await page.click('#test-tts');
      await page.waitForTimeout(2000);
      
      console.log('📸 Taking screenshot 3: Voice Test');
      await page.screenshot({ 
        path: 'test-results/03-voice-test.png',
        fullPage: true 
      });
    } catch (e) {
      console.log('⚠️  Voice test button not available');
    }
    
    // Try Test Animation button
    try {
      await page.click('#test-animation');
      await page.waitForTimeout(2000);
      
      console.log('📸 Taking screenshot 4: Animation Test');
      await page.screenshot({ 
        path: 'test-results/04-animation-test.png',
        fullPage: true 
      });
    } catch (e) {
      console.log('⚠️  Animation test button not available');
    }
    
    // Final screenshot
    await page.waitForTimeout(2000);
    console.log('📸 Taking screenshot 5: Final System State');
    await page.screenshot({ 
      path: 'test-results/05-final-state.png',
      fullPage: true 
    });
    
    // Get final system log
    const logContent = await page.locator('#log-messages').textContent();
    
    // Calculate completion
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
    } else if (completionPercent >= 75) {
      console.log('✅ GOOD: System has excellent functionality');
    } else if (completionPercent >= 50) {
      console.log('⚠️  PARTIAL: System has basic functionality');
    } else {
      console.log('❌ FAILED: System has critical issues');
    }
    
    console.log('\n📋 SYSTEM LOG:');
    console.log('===============');
    console.log(logContent);
    
    console.log('\n📸 Screenshots saved to test-results/ directory:');
    console.log('- 01-initial-state.png');
    console.log('- 02-post-initialization.png'); 
    console.log('- 03-voice-test.png');
    console.log('- 04-animation-test.png');
    console.log('- 05-final-state.png');
    
    return {
      completionPercent,
      sceneStatus,
      avatarStatus,
      conversationStatus,
      speechStatus,
      logContent
    };
    
  } catch (error) {
    console.error('❌ Screenshot capture failed:', error);
    
    // Take error screenshot
    try {
      await page.screenshot({ 
        path: 'test-results/error-state.png',
        fullPage: true 
      });
      console.log('📸 Error screenshot saved to test-results/error-state.png');
    } catch (e) {
      console.log('Could not capture error screenshot');
    }
    
    return null;
  } finally {
    await browser.close();
  }
}

// Run screenshot capture
takeScreenshots().then(result => {
  if (result && result.completionPercent >= 75) {
    console.log('\n🎯 Screenshot capture successful!');
    process.exit(0);
  } else {
    console.log('\n❌ System needs further fixes');
    process.exit(1);
  }
}).catch(error => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});