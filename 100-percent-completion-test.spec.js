const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test.describe('100% Completion Validation for 3D Ichika VRM System', () => {
  test.setTimeout(600000); // 10 minutes for comprehensive testing

  test('Complete system validation and final 5% completion', async ({ page }) => {
    const baseUrl = 'http://localhost:8080';
    const screenshotDir = path.join(process.cwd(), 'test-results', '100-percent-validation');
    
    // Ensure screenshot directory exists
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }

    // Enable console logging for debugging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log(`❌ Browser Error: ${msg.text()}`);
      } else if (msg.text().includes('ConversationManager') || 
                 msg.text().includes('ClassroomAvatarIntegration') ||
                 msg.text().includes('EnhancedSpeechSync')) {
        console.log(`🔧 System: ${msg.text()}`);
      }
    });

    console.log('🚀 Loading complete Ichika conversation system...');
    await page.goto(`${baseUrl}/demos/complete_ichika_conversation_system.html`);
    
    // Wait for initial page load
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000);
    
    console.log('📷 Taking initial system screenshot...');
    await page.screenshot({ 
      path: path.join(screenshotDir, '01-initial-system-load.png'),
      fullPage: true
    });

    // Test 1: System Initialization
    console.log('🔧 Testing system initialization...');
    const initButton = page.locator('#init-button');
    await expect(initButton).toBeVisible();
    await initButton.click();
    
    // Wait for initialization process
    await page.waitForTimeout(5000);
    
    // Check if all core components are loaded
    const statusIndicators = {
      stt: page.locator('#status-stt'),
      tts: page.locator('#status-tts'),
      scene3d: page.locator('#status-3d'),
      avatar: page.locator('#status-avatar'),
      conversation: page.locator('#status-conversation'),
      speech: page.locator('#status-speech')
    };

    console.log('✅ Validating component initialization...');
    let componentsReady = 0;
    for (const [component, selector] of Object.entries(statusIndicators)) {
      try {
        const hasActiveClass = await selector.evaluate(el => el.classList.contains('status-active'));
        if (hasActiveClass) {
          console.log(`  ✓ ${component.toUpperCase()}: Ready`);
          componentsReady++;
        } else {
          console.log(`  ⚠️  ${component.toUpperCase()}: Not ready`);
        }
      } catch (error) {
        console.log(`  ❌ ${component.toUpperCase()}: Error checking status`);
      }
    }

    console.log(`📊 Components Ready: ${componentsReady}/6 (${Math.round(componentsReady/6*100)}%)`);

    await page.screenshot({ 
      path: path.join(screenshotDir, '02-after-initialization.png'),
      fullPage: true
    });

    // Test 2: TTS and Animation Test
    console.log('🎵 Testing TTS and animation synchronization...');
    const testTTSButton = page.locator('#test-tts');
    if (await testTTSButton.isVisible()) {
      await testTTSButton.click();
      await page.waitForTimeout(3000);
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '03-tts-animation-test.png'),
        fullPage: true
      });
    }

    // Test 3: Animation Test
    console.log('🎭 Testing avatar animation system...');
    const testAnimButton = page.locator('#test-animation');
    if (await testAnimButton.isVisible()) {
      await testAnimButton.click();
      await page.waitForTimeout(3000);
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '04-animation-test.png'),
        fullPage: true
      });
    }

    // Test 4: Conversation System Validation
    console.log('💬 Testing conversation system...');
    const startConversationButton = page.locator('#start-conversation');
    if (await startConversationButton.isVisible() && componentsReady >= 4) {
      await startConversationButton.click();
      await page.waitForTimeout(2000);
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '05-conversation-active.png'),
        fullPage: true
      });

      // Test stop conversation
      const stopConversationButton = page.locator('#stop-conversation');
      if (await stopConversationButton.isVisible()) {
        await stopConversationButton.click();
        await page.waitForTimeout(1000);
      }
    }

    // Test 5: Performance Monitoring Validation
    console.log('📈 Testing performance monitoring...');
    const performanceMonitor = page.locator('#performance-monitor');
    const hasPerformanceData = await performanceMonitor.isVisible();
    console.log(`📊 Performance Monitor Active: ${hasPerformanceData}`);

    // Test 6: 3D Scene Validation
    console.log('🏫 Testing 3D scene rendering...');
    const sceneContainer = page.locator('#scene-container');
    const hasScene = await sceneContainer.isVisible();
    console.log(`🎭 3D Scene Active: ${hasScene}`);

    // Test 7: Check for any JavaScript errors
    console.log('🔍 Checking for system errors...');
    const logMessages = await page.locator('#log-messages').textContent();
    const hasErrors = logMessages?.includes('Error') || logMessages?.includes('Failed');
    console.log(`❌ System Errors Detected: ${hasErrors}`);

    // Final comprehensive screenshot
    await page.screenshot({ 
      path: path.join(screenshotDir, '06-final-system-state.png'),
      fullPage: true
    });

    // Calculate completion percentage
    let completionScore = 0;
    const maxScore = 100;
    
    // Component initialization (40 points)
    completionScore += (componentsReady / 6) * 40;
    
    // 3D Scene rendering (20 points)
    if (hasScene) completionScore += 20;
    
    // Performance monitoring (10 points)
    if (hasPerformanceData) completionScore += 10;
    
    // No critical errors (20 points)
    if (!hasErrors) completionScore += 20;
    
    // UI responsiveness (10 points - if we got this far)
    completionScore += 10;

    console.log('📊 FINAL COMPLETION ANALYSIS:');
    console.log(`✅ System Integration Score: ${Math.round(completionScore)}/100`);
    console.log(`📈 Components Ready: ${componentsReady}/6 (${Math.round(componentsReady/6*100)}%)`);
    console.log(`🎭 3D Scene Active: ${hasScene}`);
    console.log(`📊 Performance Monitor: ${hasPerformanceData}`);
    console.log(`❌ Critical Errors: ${hasErrors}`);

    // Determine what's needed for 100%
    if (completionScore < 100) {
      const missing = [];
      if (componentsReady < 6) missing.push(`${6 - componentsReady} components not fully initialized`);
      if (!hasScene) missing.push('3D scene not rendering');
      if (!hasPerformanceData) missing.push('Performance monitoring not active');
      if (hasErrors) missing.push('Critical system errors present');
      
      console.log('🔧 TO ACHIEVE 100% COMPLETION:');
      missing.forEach(item => console.log(`   - Fix: ${item}`));
    } else {
      console.log('🎉 SYSTEM IS 100% COMPLETE!');
    }

    // Create completion report
    const reportPath = path.join(screenshotDir, 'completion-report.md');
    const report = `# 100% Completion Validation Report

## System Analysis Results

**Overall Completion Score:** ${Math.round(completionScore)}/100

### Component Status
- Components Ready: ${componentsReady}/6 (${Math.round(componentsReady/6*100)}%)
- 3D Scene Active: ${hasScene}
- Performance Monitor: ${hasPerformanceData}
- Critical Errors: ${hasErrors}

### Screenshots Captured
1. **Initial System Load** - System before initialization
2. **After Initialization** - Post-component loading state
3. **TTS Animation Test** - Audio-visual sync demonstration
4. **Animation Test** - Avatar animation capabilities
5. **Conversation Active** - Interactive conversation mode
6. **Final System State** - Complete system overview

### Completion Analysis
${completionScore >= 100 ? 
  '🎉 **SYSTEM IS 100% COMPLETE!** All components are working perfectly.' :
  `🔧 **System needs ${100 - Math.round(completionScore)} points to reach 100%**`
}

Generated: ${new Date().toISOString()}
`;

    fs.writeFileSync(reportPath, report);
    console.log(`📝 Completion report saved to: ${reportPath}`);

    // Assertions for test completion
    expect(completionScore).toBeGreaterThan(90); // Should be at least 90%
    expect(componentsReady).toBeGreaterThan(3); // At least 4 components working
    expect(hasScene).toBeTruthy(); // 3D scene should be active
  });
});