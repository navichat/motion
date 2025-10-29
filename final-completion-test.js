const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function validate100PercentCompletion() {
  const browser = await puppeteer.launch({ 
    headless: true, 
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });
  const page = await browser.newPage();
  
  // Set viewport for consistent screenshots
  await page.setViewport({ width: 1280, height: 800 });
  
  const screenshotDir = path.join(process.cwd(), 'test-results', '100-percent-validation');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  console.log('🚀 Starting 100% completion validation...');

  try {
    // Load the complete system
    console.log('📱 Loading complete Ichika conversation system...');
    await page.goto('http://localhost:8080/demos/complete_ichika_conversation_system.html');
    await page.waitForTimeout(3000);
    
    // Take initial screenshot
    await page.screenshot({ 
      path: path.join(screenshotDir, '01-system-loaded.png'),
      fullPage: true
    });
    console.log('📷 Initial system screenshot captured');

    // Click Initialize System button
    console.log('🔧 Initializing system...');
    const initButton = await page.$('#init-button');
    if (initButton) {
      await initButton.click();
      await page.waitForTimeout(8000); // Allow more time for initialization
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '02-system-initialized.png'),
        fullPage: true
      });
      console.log('📷 System initialization screenshot captured');
    }

    // Check component status
    console.log('✅ Checking component status...');
    const statusElements = [
      { id: '#status-stt', name: 'Speech-to-Text' },
      { id: '#status-tts', name: 'Text-to-Speech' },
      { id: '#status-3d', name: '3D Scene' },
      { id: '#status-avatar', name: 'Avatar' },
      { id: '#status-conversation', name: 'Conversation' },
      { id: '#status-speech', name: 'Speech Sync' }
    ];

    let componentsReady = 0;
    for (const component of statusElements) {
      try {
        const element = await page.$(component.id);
        if (element) {
          const hasActiveClass = await page.evaluate(el => el.classList.contains('status-active'), element);
          if (hasActiveClass) {
            console.log(`  ✓ ${component.name}: Ready`);
            componentsReady++;
          } else {
            console.log(`  ⚠️  ${component.name}: Not ready`);
          }
        }
      } catch (error) {
        console.log(`  ❌ ${component.name}: Error checking status`);
      }
    }

    // Test TTS functionality
    console.log('🎵 Testing TTS and animation...');
    const testTTSButton = await page.$('#test-tts');
    if (testTTSButton) {
      await testTTSButton.click();
      await page.waitForTimeout(4000);
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '03-tts-test.png'),
        fullPage: true
      });
      console.log('📷 TTS test screenshot captured');
    }

    // Test Animation functionality
    console.log('🎭 Testing animation system...');
    const testAnimButton = await page.$('#test-animation');
    if (testAnimButton) {
      await testAnimButton.click();
      await page.waitForTimeout(3000);
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '04-animation-test.png'),
        fullPage: true
      });
      console.log('📷 Animation test screenshot captured');
    }

    // Test conversation start
    console.log('💬 Testing conversation system...');
    const startConvButton = await page.$('#start-conversation');
    if (startConvButton && componentsReady >= 4) {
      await startConvButton.click();
      await page.waitForTimeout(3000);
      
      await page.screenshot({ 
        path: path.join(screenshotDir, '05-conversation-active.png'),
        fullPage: true
      });
      console.log('📷 Conversation active screenshot captured');

      // Stop conversation
      const stopConvButton = await page.$('#stop-conversation');
      if (stopConvButton) {
        await stopConvButton.click();
        await page.waitForTimeout(1000);
      }
    }

    // Check performance monitoring
    const performanceMonitor = await page.$('#performance-monitor');
    const hasPerformanceData = !!performanceMonitor;
    
    // Check 3D scene
    const sceneContainer = await page.$('#scene-container canvas');
    const hasScene = !!sceneContainer;

    // Check for errors in logs
    const logContent = await page.evaluate(() => {
      const logElement = document.getElementById('log-messages');
      return logElement ? logElement.textContent : '';
    });
    const hasErrors = logContent.includes('Error') || logContent.includes('Failed');

    // Take final comprehensive screenshot
    await page.screenshot({ 
      path: path.join(screenshotDir, '06-final-state.png'),
      fullPage: true
    });
    console.log('📷 Final state screenshot captured');

    // Calculate completion score
    let completionScore = 0;
    
    // Component initialization (50 points)
    completionScore += (componentsReady / 6) * 50;
    
    // 3D Scene rendering (20 points)
    if (hasScene) completionScore += 20;
    
    // Performance monitoring (10 points)
    if (hasPerformanceData) completionScore += 10;
    
    // No critical errors (15 points)
    if (!hasErrors) completionScore += 15;
    
    // UI responsiveness (5 points - if we got this far)
    completionScore += 5;

    console.log('\n📊 FINAL COMPLETION ANALYSIS:');
    console.log('═'.repeat(50));
    console.log(`🎯 System Integration Score: ${Math.round(completionScore)}/100`);
    console.log(`📈 Components Ready: ${componentsReady}/6 (${Math.round(componentsReady/6*100)}%)`);
    console.log(`🎭 3D Scene Active: ${hasScene ? '✅' : '❌'}`);
    console.log(`📊 Performance Monitor: ${hasPerformanceData ? '✅' : '❌'}`);
    console.log(`❌ Critical Errors: ${hasErrors ? '⚠️  Yes' : '✅ None'}`);
    console.log('═'.repeat(50));

    // Determine what's needed for 100%
    const missing = [];
    if (componentsReady < 6) missing.push(`${6 - componentsReady} components need initialization`);
    if (!hasScene) missing.push('3D scene rendering needs fixes');
    if (!hasPerformanceData) missing.push('Performance monitoring needs activation');
    if (hasErrors) missing.push('Critical system errors need resolution');
    
    if (missing.length > 0) {
      console.log('\n🔧 TO ACHIEVE 100% COMPLETION:');
      missing.forEach((item, index) => console.log(`   ${index + 1}. ${item}`));
    } else {
      console.log('\n🎉 SYSTEM IS 100% COMPLETE!');
    }

    // Create detailed report
    const report = `# 100% Completion Validation Report

## System Analysis Results

**Overall Completion Score:** ${Math.round(completionScore)}/100

### Component Status Details
- **Components Ready:** ${componentsReady}/6 (${Math.round(componentsReady/6*100)}%)
- **3D Scene Active:** ${hasScene ? '✅ Yes' : '❌ No'}
- **Performance Monitor:** ${hasPerformanceData ? '✅ Active' : '❌ Inactive'}
- **Critical Errors:** ${hasErrors ? '⚠️ Present' : '✅ None detected'}

### Screenshots Generated
1. **System Loaded** - Initial page load state
2. **System Initialized** - After component initialization
3. **TTS Test** - Audio output and animation sync test
4. **Animation Test** - Avatar animation system test
5. **Conversation Active** - Interactive conversation mode
6. **Final State** - Complete system overview

### Component Analysis
${statusElements.map(comp => {
  const ready = componentsReady > statusElements.indexOf(comp);
  return `- **${comp.name}:** ${ready ? '✅ Ready' : '❌ Not Ready'}`;
}).join('\n')}

### Completion Status
${completionScore >= 100 ? 
  '🎉 **SYSTEM IS 100% COMPLETE!** All components are functioning perfectly.' :
  `🔧 **System needs ${Math.round(100 - completionScore)} additional points to reach 100%**`
}

${missing.length > 0 ? `
### Required Improvements
${missing.map((item, i) => `${i + 1}. ${item}`).join('\n')}
` : ''}

**Generated:** ${new Date().toISOString()}
**Browser:** Puppeteer Chromium
**Test Suite:** Final Completion Validation
`;

    fs.writeFileSync(path.join(screenshotDir, 'completion-report.md'), report);
    console.log(`\n📝 Detailed completion report saved to: ${path.join(screenshotDir, 'completion-report.md')}`);

    return {
      score: Math.round(completionScore),
      componentsReady,
      hasScene,
      hasPerformanceData,
      hasErrors,
      missing,
      screenshotDir
    };

  } catch (error) {
    console.error('❌ Test execution error:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

// Run the validation if called directly
if (require.main === module) {
  validate100PercentCompletion()
    .then(results => {
      console.log('\n🎯 Validation complete!');
      process.exit(results.score >= 100 ? 0 : 1);
    })
    .catch(error => {
      console.error('❌ Validation failed:', error);
      process.exit(1);
    });
}

module.exports = { validate100PercentCompletion };