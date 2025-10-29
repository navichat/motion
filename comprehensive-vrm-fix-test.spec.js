const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('Comprehensive VRM System Fix', () => {
  test.setTimeout(300000); // 5 minute timeout for shell compliance

  test('Fix VRM loading and capture working system', async ({ browser }) => {
    const context = await browser.newContext({
      permissions: ['microphone'],
      args: [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--use-angle=swiftshader-webgl',
        '--disable-web-security',
        '--disable-features=VizDisplayCompositor'
      ]
    });
    
    const page = await context.newPage();
    
    console.log('🎭 Starting comprehensive VRM system fix and validation...');
    
    // Enable verbose console logging
    page.on('console', msg => {
      const type = msg.type();
      if (['error', 'warn', 'info'].includes(type)) {
        console.log(`[${type.toUpperCase()}]`, msg.text());
      }
    });
    
    try {
      // Step 1: Test the complete demo page
      console.log('📋 Step 1: Loading complete VRM conversation system...');
      await page.goto('http://localhost:8000/demos/complete_ichika_conversation_system.html', { 
        waitUntil: 'networkidle',
        timeout: 60000 
      });
      
      // Wait for page to stabilize
      await page.waitForTimeout(3000);
      
      // Take initial screenshot
      await page.screenshot({ 
        path: 'test-results/vrm-fix/01-initial-demo-load.png',
        fullPage: true 
      });
      console.log('✅ Initial demo screenshot captured');
      
      // Step 2: Check VRM component availability
      console.log('📋 Step 2: Checking VRM component availability...');
      
      const vrmStatus = await page.evaluate(() => {
        const status = {
          VRMLoaderLite: typeof window.VRMLoaderLite !== 'undefined',
          AvatarBinder: typeof window.AvatarBinder !== 'undefined', 
          BVHTimeline: typeof window.BVHTimeline !== 'undefined',
          BVHTimelineVRMIntegration: typeof window.BVHTimelineVRMIntegration !== 'undefined',
          ClassroomAvatarIntegration: typeof window.ClassroomAvatarIntegration !== 'undefined',
          THREE: typeof window.THREE !== 'undefined'
        };
        
        console.log('🔍 VRM Components Status:', JSON.stringify(status, null, 2));
        return status;
      });
      
      // Step 3: Initialize the system
      console.log('📋 Step 3: Initializing VRM conversation system...');
      
      const initButton = page.locator('#init-button');
      if (await initButton.isVisible()) {
        await initButton.click();
        console.log('✅ Initialize button clicked');
        
        // Wait for initialization to complete
        await page.waitForTimeout(5000);
        
        // Check system status after initialization
        const systemStatus = await page.evaluate(() => {
          const status = {
            scene3D: document.querySelector('#status-3d-text')?.textContent,
            avatar: document.querySelector('#status-avatar-text')?.textContent,
            conversation: document.querySelector('#status-conversation-text')?.textContent,
            speechSync: document.querySelector('#status-speech-text')?.textContent
          };
          console.log('🎯 System Status:', JSON.stringify(status, null, 2));
          return status;
        });
        
        await page.screenshot({ 
          path: 'test-results/vrm-fix/02-post-initialization.png',
          fullPage: true 
        });
        console.log('✅ Post-initialization screenshot captured');
      }
      
      // Step 4: Test VRM debug interface
      console.log('📋 Step 4: Testing VRM debug interface...');
      
      await page.goto('http://localhost:8000/vrm_debug_test.html', {
        waitUntil: 'networkidle',
        timeout: 30000
      });
      
      await page.waitForTimeout(2000);
      
      // Test VRM loader availability
      const testLoaderButton = page.locator('button:has-text("Test VRMLoader Availability")');
      if (await testLoaderButton.isVisible()) {
        await testLoaderButton.click();
        await page.waitForTimeout(2000);
      }
      
      // Test direct VRM load
      const testVRMButton = page.locator('button:has-text("Test Direct VRM Load")');
      if (await testVRMButton.isVisible()) {
        await testVRMButton.click();
        await page.waitForTimeout(3000);
      }
      
      await page.screenshot({ 
        path: 'test-results/vrm-fix/03-vrm-debug-test.png',
        fullPage: true 
      });
      console.log('✅ VRM debug test screenshot captured');
      
      // Step 5: Analyze what needs to be fixed
      console.log('📋 Step 5: Analyzing VRM loading issues...');
      
      const vrmAnalysis = await page.evaluate(() => {
        const logs = Array.from(document.querySelectorAll('#log div')).map(div => div.textContent);
        const statusText = document.querySelector('#status-text')?.textContent || 'Unknown';
        
        const analysis = {
          statusText,
          recentLogs: logs.slice(-10), // Last 10 log entries
          vrmLoaderAvailable: typeof window.VRMLoaderLite !== 'undefined',
          threeJSAvailable: typeof window.THREE !== 'undefined',
          canvasElements: document.querySelectorAll('canvas').length
        };
        
        console.log('🔍 VRM Analysis:', JSON.stringify(analysis, null, 2));
        return analysis;
      });
      
      console.log('🎯 VRM System Analysis Complete:', vrmAnalysis);
      
      // Step 6: Generate comprehensive report
      const report = {
        timestamp: new Date().toISOString(),
        vrmStatus,
        vrmAnalysis,
        recommendations: [
          'Check Three.js version compatibility (should be 0.177.0)',
          'Verify VRM asset paths (./assets/avatars/ vs ../assets/avatars/)',
          'Ensure proper VRM ready promise awaiting',
          'Fix VRM scene assignment and positioning',
          'Implement BVH animation loading and timeline integration'
        ]
      };
      
      console.log('📊 Comprehensive VRM Fix Report:');
      console.log(JSON.stringify(report, null, 2));
      
      // Final screenshot showing analysis complete
      await page.screenshot({ 
        path: 'test-results/vrm-fix/04-analysis-complete.png',
        fullPage: true 
      });
      console.log('✅ Analysis complete screenshot captured');
      
    } catch (error) {
      console.error('❌ VRM system analysis failed:', error);
      
      // Capture error screenshot
      await page.screenshot({ 
        path: 'test-results/vrm-fix/error-state.png',
        fullPage: true 
      });
      
      throw error;
    }
    
    await context.close();
  });
});