const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('Complete Ichika VRM Classroom System', () => {
  test('should demonstrate working VRM infrastructure with walking system', async ({ page }) => {
    // Configure longer timeout for VRM system initialization
    test.setTimeout(120000);

    try {
      console.log('🎭 Testing Complete Ichika VRM Classroom System...');

      // Navigate to the complete system demo
      const demoPath = path.resolve(__dirname, '../demos/complete_ichika_vrm_classroom_system.html');
      await page.goto(`file://${demoPath}`);

      // Wait for initial page load
      await page.waitForLoadState('domcontentloaded');
      console.log('✅ Demo page loaded');

      // Take initial screenshot
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/01-initial-load.png',
        fullPage: true 
      });

      // Wait for VRM components to load (check debug console)
      await page.waitForFunction(
        () => {
          const debugConsole = document.getElementById('debug-console');
          return debugConsole && debugConsole.textContent.includes('Self-contained VRM system created');
        },
        { timeout: 15000 }
      );
      console.log('✅ VRM infrastructure detected');

      // Take screenshot showing infrastructure ready
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/02-infrastructure-ready.png',
        fullPage: true 
      });

      // Initialize the complete VRM system
      console.log('🚀 Initializing complete VRM system...');
      await page.click('#init-system-btn');

      // Wait for system initialization to progress
      await page.waitForFunction(
        () => {
          const progressText = document.getElementById('progress-text');
          return progressText && progressText.textContent.includes('Validating VRM infrastructure');
        },
        { timeout: 10000 }
      );
      console.log('✅ System initialization started');

      // Wait for scene initialization to complete
      await page.waitForFunction(
        () => {
          const progressText = document.getElementById('progress-text');
          return progressText && (
            progressText.textContent.includes('complete') ||
            progressText.textContent.includes('ready')
          );
        },
        { timeout: 30000 }
      );
      console.log('✅ VRM system initialization completed');

      // Take screenshot showing initialized system
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/03-system-initialized.png',
        fullPage: true 
      });

      // Test VRM infrastructure validation
      console.log('🔍 Testing VRM infrastructure validation...');
      await page.click('#validate-btn');
      await page.waitForTimeout(2000);

      // Take screenshot of infrastructure validation
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/04-infrastructure-validated.png',
        fullPage: true 
      });

      // Test VRM operations
      console.log('👩‍🦳 Testing Ichika VRM loading...');
      await page.click('#load-ichika-btn');
      await page.waitForTimeout(1500);

      console.log('🏫 Testing classroom loading...');
      await page.click('#load-classroom-btn');
      await page.waitForTimeout(1500);

      // Take screenshot of VRM operations
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/05-vrm-operations.png',
        fullPage: true 
      });

      // Test animation system
      console.log('🎭 Testing BVH animation system...');
      await page.click('#test-animations-btn');
      await page.waitForTimeout(1000);

      console.log('🧪 Testing VRM integration...');
      await page.click('#test-integration-btn');
      await page.waitForTimeout(1000);

      // Take screenshot of animation tests
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/06-animation-tests.png',
        fullPage: true 
      });

      // Test walking system functionality
      console.log('🚶‍♀️ Testing walking system...');
      
      // Test individual walking positions
      console.log('📍 Testing walk to blackboard...');
      await page.click('#walk-blackboard-btn');
      await page.waitForTimeout(1500);
      
      console.log('📍 Testing walk to center...');
      await page.click('#walk-center-btn');
      await page.waitForTimeout(1500);
      
      console.log('📍 Testing walk to desk...');
      await page.click('#walk-desk-btn');
      await page.waitForTimeout(1500);

      // Take screenshot of walking system
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/07-walking-system.png',
        fullPage: true 
      });

      // Start comprehensive walking demo
      console.log('🎬 Starting comprehensive walking demonstration...');
      await page.click('#start-walking-demo-btn');
      
      // Let the walking demo run for a bit
      await page.waitForTimeout(8000);

      // Take screenshot during walking demo
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/08-walking-demo-active.png',
        fullPage: true 
      });

      // Test additional walking controls
      console.log('📍 Testing additional walking positions...');
      await page.click('#walk-front-left-btn');
      await page.waitForTimeout(1500);
      
      await page.click('#walk-front-right-btn');
      await page.waitForTimeout(1500);
      
      await page.click('#walk-back-btn');
      await page.waitForTimeout(1500);

      // Test random walk
      console.log('🎲 Testing random walk...');
      await page.click('#random-walk-btn');
      await page.waitForTimeout(2000);

      // Take final screenshot showing complete system
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/09-complete-system-final.png',
        fullPage: true 
      });

      // Verify system status indicators
      const loaderStatus = await page.textContent('#loader-status');
      const adapterStatus = await page.textContent('#adapter-status');
      const binderStatus = await page.textContent('#binder-status');
      const timelineStatus = await page.textContent('#timeline-status');
      const integrationStatus = await page.textContent('#integration-status');

      console.log('📊 System Status Check:');
      console.log(`  • AdvancedVRMLoader: ${loaderStatus}`);
      console.log(`  • VRMBVHAdapter: ${adapterStatus}`);
      console.log(`  • AvatarBinder: ${binderStatus}`);
      console.log(`  • BVHTimeline: ${timelineStatus}`);
      console.log(`  • ClassroomAvatarIntegration: ${integrationStatus}`);

      // Verify debug console shows system activity
      const debugContent = await page.textContent('#debug-console');
      expect(debugContent).toContain('Complete VRM system initialization successful');
      expect(debugContent).toContain('Ichika VRM classroom system ready');
      expect(debugContent).toContain('Walking system');

      // Stop any running demos
      await page.click('#stop-walking-btn');
      await page.waitForTimeout(1000);

      console.log('🎉 Complete Ichika VRM Classroom System test completed successfully!');
      console.log('📸 Screenshots captured showing:');
      console.log('  1. Initial system load');
      console.log('  2. VRM infrastructure ready');
      console.log('  3. System fully initialized');
      console.log('  4. Infrastructure validation');
      console.log('  5. VRM operations');
      console.log('  6. Animation system tests');
      console.log('  7. Walking system functionality');
      console.log('  8. Walking demo active');
      console.log('  9. Complete system demonstration');

    } catch (error) {
      console.error('❌ Test failed:', error);
      
      // Take error screenshot
      await page.screenshot({ 
        path: 'test-results/complete-ichika-system/error-screenshot.png',
        fullPage: true 
      });
      
      // Capture debug information
      const debugContent = await page.textContent('#debug-console').catch(() => 'Debug console not available');
      console.log('Debug Console Content:', debugContent);
      
      throw error;
    }
  });
});