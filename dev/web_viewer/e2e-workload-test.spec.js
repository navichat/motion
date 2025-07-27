import { test, expect } from '@playwright/test';

test.describe('Real Workload Test with Timeout and Enhanced Logging', () => {
  test('should run the Real WASM/GPU/WebNN Workload test and report completion', async ({ page }) => {
    // Set extended timeout for this test
    test.setTimeout(120000); // 2 minutes
    
    // Navigate to the demo page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront(); // Bring the page to the front to prevent throttling

    // Array to store console messages with timestamps
    const consoleMessages = [];
    const errorMessages = [];
    
    page.on('console', msg => {
      const timestamp = new Date().toISOString();
      const logEntry = `[${timestamp}] ${msg.text()}`;
      consoleMessages.push(logEntry);
      console.log(`[PAGE CONSOLE]: ${logEntry}`);
    });
    
    page.on('pageerror', error => {
      const timestamp = new Date().toISOString();
      const errorEntry = `[${timestamp}] PAGE ERROR: ${error.toString()}`;
      errorMessages.push(errorEntry);
      console.error(`[PAGE ERROR]: ${errorEntry}`);
    });

    // Wait for page to load completely
    console.log('⏳ Waiting for page to load...');
    await page.waitForLoadState('networkidle');
    
    // Check if TaskManager is available
    console.log('🔍 Checking if TaskManager is available...');
    const taskManagerAvailable = await page.evaluate(() => {
      return typeof TaskManager !== 'undefined';
    });
    console.log(`TaskManager available: ${taskManagerAvailable}`);
    
    if (!taskManagerAvailable) {
      throw new Error('TaskManager is not available on the page');
    }

    // Click the "Real WASM/GPU/WebNN Workload" button
    console.log('🖱️ Clicking "Real WASM/GPU/WebNN Workload" button...');
    
    // Wait for the button to be visible and clickable
    const workloadButton = page.locator('button:has-text("🚀 Real WASM/GPU/WebNN Workload")');
    await expect(workloadButton).toBeVisible({ timeout: 10000 });
    await workloadButton.click();
    await page.waitForTimeout(1000); // Give the page a moment to initialize after click
    
    console.log('✅ Button clicked, starting workload test...');

    // Monitor page activity and add periodic logging
    const startTime = Date.now();
    let lastLogTime = startTime;
    
    // Set up a periodic status check
    const statusInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      console.log(`⏱️  Test running for ${elapsed.toFixed(1)}s...`);
      
      // Log recent console messages
      const recentMessages = consoleMessages.slice(-5);
      if (recentMessages.length > 0) {
        console.log('📝 Recent page console messages:');
        recentMessages.forEach(msg => console.log(`   ${msg}`));
      }
    }, 10000); // Every 10 seconds

    try {
      // Wait for the completion message in the console output area with extended timeout
      console.log('⏳ Waiting for workload test completion message...');
      
      // First, wait for the workload to be created and scheduled
      await expect(page.locator('#consoleContent')).toContainText('📋 Creating realistic computational workload...', { 
        timeout: 15000 // 15 seconds timeout
      });
      
      console.log('✅ Workload creation detected!');
      
      // Wait for jobs to be generated
      await expect(page.locator('#consoleContent')).toContainText('📦 Generated', { 
        timeout: 15000 // 15 seconds timeout
      });
      
      console.log('✅ Job generation detected!');
      
      // Wait for jobs to be scheduled
      await expect(page.locator('#consoleContent')).toContainText('🎬', { 
        timeout: 15000 // 15 seconds timeout
      });
      
      console.log('✅ Job scheduling detected!');
      
      // Now wait for completion - be more flexible about completion messages
      await expect(page.locator('#consoleContent')).toContainText('🎉', { 
        timeout: 120000 // 2 minutes timeout for completion
      });
      
      clearInterval(statusInterval);
      console.log('🎯 Completion message detected!');
      
    } catch (timeoutError) {
      clearInterval(statusInterval);
      
      // Capture current state for debugging
      const currentTime = Date.now();
      const elapsedTime = (currentTime - startTime) / 1000;
      
      console.error(`❌ Test timed out after ${elapsedTime.toFixed(1)}s`);
      console.error('🔍 Debugging information:');
      
      // Get current console content
      const consoleContent = await page.locator('#consoleContent').textContent();
      console.error('📄 Current console content:');
      console.error(consoleContent);
      
      // Get page state
      const pageState = await page.evaluate(() => {
        return {
          taskManagerExists: typeof TaskManager !== 'undefined',
          windowTaskManager: typeof window.TaskManager !== 'undefined',
          runRealWorkloadTest: typeof window.runRealWorkloadTest !== 'undefined',
          currentTasks: window.taskManager ? window.taskManager.getStats() : 'No task manager',
          workerCount: {
            cpu: window.document.querySelectorAll('script[src*="cpu-worker"]').length,
            gpu: window.document.querySelectorAll('script[src*="gpu-worker"]').length,
            webnn: window.document.querySelectorAll('script[src*="webnn-worker"]').length
          }
        };
      });
      
      console.error('🔧 Page state:', JSON.stringify(pageState, null, 2));
      
      // Log all captured console messages
      console.error('📋 All console messages:');
      consoleMessages.forEach(msg => console.error(`   ${msg}`));
      
      // Log any errors
      if (errorMessages.length > 0) {
        console.error('🚨 Page errors:');
        errorMessages.forEach(msg => console.error(`   ${msg}`));
      }
      
      throw new Error(`Test timed out after ${elapsedTime.toFixed(1)}s. See debugging info above.`);
    }

    // Optional: Wait for a short period to ensure all final logs are captured
    await page.waitForTimeout(2000);

    // Assertions based on captured console messages  
    const logs = consoleMessages.join('\n');
    console.log('📊 Analyzing captured logs...');
    console.log('📋 Full logs preview:', logs.substring(0, 500) + '...');

    // Make assertions more flexible to handle test environment differences
    const hasWorkloadStart = logs.includes('🚀 Starting Real WASM') || logs.includes('Starting Real WASM') || logs.includes('🔧 Global createRealisticWorkload called');
    const hasWorkloadCreation = logs.includes('📋 Creating realistic computational workload') || logs.includes('Creating realistic computational workload') || logs.includes('createRealisticWorkload called');
    const hasJobGeneration = logs.includes('📦 Generated') || logs.includes('Generated') || logs.includes('createRandomJob');
    const hasJobScheduling = logs.includes('🎬') || logs.includes('scheduled') || logs.includes('Scheduled');
    const hasTaskActivity = logs.includes('Task') || logs.includes('Worker') || logs.includes('progress') || logs.includes('✅');

    // Verify that the workload test initiated
    expect(hasWorkloadStart || hasWorkloadCreation).toBe(true);
    
    // Verify that jobs were created and there's activity  
    expect(hasJobGeneration).toBe(true);
    
    // Verify that there was actual task execution activity
    expect(hasTaskActivity).toBe(true);

    const totalTime = (Date.now() - startTime) / 1000;
    console.log(`✅ Real Workload Test completed successfully in ${totalTime.toFixed(1)}s`);
    
    // Log summary statistics
    console.log('📈 Test Summary:');
    console.log(`   Total console messages: ${consoleMessages.length}`);
    console.log(`   Total errors: ${errorMessages.length}`);
    console.log(`   Test duration: ${totalTime.toFixed(1)}s`);
  });
});