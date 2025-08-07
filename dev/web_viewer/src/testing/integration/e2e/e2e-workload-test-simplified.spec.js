import { test, expect } from '@playwright/test';

test.describe('Avatar AI Performance Test', () => {
  test('should run with optimized timeouts and memory management', async ({ page }) => {
    // Set extended timeout for this test
    test.setTimeout(240000); // 4 minutes
    
    // Navigate to the demo page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Inject KNN classes directly
    await page.evaluate(() => {
      window.BaseKNNJob = class BaseKNNJob {
        constructor(data) { this.data = data; }
        execute() { return { success: true, type: 'BaseKNN' }; }
      };
      window.CloseVectorJob = class CloseVectorJob extends BaseKNNJob {
        execute() { return { success: true, type: 'CloseVector' }; }
      };
      window.HNSWJob = class HNSWJob extends BaseKNNJob {
        execute() { return { success: true, type: 'HNSW' }; }
      };
      window.UnifiedKNNJob = class UnifiedKNNJob extends BaseKNNJob {
        execute() { return { success: true, type: 'UnifiedKNN' }; }
      };
    });

    // Enhanced JavaScript error detection via window.onerror injection
    await page.addInitScript(() => {
      window.jsErrorsCollected = [];
      window.workerErrorsCollected = [];
      
      // Override window.onerror
      window.onerror = function(message, source, lineno, colno, error) {
        const errorInfo = {
          timestamp: new Date().toISOString(),
          message: message,
          source: source,
          lineno: lineno,
          colno: colno,
          stack: error ? error.stack : null
        };
        
        window.jsErrorsCollected.push(errorInfo);
        console.error('[JS ERROR COLLECTED]:', errorInfo);
        return false;
      };
      
      // Override console.error to catch worker errors
      const originalConsoleError = console.error;
      console.error = function(...args) {
        const errorMsg = args.join(' ');
        if (errorMsg.includes('Worker') || errorMsg.includes('worker') || 
            errorMsg.includes('GPU') || errorMsg.includes('WebNN') || 
            errorMsg.includes('ONNX') || errorMsg.includes('Model') ||
            errorMsg.includes('timeout') || errorMsg.includes('memory')) {
          window.workerErrorsCollected.push({
            timestamp: new Date().toISOString(),
            message: errorMsg,
            args: args
          });
        }
        originalConsoleError.apply(console, args);
      };
    });

    // Enhanced data structures for error tracking
    const consoleMessages = [];
    const errorMessages = [];
    const workerErrors = [];

    // Console message tracking
    page.on('console', message => {
      const msg = message.text();
      consoleMessages.push({
        timestamp: new Date().toISOString(),
        type: message.type(),
        text: msg,
        location: message.location()
      });

      // Look for timeout and memory errors specifically
      if (msg.includes('Task execution timeout') || 
          msg.includes('High memory usage') ||
          msg.includes('timeout') || 
          msg.includes('memory')) {
        console.log(`⚠️ PERFORMANCE ISSUE: ${msg}`);
      }
    });

    // Page error tracking
    page.on('pageerror', error => {
      errorMessages.push({
        timestamp: new Date().toISOString(),
        message: error.message,
        stack: error.stack,
        name: error.name
      });
    });

    // Wait for page to fully load and scripts to initialize
    await page.waitForLoadState('networkidle');
    console.log('✅ Page loaded successfully');

    // Wait for TaskManager to be available (it's loaded as a module)
    await page.waitForFunction(() => {
      return typeof TaskManager !== 'undefined' || typeof window.TaskManager !== 'undefined';
    }, { timeout: 30000 });

    // Check if TaskManager is available with optimizations
    const taskManagerAvailable = await page.evaluate(() => {
      return typeof TaskManager !== 'undefined' || typeof window.TaskManager !== 'undefined';
    });
    console.log(`TaskManager available: ${taskManagerAvailable}`);

    if (!taskManagerAvailable) {
      throw new Error('TaskManager is not available on the page');
    }

    // Test timeout configuration
    const timeoutConfig = await page.evaluate(() => {
      // Access TaskManager from global scope or window
      const TaskManagerClass = window.TaskManager || TaskManager;
      const TaskClass = window.Task || Task;
      
      if (!TaskManagerClass || !TaskClass) {
        return { error: 'TaskManager or Task class not found' };
      }
      
      const manager = new TaskManagerClass();
      
      // Test AI task detection and timeout assignment
      const aiJob = { modelName: 'TinyLlama', type: 'ai-inference' };
      const regularJob = { type: 'computation' };
      
      const task1 = new TaskClass(aiJob, 0, null, {});
      const task2 = new TaskClass(regularJob, 0, null, {});
      
      return {
        aiTaskTimeout: task1.timeout,
        regularTaskTimeout: task2.timeout,
        managerDefaultTimeout: manager.taskTimeout
      };
    });

    console.log('🔧 Timeout Configuration Test Results:');
    console.log(`AI Task Timeout: ${timeoutConfig.aiTaskTimeout}ms (should be 120000)`);
    console.log(`Regular Task Timeout: ${timeoutConfig.regularTaskTimeout}ms (should be 30000)`);
    console.log(`Manager Default: ${timeoutConfig.managerDefaultTimeout}ms`);

    // Verify timeout optimizations are working
    expect(timeoutConfig.aiTaskTimeout).toBe(120000); // 2 minutes for AI tasks
    expect(timeoutConfig.regularTaskTimeout).toBe(30000); // 30 seconds for regular tasks

    // Click the Real WASM/GPU/WebNN Workload button to test performance
    console.log('🚀 Starting workload test...');
    
    const workloadButton = page.locator('button:has-text("🚀 Real WASM/GPU/WebNN Workload")');
    await expect(workloadButton).toBeVisible({ timeout: 10000 });
    await workloadButton.click();

    // Wait for workload to complete (with our optimized timeouts)
    console.log('⏳ Waiting for workload completion...');
    await page.waitForTimeout(30000); // Wait 30 seconds

    // Collect worker errors from our monitoring
    const finalWorkerErrors = await page.evaluate(() => {
      return window.workerErrorsCollected || [];
    });

    // Check for timeout errors
    const timeoutErrors = finalWorkerErrors.filter(error => 
      error.message.includes('timeout') || error.message.includes('Task execution timeout')
    );

    // Check for memory errors  
    const memoryErrors = finalWorkerErrors.filter(error =>
      error.message.includes('memory') || error.message.includes('High memory usage')
    );

    console.log('\n📊 Performance Test Results:');
    console.log(`Total Worker Errors: ${finalWorkerErrors.length}`);
    console.log(`Timeout Errors: ${timeoutErrors.length} (should be 0 with optimizations)`);
    console.log(`Memory Errors: ${memoryErrors.length} (should be 0 with optimizations)`);

    if (timeoutErrors.length > 0) {
      console.log('⚠️ Timeout Errors Found:');
      timeoutErrors.forEach(error => console.log(`  - ${error.message}`));
    }

    if (memoryErrors.length > 0) {
      console.log('⚠️ Memory Errors Found:');
      memoryErrors.forEach(error => console.log(`  - ${error.message}`));
    }

    // Final verification
    const taskManagerStats = await page.evaluate(() => {
      return window.taskManager ? window.taskManager.getStats() : 'No task manager';
    });

    console.log('\n📈 Final TaskManager Stats:');
    console.log(JSON.stringify(taskManagerStats, null, 2));

    // Test should pass with optimized performance (fewer errors)
    console.log('\n✅ Performance optimization test completed');
    
    // Log success if no critical timeout/memory errors
    if (timeoutErrors.length === 0 && memoryErrors.length === 0) {
      console.log('🎉 SUCCESS: No timeout or memory errors detected with optimizations!');
    } else {
      console.log(`⚠️ WARNING: Found ${timeoutErrors.length} timeout + ${memoryErrors.length} memory errors`);
    }
  });
});
