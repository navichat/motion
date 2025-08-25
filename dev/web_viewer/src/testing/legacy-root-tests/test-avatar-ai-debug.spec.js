const { test, expect } = require('@playwright/test');

test.describe('Debug Avatar AI Collection', () => {
  test('debug what happens when button is clicked', async ({ page }) => {
    test.setTimeout(60000);
    
    // Capture all console messages
    const consoleMessages = [];
    page.on('console', msg => {
      const msgText = msg.text();
      consoleMessages.push(`${msg.type()}: ${msgText}`);
      console.log(`[BROWSER ${msg.type().toUpperCase()}] ${msgText}`);
    });
    
    // Navigate to the page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    
    // Wait for page to load
    await page.waitForTimeout(2000);
    
    console.log('🔍 Checking if TaskManager is available...');
    const taskManagerAvailable = await page.evaluate(() => {
      return typeof window.TaskManager !== 'undefined';
    });
    console.log(`TaskManager available: ${taskManagerAvailable}`);
    
    console.log('🔍 Checking if runRealWorkloadTest is available...');
    const runRealWorkloadTestAvailable = await page.evaluate(() => {
      return typeof window.runRealWorkloadTest !== 'undefined';
    });
    console.log(`runRealWorkloadTest available: ${runRealWorkloadTestAvailable}`);
    
    // Try to initialize TaskManager manually if needed
    await page.evaluate(() => {
      if (typeof window.TaskManager !== 'undefined' && !window.taskManager) {
        console.log('🔧 Manually initializing TaskManager...');
        window.taskManager = new window.TaskManager();
      }
    });
    
    console.log('🔍 Checking if taskManager instance exists...');
    const taskManagerInstanceExists = await page.evaluate(() => {
      return typeof window.taskManager !== 'undefined' && window.taskManager !== null;
    });
    console.log(`TaskManager instance exists: ${taskManagerInstanceExists}`);
    
    // Click the button
    console.log('🖱️ Clicking Real Workload Test button...');
    await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
    
    // Wait and capture messages
    console.log('⏳ Waiting for tasks to execute...');
    await page.waitForTimeout(10000);
    
    // Check if any workers were created
    const workerInfo = await page.evaluate(() => {
      if (window.taskManager) {
        return {
          heapSize: window.taskManager.heap ? window.taskManager.heap.size() : 'no heap',
          tasksCount: window.taskManager.tasks ? window.taskManager.tasks.size : 'no tasks',
          runningTasksCount: window.taskManager.runningTasks ? window.taskManager.runningTasks.size : 'no running tasks',
          workerPools: Object.keys(window.taskManager.workerPools || {})
        };
      }
      return 'No taskManager';
    });
    console.log('📊 TaskManager state:', JSON.stringify(workerInfo, null, 2));
    
    // Try calling the function directly
    console.log('🔧 Trying to call runRealWorkloadTest directly...');
    const directCallResult = await page.evaluate(async () => {
      try {
        if (typeof window.runRealWorkloadTest === 'function') {
          await window.runRealWorkloadTest();
          return 'SUCCESS';
        } else {
          return 'FUNCTION_NOT_FOUND';
        }
      } catch (error) {
        return `ERROR: ${error.message}`;
      }
    });
    console.log(`Direct call result: ${directCallResult}`);
    
    await page.waitForTimeout(5000);
    
    console.log('📝 Final console messages:');
    consoleMessages.forEach((msg, index) => {
      console.log(`  ${index + 1}. ${msg}`);
    });
  });
});
