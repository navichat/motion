import { test, expect } from '@playwright/test';

test.describe('Worker Capability Debug', () => {
  test('should debug worker initialization and capabilities', async ({ page }) => {
    test.setTimeout(120000); // 2 minutes
    
    console.log('🔧 Starting Worker Capability Debug Test...');
    
    // Navigate to the demo page
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();
    await page.waitForLoadState('networkidle');
    
    // Track all console messages
    const allMessages = [];
    
    page.on('console', msg => {
      const msgText = msg.text();
      allMessages.push(msgText);
      
      if (msgText.includes('Worker') && (msgText.includes('ready') || msgText.includes('capabilities') || msgText.includes('initialized'))) {
        console.log(`🔧 WORKER DEBUG: ${msgText}`);
      } else if (msgText.includes('🔍') || msgText.includes('❌') || msgText.includes('✅')) {
        console.log(`🔍 TASK DEBUG: ${msgText}`);
      }
    });
    
    // Wait for workers to initialize
    console.log('⏳ Waiting for TaskManager and workers to initialize...');
    await page.waitForTimeout(5000);
    
    // Click the button to start a single test
    console.log('🖱️ Clicking the workload button...');
    await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
    
    // Wait for some task processing attempts
    await page.waitForTimeout(10000);
    
    // Get TaskManager state
    const debugInfo = await page.evaluate(() => {
      if (window.taskManager) {
        const workerPoolInfo = {};
        
        for (const [poolType, pool] of Object.entries(window.taskManager.workerPools)) {
          workerPoolInfo[poolType] = {
            size: pool.size,
            availableWorkers: pool.availableWorkers.length,
            busyWorkers: pool.busyWorkers.size,
            workers: pool.workers.map(w => ({
              id: w.id,
              type: w.type,
              busy: w.busy,
              capabilities: w.capabilities
            }))
          };
        }
        
        return {
          taskManagerExists: true,
          running: window.taskManager.running,
          heapSize: window.taskManager.heap ? window.taskManager.heap.size() : 'no heap',
          totalTasks: window.taskManager.tasks ? window.taskManager.tasks.size : 'no tasks map',
          runningTasks: window.taskManager.runningTasks ? window.taskManager.runningTasks.size : 'no running map',
          completedTasks: window.taskManager.completedTasks ? window.taskManager.completedTasks.size : 'no completed map',
          failedTasks: window.taskManager.failedTasks ? window.taskManager.failedTasks.size : 'no failed map',
          workerPools: workerPoolInfo,
          maxConcurrentTasks: window.taskManager.maxConcurrentTasks
        };
      }
      return { taskManagerExists: false };
    });
    
    console.log('\n🔧 DETAILED WORKER POOL DEBUG:');
    console.log(JSON.stringify(debugInfo, null, 2));
    
    if (debugInfo.taskManagerExists) {
      console.log('\n📊 WORKER POOL SUMMARY:');
      for (const [poolType, poolInfo] of Object.entries(debugInfo.workerPools)) {
        console.log(`  ${poolType}: ${poolInfo.availableWorkers}/${poolInfo.size} available, ${poolInfo.busyWorkers} busy`);
        
        for (const worker of poolInfo.workers) {
          console.log(`    - ${worker.id}: busy=${worker.busy}, capabilities=${JSON.stringify(worker.capabilities)}`);
        }
      }
      
      console.log(`\n⚙️ TaskManager Config: maxConcurrentTasks=${debugInfo.maxConcurrentTasks}`);
      console.log(`📋 Task State: heap=${debugInfo.heapSize}, total=${debugInfo.totalTasks}, running=${debugInfo.runningTasks}, completed=${debugInfo.completedTasks}, failed=${debugInfo.failedTasks}`);
    } else {
      console.log('❌ TaskManager not found!');
    }
    
    // Always pass the test - this is just for debugging
    expect(true).toBe(true);
  });
});
