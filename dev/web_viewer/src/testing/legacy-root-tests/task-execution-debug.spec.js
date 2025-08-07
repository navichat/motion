const { test, expect } = require('@playwright/test');

test('debug task execution', async ({ page }) => {
  test.setTimeout(90000);
  
  const consoleMessages = [];
  const avatarMessages = [];
  
  page.on('console', msg => {
    const msgText = msg.text();
    consoleMessages.push(`${msg.type()}: ${msgText}`);
    
    if (msgText.includes('AVATAR AI COLLECTED')) {
      avatarMessages.push(msgText);
      console.log(`🎯 FOUND AVATAR MESSAGE: ${msgText}`);
    }
    
    if (msgText.includes('Task') || msgText.includes('Worker') || msgText.includes('running')) {
      console.log(`[TASK] ${msgText}`);
    }
  });
  
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
  await page.waitForTimeout(3000);
  
  // Click the button
  console.log('🖱️ Clicking Real Workload Test button...');
  await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
  
  await page.waitForTimeout(3000);
  
  // Also try calling the function directly
  console.log('🔧 Calling runRealWorkloadTest directly...');
  const directResult = await page.evaluate(async () => {
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
  console.log(`Direct call result: ${directResult}`);
  
  // Wait for tasks to process
  console.log('⏳ Waiting for tasks to execute...');
  await page.waitForTimeout(20000);
  
  // Check final state
  const finalState = await page.evaluate(() => {
    if (window.taskManager) {
      return {
        running: window.taskManager.running,
        heapSize: window.taskManager.heap ? window.taskManager.heap.size() : 'no heap',
        tasksCount: window.taskManager.tasks ? window.taskManager.tasks.size : 'no tasks',
        runningTasksCount: window.taskManager.runningTasks ? window.taskManager.runningTasks.size : 'no running tasks',
        completedTasksCount: window.taskManager.completedTasks ? window.taskManager.completedTasks.size : 'no completed tasks',
        failedTasksCount: window.taskManager.failedTasks ? window.taskManager.failedTasks.size : 'no failed tasks'
      };
    }
    return 'No taskManager';
  });
  
  console.log('📊 Final TaskManager state:', JSON.stringify(finalState, null, 2));
  console.log(`🎯 Total AVATAR AI COLLECTED messages: ${avatarMessages.length}`);
  avatarMessages.forEach((msg, index) => {
    console.log(`  ${index + 1}. ${msg}`);
  });
});
