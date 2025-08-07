import { test, expect } from '@playwright/test';

test.describe('Full Debug E2E Test', () => {
  test('should debug the complete AI model collection flow', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes
    
    console.log('🤖 Starting Full Debug E2E Test...');
    
    // Navigate to the demo page
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();
    
    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle');
    console.log('📄 Page loaded completely');
    
    // Check if button exists
    const button = page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' });
    const buttonExists = await button.count();
    console.log(`🔍 Button exists: ${buttonExists > 0 ? 'YES' : 'NO'}`);
    
    if (buttonExists > 0) {
      console.log('🔘 Button text:', await button.textContent());
    }
    
    // Listen for ALL console messages
    const consoleMessages = [];
    page.on('console', msg => {
      const msgText = msg.text();
      consoleMessages.push(msgText);
      
      // Log specific types of messages
      if (msgText.includes('AVATAR AI COLLECTED')) {
        console.log('🎯 FOUND AVATAR MESSAGE:', msgText);
      } else if (msgText.includes('TaskManager')) {
        console.log('🔧 TaskManager message:', msgText);
      } else if (msgText.includes('Worker')) {
        console.log('👷 Worker message:', msgText);
      } else if (msgText.includes('Task ')) {
        console.log('📋 Task message:', msgText);
      }
    });
    
    // Check if runRealWorkloadTest function exists
    const functionExists = await page.evaluate(() => {
      return typeof window.runRealWorkloadTest === 'function';
    });
    console.log(`🔍 runRealWorkloadTest function exists: ${functionExists ? 'YES' : 'NO'}`);
    
    // Check if TaskManager exists
    const taskManagerExists = await page.evaluate(() => {
      return typeof window.TaskManager === 'function';
    });
    console.log(`🔍 TaskManager class exists: ${taskManagerExists ? 'YES' : 'NO'}`);
    
    if (buttonExists > 0) {
      console.log('🖱️ Clicking the button...');
      await button.click();
      
      // Wait and show progress
      for (let i = 0; i < 12; i++) { // 60 seconds total, 5 second intervals
        await page.waitForTimeout(5000);
        console.log(`⏰ Waiting... ${(i + 1) * 5}s elapsed`);
        
        // Check TaskManager state
        const taskManagerState = await page.evaluate(() => {
          // Check multiple possible TaskManager instances
          const instances = {
            globalTaskManager: window.globalTaskManager,
            taskManager: window.taskManager,
            TaskManagerClass: typeof window.TaskManager
          };
          
          if (window.taskManager) {
            return {
              source: 'window.taskManager',
              running: window.taskManager.running,
              heapSize: window.taskManager.heap ? window.taskManager.heap.size() : 'no heap',
              completedTasks: window.taskManager.completedTasks ? window.taskManager.completedTasks.length : 'no completed array',
              failedTasks: window.taskManager.failedTasks ? window.taskManager.failedTasks.length : 'no failed array'
            };
          } else if (window.globalTaskManager) {
            return {
              source: 'window.globalTaskManager',
              running: window.globalTaskManager.running,
              heapSize: window.globalTaskManager.heap ? window.globalTaskManager.heap.size() : 'no heap',
              completedTasks: window.globalTaskManager.completedTasks ? window.globalTaskManager.completedTasks.length : 'no completed array',
              failedTasks: window.globalTaskManager.failedTasks ? window.globalTaskManager.failedTasks.length : 'no failed array'
            };
          }
          return { instances, noManager: true };
        });
        
        if (taskManagerState && !taskManagerState.noManager) {
          console.log(`📊 TaskManager state: ${taskManagerState.source} - running=${taskManagerState.running}, heap=${taskManagerState.heapSize}, completed=${taskManagerState.completedTasks}, failed=${taskManagerState.failedTasks}`);
        } else {
          console.log(`📊 No TaskManager found:`, taskManagerState);
        }
      }
    }
    
    console.log(`📝 Total console messages captured: ${consoleMessages.length}`);
    
    // Count AVATAR AI COLLECTED messages
    const avatarMessages = consoleMessages.filter(msg => msg.includes('AVATAR AI COLLECTED'));
    console.log(`🎯 Total AVATAR AI COLLECTED messages: ${avatarMessages.length}`);
    
    avatarMessages.forEach((msg, index) => {
      console.log(`  ${index + 1}. ${msg.substring(0, 200)}...`);
    });
  });
});
