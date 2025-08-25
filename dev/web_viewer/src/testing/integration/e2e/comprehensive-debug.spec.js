import { test, expect } from '@playwright/test';

test.describe('Comprehensive AI Model Debug', () => {
  test('should debug all AI model task assignments and completions', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes
    
    console.log('🔬 Starting Comprehensive AI Model Debug Test...');
    
    // Navigate to the demo page
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();
    await page.waitForLoadState('networkidle');
    
    // Track all console messages
    const allMessages = [];
    const avatarMessages = [];
    const taskMessages = [];
    const workerMessages = [];
    
    page.on('console', msg => {
      const msgText = msg.text();
      allMessages.push(msgText);
      
      if (msgText.includes('AVATAR AI COLLECTED')) {
        avatarMessages.push(msgText);
        console.log(`🎯 AVATAR COLLECTED #${avatarMessages.length}: ${msgText.substring(0, 100)}...`);
      } else if (msgText.includes('✅ Submitted') && msgText.includes('job with')) {
        taskMessages.push(msgText);
        console.log(`📋 TASK SUBMITTED: ${msgText}`);
      } else if (msgText.includes('assigned to') && msgText.includes('worker')) {
        workerMessages.push(msgText);
        console.log(`👷 WORKER ASSIGNED: ${msgText}`);
      }
    });
    
    // Click the button to start
    console.log('🖱️ Clicking the workload button...');
    await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
    
    // Monitor task distribution and execution
    for (let i = 0; i < 24; i++) { // 2 minutes total, 5 second intervals
      await page.waitForTimeout(5000);
      
      // Get TaskManager state
      const taskManagerState = await page.evaluate(() => {
        if (window.taskManager) {
          return {
            running: window.taskManager.running,
            heapSize: window.taskManager.heap ? window.taskManager.heap.size() : 'no heap',
            totalTasks: window.taskManager.tasks ? window.taskManager.tasks.size : 'no tasks map',
            runningTasks: window.taskManager.runningTasks ? window.taskManager.runningTasks.size : 'no running map',
            completedTasks: window.taskManager.completedTasks ? window.taskManager.completedTasks.size : 'no completed map',
            failedTasks: window.taskManager.failedTasks ? window.taskManager.failedTasks.size : 'no failed map'
          };
        }
        return { error: 'TaskManager not found' };
      });
      
      console.log(`⏰ ${(i + 1) * 5}s - TaskManager: running=${taskManagerState.running}, heap=${taskManagerState.heapSize}, total=${taskManagerState.totalTasks}, running=${taskManagerState.runningTasks}, completed=${taskManagerState.completedTasks}, failed=${taskManagerState.failedTasks}`);
      console.log(`📊 Messages so far - Avatar: ${avatarMessages.length}, Tasks: ${taskMessages.length}, Workers: ${workerMessages.length}, Total: ${allMessages.length}`);
      
      // If we have good results and heap is empty, we might be done
      if (avatarMessages.length >= 10 && taskManagerState.heapSize === 0) {
        console.log(`🏁 Looks like we have substantial results (${avatarMessages.length}) and empty heap. Breaking early.`);
        break;
      }
    }
    
    // Final summary
    console.log(`\n📈 FINAL SUMMARY:`);
    console.log(`🎯 Total AVATAR AI COLLECTED messages: ${avatarMessages.length}`);
    console.log(`📋 Total task submissions: ${taskMessages.length}`);
    console.log(`👷 Total worker assignments: ${workerMessages.length}`);
    
    // Show task distribution
    console.log(`\n📋 TASK SUBMISSIONS:`);
    taskMessages.forEach((msg, index) => {
      console.log(`  ${index + 1}. ${msg}`);
    });
    
    // Show worker assignments
    console.log(`\n👷 WORKER ASSIGNMENTS:`);
    workerMessages.forEach((msg, index) => {
      console.log(`  ${index + 1}. ${msg}`);
    });
    
    // Show all avatar messages
    console.log(`\n🎯 AVATAR AI COLLECTED MESSAGES:`);
    avatarMessages.forEach((msg, index) => {
      console.log(`  ${index + 1}. ${msg.substring(0, 200)}...`);
    });
    
    // Ensure we captured a good number
    expect(avatarMessages.length).toBeGreaterThanOrEqual(8);
  });
});
