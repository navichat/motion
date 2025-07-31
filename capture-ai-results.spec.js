import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

test('Capture and Save All AI Model Results', async ({ page }) => {
  test.setTimeout(300000); // 5 minutes
  
  // Navigate to the demo page
  console.log('🤖 Starting AI Model Results Capture...');
  await page.goto('http://localhost:8081/task-manager-demo.html');
  
  // Wait for page to fully load
  await page.waitForTimeout(5000);
  
  // Click the workload button and wait for completion
  console.log('🖱️ Starting workload...');
  const workloadButton = page.locator('button:has-text("🚀 Real WASM/GPU/WebNN Workload")');
  await expect(workloadButton).toBeVisible({ timeout: 10000 });
  await workloadButton.click();
  
  // Wait for tasks to complete
  console.log('⏳ Waiting for tasks to complete...');
  await page.waitForTimeout(180000); // 3 minutes
  
  // Capture ALL results from the browser
  console.log('🔍 Capturing all results from browser...');
  const allResults = await page.evaluate(() => {
    const results = {
      completedTasks: [],
      taskManagerState: null,
      allJobTypes: [],
      timestamp: new Date().toISOString()
    };
    
    // Get TaskManager state and completed tasks
    if (window.taskManager) {
      results.taskManagerState = {
        totalTasks: window.taskManager.tasks ? window.taskManager.tasks.length : 0,
        completedCount: window.taskManager.completedTasks ? window.taskManager.completedTasks.length : 0,
        runningCount: window.taskManager.runningTasks ? window.taskManager.runningTasks.length : 0,
        pendingCount: window.taskManager.pendingTasks ? window.taskManager.pendingTasks.length : 0
      };
      
      // Collect all completed tasks
      if (window.taskManager.completedTasks) {
        window.taskManager.completedTasks.forEach(task => {
          if (task && task.job) {
            results.completedTasks.push({
              jobType: task.job.type || task.job.constructor.name,
              jobId: task.job.id,
              result: task.result,
              executionTime: task.endTime - task.startTime,
              success: task.status === 'completed',
              worker: task.worker ? task.worker.type : 'unknown'
            });
          }
        });
      }
    }
    
    // Collect all job types that were generated
    const allJobTypesSet = new Set();
    results.completedTasks.forEach(task => {
      allJobTypesSet.add(task.jobType);
    });
    results.allJobTypes = Array.from(allJobTypesSet);
    
    return results;
  });
  
  console.log('📊 Results captured:', {
    completedTasks: allResults.completedTasks.length,
    uniqueJobTypes: allResults.allJobTypes.length,
    taskManagerState: allResults.taskManagerState
  });
  
  // Create results directory
  const resultsDir = path.join(process.cwd(), 'ai-inference-results');
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
  }
  
  // Save complete results
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const resultsFile = path.join(resultsDir, `complete-ai-results-${timestamp}.json`);
  fs.writeFileSync(resultsFile, JSON.stringify(allResults, null, 2));
  
  // Create job type summary
  const jobTypeCounts = {};
  allResults.completedTasks.forEach(task => {
    jobTypeCounts[task.jobType] = (jobTypeCounts[task.jobType] || 0) + 1;
  });
  
  const summaryData = {
    timestamp: allResults.timestamp,
    totalCompletedTasks: allResults.completedTasks.length,
    uniqueJobTypes: allResults.allJobTypes.length,
    jobTypeCounts: jobTypeCounts,
    taskManagerState: allResults.taskManagerState,
    jobTypesList: allResults.allJobTypes.sort()
  };
  
  const summaryFile = path.join(resultsDir, `job-summary-${timestamp}.json`);
  fs.writeFileSync(summaryFile, JSON.stringify(summaryData, null, 2));
  
  console.log('💾 Results saved to:');
  console.log(`   📄 Complete results: ${resultsFile}`);
  console.log(`   📋 Summary: ${summaryFile}`);
  
  // Display summary
  console.log('\n📊 JOB TYPE SUMMARY:');
  console.log(`   Total completed tasks: ${allResults.completedTasks.length}`);
  console.log(`   Unique job types: ${allResults.allJobTypes.length}`);
  console.log('   Job type distribution:');
  Object.entries(jobTypeCounts).forEach(([type, count]) => {
    console.log(`     ${type}: ${count} tasks`);
  });
  
  // Test assertions
  expect(allResults.completedTasks.length).toBeGreaterThan(0);
  expect(allResults.allJobTypes.length).toBeGreaterThan(5);
  
  console.log('\n✅ AI model results capture completed successfully!');
});
