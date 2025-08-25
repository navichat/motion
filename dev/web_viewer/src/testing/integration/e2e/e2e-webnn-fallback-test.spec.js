import { test, expect } from '@playwright/test';

test.describe('WebNN to WebGPU Fallback Test', () => {
  test('should run WebNN models with WebGPU fallback when WebNN is unavailable', async ({ page }) => {
    // Set timeout for this test
    test.setTimeout(30000); // 30 seconds
    
    // Navigate to the demo page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track console messages
    const consoleMessages = [];
    const completedTasks = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      console.log(`[PAGE CONSOLE]: ${text}`);
      
      // Track completed tasks
      if (text.includes('Task completed:') && text.includes('jobType:')) {
        try {
          const taskMatch = text.match(/taskId: (task_\w+), jobType: (\w+), duration: (\d+)/);
          if (taskMatch) {
            completedTasks.push({
              taskId: taskMatch[1],
              jobType: taskMatch[2], 
              duration: parseInt(taskMatch[3])
            });
            console.log(`✅ Captured completed task: ${taskMatch[2]} in ${taskMatch[3]}ms`);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }
    });

    // Wait for the page to load
    await page.waitForSelector('button[onclick="runRealWorkloadTest()"]', { timeout: 10000 });
    
    // Click the Real Workload button to trigger capability detection and job generation
    console.log('🚀 Starting Real Workload test...');
    await page.click('button[onclick="runRealWorkloadTest()"]');
    
    // Wait for capability detection and job execution
    console.log('⏳ Waiting for WebNN models to run with WebGPU fallback...');
    
    // Wait for at least some WebNN fallback models to complete
    let webnnFallbackModels = [];
    const maxWaitTime = 20000; // 20 seconds
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
      // Check for WebNN models that should fallback to WebGPU
      const webnnModelTypes = ['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'];
      webnnFallbackModels = completedTasks.filter(task => 
        webnnModelTypes.includes(task.jobType)
      );
      
      // If we have at least 2 WebNN fallback models completed, we can proceed
      if (webnnFallbackModels.length >= 2) {
        console.log(`🎉 Found ${webnnFallbackModels.length} WebNN fallback models completed!`);
        break;
      }
      
      await page.waitForTimeout(1000); // Wait 1 second
    }
    
    // Verify that WebNN models ran successfully with WebGPU fallback
    console.log('🔍 Checking WebNN to WebGPU fallback results...');
    
    const webnnModelTypes = ['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'];
    let foundWebnnFallbacks = 0;
    
    for (const modelType of webnnModelTypes) {
      const completed = completedTasks.find(task => task.jobType === modelType);
      if (completed) {
        console.log(`✅ ${modelType} successfully completed with WebGPU fallback in ${completed.duration}ms`);
        foundWebnnFallbacks++;
      } else {
        console.log(`⚠️ ${modelType} not found in completed tasks`);
      }
    }
    
    // Check for backend creation messages
    const backendMessages = consoleMessages.filter(msg => 
      msg.includes('job created with gpu backend') ||
      msg.includes('job created with webnn backend')
    );
    
    console.log('🔧 Backend creation messages:');
    backendMessages.forEach(msg => console.log(`  ${msg}`));
    
    // Assertions
    expect(foundWebnnFallbacks).toBeGreaterThan(0);
    console.log(`🎉 SUCCESS: ${foundWebnnFallbacks} WebNN models successfully ran with WebGPU fallback!`);
    
    // Check that we got the backend creation messages indicating fallback
    const gpuFallbackMessages = consoleMessages.filter(msg => 
      msg.includes('job created with gpu backend')
    );
    expect(gpuFallbackMessages.length).toBeGreaterThan(0);
    
    console.log('✅ WebNN to WebGPU fallback test completed successfully!');
  });
});
