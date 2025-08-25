import { test, expect } from '@playwright/test';

test.describe('Simple WebGPU Model Verification', () => {
  test('should verify WebNN models work with WebGPU fallback', async ({ page }) => {
    test.setTimeout(60000);
    
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track backend creation messages
    const backendMessages = [];
    const completedTasks = [];
    
    page.on('console', msg => {
      const text = msg.text();
      
      if (text.includes('job created with') && text.includes('backend')) {
        backendMessages.push(text);
        console.log(`🔧 ${text}`);
      }
      
      if (text.includes('Task completed:') && text.includes('jobType:')) {
        try {
          const taskMatch = text.match(/taskId: (task_\w+), jobType: (\w+), duration: (\d+)/);
          if (taskMatch) {
            completedTasks.push({
              taskId: taskMatch[1],
              jobType: taskMatch[2], 
              duration: parseInt(taskMatch[3])
            });
            console.log(`✅ Task completed: ${taskMatch[2]} in ${taskMatch[3]}ms`);
          }
        } catch (e) {}
      }
    });

    // Wait for page to fully load
    await page.waitForSelector('button[onclick="runRealWorkloadTest()"]', { timeout: 10000 });
    
    // Wait a bit more for all scripts to load
    await page.waitForTimeout(2000);
    
    // Test capability detection first
    console.log('🔍 Testing browser capabilities...');
    const capabilities = await page.evaluate(() => {
      return {
        webgpu: !!navigator.gpu,
        webnn: !!navigator.ml,
        aiModelJobs: !!window.AIModelJobs
      };
    });
    
    console.log('📋 Browser capabilities:', capabilities);
    
    // Test WebGPU device availability
    const webgpuTest = await page.evaluate(async () => {
      try {
        if (!navigator.gpu) return { available: false, reason: 'WebGPU not supported' };
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { available: false, reason: 'No adapter' };
        
        const device = await adapter.requestDevice();
        return { available: true, adapter: !!adapter, device: !!device };
      } catch (error) {
        return { available: false, reason: error.message };
      }
    });
    
    console.log('🖥️ WebGPU test:', webgpuTest);
    
    // If AIModelJobs is available, test individual model creation
    if (capabilities.aiModelJobs) {
      console.log('🧪 Testing individual model creation...');
      
      const modelTests = await page.evaluate(() => {
        const results = [];
        const models = ['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'];
        
        for (const modelName of models) {
          try {
            let job;
            switch (modelName) {
              case 'FaceFormer':
                job = new window.AIModelJobs.FaceFormerJob('gpu');
                break;
              case 'RSMT':
                job = new window.AIModelJobs.RSMTJob('gpu');
                break;
              case 'Kokoro':
                job = new window.AIModelJobs.KokoroJob('gpu');
                break;
              case 'TinyLlama':
                job = new window.AIModelJobs.TinyLlamaJob('gpu');
                break;
            }
            
            results.push({
              model: modelName,
              success: true,
              backend: job.backend || 'default',
              jobType: job.constructor.name
            });
          } catch (error) {
            results.push({
              model: modelName,
              success: false,
              error: error.message
            });
          }
        }
        
        return results;
      });
      
      console.log('📊 Individual model test results:');
      modelTests.forEach(result => {
        if (result.success) {
          console.log(`✅ ${result.model}: Created with ${result.backend} backend`);
        } else {
          console.log(`❌ ${result.model}: Failed - ${result.error}`);
        }
      });
      
      // Verify WebNN models are using GPU backend
      const gpuBackendModels = modelTests.filter(r => r.success && r.backend === 'gpu');
      console.log(`🔄 Models using GPU backend: ${gpuBackendModels.length}/4`);
    }
    
    // Run the workload test to see models in action
    console.log('🚀 Starting workload test...');
    await page.click('button[onclick="runRealWorkloadTest()"]');
    
    // Wait for WebNN models to be created with WebGPU backend
    console.log('⏳ Waiting for WebNN models to use WebGPU fallback...');
    
    const maxWaitTime = 15000; // 15 seconds
    const startTime = Date.now();
    let webnnModelsFound = [];
    
    while (Date.now() - startTime < maxWaitTime) {
      webnnModelsFound = backendMessages.filter(msg => 
        (msg.includes('FaceFormer') || msg.includes('RSMT') || 
         msg.includes('Kokoro') || msg.includes('TinyLlama')) &&
        msg.includes('gpu backend')
      );
      
      if (webnnModelsFound.length >= 4) { // All 4 model types found
        console.log(`🎉 Found all WebNN model types using WebGPU fallback!`);
        break;
      }
      
      await page.waitForTimeout(1000);
    }
    
    // Final results
    console.log(`\n📊 Final Results:`);
    console.log(`- WebGPU Available: ${capabilities.webgpu}`);
    console.log(`- WebNN Available: ${capabilities.webnn}`);
    console.log(`- WebGPU Device Working: ${webgpuTest.available}`);
    console.log(`- Total Backend Messages: ${backendMessages.length}`);
    console.log(`- WebNN Models with GPU Backend: ${webnnModelsFound.length}`);
    console.log(`- Completed Tasks: ${completedTasks.length}`);
    
    // Check each WebNN model type
    const modelTypes = ['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'];
    const foundModelTypes = modelTypes.filter(type => 
      backendMessages.some(msg => msg.includes(type) && msg.includes('gpu backend'))
    );
    
    console.log(`\n🔍 WebNN Model Types Using WebGPU:`);
    modelTypes.forEach(type => {
      const found = foundModelTypes.includes(type);
      console.log(`- ${type}: ${found ? '✅' : '❌'}`);
    });
    
    // Assertions
    expect(capabilities.webgpu).toBe(true);
    expect(webgpuTest.available).toBe(true);
    expect(webnnModelsFound.length).toBeGreaterThan(0);
    expect(foundModelTypes.length).toBeGreaterThanOrEqual(3); // At least 3 of 4 model types
    
    console.log(`\n🎉 SUCCESS: ${foundModelTypes.length}/4 WebNN model types successfully using WebGPU fallback!`);
    
    if (completedTasks.length > 0) {
      console.log(`\n✅ BONUS: ${completedTasks.length} tasks completed successfully:`);
      completedTasks.forEach(task => console.log(`  - ${task.jobType}: ${task.duration}ms`));
    }
  });
});
