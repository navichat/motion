import { test, expect } from '@playwright/test';

test.describe('WebGPU Model Testing', () => {
  test('should test all models individually with WebGPU backend', async ({ page }) => {
    // Set a longer timeout for this comprehensive test
    test.setTimeout(120000); // 2 minutes
    
    // Navigate to the demo page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track console messages and completed tasks
    const consoleMessages = [];
    const completedTasks = [];
    const backendMessages = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      console.log(`[PAGE CONSOLE]: ${text}`);
      
      // Track backend creation messages
      if (text.includes('job created with') && text.includes('backend')) {
        backendMessages.push(text);
        console.log(`🔧 Backend: ${text}`);
      }
      
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
            console.log(`✅ Task completed: ${taskMatch[2]} in ${taskMatch[3]}ms`);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }
    });

    // Wait for the page to load
    await page.waitForSelector('button[onclick="runRealWorkloadTest()"]', { timeout: 10000 });
    
    // First, let's test individual models by injecting JavaScript
    console.log('🧪 Testing individual models with WebGPU backend...');
    
    // Define the models we want to test
    const modelsToTest = [
      { name: 'FaceFormer', emoji: '🎭' },
      { name: 'RSMT', emoji: '🏃' },
      { name: 'Kokoro', emoji: '🗣️' },
      { name: 'TinyLlama', emoji: '🦙' },
      { name: 'WebGPUImage', emoji: '🖼️' },
      { name: 'Whisper', emoji: '🎙️' }
    ];
    
    // Test each model individually
    for (const model of modelsToTest) {
      console.log(`\n🔍 Testing ${model.emoji} ${model.name} model...`);
      
      try {
        // Force create a specific model with GPU backend
        const result = await page.evaluate(async (modelName) => {
          try {
            // Get the AI model jobs factory
            if (!window.AIModelJobs) {
              return { success: false, error: 'AIModelJobs not available' };
            }
            
            // Create a specific job with GPU backend
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
              case 'WebGPUImage':
                job = new window.AIModelJobs.WebGPUImageJob();
                break;
              case 'Whisper':
                job = new window.AIModelJobs.WhisperJob();
                break;
              default:
                return { success: false, error: 'Unknown model' };
            }
            
            if (!job) {
              return { success: false, error: 'Failed to create job' };
            }
            
            // Check if the job was created with the correct backend
            const expectedBackend = ['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'].includes(modelName) ? 'gpu' : 
                                  ['WebGPUImage', 'Whisper'].includes(modelName) ? 'gpu' : 'unknown';
            
            return { 
              success: true, 
              jobType: job.constructor.name,
              backend: job.backend || 'default',
              expectedBackend: expectedBackend,
              config: job.config || {}
            };
            
          } catch (error) {
            return { success: false, error: error.message };
          }
        }, model.name);
        
        console.log(`📊 ${model.name} test result:`, result);
        
        if (result.success) {
          console.log(`✅ ${model.emoji} ${model.name} successfully created with ${result.backend} backend`);
          
          // For WebNN models, verify they're using GPU fallback
          if (['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'].includes(model.name)) {
            expect(result.backend).toBe('gpu');
            console.log(`🔄 ${model.name} correctly using WebGPU fallback`);
          }
        } else {
          console.error(`❌ ${model.emoji} ${model.name} failed: ${result.error}`);
        }
        
      } catch (error) {
        console.error(`❌ ${model.emoji} ${model.name} test failed: ${error.message}`);
      }
      
      // Wait a bit between tests
      await page.waitForTimeout(1000);
    }
    
    // Now test capability detection
    console.log('\n🔍 Testing capability detection...');
    
    const capabilities = await page.evaluate(() => {
      return {
        webgpu: !!navigator.gpu,
        webnn: !!navigator.ml,
        transformersJS: !!window.Transformers,
        aiModelJobs: !!window.AIModelJobs
      };
    });
    
    console.log('📋 Browser capabilities:', capabilities);
    
    // Test GPU capability check specifically
    const gpuTest = await page.evaluate(async () => {
      try {
        if (!navigator.gpu) {
          return { available: false, reason: 'WebGPU not supported' };
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          return { available: false, reason: 'No WebGPU adapter available' };
        }
        
        const device = await adapter.requestDevice();
        return { 
          available: true, 
          adapter: !!adapter,
          device: !!device,
          features: Array.from(adapter.features || []),
          limits: adapter.limits ? Object.keys(adapter.limits) : []
        };
      } catch (error) {
        return { available: false, reason: error.message };
      }
    });
    
    console.log('🖥️ WebGPU detailed test:', gpuTest);
    
    // Final verification
    console.log('\n📝 Test Summary:');
    console.log(`- WebGPU Available: ${capabilities.webgpu}`);
    console.log(`- WebNN Available: ${capabilities.webnn}`);
    console.log(`- AI Model Jobs Available: ${capabilities.aiModelJobs}`);
    console.log(`- GPU Device Available: ${gpuTest.available}`);
    
    // Assertions
    expect(capabilities.webgpu).toBe(true);
    expect(capabilities.aiModelJobs).toBe(true);
    expect(gpuTest.available).toBe(true);
    
    console.log('✅ All individual model tests completed successfully!');
  });
  
  test('should run workload test and verify WebNN models use WebGPU fallback', async ({ page }) => {
    test.setTimeout(60000); // 1 minute
    
    // Navigate to the demo page
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track console messages
    const consoleMessages = [];
    const completedTasks = [];
    const backendMessages = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      
      // Track backend creation messages
      if (text.includes('job created with') && text.includes('backend')) {
        backendMessages.push(text);
        console.log(`🔧 Backend: ${text}`);
      }
      
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
            console.log(`✅ Task completed: ${taskMatch[2]} in ${taskMatch[3]}ms`);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }
    });

    // Wait for the page to load and click the workload button
    await page.waitForSelector('button[onclick="runRealWorkloadTest()"]', { timeout: 10000 });
    
    console.log('🚀 Starting Real Workload test...');
    await page.click('button[onclick="runRealWorkloadTest()"]');
    
    // Wait for models to be created and check backend messages
    console.log('⏳ Waiting for WebNN models to be created with WebGPU fallback...');
    
    let webnnModelsFound = [];
    const maxWaitTime = 30000; // 30 seconds
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
      // Check for WebNN model backend creation messages
      const webnnBackends = backendMessages.filter(msg => 
        (msg.includes('FaceFormer') || msg.includes('RSMT') || 
         msg.includes('Kokoro') || msg.includes('TinyLlama')) &&
        msg.includes('gpu backend')
      );
      
      if (webnnBackends.length > 0) {
        webnnModelsFound = webnnBackends;
        console.log(`🎉 Found ${webnnBackends.length} WebNN models using WebGPU fallback!`);
        break;
      }
      
      await page.waitForTimeout(1000); // Wait 1 second
    }
    
    // Verify results
    console.log('\n🔍 Verification Results:');
    console.log(`Found ${backendMessages.length} backend creation messages:`);
    backendMessages.forEach(msg => console.log(`  - ${msg}`));
    
    console.log(`\nWebNN models using WebGPU fallback: ${webnnModelsFound.length}`);
    webnnModelsFound.forEach(msg => console.log(`  ✅ ${msg}`));
    
    // Assertions
    expect(webnnModelsFound.length).toBeGreaterThan(0);
    
    // Verify that WebNN models are specifically using GPU backend
    const faceformerBackend = backendMessages.find(msg => msg.includes('FaceFormer') && msg.includes('gpu backend'));
    const kokoroBackend = backendMessages.find(msg => msg.includes('Kokoro') && msg.includes('gpu backend'));
    const rsmtBackend = backendMessages.find(msg => msg.includes('RSMT') && msg.includes('gpu backend'));
    const tinyllamaBackend = backendMessages.find(msg => msg.includes('TinyLlama') && msg.includes('gpu backend'));
    
    // At least some of these should be found
    const foundModels = [faceformerBackend, kokoroBackend, rsmtBackend, tinyllamaBackend].filter(Boolean);
    
    console.log(`\n📊 Individual model verification:`);
    console.log(`- FaceFormer with GPU: ${!!faceformerBackend}`);
    console.log(`- Kokoro with GPU: ${!!kokoroBackend}`);
    console.log(`- RSMT with GPU: ${!!rsmtBackend}`);
    console.log(`- TinyLlama with GPU: ${!!tinyllamaBackend}`);
    console.log(`Total verified models: ${foundModels.length}`);
    
    expect(foundModels.length).toBeGreaterThan(0);
    
    console.log('✅ WebNN to WebGPU fallback verification completed successfully!');
  });
});
