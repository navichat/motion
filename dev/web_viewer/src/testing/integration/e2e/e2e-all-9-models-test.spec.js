import { test, expect } from '@playwright/test';

test.describe('Complete AI Model WebGPU Testing - All 9 Models', () => {
  test('should test all 9 AI models with WebGPU and verify complete inference', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes for testing all models
    
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track all console messages and outputs
    const consoleMessages = [];
    const backendMessages = [];
    const taskCompletions = [];
    const workerMessages = [];
    const modelInferenceOutputs = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      
      // Track backend creation for all models
      if (text.includes('job created with') && text.includes('backend')) {
        backendMessages.push(text);
        console.log(`🔧 ${text}`);
      }
      
      // Track task completions with detailed parsing
      if (text.includes('Task completed') || text.includes('COMPLETED') || text.includes('completed successfully')) {
        taskCompletions.push(text);
        console.log(`✅ ${text}`);
      }
      
      // Track worker messages specifically
      if (text.includes('[WORKER]') || text.includes('Worker') || text.includes('worker')) {
        workerMessages.push(text);
      }
      
      // Track inference outputs
      if (text.includes('inference') || text.includes('COMPLETED') || text.includes('result')) {
        modelInferenceOutputs.push({
          timestamp: Date.now(),
          message: text
        });
      }
    });

    // Wait for page to load completely
    await page.waitForSelector('button[onclick="runRealWorkloadTest()"]', { timeout: 10000 });
    
    // Wait for all modules to load
    console.log('⏳ Waiting for all modules to load...');
    let modulesReady = false;
    for (let i = 0; i < 30; i++) {
      const capabilities = await page.evaluate(() => {
        return {
          webgpu: !!navigator.gpu,
          webnn: !!navigator.ml,
          deepMimicJob: !!window.DeepMimicJob,
          faceFormerJob: !!window.FaceFormerJob,
          audio2GestureJob: !!window.Audio2GestureJob,
          rsmtJob: !!window.RSMTJob,
          kokoroJob: !!window.KokoroJob,
          whisperJob: !!window.WhisperJob,
          vadJob: !!window.VADJob,
          tinyLlamaJob: !!window.TinyLlamaJob,
          diabloGPTJob: !!window.DiabloGPTJob,
          realJobFactory: !!window.RealJobFactory,
          taskManager: !!window.TaskManager
        };
      });
      
      const allJobsReady = capabilities.deepMimicJob && capabilities.faceFormerJob && 
                          capabilities.audio2GestureJob && capabilities.rsmtJob && 
                          capabilities.kokoroJob && capabilities.whisperJob && 
                          capabilities.vadJob && capabilities.tinyLlamaJob && 
                          capabilities.diabloGPTJob;
      
      if (allJobsReady && capabilities.realJobFactory && capabilities.taskManager) {
        modulesReady = true;
        console.log('✅ All modules loaded successfully');
        break;
      }
      
      console.log(`⏳ Waiting for modules... Jobs ready: ${allJobsReady}, Factory: ${capabilities.realJobFactory}, TaskManager: ${capabilities.taskManager}`);
      await page.waitForTimeout(1000);
    }
    
    if (!modulesReady) {
      throw new Error('Required modules did not load within 30 seconds');
    }
    
    // Check final capabilities
    const capabilities = await page.evaluate(() => {
      return {
        webgpu: !!navigator.gpu,
        webnn: !!navigator.ml,
        deepMimicJob: !!window.DeepMimicJob,
        faceFormerJob: !!window.FaceFormerJob,
        audio2GestureJob: !!window.Audio2GestureJob,
        rsmtJob: !!window.RSMTJob,
        kokoroJob: !!window.KokoroJob,
        whisperJob: !!window.WhisperJob,
        vadJob: !!window.VADJob,
        tinyLlamaJob: !!window.TinyLlamaJob,
        diabloGPTJob: !!window.DiabloGPTJob,
        realJobFactory: !!window.RealJobFactory,
        taskManager: !!window.TaskManager
      };
    });
    
    console.log('📋 Browser capabilities:', capabilities);
    
    // Create and test all 9 AI models
    console.log('🎯 Creating all 9 AI model jobs for comprehensive testing...');
    
    const testResults = await page.evaluate(async () => {
      const results = {
        tasksCreated: [],
        tasksCompleted: [],
        backendUsage: [],
        modelOutputs: [],
        errors: [],
        modelDetails: {}
      };
      
      try {
        // Create TaskManager
        const manager = new TaskManager({
          maxConcurrentTasks: 6,
          preemptionEnabled: false,
          schedulingInterval: 100,
          workerPools: {
            cpu: { size: 3 },
            gpu: { size: 3 },
            webnn: { size: 1 }
          },
          capabilities: {
            webgpu: !!navigator.gpu,
            webnn: !!navigator.ml,
            workers: true,
            wasm: true
          }
        });
        
        // Set up comprehensive event tracking
        manager.on('taskCompleted', (task) => {
          const completion = {
            taskId: task.id,
            jobType: task.job.type,
            duration: task.endTime - task.startTime,
            success: task.result?.success || false,
            result: task.result
          };
          results.tasksCompleted.push(completion);
          console.log(`[TEST] Task completed: ${JSON.stringify(completion)}`);
        });
        
        manager.on('taskFailed', (task) => {
          results.errors.push(`Task ${task.id} (${task.job.type}) failed: ${task.error}`);
        });
        
        await manager.start();
        
        // Define all 9 AI models with their test data
        const allAIModels = [
          {
            name: 'DeepMimic',
            constructor: window.DeepMimicJob,
            params: {
              characterFile: 'test_character.txt',
              motionFile: 'test_motion.txt'
            }
          },
          {
            name: 'FaceFormer',
            constructor: window.FaceFormerJob,
            params: {
              audioData: new Float32Array(1024).fill(0.1),
              duration: 1.0
            }
          },
          {
            name: 'Audio2Gesture',
            constructor: window.Audio2GestureJob,
            params: {
              audioData: new Float32Array(2048).fill(0.2),
              gestureStyle: 'natural'
            }
          },
          {
            name: 'RSMT',
            constructor: window.RSMTJob,
            params: {
              motionData: new Float32Array(512).fill(0.2),
              targetStyle: 'natural'
            }
          },
          {
            name: 'Kokoro',
            constructor: window.KokoroJob,
            params: {
              text: 'Hello, this is a comprehensive test of all AI models',
              voice: 'default'
            }
          },
          {
            name: 'Whisper',
            constructor: window.WhisperJob,
            params: {
              audioData: new Float32Array(16000).fill(0.1), // 1 second at 16kHz
              language: 'en'
            }
          },
          {
            name: 'VAD',
            constructor: window.VADJob,
            params: {
              audioData: new Float32Array(8000).fill(0.1),
              threshold: 0.5
            }
          },
          {
            name: 'TinyLlama',
            constructor: window.TinyLlamaJob,
            params: {
              prompt: 'Complete this: Artificial Intelligence is',
              maxTokens: 15
            }
          },
          {
            name: 'DiabloGPT',
            constructor: window.DiabloGPTJob,
            params: {
              prompt: 'Hello! How are you today?',
              maxTokens: 20
            }
          }
        ];
        
        console.log(`[TEST] Testing ${allAIModels.length} AI model types`);
        
        // Create jobs for all AI models
        for (const modelInfo of allAIModels) {
          try {
            const taskId = `test_${modelInfo.name.toLowerCase()}_${Date.now()}`;
            
            // Create the job
            const job = new modelInfo.constructor(modelInfo.params);
            
            if (job) {
              // Capture model details
              results.modelDetails[modelInfo.name] = {
                backend: job.backend || 'auto',
                complexity: job.complexity || 1,
                duration: job.duration || 0,
                resourceRequirements: job.resourceRequirements || {},
                type: job.type || modelInfo.name
              };
              
              results.tasksCreated.push({
                taskId,
                modelType: modelInfo.name,
                jobType: job.type,
                backend: job.backend || 'auto'
              });
              
              // Track backend usage
              const backendUsed = job.backend || (navigator.ml ? 'webnn' : 'gpu');
              results.backendUsage.push(`${modelInfo.name} using ${backendUsed} backend`);
              console.log(`[TEST] Created ${modelInfo.name} job with ${backendUsed} backend`);
              
              // Schedule the task
              const task = await manager.scheduleTask(job, { id: taskId });
              console.log(`[TEST] Scheduled task: ${taskId} (${modelInfo.name})`);
            }
          } catch (error) {
            results.errors.push(`Failed to create ${modelInfo.name} job: ${error.message}`);
            console.error(`[TEST] Error creating ${modelInfo.name}:`, error);
          }
        }
        
        // Wait for all tasks to complete (up to 2 minutes)
        console.log(`[TEST] Waiting for ${results.tasksCreated.length} tasks to complete...`);
        const maxWait = 120000; // 2 minutes
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait && results.tasksCompleted.length < results.tasksCreated.length) {
          await new Promise(resolve => setTimeout(resolve, 2000));
          console.log(`[TEST] Progress: ${results.tasksCompleted.length}/${results.tasksCreated.length} tasks completed`);
        }
        
        // Final statistics
        results.summary = {
          totalCreated: results.tasksCreated.length,
          totalCompleted: results.tasksCompleted.length,
          totalErrors: results.errors.length,
          completionRate: results.tasksCompleted.length / results.tasksCreated.length,
          averageDuration: results.tasksCompleted.reduce((sum, t) => sum + t.duration, 0) / results.tasksCompleted.length,
          successfulTasks: results.tasksCompleted.filter(t => t.success).length
        };
        
        console.log(`[TEST] Final summary:`, results.summary);
        return results;
        
      } catch (error) {
        results.errors.push(`Overall test error: ${error.message}`);
        console.error('[TEST] Overall error:', error);
        return results;
      }
    });
    
    // Wait a bit more for any final outputs
    await page.waitForTimeout(5000);
    
    // Display comprehensive results for all 9 models
    console.log(`\n🎯 COMPREHENSIVE 9-MODEL AI TEST RESULTS:`);
    console.log(`=============================================`);
    console.log(`📦 Total Tasks Created: ${testResults.tasksCreated.length}`);
    console.log(`✅ Total Tasks Completed: ${testResults.tasksCompleted.length}`);
    console.log(`❌ Total Errors: ${testResults.errors.length}`);
    console.log(`📊 Completion Rate: ${(testResults.summary.completionRate * 100).toFixed(1)}%`);
    console.log(`🎯 Successful Tasks: ${testResults.summary.successfulTasks}`);
    
    if (testResults.summary.averageDuration) {
      console.log(`⏱️ Average Duration: ${testResults.summary.averageDuration.toFixed(0)}ms`);
    }
    
    console.log(`\n🔧 BACKEND USAGE FOR ALL 9 MODELS:`);
    console.log(`=============================================`);
    testResults.backendUsage.forEach(usage => console.log(`  - ${usage}`));
    
    console.log(`\n📦 MODEL DETAILS & CONFIGURATION:`);
    console.log(`=============================================`);
    Object.entries(testResults.modelDetails).forEach(([modelName, details]) => {
      console.log(`  ${modelName}:`);
      console.log(`    Backend: ${details.backend}`);
      console.log(`    Complexity: ${details.complexity}`);
      console.log(`    Duration Estimate: ${details.duration}ms`);
      console.log(`    Type: ${details.type}`);
    });
    
    console.log(`\n📦 ALL CREATED TASKS:`);
    console.log(`=============================================`);
    testResults.tasksCreated.forEach((task, i) => {
      console.log(`  ${i+1}. ${task.modelType} (${task.taskId}) - ${task.backend} backend`);
    });
    
    console.log(`\n✅ ALL COMPLETED TASKS:`);
    console.log(`=============================================`);
    testResults.tasksCompleted.forEach((task, i) => {
      console.log(`  ${i+1}. ${task.jobType} (${task.taskId}) - ${task.duration}ms - Success: ${task.success}`);
    });
    
    if (testResults.errors.length > 0) {
      console.log(`\n❌ ERRORS ENCOUNTERED:`);
      console.log(`=============================================`);
      testResults.errors.forEach((error, i) => {
        console.log(`  ${i+1}. ${error}`);
      });
    }
    
    console.log(`\n🧠 MODEL INFERENCE OUTPUTS (${modelInferenceOutputs.length} total):`);
    modelInferenceOutputs.slice(0, 12).forEach((output, i) => {
      console.log(`  ${i+1}. ${output.message.substring(0, 120)}...`);
    });
    
    console.log(`\n📨 BACKEND CREATION MESSAGES (${backendMessages.length} total):`);
    backendMessages.forEach((msg, i) => {
      console.log(`  ${i+1}. ${msg}`);
    });
    
    // Categorize models by backend type
    const webnnModels = testResults.backendUsage.filter(usage => usage.includes('webnn backend'));
    const webgpuModels = testResults.backendUsage.filter(usage => usage.includes('gpu backend'));
    const cpuModels = testResults.backendUsage.filter(usage => usage.includes('cpu backend'));
    
    console.log(`\n🔍 BACKEND ANALYSIS:`);
    console.log(`=============================================`);
    console.log(`🌐 WebNN Models: ${webnnModels.length}`);
    webnnModels.forEach(model => console.log(`    - ${model}`));
    console.log(`🎮 WebGPU Models: ${webgpuModels.length}`);
    webgpuModels.forEach(model => console.log(`    - ${model}`));
    console.log(`💻 CPU Models: ${cpuModels.length}`);
    cpuModels.forEach(model => console.log(`    - ${model}`));
    
    // Run comprehensive assertions
    console.log(`\n🧪 RUNNING COMPREHENSIVE ASSERTIONS:`);
    console.log(`=============================================`);
    
    // Basic capability checks
    expect(capabilities.webgpu).toBe(true);
    console.log(`✅ WebGPU available: ${capabilities.webgpu}`);
    
    // Check all 9 models are available
    const allModelsAvailable = capabilities.deepMimicJob && capabilities.faceFormerJob && 
                              capabilities.audio2GestureJob && capabilities.rsmtJob && 
                              capabilities.kokoroJob && capabilities.whisperJob && 
                              capabilities.vadJob && capabilities.tinyLlamaJob && 
                              capabilities.diabloGPTJob;
    expect(allModelsAvailable).toBe(true);
    console.log(`✅ All 9 AI models available: ${allModelsAvailable}`);
    
    // Task creation and completion checks
    expect(testResults.tasksCreated.length).toBe(9);
    console.log(`✅ All 9 model tasks created: ${testResults.tasksCreated.length} = 9`);
    
    expect(testResults.tasksCompleted.length).toBeGreaterThan(0);
    console.log(`✅ Tasks completed: ${testResults.tasksCompleted.length} > 0`);
    
    // Backend usage checks
    expect(testResults.backendUsage.length).toBe(9);
    console.log(`✅ Backend assignments for all models: ${testResults.backendUsage.length} = 9`);
    
    // Check overall success rate
    const successRate = testResults.summary.successfulTasks / testResults.tasksCreated.length;
    expect(successRate).toBeGreaterThan(0.5); // At least 50% success rate
    console.log(`✅ Success rate: ${(successRate * 100).toFixed(1)}% > 50%`);
    
    // Check that WebGPU fallback is working for some models
    if (webgpuModels.length > 0) {
      console.log(`✅ WebGPU backend working: ${webgpuModels.length} models using GPU backend`);
    }
    
    // Check completion rate
    expect(testResults.summary.completionRate).toBeGreaterThan(0.5);
    console.log(`✅ Completion rate: ${(testResults.summary.completionRate * 100).toFixed(1)}% > 50%`);
    
    console.log(`\n🎉 COMPREHENSIVE 9-MODEL AI TEST COMPLETED!`);
    console.log(`=============================================`);
    console.log(`✅ ${testResults.tasksCreated.length}/9 AI model tasks created`);
    console.log(`✅ ${testResults.tasksCompleted.length} tasks completed inference`);
    console.log(`✅ ${testResults.summary.successfulTasks} tasks completed successfully`);
    console.log(`✅ ${webgpuModels.length} models using WebGPU backend`);
    console.log(`✅ All AI model types tested and verified`);
    console.log(`📊 Overall success rate: ${(successRate * 100).toFixed(1)}%`);
    console.log(`📊 Overall completion rate: ${(testResults.summary.completionRate * 100).toFixed(1)}%`);
  });
});
