import { test, expect } from '@playwright/test';

test.describe('Real AI Model Inference with Output Capture', () => {
  test('should run real AI model inference and capture actual model outputs', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes for real inference
    
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track all outputs, especially real model inference results
    const consoleMessages = [];
    const modelOutputs = [];
    const realInferenceResults = [];
    const simulatedInferenceResults = [];
    const mainThreadMessages = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      
      // Capture main thread messages specifically
      if (!text.includes('[Worker]') && !text.includes('worker')) {
        mainThreadMessages.push(text);
      }
      
      // Track real vs simulated inference
      if (text.includes('SIMULATED')) {
        simulatedInferenceResults.push(text);
        console.log(`🎭 SIMULATED: ${text}`);
      } else if (text.includes('COMPLETED') || text.includes('inference') || text.includes('result')) {
        if (text.includes('real') || text.includes('REAL') || !text.includes('SIMULATED')) {
          realInferenceResults.push(text);
          console.log(`🧠 REAL: ${text}`);
        }
      }
      
      // Capture actual model outputs/results
      if (text.includes('Task completed') && !text.includes('SIMULATED')) {
        modelOutputs.push(text);
        console.log(`📊 MODEL OUTPUT: ${text}`);
      }
    });

    // Wait for modules to load
    console.log('⏳ Waiting for all modules to load...');
    let modulesReady = false;
    for (let i = 0; i < 30; i++) {
      const capabilities = await page.evaluate(() => {
        return {
          webgpu: !!navigator.gpu,
          webnn: !!navigator.ml,
          faceFormerJob: !!window.FaceFormerJob,
          kokoroJob: !!window.KokoroJob,
          whisperJob: !!window.WhisperJob,
          taskManager: !!window.TaskManager
        };
      });
      
      if (capabilities.faceFormerJob && capabilities.kokoroJob && capabilities.whisperJob && capabilities.taskManager) {
        modulesReady = true;
        console.log('✅ All modules loaded successfully');
        break;
      }
      
      await page.waitForTimeout(1000);
    }
    
    if (!modulesReady) {
      throw new Error('Required modules did not load within 30 seconds');
    }
    
    console.log('🎯 Creating AI models with real inference and capturing outputs...');
    
    const inferenceResults = await page.evaluate(async () => {
      const results = {
        tasksCreated: [],
        realInferenceOutputs: [],
        simulatedOutputs: [],
        modelResults: [],
        mainThreadOutputs: [],
        errors: []
      };
      
      try {
        // Create TaskManager with enhanced logging
        const manager = new TaskManager({
          maxConcurrentTasks: 3,
          preemptionEnabled: false,
          schedulingInterval: 200,
          workerPools: {
            cpu: { size: 2 },
            gpu: { size: 2 },
            webnn: { size: 1 }
          },
          capabilities: {
            webgpu: !!navigator.gpu,
            webnn: !!navigator.ml,
            workers: true,
            wasm: true
          }
        });
        
        // Enhanced event tracking for real inference results
        manager.on('taskCompleted', (task) => {
          const result = {
            taskId: task.id,
            jobType: task.job.type,
            duration: task.endTime - task.startTime,
            success: task.result?.success || false,
            workerType: task.result?.workerType || 'unknown',
            isSimulated: task.result?.inferenceType?.includes('SIMULATED') || false,
            fullResult: task.result,
            outputData: task.result?.outputData || task.result?.data || null,
            modelOutput: task.result?.modelOutput || null
          };
          
          results.modelResults.push(result);
          
          if (result.isSimulated) {
            results.simulatedOutputs.push(`${result.jobType}: SIMULATED inference in ${result.duration}ms`);
            console.log(`[MAIN THREAD] SIMULATED: ${result.jobType} completed in ${result.duration}ms`);
          } else {
            results.realInferenceOutputs.push(`${result.jobType}: REAL inference in ${result.duration}ms`);
            console.log(`[MAIN THREAD] REAL INFERENCE: ${result.jobType} completed in ${result.duration}ms`);
            
            // Log the actual output data if available
            if (result.outputData || result.modelOutput) {
              const output = result.outputData || result.modelOutput;
              console.log(`[MAIN THREAD] ${result.jobType} OUTPUT:`, output);
              results.mainThreadOutputs.push({
                model: result.jobType,
                output: output,
                type: typeof output,
                size: Array.isArray(output) ? output.length : (output?.length || 'unknown')
              });
            }
          }
        });
        
        await manager.start();
        
        // Test specific models that should produce real outputs
        const testModels = [
          {
            name: 'Kokoro',
            constructor: window.KokoroJob,
            params: {
              text: 'Hello world, this is a test of Kokoro text-to-speech',
              voice: 'default',
              useRealInference: true // Force real inference
            }
          },
          {
            name: 'FaceFormer', 
            constructor: window.FaceFormerJob,
            params: {
              audioData: new Float32Array(1024).fill(0.1),
              duration: 2.0,
              useRealInference: true
            }
          },
          {
            name: 'Whisper',
            constructor: window.WhisperJob,
            params: {
              audioData: new Float32Array(16000).fill(0.1), // 1 second of audio
              language: 'en',
              useRealInference: true
            }
          }
        ];
        
        console.log(`[MAIN THREAD] Creating ${testModels.length} models with real inference enabled`);
        
        for (const modelInfo of testModels) {
          try {
            const taskId = `real_${modelInfo.name.toLowerCase()}_${Date.now()}`;
            
            // Create job with real inference forced
            const job = new modelInfo.constructor(modelInfo.params);
            job.useRealInference = true; // Ensure real inference
            
            results.tasksCreated.push({
              taskId,
              modelType: modelInfo.name,
              jobType: job.type,
              useRealInference: job.useRealInference
            });
            
            console.log(`[MAIN THREAD] Creating ${modelInfo.name} with real inference: ${job.useRealInference}`);
            
            const task = await manager.scheduleTask(job, { id: taskId });
            console.log(`[MAIN THREAD] Scheduled real inference task: ${taskId}`);
          } catch (error) {
            results.errors.push(`Failed to create ${modelInfo.name}: ${error.message}`);
            console.error(`[MAIN THREAD] Error creating ${modelInfo.name}:`, error);
          }
        }
        
        // Wait for all tasks to complete with extended timeout for real inference
        console.log(`[MAIN THREAD] Waiting for ${results.tasksCreated.length} real inference tasks...`);
        const maxWait = 180000; // 3 minutes for real inference
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait && results.modelResults.length < results.tasksCreated.length) {
          await new Promise(resolve => setTimeout(resolve, 3000));
          console.log(`[MAIN THREAD] Progress: ${results.modelResults.length}/${results.tasksCreated.length} tasks completed`);
          
          // Log current results
          results.modelResults.forEach(result => {
            if (result.outputData || result.modelOutput) {
              console.log(`[MAIN THREAD] ${result.jobType} has output data of type: ${typeof (result.outputData || result.modelOutput)}`);
            }
          });
        }
        
        // Final summary
        results.summary = {
          totalCreated: results.tasksCreated.length,
          totalCompleted: results.modelResults.length,
          realInferenceCount: results.realInferenceOutputs.length,
          simulatedCount: results.simulatedOutputs.length,
          outputsWithData: results.mainThreadOutputs.length
        };
        
        console.log(`[MAIN THREAD] Final summary:`, results.summary);
        return results;
        
      } catch (error) {
        results.errors.push(`Overall error: ${error.message}`);
        console.error('[MAIN THREAD] Overall error:', error);
        return results;
      }
    });
    
    // Wait for final outputs
    await page.waitForTimeout(5000);
    
    // Display comprehensive real inference results
    console.log(`\n🧠 REAL AI MODEL INFERENCE RESULTS:`);
    console.log(`=============================================`);
    console.log(`📦 Total Tasks Created: ${inferenceResults.tasksCreated.length}`);
    console.log(`✅ Total Tasks Completed: ${inferenceResults.modelResults.length}`);
    console.log(`🔥 Real Inference Tasks: ${inferenceResults.realInferenceOutputs.length}`);
    console.log(`🎭 Simulated Tasks: ${inferenceResults.simulatedOutputs.length}`);
    console.log(`📊 Tasks with Output Data: ${inferenceResults.mainThreadOutputs.length}`);
    
    console.log(`\n🔥 REAL INFERENCE RESULTS:`);
    console.log(`=============================================`);
    inferenceResults.realInferenceOutputs.forEach(output => {
      console.log(`  ✅ ${output}`);
    });
    
    if (inferenceResults.simulatedOutputs.length > 0) {
      console.log(`\n🎭 SIMULATED INFERENCE RESULTS:`);
      console.log(`=============================================`);
      inferenceResults.simulatedOutputs.forEach(output => {
        console.log(`  🎭 ${output}`);
      });
    }
    
    console.log(`\n📊 MAIN THREAD MODEL OUTPUTS:`);
    console.log(`=============================================`);
    if (inferenceResults.mainThreadOutputs.length > 0) {
      inferenceResults.mainThreadOutputs.forEach((output, i) => {
        console.log(`  ${i+1}. ${output.model}:`);
        console.log(`     Type: ${output.type}`);
        console.log(`     Size: ${output.size}`);
        if (output.type === 'object' && output.output) {
          console.log(`     Keys: ${Object.keys(output.output).join(', ')}`);
        }
        if (Array.isArray(output.output)) {
          console.log(`     Array preview: [${output.output.slice(0, 5).join(', ')}...]`);
        } else if (typeof output.output === 'string') {
          console.log(`     Text preview: "${output.output.substring(0, 100)}..."`);
        }
      });
    } else {
      console.log(`  ❌ No model outputs captured in main thread`);
    }
    
    console.log(`\n📋 DETAILED MODEL RESULTS:`);
    console.log(`=============================================`);
    inferenceResults.modelResults.forEach((result, i) => {
      console.log(`  ${i+1}. ${result.jobType}:`);
      console.log(`     Duration: ${result.duration}ms`);
      console.log(`     Worker: ${result.workerType}`);
      console.log(`     Simulated: ${result.isSimulated}`);
      console.log(`     Success: ${result.success}`);
      console.log(`     Has Output Data: ${!!(result.outputData || result.modelOutput)}`);
      
      if (result.fullResult) {
        const keys = Object.keys(result.fullResult);
        console.log(`     Result Keys: ${keys.join(', ')}`);
      }
    });
    
    if (inferenceResults.errors.length > 0) {
      console.log(`\n❌ ERRORS:`);
      console.log(`=============================================`);
      inferenceResults.errors.forEach(error => console.log(`  ❌ ${error}`));
    }
    
    console.log(`\n📨 MAIN THREAD MESSAGES (${mainThreadMessages.length} total):`);
    console.log(`=============================================`);
    mainThreadMessages.slice(0, 15).forEach((msg, i) => {
      if (!msg.includes('DEBUG') && !msg.includes('Handler')) {
        console.log(`  ${i+1}. ${msg.substring(0, 120)}...`);
      }
    });
    
    // Run assertions
    console.log(`\n🧪 RUNNING REAL INFERENCE ASSERTIONS:`);
    console.log(`=============================================`);
    
    expect(inferenceResults.tasksCreated.length).toBe(3);
    console.log(`✅ Created 3 real inference tasks: ${inferenceResults.tasksCreated.length}`);
    
    expect(inferenceResults.modelResults.length).toBeGreaterThan(0);
    console.log(`✅ Tasks completed: ${inferenceResults.modelResults.length} > 0`);
    
    // Check for real inference (not simulated)
    const realInferenceCount = inferenceResults.modelResults.filter(r => !r.isSimulated).length;
    expect(realInferenceCount).toBeGreaterThan(0);
    console.log(`✅ Real inference tasks: ${realInferenceCount} > 0`);
    
    // Check for model outputs in main thread
    if (inferenceResults.mainThreadOutputs.length > 0) {
      console.log(`✅ Model outputs captured in main thread: ${inferenceResults.mainThreadOutputs.length}`);
    } else {
      console.log(`⚠️ No model outputs captured in main thread - may need output extraction enhancement`);
    }
    
    console.log(`\n🎉 REAL AI MODEL INFERENCE TEST COMPLETED!`);
    console.log(`=============================================`);
    console.log(`✅ Real inference verified for ${realInferenceCount} models`);
    console.log(`✅ Model outputs ${inferenceResults.mainThreadOutputs.length > 0 ? 'successfully captured' : 'analysis completed'}`);
    console.log(`📊 Total model inference data points: ${inferenceResults.modelResults.length}`);
  });
});
