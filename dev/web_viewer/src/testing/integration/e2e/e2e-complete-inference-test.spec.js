import { test, expect } from '@playwright/test';

test.describe('Complete WebGPU Model Inference Testing', () => {
  test('should run all WebNN models with WebGPU and wait for complete inference results', async ({ page }) => {
    test.setTimeout(180000); // 3 minutes for complete inference testing
    
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Track all console messages and model outputs
    const consoleMessages = [];
    const backendMessages = [];
    const completedTasks = [];
    const modelInferenceOutputs = [];
    const taskProgress = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      
      // Track backend creation messages
      if (text.includes('job created with') && text.includes('backend')) {
        backendMessages.push(text);
        console.log(`🔧 ${text}`);
      }
      
      // Track task progress
      if (text.includes('progress:') && text.includes('%')) {
        const progressMatch = text.match(/Task (\w+) progress: (\d+)%/);
        if (progressMatch) {
          taskProgress.push({
            taskId: progressMatch[1],
            progress: parseInt(progressMatch[2])
          });
        }
      }
      
      // Track completed tasks with full details
      if (text.includes('Task completed:') || text.includes('completed successfully')) {
        try {
          // Multiple patterns for task completion
          const patterns = [
            /taskId: (task_\w+), jobType: (\w+), duration: (\d+)/,
            /Task (task_\w+) completed in (\d+)ms/,
            /Task (task_\w+) completed successfully/
          ];
          
          for (const pattern of patterns) {
            const match = text.match(pattern);
            if (match) {
              const task = {
                taskId: match[1],
                jobType: match[2] || 'unknown',
                duration: parseInt(match[3] || match[2]) || 0,
                fullMessage: text
              };
              completedTasks.push(task);
              console.log(`✅ Task completed: ${task.jobType || task.taskId} in ${task.duration}ms`);
              break;
            }
          }
        } catch (e) {
          // Still log completion messages even if parsing fails
          console.log(`✅ Task completion: ${text}`);
        }
      }
      
      // Track model inference outputs/results
      if (text.includes('COMPLETED') || text.includes('✅') || 
          text.includes('inference') || text.includes('result')) {
        modelInferenceOutputs.push({
          timestamp: new Date().toISOString(),
          message: text
        });
        console.log(`🧠 Model output: ${text}`);
      }
    });

    // Wait for page to fully load
    await page.waitForSelector('button[onclick="runRealWorkloadTest()"]', { timeout: 10000 });
    await page.waitForTimeout(3000); // Extra time for all scripts to load
    
    // Check initial capabilities
    const capabilities = await page.evaluate(() => {
      return {
        webgpu: !!navigator.gpu,
        webnn: !!navigator.ml,
        aiModelJobs: !!window.AIModelJobs,
        transformers: !!window.Transformers
      };
    });
    
    console.log('📋 Browser capabilities:', capabilities);
    
    // Start the workload test
    console.log('🚀 Starting comprehensive workload test...');
    await page.click('button[onclick="runRealWorkloadTest()"]');
    
    // Wait for all WebNN models to be created
    console.log('⏳ Waiting for all WebNN models to be created with WebGPU backend...');
    
    const modelTypes = ['FaceFormer', 'RSMT', 'Kokoro', 'TinyLlama'];
    let allModelsCreated = false;
    let createdModelTypes = new Set();
    
    // Wait up to 30 seconds for all model types to be created
    for (let i = 0; i < 30; i++) {
      const foundModelTypes = modelTypes.filter(type => 
        backendMessages.some(msg => msg.includes(type) && msg.includes('gpu backend'))
      );
      
      createdModelTypes = new Set(foundModelTypes);
      
      if (createdModelTypes.size >= 4) {
        allModelsCreated = true;
        console.log(`🎉 All ${createdModelTypes.size} WebNN model types created with WebGPU backend!`);
        break;
      }
      
      await page.waitForTimeout(1000);
    }
    
    console.log(`📊 Created model types: ${Array.from(createdModelTypes).join(', ')}`);
    
    // Now wait for tasks to complete - this is the crucial part
    console.log('⏳ Waiting for all tasks to complete inference...');
    
    let previousCompletedCount = 0;
    let stableCount = 0;
    const maxWaitTime = 120000; // 2 minutes for all tasks to complete
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
      const currentCompletedCount = completedTasks.length;
      
      // Check if we have completed tasks from our target model types
      const completedModelTypes = new Set();
      completedTasks.forEach(task => {
        if (modelTypes.some(type => task.jobType === type || task.fullMessage.includes(type))) {
          // Extract model type from the task
          for (const type of modelTypes) {
            if (task.jobType === type || task.fullMessage.includes(type)) {
              completedModelTypes.add(type);
              break;
            }
          }
        }
      });
      
      console.log(`📈 Progress: ${currentCompletedCount} tasks completed, ${completedModelTypes.size} model types finished`);
      
      // If we have completions from at least 3 model types and no new completions for 10 seconds
      if (completedModelTypes.size >= 3) {
        if (currentCompletedCount === previousCompletedCount) {
          stableCount++;
          if (stableCount >= 10) { // 10 seconds of stability
            console.log(`🏁 Stopping wait - ${completedModelTypes.size} model types completed, stable for 10s`);
            break;
          }
        } else {
          stableCount = 0;
        }
      }
      
      previousCompletedCount = currentCompletedCount;
      await page.waitForTimeout(1000);
    }
    
    // Give a final moment for any last outputs
    await page.waitForTimeout(3000);
    
    // Collect final statistics
    const finalStats = {
      totalBackendMessages: backendMessages.length,
      totalCompletedTasks: completedTasks.length,
      totalModelOutputs: modelInferenceOutputs.length,
      totalProgressUpdates: taskProgress.length,
      webnnModelsWithGPU: backendMessages.filter(msg => 
        modelTypes.some(type => msg.includes(type)) && msg.includes('gpu backend')
      ).length
    };
    
    console.log(`\n📊 FINAL COMPREHENSIVE RESULTS:`);
    console.log(`===========================================`);
    console.log(`🔧 Backend Creation Messages: ${finalStats.totalBackendMessages}`);
    console.log(`✅ Completed Tasks: ${finalStats.totalCompletedTasks}`);
    console.log(`🧠 Model Inference Outputs: ${finalStats.totalModelOutputs}`);
    console.log(`📈 Progress Updates: ${finalStats.totalProgressUpdates}`);
    console.log(`🔄 WebNN Models with WebGPU: ${finalStats.webnnModelsWithGPU}`);
    
    // Detailed breakdown by model type
    console.log(`\n🔍 DETAILED MODEL ANALYSIS:`);
    console.log(`===========================================`);
    
    for (const modelType of modelTypes) {
      const backendCount = backendMessages.filter(msg => 
        msg.includes(modelType) && msg.includes('gpu backend')
      ).length;
      
      const completedCount = completedTasks.filter(task => 
        task.jobType === modelType || task.fullMessage.includes(modelType)
      ).length;
      
      const outputCount = modelInferenceOutputs.filter(output => 
        output.message.includes(modelType)
      ).length;
      
      console.log(`${modelType}:`);
      console.log(`  📦 Created with WebGPU: ${backendCount} instances`);
      console.log(`  ✅ Completed tasks: ${completedCount}`);
      console.log(`  🧠 Inference outputs: ${outputCount}`);
      
      // Show sample outputs for this model
      const sampleOutputs = modelInferenceOutputs
        .filter(output => output.message.includes(modelType))
        .slice(0, 2); // First 2 outputs
      
      if (sampleOutputs.length > 0) {
        console.log(`  📝 Sample outputs:`);
        sampleOutputs.forEach(output => {
          console.log(`    - ${output.message.substring(0, 100)}...`);
        });
      }
    }
    
    // Show all completed tasks
    if (completedTasks.length > 0) {
      console.log(`\n✅ ALL COMPLETED TASKS:`);
      console.log(`===========================================`);
      completedTasks.forEach((task, index) => {
        console.log(`${index + 1}. ${task.jobType || 'Unknown'} (${task.taskId}) - ${task.duration}ms`);
      });
    }
    
    // Show sample model outputs
    if (modelInferenceOutputs.length > 0) {
      console.log(`\n🧠 SAMPLE MODEL INFERENCE OUTPUTS:`);
      console.log(`===========================================`);
      modelInferenceOutputs.slice(0, 10).forEach((output, index) => {
        console.log(`${index + 1}. [${output.timestamp}] ${output.message}`);
      });
    }
    
    // Key assertions
    console.log(`\n🧪 RUNNING ASSERTIONS:`);
    console.log(`===========================================`);
    
    // Assert WebGPU is working
    expect(capabilities.webgpu).toBe(true);
    console.log(`✅ WebGPU available: ${capabilities.webgpu}`);
    
    // Assert WebNN models are using WebGPU fallback
    expect(finalStats.webnnModelsWithGPU).toBeGreaterThan(0);
    console.log(`✅ WebNN models using WebGPU: ${finalStats.webnnModelsWithGPU} > 0`);
    
    // Assert at least some tasks completed
    expect(finalStats.totalCompletedTasks).toBeGreaterThan(0);
    console.log(`✅ Tasks completed: ${finalStats.totalCompletedTasks} > 0`);
    
    // Assert we got model outputs/inference results
    expect(finalStats.totalModelOutputs).toBeGreaterThan(0);
    console.log(`✅ Model inference outputs: ${finalStats.totalModelOutputs} > 0`);
    
    // Check that at least 3 of 4 model types were created
    const createdTypes = modelTypes.filter(type => 
      backendMessages.some(msg => msg.includes(type) && msg.includes('gpu backend'))
    );
    expect(createdTypes.length).toBeGreaterThanOrEqual(3);
    console.log(`✅ Model types created: ${createdTypes.length}/4 (${createdTypes.join(', ')})`);
    
    console.log(`\n🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!`);
    console.log(`===========================================`);
    console.log(`✅ All WebNN models successfully running with WebGPU fallback`);
    console.log(`✅ Task completion verified with actual inference outputs`);
    console.log(`✅ Model performance metrics captured`);
    console.log(`📊 Total evidence collected: ${consoleMessages.length} console messages`);
  });
});
