import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

test('Collect All AI Model Outputs', async ({ page }) => {
  const modelOutputs = {};
  const consoleMessages = [];
  
  // Capture all console messages
  page.on('console', msg => {
    const text = msg.text();
    consoleMessages.push({
      type: msg.type(),
      text: text,
      timestamp: new Date().toISOString()
    });
    console.log(`[${msg.type()}] ${text}`);
  });

  // Navigate to the test page
  await page.goto('http://localhost:8000/web_viewer/');
  
  // Wait for initial load
  await page.waitForTimeout(3000);
  
  console.log('\n=== STARTING AI MODEL OUTPUT COLLECTION ===\n');
  
  // Test all AI models with detailed output collection
  const modelTests = [
    {
      name: 'TinyLlama',
      action: async () => {
        console.log('\n--- Testing TinyLlama Language Model ---');
        await page.click('button:has-text("Load TinyLlama")');
        await page.waitForTimeout(8000);
        
        // Try to generate text
        await page.fill('input[placeholder*="text"], textarea[placeholder*="text"], input[type="text"]', 'Hello world');
        await page.click('button:has-text("Generate"), button:has-text("Infer"), button:has-text("Process")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'DiabloGPT',
      action: async () => {
        console.log('\n--- Testing DiabloGPT Conversational Model ---');
        await page.click('button:has-text("Load DiabloGPT"), button:has-text("DiabloGPT")');
        await page.waitForTimeout(8000);
        
        await page.fill('input[placeholder*="message"], input[placeholder*="chat"]', 'How are you?');
        await page.click('button:has-text("Send"), button:has-text("Chat"), button:has-text("Generate")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'Whisper',
      action: async () => {
        console.log('\n--- Testing Whisper Speech Recognition ---');
        await page.click('button:has-text("Load Whisper"), button:has-text("Whisper")');
        await page.waitForTimeout(8000);
        
        // Simulate audio input
        await page.click('button:has-text("Record"), button:has-text("Start"), button:has-text("Audio")');
        await page.waitForTimeout(3000);
        await page.click('button:has-text("Stop"), button:has-text("Process")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'VAD',
      action: async () => {
        console.log('\n--- Testing VAD Voice Activity Detection ---');
        await page.click('button:has-text("VAD"), button:has-text("Voice")');
        await page.waitForTimeout(5000);
        
        await page.click('button:has-text("Detect"), button:has-text("Analyze")');
        await page.waitForTimeout(3000);
      }
    },
    {
      name: 'Kokoro',
      action: async () => {
        console.log('\n--- Testing Kokoro TTS ---');
        await page.click('button:has-text("Kokoro"), button:has-text("TTS")');
        await page.waitForTimeout(8000);
        
        await page.fill('input[placeholder*="text"], textarea[placeholder*="speak"]', 'Hello, this is a test');
        await page.click('button:has-text("Speak"), button:has-text("Generate Audio")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'SpeechT5',
      action: async () => {
        console.log('\n--- Testing SpeechT5 ---');
        await page.click('button:has-text("SpeechT5"), button:has-text("Speech")');
        await page.waitForTimeout(8000);
        
        await page.fill('textarea, input[type="text"]', 'Test speech synthesis');
        await page.click('button:has-text("Synthesize"), button:has-text("Generate")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'FaceFormer',
      action: async () => {
        console.log('\n--- Testing FaceFormer Facial Animation ---');
        await page.click('button:has-text("FaceFormer"), button:has-text("Face")');
        await page.waitForTimeout(8000);
        
        await page.click('button:has-text("Animate"), button:has-text("Generate")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'Audio2Gesture',
      action: async () => {
        console.log('\n--- Testing Audio2Gesture ---');
        await page.click('button:has-text("Audio2Gesture"), button:has-text("Gesture")');
        await page.waitForTimeout(8000);
        
        await page.click('button:has-text("Generate Gesture"), button:has-text("Process")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'RSMT',
      action: async () => {
        console.log('\n--- Testing RSMT Motion Transition ---');
        await page.click('button:has-text("RSMT"), button:has-text("Motion")');
        await page.waitForTimeout(8000);
        
        await page.click('button:has-text("Transition"), button:has-text("Process")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'DeepMimic',
      action: async () => {
        console.log('\n--- Testing DeepMimic ---');
        await page.click('button:has-text("DeepMimic"), button:has-text("Mimic")');
        await page.waitForTimeout(8000);
        
        await page.click('button:has-text("Train"), button:has-text("Simulate")');
        await page.waitForTimeout(5000);
      }
    },
    {
      name: 'WASMMatrix',
      action: async () => {
        console.log('\n--- Testing WASMMatrix Compute ---');
        await page.click('button:has-text("WASM"), button:has-text("Matrix")');
        await page.waitForTimeout(5000);
        
        await page.click('button:has-text("Compute"), button:has-text("Calculate")');
        await page.waitForTimeout(3000);
      }
    },
    {
      name: 'WASMPrime',
      action: async () => {
        console.log('\n--- Testing WASMPrime ---');
        await page.click('button:has-text("Prime")');
        await page.waitForTimeout(5000);
        
        await page.click('button:has-text("Calculate"), button:has-text("Compute")');
        await page.waitForTimeout(3000);
      }
    },
    {
      name: 'WASMFractal',
      action: async () => {
        console.log('\n--- Testing WASMFractal ---');
        await page.click('button:has-text("Fractal")');
        await page.waitForTimeout(5000);
        
        await page.click('button:has-text("Generate"), button:has-text("Render")');
        await page.waitForTimeout(3000);
      }
    }
  ];

  // Run each model test and collect outputs
  for (const modelTest of modelTests) {
    try {
      console.log(`\n🔄 Starting ${modelTest.name} test...`);
      const startTime = Date.now();
      const messagesBefore = consoleMessages.length;
      
      await modelTest.action();
      
      const messagesAfter = consoleMessages.length;
      const modelMessages = consoleMessages.slice(messagesBefore, messagesAfter);
      const duration = Date.now() - startTime;
      
      modelOutputs[modelTest.name] = {
        status: 'completed',
        duration: `${duration}ms`,
        messageCount: modelMessages.length,
        messages: modelMessages,
        outputs: modelMessages.filter(msg => 
          msg.text.includes('neural_network') ||
          msg.text.includes('model_output') ||
          msg.text.includes('inference') ||
          msg.text.includes('result') ||
          msg.text.includes('webgpu') ||
          msg.text.includes('webnn') ||
          msg.text.includes('generated') ||
          msg.text.includes('processed') ||
          msg.text.includes('tensor') ||
          msg.text.includes('execution') ||
          msg.text.includes('completed') ||
          msg.text.includes('success') ||
          msg.text.includes('output')
        )
      };
      
      console.log(`✅ ${modelTest.name} completed in ${duration}ms with ${modelMessages.length} messages`);
      
    } catch (error) {
      console.log(`❌ ${modelTest.name} failed: ${error.message}`);
      modelOutputs[modelTest.name] = {
        status: 'failed',
        error: error.message,
        messages: []
      };
    }
    
    // Wait between tests
    await page.waitForTimeout(2000);
  }

  // Final collection sweep
  console.log('\n🔍 Running final output collection sweep...');
  await page.waitForTimeout(3000);
  
  // Collect any additional outputs
  const finalMessages = consoleMessages.slice(-50); // Last 50 messages
  
  // Save all collected data
  const outputData = {
    timestamp: new Date().toISOString(),
    testDuration: `${Date.now() - test.info().startTime}ms`,
    totalModels: Object.keys(modelOutputs).length,
    totalMessages: consoleMessages.length,
    modelResults: modelOutputs,
    allConsoleMessages: consoleMessages,
    finalCollectionSweep: finalMessages,
    summary: {
      successfulModels: Object.keys(modelOutputs).filter(k => modelOutputs[k].status === 'completed').length,
      failedModels: Object.keys(modelOutputs).filter(k => modelOutputs[k].status === 'failed').length,
      totalOutputMessages: Object.values(modelOutputs).reduce((sum, model) => sum + (model.outputs?.length || 0), 0)
    }
  };

  // Write outputs to file
  const outputFile = path.join(process.cwd(), 'collected-model-outputs.json');
  fs.writeFileSync(outputFile, JSON.stringify(outputData, null, 2));
  
  console.log('\n=== COLLECTION COMPLETE ===');
  console.log(`📁 All model outputs saved to: ${outputFile}`);
  console.log(`📊 Summary: ${outputData.summary.successfulModels}/${outputData.totalModels} models successful`);
  console.log(`📝 Total output messages: ${outputData.summary.totalOutputMessages}`);
  console.log(`🕒 Test duration: ${outputData.testDuration}`);
  
  // Also create a readable summary file
  const summaryFile = path.join(process.cwd(), 'model-outputs-summary.txt');
  let summaryText = `AI Model Output Collection Summary\n`;
  summaryText += `Generated: ${outputData.timestamp}\n`;
  summaryText += `Duration: ${outputData.testDuration}\n`;
  summaryText += `Models Tested: ${outputData.totalModels}\n`;
  summaryText += `Successful: ${outputData.summary.successfulModels}\n`;
  summaryText += `Failed: ${outputData.summary.failedModels}\n\n`;
  
  for (const [modelName, modelData] of Object.entries(modelOutputs)) {
    summaryText += `\n--- ${modelName} ---\n`;
    summaryText += `Status: ${modelData.status}\n`;
    if (modelData.status === 'completed') {
      summaryText += `Duration: ${modelData.duration}\n`;
      summaryText += `Messages: ${modelData.messageCount}\n`;
      summaryText += `Outputs: ${modelData.outputs?.length || 0}\n`;
      
      if (modelData.outputs && modelData.outputs.length > 0) {
        summaryText += `Key Outputs:\n`;
        modelData.outputs.slice(0, 3).forEach(output => {
          summaryText += `  - ${output.text.substring(0, 100)}...\n`;
        });
      }
    } else {
      summaryText += `Error: ${modelData.error}\n`;
    }
  }
  
  fs.writeFileSync(summaryFile, summaryText);
  console.log(`📄 Readable summary saved to: ${summaryFile}`);
  
  // Verify we collected meaningful data
  expect(outputData.totalModels).toBeGreaterThan(10);
  expect(outputData.summary.successfulModels).toBeGreaterThan(8);
});
