import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

test('Collect All AI Model Outputs', async ({ page }) => {
  const allModelOutputs = [];
  const detailedOutputs = {};
  
  // Enhanced console capture with model detection
  page.on('console', msg => {
    const text = msg.text();
    const timestamp = new Date().toISOString();
    
    const logEntry = {
      type: msg.type(),
      text: text,
      timestamp: timestamp
    };
    
    allModelOutputs.push(logEntry);
    
    // Detect which model this output belongs to
    const modelPatterns = {
      'TinyLlama': /tinyllama|tiny.*llama|language.*model|text.*generation/i,
      'DiabloGPT': /diablogpt|diablo.*gpt|conversational/i,
      'Whisper': /whisper|speech.*recognition|audio.*text/i,
      'VAD': /vad|voice.*activity|voice.*detection/i,
      'Kokoro': /kokoro|tts|text.*speech|speech.*synthesis/i,
      'SpeechT5': /speecht5|speech.*t5|speech.*transformer/i,
      'FaceFormer': /faceformer|face.*animation|facial.*animation/i,
      'Audio2Gesture': /audio2gesture|audio.*gesture|gesture.*generation/i,
      'RSMT': /rsmt|motion.*transition|stylized.*motion/i,
      'DeepMimic': /deepmimic|deep.*mimic|reinforcement.*learning/i,
      'WASMMatrix': /wasm.*matrix|matrix.*computation/i,
      'WASMPrime': /wasm.*prime|prime.*calculation/i,
      'WASMFractal': /wasm.*fractal|fractal.*generation/i
    };
    
    // Check for neural network and AI indicators
    const aiIndicators = [
      'neural_network_used',
      'model_output',
      'inference_result',
      'webgpu',
      'webnn',
      'tensor',
      'execution_provider',
      'layers_processed',
      'attention_weights',
      'transformer_blocks',
      'gpu_memory',
      'model_path',
      'quantization',
      'embedding',
      'mel_spectrogram',
      'vocoder',
      'conv1d_layers',
      'lstm_layers',
      'gru_layers',
      'policy_network',
      'value_network',
      'reinforcement_learning'
    ];
    
    const hasAIIndicators = aiIndicators.some(indicator => 
      text.toLowerCase().includes(indicator.toLowerCase())
    );
    
    if (hasAIIndicators) {
      logEntry.isAIOutput = true;
      logEntry.aiIndicators = aiIndicators.filter(indicator => 
        text.toLowerCase().includes(indicator.toLowerCase())
      );
    }
    
    // Categorize by model
    for (const [modelName, pattern] of Object.entries(modelPatterns)) {
      if (pattern.test(text)) {
        if (!detailedOutputs[modelName]) {
          detailedOutputs[modelName] = [];
        }
        detailedOutputs[modelName].push(logEntry);
      }
    }
    
    console.log(`[${msg.type()}] ${text}`);
  });

  console.log('🚀 Starting comprehensive AI model output collection...');
  
  // Navigate to the test page
  await page.goto('http://localhost:8000/web_viewer/');
  
  // Wait for page to load
  await page.waitForTimeout(5000);
  
  console.log('📡 Page loaded, listening for AI model outputs...');
  
  // Click anywhere to potentially trigger model loading
  await page.click('body');
  await page.waitForTimeout(2000);
  
  // Try to trigger various model activities by interacting with elements
  const interactions = [
    async () => {
      // Look for and click any model-related buttons
      const buttons = await page.$$('button');
      console.log(`Found ${buttons.length} buttons to interact with`);
      
      for (let i = 0; i < Math.min(buttons.length, 20); i++) {
        try {
          const button = buttons[i];
          const text = await button.textContent();
          console.log(`Clicking button: ${text}`);
          await button.click();
          await page.waitForTimeout(3000);
        } catch (error) {
          console.log(`Button click failed: ${error.message}`);
        }
      }
    },
    
    async () => {
      // Look for input fields and fill them
      const inputs = await page.$$('input, textarea');
      console.log(`Found ${inputs.length} input fields`);
      
      for (let i = 0; i < Math.min(inputs.length, 10); i++) {
        try {
          const input = inputs[i];
          await input.fill('Test input for AI model processing');
          await page.waitForTimeout(1000);
        } catch (error) {
          console.log(`Input fill failed: ${error.message}`);
        }
      }
    },
    
    async () => {
      // Try keyboard shortcuts that might trigger models
      await page.keyboard.press('Enter');
      await page.waitForTimeout(1000);
      await page.keyboard.press('Space');
      await page.waitForTimeout(1000);
    },
    
    async () => {
      // Scroll and wait to trigger any lazy-loaded models
      await page.mouse.wheel(0, 1000);
      await page.waitForTimeout(2000);
      await page.mouse.wheel(0, -1000);
      await page.waitForTimeout(2000);
    }
  ];
  
  // Execute all interactions
  for (let i = 0; i < interactions.length; i++) {
    console.log(`🔄 Running interaction set ${i + 1}/${interactions.length}`);
    try {
      await interactions[i]();
    } catch (error) {
      console.log(`Interaction ${i + 1} failed: ${error.message}`);
    }
    await page.waitForTimeout(3000);
  }
  
  // Extended waiting period to collect outputs
  console.log('⏳ Extended waiting period for model outputs...');
  for (let i = 0; i < 30; i++) {
    console.log(`Waiting... ${i + 1}/30 (${allModelOutputs.length} messages collected)`);
    await page.waitForTimeout(2000);
  }
  
  // Final collection summary
  const aiOutputs = allModelOutputs.filter(output => output.isAIOutput);
  const uniqueModels = Object.keys(detailedOutputs);
  
  const collectionSummary = {
    timestamp: new Date().toISOString(),
    totalMessages: allModelOutputs.length,
    aiRelatedMessages: aiOutputs.length,
    modelsDetected: uniqueModels.length,
    modelList: uniqueModels,
    detailedOutputsByModel: detailedOutputs,
    allMessages: allModelOutputs,
    aiOutputs: aiOutputs
  };
  
  // Save comprehensive output file
  const outputFile = path.join(process.cwd(), 'comprehensive-model-outputs.json');
  fs.writeFileSync(outputFile, JSON.stringify(collectionSummary, null, 2));
  
  // Create readable summary
  let summaryText = `=== AI Model Output Collection Summary ===\n`;
  summaryText += `Generated: ${collectionSummary.timestamp}\n`;
  summaryText += `Total Messages: ${collectionSummary.totalMessages}\n`;
  summaryText += `AI-Related Messages: ${collectionSummary.aiRelatedMessages}\n`;
  summaryText += `Models Detected: ${collectionSummary.modelsDetected}\n`;
  summaryText += `Model List: ${uniqueModels.join(', ')}\n\n`;
  
  // Detailed breakdown by model
  for (const [modelName, outputs] of Object.entries(detailedOutputs)) {
    summaryText += `\n--- ${modelName} (${outputs.length} messages) ---\n`;
    
    const aiOutputsForModel = outputs.filter(o => o.isAIOutput);
    summaryText += `AI-related outputs: ${aiOutputsForModel.length}\n`;
    
    if (aiOutputsForModel.length > 0) {
      summaryText += `Sample outputs:\n`;
      aiOutputsForModel.slice(0, 3).forEach((output, idx) => {
        summaryText += `  ${idx + 1}. [${output.type}] ${output.text.substring(0, 150)}...\n`;
        if (output.aiIndicators && output.aiIndicators.length > 0) {
          summaryText += `     AI Indicators: ${output.aiIndicators.join(', ')}\n`;
        }
      });
    }
    
    summaryText += `\nAll outputs for ${modelName}:\n`;
    outputs.forEach((output, idx) => {
      summaryText += `  ${idx + 1}. [${output.timestamp}] [${output.type}] ${output.text}\n`;
    });
  }
  
  // Add section for all AI outputs regardless of model
  summaryText += `\n\n=== ALL AI-RELATED OUTPUTS ===\n`;
  aiOutputs.forEach((output, idx) => {
    summaryText += `${idx + 1}. [${output.timestamp}] [${output.type}] ${output.text}\n`;
    if (output.aiIndicators && output.aiIndicators.length > 0) {
      summaryText += `   Indicators: ${output.aiIndicators.join(', ')}\n`;
    }
  });
  
  const summaryFile = path.join(process.cwd(), 'model-outputs-detailed-summary.txt');
  fs.writeFileSync(summaryFile, summaryText);
  
  console.log('\n🎉 COLLECTION COMPLETE!');
  console.log(`📁 Comprehensive data: ${outputFile}`);
  console.log(`📄 Readable summary: ${summaryFile}`);
  console.log(`📊 Stats: ${collectionSummary.totalMessages} total, ${collectionSummary.aiRelatedMessages} AI-related, ${collectionSummary.modelsDetected} models detected`);
  
  // Verify we collected some meaningful data
  expect(allModelOutputs.length).toBeGreaterThan(10);
  
});
