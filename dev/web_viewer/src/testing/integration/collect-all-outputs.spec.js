import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

test.describe('Avatar AI Model Output Collection', () => {
  test('should collect and save all AI model outputs for inspection', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes
    
    console.log('🤖 Starting COMPREHENSIVE Avatar AI Model Output Collection...');
    console.log('🌐 Navigating to task-manager-demo.html...');
    
    // Data collection structures
    const allConsoleMessages = [];
    const errorMessages = [];
    const modelOutputs = {
      languageModels: { tinyLlama: [], diabloGPT: [] },
      audioProcessing: { whisper: [], vad: [], kokoro: [], speechT5: [] },
      motionModels: { rsmt: [], deepMimic: [], faceFormer: [], audio2Gesture: [] },
      computeModels: { wasmMatrix: [], wasmPrime: [], wasmFractal: [] },
      rawOutputs: [],
      neuralNetworkOutputs: [],
      modelSpecificOutputs: {}
    };
    
    // Enhanced console capture for model outputs
    page.on('console', msg => {
      const timestamp = new Date().toISOString();
      const text = msg.text();
      const logEntry = {
        timestamp,
        type: msg.type(),
        text,
        url: msg.location()?.url || 'unknown'
      };
      
      allConsoleMessages.push(logEntry);
      
      // Check for AI/Neural Network indicators
      const neuralIndicators = [
        'neural_network_used', 'model_output', 'inference_result', 'webgpu', 'webnn',
        'tensor', 'execution_provider', 'layers_processed', 'attention_weights',
        'transformer_blocks', 'gpu_memory', 'model_path', 'quantization', 'embedding',
        'mel_spectrogram', 'vocoder', 'conv1d_layers', 'lstm_layers', 'gru_layers',
        'policy_network', 'value_network', 'reinforcement_learning', 'facial_landmark',
        'blendshape', 'motion_encoder', 'style_encoder', 'attention_mechanism'
      ];
      
      const hasNeuralIndicators = neuralIndicators.some(indicator => 
        text.toLowerCase().includes(indicator.toLowerCase())
      );
      
      if (hasNeuralIndicators) {
        const neuralEntry = { ...logEntry, indicators: neuralIndicators.filter(ind => 
          text.toLowerCase().includes(ind.toLowerCase())
        )};
        modelOutputs.neuralNetworkOutputs.push(neuralEntry);
      }
      
      // Categorize by model type
      const modelPatterns = {
        'TinyLlama': /tinyllama|tiny.*llama/i,
        'DiabloGPT': /diablogpt|diablo.*gpt/i,
        'Whisper': /whisper|speech.*recognition/i,
        'VAD': /vad|voice.*activity/i,
        'Kokoro': /kokoro|tts.*kokoro/i,
        'SpeechT5': /speecht5|speech.*t5/i,
        'FaceFormer': /faceformer|face.*animation/i,
        'Audio2Gesture': /audio2gesture|audio.*gesture/i,
        'RSMT': /rsmt|motion.*transition/i,
        'DeepMimic': /deepmimic|deep.*mimic/i,
        'WASMMatrix': /wasm.*matrix/i,
        'WASMPrime': /wasm.*prime/i,
        'WASMFractal': /wasm.*fractal/i
      };
      
      for (const [modelName, pattern] of Object.entries(modelPatterns)) {
        if (pattern.test(text)) {
          if (!modelOutputs.modelSpecificOutputs[modelName]) {
            modelOutputs.modelSpecificOutputs[modelName] = [];
          }
          modelOutputs.modelSpecificOutputs[modelName].push(logEntry);
        }
      }
      
      // Always add to raw outputs
      modelOutputs.rawOutputs.push(logEntry);
      
      console.log(`[COLLECT] [${msg.type()}] ${text}`);
    });
    
    // Error capture
    page.on('pageerror', error => {
      const errorEntry = {
        timestamp: new Date().toISOString(),
        message: error.message,
        stack: error.stack
      };
      errorMessages.push(errorEntry);
      console.log(`[ERROR] ${error.message}`);
    });
    
    // Navigate and start collection
    await page.goto('http://localhost:8080/dev/web_viewer/demos/html-tests/task-manager-demo.html');
    await page.bringToFront();
    
    console.log('📡 Page loaded, starting AI model output collection...');
    await page.waitForTimeout(5000);
    
    // Trigger model loading and inference
    console.log('🔄 Triggering AI model activities...');
    
    // Click various elements to trigger models
    try {
      await page.click('body');
      await page.waitForTimeout(2000);
      
      // Look for buttons to click
      const buttons = await page.$$('button');
      console.log(`Found ${buttons.length} buttons to interact with`);
      
      for (let i = 0; i < Math.min(buttons.length, 30); i++) {
        try {
          const button = buttons[i];
          const buttonText = await button.textContent();
          if (buttonText) {
            console.log(`Clicking button: "${buttonText.trim()}"`);
            await button.click();
            await page.waitForTimeout(3000);
          }
        } catch (error) {
          console.log(`Button ${i} click failed: ${error.message}`);
        }
      }
      
      // Try other interactions
      await page.keyboard.press('Enter');
      await page.waitForTimeout(2000);
      
      // Scroll to trigger lazy loading
      await page.mouse.wheel(0, 1000);
      await page.waitForTimeout(2000);
      
    } catch (error) {
      console.log(`Interaction error: ${error.message}`);
    }
    
    // Extended collection period
    console.log('⏳ Extended collection period for all AI model outputs...');
    for (let i = 0; i < 60; i++) {
      console.log(`Collecting... ${i + 1}/60 (${allConsoleMessages.length} total messages, ${modelOutputs.neuralNetworkOutputs.length} neural outputs)`);
      await page.waitForTimeout(2000);
    }
    
    // Final analysis and file generation
    const collectionSummary = {
      timestamp: new Date().toISOString(),
      collection: {
        totalMessages: allConsoleMessages.length,
        errorMessages: errorMessages.length,
        neuralNetworkOutputs: modelOutputs.neuralNetworkOutputs.length,
        modelsDetected: Object.keys(modelOutputs.modelSpecificOutputs).length,
        modelList: Object.keys(modelOutputs.modelSpecificOutputs)
      },
      outputs: modelOutputs,
      errors: errorMessages,
      allMessages: allConsoleMessages
    };
    
    // Save comprehensive JSON file
    const jsonFile = path.join(process.cwd(), 'collected-ai-model-outputs.json');
    fs.writeFileSync(jsonFile, JSON.stringify(collectionSummary, null, 2));
    
    // Create detailed text report
    let report = `AI MODEL OUTPUT COLLECTION REPORT\n`;
    report += `=====================================\n`;
    report += `Generated: ${collectionSummary.timestamp}\n`;
    report += `Total Messages: ${collectionSummary.collection.totalMessages}\n`;
    report += `Neural Network Outputs: ${collectionSummary.collection.neuralNetworkOutputs}\n`;
    report += `Models Detected: ${collectionSummary.collection.modelsDetected}\n`;
    report += `Model Types: ${collectionSummary.collection.modelList.join(', ')}\n`;
    report += `Errors: ${collectionSummary.collection.errorMessages}\n\n`;
    
    // Neural Network Outputs Section
    if (modelOutputs.neuralNetworkOutputs.length > 0) {
      report += `NEURAL NETWORK OUTPUTS (${modelOutputs.neuralNetworkOutputs.length} entries)\n`;
      report += `=================================\n`;
      modelOutputs.neuralNetworkOutputs.forEach((output, idx) => {
        report += `${idx + 1}. [${output.timestamp}] [${output.type}]\n`;
        report += `   Text: ${output.text}\n`;
        report += `   Indicators: ${output.indicators.join(', ')}\n\n`;
      });
    }
    
    // Model-Specific Outputs
    for (const [modelName, outputs] of Object.entries(modelOutputs.modelSpecificOutputs)) {
      report += `\\n${modelName} OUTPUTS (${outputs.length} entries)\\n`;
      report += `${'='.repeat(modelName.length + 20)}\\n`;
      outputs.forEach((output, idx) => {
        report += `${idx + 1}. [${output.timestamp}] [${output.type}]\\n`;
        report += `   ${output.text}\\n\\n`;
      });
    }
    
    // All Raw Outputs Section
    report += `\\nALL RAW CONSOLE OUTPUTS (${allConsoleMessages.length} entries)\\n`;
    report += `======================================\\n`;
    allConsoleMessages.forEach((msg, idx) => {
      report += `${idx + 1}. [${msg.timestamp}] [${msg.type}] ${msg.text}\\n`;
    });
    
    // Error Section
    if (errorMessages.length > 0) {
      report += `\\nERRORS (${errorMessages.length} entries)\\n`;
      report += `=================\\n`;
      errorMessages.forEach((error, idx) => {
        report += `${idx + 1}. [${error.timestamp}] ${error.message}\\n`;
        if (error.stack) {
          report += `   Stack: ${error.stack}\\n`;
        }
        report += `\\n`;
      });
    }
    
    const reportFile = path.join(process.cwd(), 'ai-model-outputs-report.txt');
    fs.writeFileSync(reportFile, report);
    
    // Create simplified summary for quick inspection
    const quickSummary = {
      summary: collectionSummary.collection,
      neuralOutputs: modelOutputs.neuralNetworkOutputs.map(o => ({
        text: o.text,
        indicators: o.indicators,
        timestamp: o.timestamp
      })),
      modelOutputsByType: Object.fromEntries(
        Object.entries(modelOutputs.modelSpecificOutputs).map(([model, outputs]) => [
          model, outputs.map(o => ({ text: o.text, timestamp: o.timestamp }))
        ])
      )
    };
    
    const summaryFile = path.join(process.cwd(), 'ai-model-outputs-quick-summary.json');
    fs.writeFileSync(summaryFile, JSON.stringify(quickSummary, null, 2));
    
    console.log('\\n🎉 COLLECTION COMPLETE!');
    console.log(`📁 Full data: ${jsonFile}`);
    console.log(`📄 Detailed report: ${reportFile}`);
    console.log(`📋 Quick summary: ${summaryFile}`);
    console.log(`📊 Collected: ${collectionSummary.collection.totalMessages} total, ${collectionSummary.collection.neuralNetworkOutputs} neural, ${collectionSummary.collection.modelsDetected} models`);
    
    // Verify we collected data
    expect(allConsoleMessages.length).toBeGreaterThan(0);
  });
});
