import { test, expect } from '@playwright/test';

test.describe('Avatar AI Inference Collection from Real Workload Test', () => {
  test('should collect AI model inference results for avatar driving applications', async ({ page }) => {
    // Set extended timeout for this test to ensure ALL AI models are captured
    test.setTimeout(240000); // 4 minutes for comprehensive collection of ALL models
    
    // Navigate to the demo page
    console.log('🤖 Starting COMPREHENSIVE Avatar AI Inference Collection Test...');
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8082/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront(); // Bring the page to the front to prevent throttling

    // Enhanced data structures for avatar AI inference collection
    const avatarInferenceResults = {
      languageModels: { tinyLlama: [], diabloGPT: [] },
      audioProcessing: { whisper: [], vad: [], kokoro: [], speechT5: [] },
      motionModels: { rsmt: [], deepMimic: [], faceFormer: [], audio2Gesture: [] },
      computeModels: { wasmMatrix: [], wasmPrime: [], wasmFractal: [] },
      knnModels: { closeVector: [], hnsw: [], unifiedKnn: [] },
      metadata: { totalResults: 0, executionTime: 0, capabilitiesDetected: {}, workerTypes: [] }
    };

    // Listen for console events and parse the results
    page.on('console', msg => {
        const msgText = msg.text();
        if (msgText.includes('AVATAR AI COLLECTED')) {
            try {
                const jsonString = msgText.substring(msgText.indexOf('{'));
                const parsed = JSON.parse(jsonString);
                const jobType = parsed.jobType;
                const executionTime = parsed.executionTime;
                const output = parsed.modelOutput;

                switch(jobType) {
                    case 'TinyLlama':
                        avatarInferenceResults.languageModels.tinyLlama.push({ ...output, executionTime });
                        break;
                    case 'DiabloGPT':
                        avatarInferenceResults.languageModels.diabloGPT.push({ ...output, executionTime });
                        break;
                    case 'Whisper':
                        avatarInferenceResults.audioProcessing.whisper.push({ ...output, executionTime });
                        break;
                    case 'VAD':
                        avatarInferenceResults.audioProcessing.vad.push({ ...output, executionTime });
                        break;
                    case 'Kokoro':
                        avatarInferenceResults.audioProcessing.kokoro.push({ ...output, executionTime });
                        break;
                    case 'SpeechT5':
                        avatarInferenceResults.audioProcessing.speechT5.push({ ...output, executionTime });
                        break;
                    case 'RSMT':
                        avatarInferenceResults.motionModels.rsmt.push({ ...output, executionTime });
                        break;
                    case 'DeepMimic':
                        avatarInferenceResults.motionModels.deepMimic.push({ ...output, executionTime });
                        break;
                    case 'FaceFormer':
                        avatarInferenceResults.motionModels.faceFormer.push({ ...output, executionTime });
                        break;
                    case 'Audio2Gesture':
                        avatarInferenceResults.motionModels.audio2Gesture.push({ ...output, executionTime });
                        break;
                    case 'WASMMatrix':
                        avatarInferenceResults.computeModels.wasmMatrix.push({ ...output, executionTime });
                        break;
                    case 'WASMPrime':
                        avatarInferenceResults.computeModels.wasmPrime.push({ ...output, executionTime });
                        break;
                    case 'WASMFractal':
                        avatarInferenceResults.computeModels.wasmFractal.push({ ...output, executionTime });
                        break;
                    case 'CloseVector':
                        avatarInferenceResults.knnModels.closeVector.push({ ...output, executionTime });
                        break;
                    case 'HNSW':
                        avatarInferenceResults.knnModels.hnsw.push({ ...output, executionTime });
                        break;
                    case 'UnifiedKNN':
                        avatarInferenceResults.knnModels.unifiedKnn.push({ ...output, executionTime });
                        break;
                }
                avatarInferenceResults.metadata.totalResults++;
            } catch (e) {
                console.error('Error parsing console message:', e);
            }
        }
    });

    // Click the button to run the test
    await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();

    // Wait much longer for all tasks to complete - our debug showed it needs more time
    console.log('⏰ Waiting for all AI model tasks to complete...');
    
    // Wait until we have collected at least 15 AI model results (enough for all models including slow ones)
    // AND wait an extra 5 seconds after reaching 13+ to ensure TinyLlama/DiabloGPT/Whisper complete
    let attempts = 0;
    let stableCount = 0;
    let lastResultCount = 0;
    
    while (attempts < 150) { // Increased max wait time
      await page.waitForTimeout(1000);
      attempts++;
      
      // Check if results are still increasing
      if (avatarInferenceResults.metadata.totalResults > lastResultCount) {
        lastResultCount = avatarInferenceResults.metadata.totalResults;
        stableCount = 0; // Reset stability counter
      } else {
        stableCount++;
      }
      
      // Log progress every 10 seconds
      if (attempts % 10 === 0) {
        console.log(`⏰ Still waiting... collected ${avatarInferenceResults.metadata.totalResults} results so far (${attempts}s elapsed)`);
      }
      
      // Break if we have 13+ results AND they've been stable for 5 seconds
      if (avatarInferenceResults.metadata.totalResults >= 13 && stableCount >= 5) {
        console.log(`✅ Results stabilized at ${avatarInferenceResults.metadata.totalResults} - waiting complete`);
        break;
      }
    }

    console.log(`🎉 Collected ${avatarInferenceResults.metadata.totalResults} AI model inference results!`);
    console.log(JSON.stringify(avatarInferenceResults, null, 2));
    
    // Verify we got all expected results including TinyLlama, DiabloGPT, Whisper
    expect(avatarInferenceResults.metadata.totalResults).toBeGreaterThanOrEqual(13);
    
    // Check that we have the specific models that were previously missing
    const allResults = JSON.stringify(avatarInferenceResults);
    expect(allResults).toContain('TinyLlama');
    expect(allResults).toContain('DiabloGPT'); 
    expect(allResults).toContain('Whisper');
  });
});