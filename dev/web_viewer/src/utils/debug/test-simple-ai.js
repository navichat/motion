const { chromium } = require('playwright');

async function testSimpleAICollection() {
  console.log('🧪 Testing simple AI collection...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const avatarInferenceResults = {
    languageModels: { tinyLlama: [], diabloGPT: [] },
    audioProcessing: { whisper: [], vad: [], kokoro: [], speechT5: [] },
    motionModels: { rsmt: [], deepMimic: [], faceFormer: [], audio2Gesture: [] },
    computeModels: { wasmMatrix: [], wasmPrime: [], wasmFractal: [] },
    knnModels: { closeVector: [], hnsw: [], unifiedKnn: [] },
    metadata: { totalResults: 0, executionTime: 0, capabilitiesDetected: {}, workerTypes: [] }
  };

  // Listen for console events and parse the results (same as real test)
  page.on('console', msg => {
    const msgText = msg.text();
    console.log(`[CONSOLE]: ${msgText}`);
    
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
  
  try {
    await page.goto('http://localhost:8000/dev/web_viewer/simple-test.html');
    console.log('✅ Simple test page loaded');
    
    // Click the button using same selector as real test
    await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
    console.log('✅ Button clicked');
    
    // Wait for all simulated results
    await page.waitForTimeout(6000);
    
    // Log the results (same as real test)
    console.log('\n📊 RESULTS:');
    console.log(`Total results: ${avatarInferenceResults.metadata.totalResults}`);
    console.log(`TinyLlama: ${avatarInferenceResults.languageModels.tinyLlama.length}`);
    console.log(`Whisper: ${avatarInferenceResults.audioProcessing.whisper.length}`);
    console.log(`FaceFormer: ${avatarInferenceResults.motionModels.faceFormer.length}`);
    console.log(`VAD: ${avatarInferenceResults.audioProcessing.vad.length}`);
    console.log(`Audio2Gesture: ${avatarInferenceResults.motionModels.audio2Gesture.length}`);
    
    if (avatarInferenceResults.metadata.totalResults > 0) {
      console.log('✅ SUCCESS: AVATAR AI COLLECTED messages are working!');
    } else {
      console.log('❌ FAILED: No AVATAR AI COLLECTED messages captured');
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
  
  await browser.close();
  console.log('🏁 Simple test completed');
}

testSimpleAICollection();
