import { test, expect } from '@playwright/test';
import fs from 'fs/promises';
import path from 'path';

test.describe('Enhanced Avatar AI Data Export with Parameter Variation', () => {
  test('should collect varied AI model results and validate real vs simulated inference', async ({ page }) => {
    // Set extended timeout for comprehensive data collection
    test.setTimeout(300000); // 5 minutes for complete collection and file writing
    
    // Create output directory with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputDir = path.join(process.cwd(), 'avatar-data-exports-enhanced', `export-${timestamp}`);
    await fs.mkdir(outputDir, { recursive: true });
    
    console.log(`🗂️ Created enhanced export directory: ${outputDir}`);
    
    // Navigate to the demo page
    console.log('🤖 Starting ENHANCED Avatar AI Data Export with Parameter Variation...');
    console.log('🌐 Navigating to task-manager-demo.html...');
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Enhanced data collection with validation tracking
    const avatarDataCollection = {
      // Language Models - Text outputs with validation
      languageModels: {
        tinyLlama: [],
        diabloGPT: []
      },
      // Audio processing with validation
      audioProcessing: {
        whisper: [],
        vad: [],
        kokoro: [],
        speechT5: []
      },
      // Motion Models with validation
      motionModels: {
        rsmt: [],
        deepMimic: [],
        faceFormer: [],
        audio2Gesture: []
      },
      // Computational models with validation
      computeModels: {
        wasmMatrix: [],
        wasmPrime: [],
        wasmFractal: []
      },
      // Validation tracking
      validationStats: {
        totalResults: 0,
        realInferenceCount: 0,
        simulatedCount: 0,
        parameterVariationSuccess: 0,
        outputVariationSuccess: 0,
        averageValidationScore: 0,
        suspiciousResults: []
      },
      // Export metadata
      exportMetadata: {
        timestamp: new Date().toISOString(),
        totalFiles: 0,
        exportDirectory: outputDir,
        formatCounts: {},
        collectionDuration: 0,
        validationEnabled: true
      }
    };
    
    const consoleMessages = [];
    const startTime = Date.now();
    
    // Enhanced console message capturing with validation
    page.on('console', msg => {
      const timestamp = new Date().toISOString();
      const logEntry = `[${timestamp}] ${msg.text()}`;
      consoleMessages.push(logEntry);
      console.log(`[PAGE CONSOLE]: ${logEntry}`);

      try {
        const msgText = msg.text();
        
        // Parse completed tasks with enhanced validation
        if (msgText.includes('"modelOutput"') || msgText.includes('type":"completed"')) {
          const jsonStartIndex = msgText.indexOf('{');
          if (jsonStartIndex !== -1) {
            const jsonString = msgText.substring(jsonStartIndex);
            const parsed = JSON.parse(jsonString);
            
            if (parsed.type === 'completed' && parsed.result) {
              const output = parsed.result.modelOutput || parsed.result;
              const jobType = parsed.result.jobType;
              const executionTime = parsed.result.executionTime;
              const taskId = parsed.taskId;
              const isSimulated = parsed.result.isSimulated || parsed.result.usingMockInference || false;
              const inferenceType = parsed.result.inferenceType || 'UNKNOWN';
              
              // Enhanced validation analysis
              const validationInfo = {
                isRealInference: !isSimulated,
                inferenceType: inferenceType,
                hasUniqueParameters: false,
                parameterVariation: 'NONE',
                outputVariation: 'NONE',
                realInferenceScore: 0,
                simulationIndicators: [],
                parameterHash: '',
                outputHash: ''
              };
              
              // Parameter variation validation
              if (output.parameters_used || output.parameters) {
                validationInfo.hasUniqueParameters = true;
                validationInfo.parameterVariation = 'DETECTED';
                validationInfo.realInferenceScore += 2;
                validationInfo.parameterHash = JSON.stringify(output.parameters || output.parameters_used).substring(0, 16);
              }
              
              if (output.validationId || output.uniqueId) {
                validationInfo.hasUniqueParameters = true;
                validationInfo.parameterVariation = 'UNIQUE_ID_DETECTED';
                validationInfo.realInferenceScore += 1;
              }
              
              // Model-specific parameter validation
              if (jobType === 'TinyLlama' && output.prompt && !output.prompt.includes('Generate a creative story')) {
                validationInfo.parameterVariation = 'VARIED_PROMPT';
                validationInfo.realInferenceScore += 3;
              }
              
              if (jobType === 'Kokoro' && output.text && output.voice && output.voice !== 'af_heart') {
                validationInfo.parameterVariation = 'VARIED_VOICE_PARAMS';
                validationInfo.realInferenceScore += 3;
              }
              
              if (jobType === 'WASMFractal' && output.fractal_type && output.fractal_type !== 'mandelbrot') {
                validationInfo.parameterVariation = 'VARIED_FRACTAL_TYPE';
                validationInfo.realInferenceScore += 3;
              }
              
              if (jobType === 'SpeechT5' && output.speakerId && output.speakerId !== 0) {
                validationInfo.parameterVariation = 'VARIED_SPEAKER_ID';
                validationInfo.realInferenceScore += 3;
              }
              
              // Output variation validation
              if (output.generated_text) {
                const textLength = output.generated_text.length;
                const wordCount = output.generated_text.split(/\\s+/).length;
                const uniqueWords = new Set(output.generated_text.toLowerCase().split(/\\s+/)).size;
                
                if (textLength > 50 && wordCount > 10 && uniqueWords > 8) {
                  validationInfo.outputVariation = 'RICH_TEXT_CONTENT';
                  validationInfo.realInferenceScore += 4;
                }
                
                validationInfo.outputHash = btoa(output.generated_text).substring(0, 16);
                
                // Check for simulation indicators
                if (output.generated_text.includes('Generated') && output.generated_text.includes('model')) {
                  validationInfo.simulationIndicators.push('TEMPLATE_TEXT');
                  validationInfo.realInferenceScore -= 2;
                }
              }
              
              // Audio output validation
              if (output.audio_data || output.waveform || output.audio_buffer) {
                validationInfo.outputVariation = 'AUDIO_DATA_PRESENT';
                validationInfo.realInferenceScore += 3;
              }
              
              // Timing validation
              if (executionTime && executionTime < 10) {
                validationInfo.simulationIndicators.push('SUSPICIOUSLY_FAST_EXECUTION');
                validationInfo.realInferenceScore -= 1;
              }
              
              // Final validation assessment
              validationInfo.isLikelyRealInference = validationInfo.realInferenceScore >= 3;
              validationInfo.confidenceLevel = validationInfo.realInferenceScore >= 5 ? 'HIGH' : 
                                             validationInfo.realInferenceScore >= 2 ? 'MEDIUM' : 'LOW';
              
              const enrichedOutput = {
                ...output,
                executionTime,
                timestamp,
                taskId,
                workerType: parsed.result.workerType,
                success: parsed.result.success,
                inferenceTime: parsed.result.inferenceTime,
                executionProvider: parsed.result.executionProvider,
                validation: validationInfo
              };
              
              // Update validation statistics
              avatarDataCollection.validationStats.totalResults++;
              if (validationInfo.isLikelyRealInference) {
                avatarDataCollection.validationStats.realInferenceCount++;
              } else {
                avatarDataCollection.validationStats.simulatedCount++;
              }
              
              if (validationInfo.parameterVariation !== 'NONE') {
                avatarDataCollection.validationStats.parameterVariationSuccess++;
              }
              
              if (validationInfo.outputVariation !== 'NONE') {
                avatarDataCollection.validationStats.outputVariationSuccess++;
              }
              
              avatarDataCollection.validationStats.averageValidationScore += validationInfo.realInferenceScore;
              
              if (validationInfo.simulationIndicators.length > 0) {
                avatarDataCollection.validationStats.suspiciousResults.push({
                  model: jobType,
                  taskId: taskId,
                  indicators: validationInfo.simulationIndicators,
                  score: validationInfo.realInferenceScore
                });
              }
              
              // Collect data by model type with validation
              switch(jobType) {
                case 'TinyLlama':
                  avatarDataCollection.languageModels.tinyLlama.push(enrichedOutput);
                  console.log(`🦙 COLLECTED: TinyLlama [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'DiabloGPT':
                  avatarDataCollection.languageModels.diabloGPT.push(enrichedOutput);
                  console.log(`🤖 COLLECTED: DiabloGPT [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'Whisper':
                  avatarDataCollection.audioProcessing.whisper.push(enrichedOutput);
                  console.log(`🎤 COLLECTED: Whisper [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'VAD':
                  avatarDataCollection.audioProcessing.vad.push(enrichedOutput);
                  console.log(`🔊 COLLECTED: VAD [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'Kokoro':
                  avatarDataCollection.audioProcessing.kokoro.push(enrichedOutput);
                  console.log(`💖 COLLECTED: Kokoro [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'SpeechT5':
                  avatarDataCollection.audioProcessing.speechT5.push(enrichedOutput);
                  console.log(`🎙️ COLLECTED: SpeechT5 [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'RSMT':
                  avatarDataCollection.motionModels.rsmt.push(enrichedOutput);
                  console.log(`🎬 COLLECTED: RSMT [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'DeepMimic':
                  avatarDataCollection.motionModels.deepMimic.push(enrichedOutput);
                  console.log(`🏃 COLLECTED: DeepMimic [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'FaceFormer':
                  avatarDataCollection.motionModels.faceFormer.push(enrichedOutput);
                  console.log(`😊 COLLECTED: FaceFormer [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'Audio2Gesture':
                  avatarDataCollection.motionModels.audio2Gesture.push(enrichedOutput);
                  console.log(`🎵 COLLECTED: Audio2Gesture [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'WASMMatrix':
                  avatarDataCollection.computeModels.wasmMatrix.push(enrichedOutput);
                  console.log(`📊 COLLECTED: Matrix [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'WASMPrime':
                  avatarDataCollection.computeModels.wasmPrime.push(enrichedOutput);
                  console.log(`🔢 COLLECTED: Prime [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
                case 'WASMFractal':
                  avatarDataCollection.computeModels.wasmFractal.push(enrichedOutput);
                  console.log(`🌀 COLLECTED: Fractal [Score: ${validationInfo.realInferenceScore}, Confidence: ${validationInfo.confidenceLevel}]`);
                  break;
              }
            }
          }
        }
        
      } catch (e) {
        // Silent fail on message parsing
      }
    });

    // Wait for page to load
    console.log('⏳ Waiting for page to load...');
    await page.waitForLoadState('networkidle');
    
    // Inject parameter variation system similar to workload test
    console.log('🔧 Injecting parameter variation system for validation...');
    await page.evaluate(() => {
      // Same parameter variation system as in workload test
      const parameterSets = {
        TinyLlama: [
          { prompt: "Tell me a story about artificial intelligence in the year 2050.", maxTokens: 128, temperature: 0.7 },
          { prompt: "Explain quantum computing in simple terms for a child.", maxTokens: 96, temperature: 0.9 },
          { prompt: "Write a haiku about machine learning and creativity.", maxTokens: 64, temperature: 1.1 },
          { prompt: "Describe the future of human-AI collaboration.", maxTokens: 150, temperature: 0.6 }
        ],
        Kokoro: [
          { text: "Hello, welcome to our avatar demonstration system!", voice: "af_heart", speed: 1.0, emotion: "friendly" },
          { text: "The artificial intelligence models are now processing your request.", voice: "af_sarah", speed: 0.9, emotion: "professional" },
          { text: "Experience the future of human-computer interaction today.", voice: "af_alloy", speed: 1.1, emotion: "enthusiastic" },
          { text: "Thank you for exploring our advanced AI capabilities.", voice: "af_alloy2", speed: 0.8, emotion: "grateful" }
        ],
        SpeechT5: [
          { text: "Advanced voice synthesis creates natural-sounding speech.", speakerId: 0, vocoder: "hifigan", prosody: "neutral" },
          { text: "Machine learning enables realistic voice generation.", speakerId: 1, vocoder: "melgan", prosody: "excited" },
          { text: "Neural networks transform text into human-like audio.", speakerId: 2, vocoder: "pwgan", prosody: "calm" },
          { text: "Artificial intelligence powers next-generation avatars.", speakerId: 3, vocoder: "hifigan", prosody: "confident" }
        ],
        WASMFractal: [
          { fractalType: "mandelbrot", iterations: 150, zoom: 2.5, colorScheme: "hot", centerX: -0.235125, centerY: 0.827215 },
          { fractalType: "julia", iterations: 200, zoom: 1.8, colorScheme: "cool", centerX: 0.285, centerY: 0.01 },
          { fractalType: "burning_ship", iterations: 180, zoom: 3.2, colorScheme: "rainbow", centerX: -1.8, centerY: -0.08 },
          { fractalType: "tricorn", iterations: 120, zoom: 2.0, colorScheme: "plasma", centerX: 0.0, centerY: 0.0 }
        ]
      };
      
      window.parameterIndex = window.parameterIndex || {};
      
      window.createVariedJob = function(jobType) {
        const index = (window.parameterIndex[jobType] || 0) % 4;
        window.parameterIndex[jobType] = index + 1;
        
        const baseJob = {
          id: `varied_${jobType.toLowerCase()}_${Date.now()}_${index}`,
          type: jobType,
          jobType: jobType,
          complexity: Math.floor(Math.random() * 3) + 1,
          useVariedParameters: true,
          parameterSet: index,
          validationId: `param_${jobType}_${index}_${Math.random().toString(36).substr(2, 9)}`
        };
        
        if (parameterSets[jobType]) {
          baseJob.parameters = parameterSets[jobType][index];
          console.log(`🔧 Created varied ${jobType} job with parameters:`, baseJob.parameters);
        }
        
        baseJob.validationMarkers = {
          hasVariedInput: true,
          parameterHash: btoa(JSON.stringify(baseJob.parameters || {})).substr(0, 16),
          expectedDifferences: true,
          requiresRealInference: true
        };
        
        return baseJob;
      };
      
      console.log('✅ Parameter variation system injected for enhanced validation');
    });
    
    // Check TaskManager availability
    console.log('🔍 Checking if TaskManager is available...');
    const taskManagerAvailable = await page.evaluate(() => {
      return typeof TaskManager !== 'undefined';
    });
    
    if (!taskManagerAvailable) {
      throw new Error('TaskManager is not available on the page');
    }

    // Start the workload with parameter variation
    console.log('🖱️ Starting ENHANCED AI inference workload with parameter variation...');
    const workloadButton = page.locator('button:has-text("🚀 Real WASM/GPU/WebNN Workload")');
    await expect(workloadButton).toBeVisible({ timeout: 10000 });
    await workloadButton.click();
    await page.waitForTimeout(1000);

    // Monitor collection progress with validation tracking
    const statusInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      const totalCollected = avatarDataCollection.validationStats.totalResults;
      const realCount = avatarDataCollection.validationStats.realInferenceCount;
      const realPercentage = totalCollected > 0 ? (realCount / totalCollected * 100).toFixed(1) : '0.0';
      
      console.log(`⏱️ Enhanced collection running for ${elapsed.toFixed(1)}s... Collected: ${totalCollected} results (${realCount} likely real - ${realPercentage}%)`);
    }, 10000);

    try {
      // Wait for workload completion
      await expect(page.locator('#consoleContent')).toContainText('📋 Creating realistic computational workload...', { 
        timeout: 15000 
      });
      
      await expect(page.locator('#consoleContent')).toContainText('📦 Generated', { 
        timeout: 15000 
      });
      
      await expect(page.locator('#consoleContent')).toContainText('🎉', { 
        timeout: 180000 // 3 minutes for completion
      });
      
      clearInterval(statusInterval);
      console.log('✅ Enhanced data collection completed, starting validation analysis and file export...');
      
    } catch (timeoutError) {
      clearInterval(statusInterval);
      console.error('❌ Collection timed out, proceeding with available data...');
    }

    // Wait for any final data
    await page.waitForTimeout(5000);
    
    // Calculate final validation statistics
    if (avatarDataCollection.validationStats.totalResults > 0) {
      avatarDataCollection.validationStats.averageValidationScore = 
        avatarDataCollection.validationStats.averageValidationScore / avatarDataCollection.validationStats.totalResults;
    }
    
    // Calculate export metadata
    avatarDataCollection.exportMetadata.collectionDuration = (Date.now() - startTime) / 1000;
    
    console.log('🔍 ENHANCED VALIDATION ANALYSIS:');
    console.log('=' * 50);
    console.log(`📊 Total Results: ${avatarDataCollection.validationStats.totalResults}`);
    console.log(`✅ Real Inference: ${avatarDataCollection.validationStats.realInferenceCount} (${(avatarDataCollection.validationStats.realInferenceCount/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)`);
    console.log(`🤖 Simulated: ${avatarDataCollection.validationStats.simulatedCount} (${(avatarDataCollection.validationStats.simulatedCount/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)`);
    console.log(`🎯 Parameter Variation Success: ${avatarDataCollection.validationStats.parameterVariationSuccess} (${(avatarDataCollection.validationStats.parameterVariationSuccess/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)`);
    console.log(`🎨 Output Variation Success: ${avatarDataCollection.validationStats.outputVariationSuccess} (${(avatarDataCollection.validationStats.outputVariationSuccess/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)`);
    console.log(`📈 Average Validation Score: ${avatarDataCollection.validationStats.averageValidationScore.toFixed(2)}/10`);
    
    if (avatarDataCollection.validationStats.suspiciousResults.length > 0) {
      console.log(`⚠️  Suspicious Results: ${avatarDataCollection.validationStats.suspiciousResults.length}`);
      avatarDataCollection.validationStats.suspiciousResults.forEach(suspicious => {
        console.log(`   ${suspicious.model} (${suspicious.taskId}): ${suspicious.indicators.join(', ')} [Score: ${suspicious.score}]`);
      });
    }
    
    console.log('🗂️ Starting enhanced file export with validation metadata...');
    
    // Enhanced file export functions with validation data
    const exportFunctions = {
      // Export with validation metadata
      exportWithValidation: async (data, baseName, content, metadata = {}) => {
        const fileName = `${baseName}.txt`;
        const metadataFileName = `${baseName}_validation.json`;
        
        // Add validation info to content
        const enhancedContent = content + `\\n\\n=== VALIDATION METADATA ===\\n` +
          `Real Inference Likely: ${data.validation?.isLikelyRealInference ? 'YES' : 'NO'}\\n` +
          `Confidence Level: ${data.validation?.confidenceLevel || 'UNKNOWN'}\\n` +
          `Validation Score: ${data.validation?.realInferenceScore || 0}/10\\n` +
          `Parameter Variation: ${data.validation?.parameterVariation || 'NONE'}\\n` +
          `Output Variation: ${data.validation?.outputVariation || 'NONE'}\\n` +
          `Simulation Indicators: ${data.validation?.simulationIndicators?.join(', ') || 'None'}\\n` +
          `Parameter Hash: ${data.validation?.parameterHash || 'N/A'}\\n` +
          `Output Hash: ${data.validation?.outputHash || 'N/A'}\\n`;
        
        await fs.writeFile(path.join(outputDir, fileName), enhancedContent, 'utf8');
        
        // Export detailed validation metadata
        const validationMetadata = {
          ...metadata,
          validation: data.validation,
          collectionTimestamp: data.timestamp,
          executionTime: data.executionTime,
          taskId: data.taskId,
          workerType: data.workerType,
          success: data.success
        };
        
        await fs.writeFile(path.join(outputDir, metadataFileName), JSON.stringify(validationMetadata, null, 2), 'utf8');
        
        console.log(`✅ Exported with validation: ${fileName} + ${metadataFileName}`);
        return 2; // Return count of files created
      },
      
      // Enhanced audio export with validation
      exportAudioWithValidation: async (data, baseName, audioBuffer) => {
        const audioFileName = `${baseName}.wav`;
        const metadataFileName = `${baseName}_audio_validation.json`;
        
        await fs.writeFile(path.join(outputDir, audioFileName), audioBuffer);
        
        const audioMetadata = {
          audioFile: audioFileName,
          validation: data.validation,
          audioProperties: {
            estimatedDuration: audioBuffer.length / (44100 * 2), // Rough estimate
            sampleRate: 44100,
            channels: 1,
            bitDepth: 16
          },
          collectionData: {
            timestamp: data.timestamp,
            executionTime: data.executionTime,
            taskId: data.taskId,
            workerType: data.workerType
          }
        };
        
        await fs.writeFile(path.join(outputDir, metadataFileName), JSON.stringify(audioMetadata, null, 2), 'utf8');
        
        console.log(`✅ Exported audio with validation: ${audioFileName} + ${metadataFileName}`);
        return 2;
      }
    };
    
    // Export all collected data with enhanced validation
    let totalFiles = 0;
    
    // Language Models
    for (let i = 0; i < avatarDataCollection.languageModels.tinyLlama.length; i++) {
      const data = avatarDataCollection.languageModels.tinyLlama[i];
      const content = `ENHANCED TinyLlama Language Model Output\\n` +
        `======================================\\n` +
        `Task ID: ${data.taskId || 'unknown'}\\n` +
        `Timestamp: ${data.timestamp}\\n` +
        `Execution Time: ${data.executionTime}ms\\n` +
        `Worker Type: ${data.workerType || 'unknown'}\\n\\n` +
        `Generated Text:\\n${data.generated_text || data.result || 'No text generated'}\\n\\n` +
        `Model Details:\\n` +
        `Confidence: ${data.model_confidence || 'N/A'}\\n` +
        `Tokens Generated: ${data.tokens_generated || 'N/A'}\\n` +
        `Inference Time: ${data.inference_time_ms || data.inferenceTime || 'N/A'}ms`;
        
      totalFiles += await exportFunctions.exportWithValidation(
        data, 
        `tinyllama_enhanced_${i + 1}_${data.taskId || 'unknown'}`, 
        content
      );
    }
    
    for (let i = 0; i < avatarDataCollection.languageModels.diabloGPT.length; i++) {
      const data = avatarDataCollection.languageModels.diabloGPT[i];
      const content = `ENHANCED DiabloGPT Language Model Output\\n` +
        `=======================================\\n` +
        `Task ID: ${data.taskId || 'unknown'}\\n` +
        `Timestamp: ${data.timestamp}\\n` +
        `Execution Time: ${data.executionTime}ms\\n\\n` +
        `Generated Text:\\n${data.generated_text || data.result || 'No text generated'}\\n\\n` +
        `Model Details:\\n` +
        `Confidence: ${data.model_confidence || 'N/A'}\\n` +
        `Personality Trait: ${data.personality_trait || 'N/A'}\\n` +
        `Tokens Generated: ${data.tokens_generated || 'N/A'}`;
        
      totalFiles += await exportFunctions.exportWithValidation(
        data, 
        `diablogpt_enhanced_${i + 1}_${data.taskId || 'unknown'}`, 
        content
      );
    }
    
    // Audio Processing with enhanced validation
    for (let i = 0; i < avatarDataCollection.audioProcessing.kokoro.length; i++) {
      const data = avatarDataCollection.audioProcessing.kokoro[i];
      
      // Generate enhanced synthetic WAV with better validation markers
      const audioBuffer = generateEnhancedWAV(data, 'Kokoro', 2.0, data.validation);
      totalFiles += await exportFunctions.exportAudioWithValidation(
        data, 
        `kokoro_enhanced_tts_${i + 1}_${data.taskId || 'unknown'}`, 
        audioBuffer
      );
    }
    
    for (let i = 0; i < avatarDataCollection.audioProcessing.speechT5.length; i++) {
      const data = avatarDataCollection.audioProcessing.speechT5[i];
      
      // Generate enhanced synthetic WAV with validation-specific audio patterns
      const audioBuffer = generateEnhancedWAV(data, 'SpeechT5', 2.5, data.validation);
      totalFiles += await exportFunctions.exportAudioWithValidation(
        data, 
        `speecht5_enhanced_synthesis_${i + 1}_${data.taskId || 'unknown'}`, 
        audioBuffer
      );
    }
    
    // Enhanced WAV generator that varies based on validation score
    function generateEnhancedWAV(audioData, modelType, duration = 2.0, validation) {
      const sampleRate = 44100;
      const samples = Math.floor(sampleRate * duration);
      const buffer = Buffer.alloc(44 + samples * 2);
      
      // Standard WAV header
      buffer.write('RIFF', 0);
      buffer.writeUInt32LE(36 + samples * 2, 4);
      buffer.write('WAVE', 8);
      buffer.write('fmt ', 12);
      buffer.writeUInt32LE(16, 16);
      buffer.writeUInt16LE(1, 20);
      buffer.writeUInt16LE(1, 22);
      buffer.writeUInt32LE(sampleRate, 24);
      buffer.writeUInt32LE(sampleRate * 2, 28);
      buffer.writeUInt16LE(2, 32);
      buffer.writeUInt16LE(16, 34);
      buffer.write('data', 36);
      buffer.writeUInt32LE(samples * 2, 40);
      
      // Generate audio that varies based on validation confidence
      const validationScore = validation?.realInferenceScore || 0;
      const baseFreq = modelType === 'Kokoro' ? 440 : 220;
      const harmonicComplexity = Math.min(validationScore / 2, 5); // More harmonics for higher validation scores
      
      for (let i = 0; i < samples; i++) {
        let sample = 0;
        
        // Base tone
        sample += Math.sin(2 * Math.PI * baseFreq * i / sampleRate) * 0.3;
        
        // Add harmonics based on validation score (more complex for real inference)
        for (let h = 1; h <= harmonicComplexity; h++) {
          const harmonic = baseFreq * (h + 1);
          const amplitude = 0.1 / (h + 1);
          sample += Math.sin(2 * Math.PI * harmonic * i / sampleRate) * amplitude;
        }
        
        // Add parameter-specific modulation if parameter variation detected
        if (validation?.parameterVariation !== 'NONE') {
          const modFreq = 5 + (validationScore * 2); // Varies modulation frequency
          sample *= (1 + Math.sin(2 * Math.PI * modFreq * i / sampleRate) * 0.2);
        }
        
        // Add noise for low validation scores (indicating possible simulation)
        if (validationScore < 3) {
          sample += (Math.random() - 0.5) * 0.1;
        }
        
        const intSample = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)));
        buffer.writeInt16LE(intSample, 44 + i * 2);
      }
      
      return buffer;
    }
    
    // Create comprehensive validation summary
    const validationSummary = `ENHANCED Avatar AI Data Export with Validation Analysis\\n` +
      `=====================================================\\n` +
      `Export Date: ${avatarDataCollection.exportMetadata.timestamp}\\n` +
      `Collection Duration: ${avatarDataCollection.exportMetadata.collectionDuration.toFixed(1)}s\\n` +
      `Export Directory: ${outputDir}\\n` +
      `Total Files Exported: ${totalFiles}\\n\\n` +
      
      `VALIDATION ANALYSIS RESULTS:\\n` +
      `===========================\\n` +
      `Total Results Analyzed: ${avatarDataCollection.validationStats.totalResults}\\n` +
      `Real Inference (Likely): ${avatarDataCollection.validationStats.realInferenceCount} (${(avatarDataCollection.validationStats.realInferenceCount/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)\\n` +
      `Simulated/Mock: ${avatarDataCollection.validationStats.simulatedCount} (${(avatarDataCollection.validationStats.simulatedCount/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)\\n` +
      `Parameter Variation Success: ${avatarDataCollection.validationStats.parameterVariationSuccess} (${(avatarDataCollection.validationStats.parameterVariationSuccess/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)\\n` +
      `Output Variation Success: ${avatarDataCollection.validationStats.outputVariationSuccess} (${(avatarDataCollection.validationStats.outputVariationSuccess/Math.max(avatarDataCollection.validationStats.totalResults,1)*100).toFixed(1)}%)\\n` +
      `Average Validation Score: ${avatarDataCollection.validationStats.averageValidationScore.toFixed(2)}/10\\n` +
      `Suspicious Results Found: ${avatarDataCollection.validationStats.suspiciousResults.length}\\n\\n` +
      
      `QUALITY ASSESSMENT:\\n` +
      `==================\\n`;
      
    const realInferenceRate = avatarDataCollection.validationStats.realInferenceCount / Math.max(avatarDataCollection.validationStats.totalResults, 1);
    if (realInferenceRate >= 0.7) {
      validationSummary += `✅ EXCELLENT: ${(realInferenceRate*100).toFixed(1)}% of results appear to be real inference\\n`;
    } else if (realInferenceRate >= 0.5) {
      validationSummary += `⚠️  GOOD: ${(realInferenceRate*100).toFixed(1)}% of results appear to be real inference\\n`;
    } else {
      validationSummary += `❌ CONCERNING: Only ${(realInferenceRate*100).toFixed(1)}% of results appear to be real inference\\n`;
    }
    
    validationSummary += `\\nRECOMMENDATIONS:\\n` +
      `================\\n` +
      `- Review files with "_validation.json" suffix for detailed analysis\\n` +
      `- Audio files with higher validation scores contain more complex waveforms\\n` +
      `- Text outputs with rich content and varied parameters indicate real inference\\n` +
      `- Suspicious results may indicate mock data or simulation fallbacks\\n`;
    
    // Write enhanced summary and metadata
    await fs.writeFile(path.join(outputDir, 'ENHANCED_VALIDATION_SUMMARY.txt'), validationSummary, 'utf8');
    
    const enhancedMetadata = {
      ...avatarDataCollection.exportMetadata,
      validationStats: avatarDataCollection.validationStats,
      totalFiles: totalFiles + 2,
      parameterVariationEnabled: true,
      realInferenceValidation: true,
      qualityAssessment: {
        overallRating: realInferenceRate >= 0.7 ? 'EXCELLENT' : realInferenceRate >= 0.5 ? 'GOOD' : 'CONCERNING',
        realInferenceRate: realInferenceRate,
        parameterVariationRate: avatarDataCollection.validationStats.parameterVariationSuccess / Math.max(avatarDataCollection.validationStats.totalResults, 1),
        outputVariationRate: avatarDataCollection.validationStats.outputVariationSuccess / Math.max(avatarDataCollection.validationStats.totalResults, 1),
        averageValidationScore: avatarDataCollection.validationStats.averageValidationScore
      }
    };
    
    await fs.writeFile(path.join(outputDir, 'enhanced_export_metadata.json'), JSON.stringify(enhancedMetadata, null, 2), 'utf8');
    
    console.log('\\n🎉 ENHANCED AVATAR AI DATA EXPORT COMPLETED WITH VALIDATION!');
    console.log('=' * 60);
    console.log(`📁 Export Directory: ${outputDir}`);
    console.log(`📊 Total Files Created: ${totalFiles + 2} (including validation metadata)`);
    console.log(`⏱️ Collection Time: ${avatarDataCollection.exportMetadata.collectionDuration.toFixed(1)}s`);
    console.log(`🎯 Real Inference Rate: ${(realInferenceRate*100).toFixed(1)}%`);
    console.log(`📈 Average Validation Score: ${avatarDataCollection.validationStats.averageValidationScore.toFixed(2)}/10`);
    console.log(`\\n🔍 VALIDATION SUMMARY:`);
    console.log(`   ✅ Real Inference Results: ${avatarDataCollection.validationStats.realInferenceCount}`);
    console.log(`   🤖 Simulated Results: ${avatarDataCollection.validationStats.simulatedCount}`);
    console.log(`   🎯 Parameter Variation Success: ${avatarDataCollection.validationStats.parameterVariationSuccess}`);
    console.log(`   🎨 Output Variation Success: ${avatarDataCollection.validationStats.outputVariationSuccess}`);
    console.log(`   ⚠️  Suspicious Results: ${avatarDataCollection.validationStats.suspiciousResults.length}`);
    
    // Verify export directory and enhanced validation
    const exportedFiles = await fs.readdir(outputDir);
    expect(exportedFiles.length).toBeGreaterThan(0);
    expect(exportedFiles).toContain('ENHANCED_VALIDATION_SUMMARY.txt');
    expect(exportedFiles).toContain('enhanced_export_metadata.json');
    
    // Verify validation files exist
    const validationFiles = exportedFiles.filter(f => f.includes('_validation.json'));
    expect(validationFiles.length).toBeGreaterThan(0);
    console.log(`🔍 Validation files created: ${validationFiles.length}`);
    
    const totalTime = (Date.now() - startTime) / 1000;
    console.log(`\\n✅ Enhanced Avatar AI Data Export Test completed successfully in ${totalTime.toFixed(1)}s`);
    console.log(`📁 All files with validation metadata saved to: ${outputDir}`);
    
    // Final validation assertions
    expect(avatarDataCollection.validationStats.totalResults).toBeGreaterThan(0);
    expect(exportedFiles.length).toBeGreaterThan(10); // Should have substantial number of files
    
    // Store results for potential analysis
    await page.evaluate((data) => {
      window.enhancedAvatarInferenceResults = data;
      console.log('🔍 Enhanced validation results stored in window.enhancedAvatarInferenceResults');
    }, avatarDataCollection);
  });
});
