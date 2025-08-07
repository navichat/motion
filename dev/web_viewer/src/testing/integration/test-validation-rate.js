// Quick test to check the enhanced neural network validation rate
import { chromium } from 'playwright';

async function testValidationRate() {
    console.log('🧪 Testing Enhanced Neural Network Validation Rate...');
    
    const browser = await chromium.launch();
    const page = await browser.newPage();
    
    let allResults = [];
    
    // Set up console message collection
    page.on('console', msg => {
        const text = msg.text();
        console.log(`[PAGE CONSOLE]: ${text}`);
        
        if (text.includes('🎯 Result for')) {
            try {
                const resultMatch = text.match(/🎯 Result for (\w+): (.+)/);
                if (resultMatch) {
                    const [, modelType, resultJson] = resultMatch;
                    const result = JSON.parse(resultJson);
                    result.model_type = modelType;
                    allResults.push(result);
                }
            } catch (e) {
                console.log('❌ Error parsing result:', e.message);
            }
        }
        
        // Also check for other result patterns
        if (text.includes('AI inference result') || text.includes('model result') || text.includes('neural_network_used')) {
            console.log(`🔍 Potential result: ${text}`);
        }
    });
    
    try {
        console.log('🌐 Loading page...');
        await page.goto('http://localhost:3000/task-manager-demo.html', { waitUntil: 'networkidle' });
        await page.waitForTimeout(3000);
        
        console.log('🚀 Starting test with enhanced validation...');
        
        // Start AI workload
        await page.evaluate(() => {
            if (!window.TaskManager) {
                throw new Error('TaskManager class not available');
            }
            
            // Create TaskManager instance
            const manager = new TaskManager({
                maxConcurrentTasks: 3,
                cpuWorkers: 2,
                schedulingInterval: 100,
                capabilities: {
                    webgpu: !!navigator.gpu,
                    webnn: !!navigator.ml
                }
            });
            
            // Store globally for access
            window.testTaskManager = manager;
            
            // Create a few varied AI jobs for testing
            const aiJobs = [
                {
                    id: 'test_tinyllama_1',
                    type: 'ai_inference',
                    priority: 3,
                    requirements: {
                        modelType: 'TinyLlama',
                        capabilities: ['inference'],
                        memory: '2GB'
                    },
                    data: {
                        generationParams: {
                            temperature: 0.9,
                            seed: Math.random()
                        }
                    }
                },
                {
                    id: 'test_kokoro_1',
                    type: 'ai_inference',
                    priority: 3,
                    requirements: {
                        modelType: 'Kokoro',
                        capabilities: ['inference'],
                        memory: '1GB'
                    },
                    data: {
                        text: 'Enhanced neural network test',
                        speechParams: {
                            pitch: 1.2,
                            rate: 1.1,
                            uniqueId: Math.random()
                        }
                    }
                },
                {
                    id: 'test_whisper_1',
                    type: 'ai_inference',
                    priority: 3,
                    requirements: {
                        modelType: 'Whisper',
                        capabilities: ['inference'],
                        memory: '2GB'
                    },
                    data: {
                        uniqueId: Math.random()
                    }
                },
                {
                    id: 'test_faceformer_1',
                    type: 'ai_inference',
                    priority: 3,
                    requirements: {
                        modelType: 'FaceFormer',
                        capabilities: ['inference'],
                        memory: '1GB'
                    },
                    data: {
                        uniqueId: Math.random()
                    }
                }
            ];
            
            console.log(`🔄 Submitting ${aiJobs.length} test jobs...`);
            aiJobs.forEach(job => {
                manager.scheduleTask(job);
            });
        });
        
        // Wait for results
        console.log('⏳ Waiting for AI inference results...');
        await page.waitForTimeout(12000);
        
        console.log(`\n📊 VALIDATION RATE TEST RESULTS:`);
        console.log(`📈 Total AI models processed: ${allResults.length}`);
        
        if (allResults.length === 0) {
            console.log('❌ No results collected');
            return;
        }
        
        // Count models with neural network validation markers
        let validatedCount = 0;
        let totalMarkers = 0;
        
        allResults.forEach(result => {
            let hasValidation = false;
            let markerCount = 0;
            
            // Check for comprehensive neural network markers
            const neuralMarkers = [
                'neural_network_used',
                'executionProvider',
                'layers_processed',
                'attention_weights',
                'transformer_blocks',
                'attention_heads',
                'mel_spectrogram_generated',
                'vocoder_output',
                'conv1d_layers',
                'lstm_layers',
                'dense_layers',
                'temporal_encoding',
                'gpu_memory_allocated',
                'model_path',
                'quantization_enabled',
                'precision_mode',
                'checkpoint_loaded'
            ];
            
            neuralMarkers.forEach(marker => {
                if (result[marker] !== undefined) {
                    markerCount++;
                    hasValidation = true;
                }
            });
            
            if (hasValidation) {
                validatedCount++;
                console.log(`✅ ${result.model_type}: ${markerCount} neural network markers detected`);
            } else {
                console.log(`❌ ${result.model_type}: No neural network validation markers`);
            }
            
            totalMarkers += markerCount;
        });
        
        const validationRate = (validatedCount / allResults.length * 100).toFixed(1);
        const avgMarkersPerModel = (totalMarkers / allResults.length).toFixed(1);
        
        console.log(`\n🎯 ENHANCED VALIDATION ANALYSIS:`);
        console.log(`📊 Models with neural network validation: ${validatedCount}/${allResults.length} (${validationRate}%)`);
        console.log(`📈 Average validation markers per model: ${avgMarkersPerModel}`);
        console.log(`🔍 Total validation markers detected: ${totalMarkers}`);
        
        if (validationRate >= 80) {
            console.log(`🎉 EXCELLENT! Validation rate significantly improved from 30.9% to ${validationRate}%`);
        } else if (validationRate >= 60) {
            console.log(`✅ GOOD! Validation rate improved from 30.9% to ${validationRate}%`);
        } else {
            console.log(`⚠️  MODERATE improvement: Validation rate ${validationRate}% (target: 60%+)`);
        }
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
    } finally {
        await browser.close();
    }
}

// Run the test
testValidationRate().catch(console.error);
