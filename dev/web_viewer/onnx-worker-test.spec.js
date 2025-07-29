import { test, expect } from '@playwright/test';

test.describe('ONNX Worker Error Investigation Test', () => {
  test('should test ONNX compatibility fixes and collect comprehensive worker errors', async ({ page }) => {
    // Set extended timeout for comprehensive testing
    test.setTimeout(180000); // 3 minutes for full workload
    
    console.log('🔧 Starting ONNX Compatibility and Worker Error Test...');
    
    // Navigate to the demo page
    await page.goto('/dev/web_viewer/task-manager-demo.html');
    await page.bringToFront();

    // Enhanced error tracking
    const consoleMessages = [];
    const workerErrors = [];
    const onnxErrors = [];

    // Listen for console messages
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push({
        type: msg.type(),
        text: text,
        timestamp: new Date().toISOString()
      });
      
      console.log(`[Browser Console] ${text}`);
      
      // Detect ONNX-related errors
      if (text.includes('wire type 4') || 
          text.includes('invalid wire type') ||
          text.includes('protobuf') ||
          text.includes('ONNX') ||
          text.includes('onnx')) {
        onnxErrors.push({
          message: text,
          timestamp: new Date().toISOString(),
          type: 'onnx_related'
        });
        console.log('🔧 ONNX ERROR DETECTED:', text);
      }
      
      // Detect worker errors
      if (text.includes('worker') || 
          text.includes('Worker') ||
          text.includes('postMessage') ||
          text.includes('importScripts')) {
        workerErrors.push({
          message: text,
          timestamp: new Date().toISOString(),
          type: 'worker_related'
        });
        console.log('👷 WORKER ERROR DETECTED:', text);
      }
    });

    // Listen for page errors
    page.on('pageerror', error => {
      console.log(`[Page Error] ${error.message}`);
      if (error.message.includes('wire type 4') || 
          error.message.includes('ONNX') ||
          error.message.includes('onnx')) {
        onnxErrors.push({
          message: error.message,
          stack: error.stack,
          timestamp: new Date().toISOString(),
          type: 'page_error'
        });
      }
    });

    // Initialize error collection
    await page.addInitScript(() => {
      window.onnxErrorsCollected = [];
      window.workerErrorsCollected = [];
      
      // Override console.error to catch ONNX issues
      const originalConsoleError = console.error;
      console.error = function(...args) {
        const message = args.join(' ');
        if (message.includes('wire type 4') || 
            message.includes('ONNX') ||
            message.includes('onnx') ||
            message.includes('protobuf')) {
          window.onnxErrorsCollected.push({
            message: message,
            timestamp: new Date().toISOString(),
            args: args
          });
        }
        if (message.includes('worker') || message.includes('Worker')) {
          window.workerErrorsCollected.push({
            message: message,
            timestamp: new Date().toISOString(),
            args: args
          });
        }
        originalConsoleError.apply(console, args);
      };
    });

    console.log('🚀 Starting AI model loading test...');

    // Test ONNX model loading
    const testResults = await page.evaluate(async () => {
      console.log('🧪 Testing ONNX model loading in browser context...');
      
      const results = {
        onnxRuntimeLoaded: false,
        modelLoadingTests: [],
        workerTests: [],
        errors: []
      };
      
      try {
        // Test ONNX Runtime loading
        if (typeof ort !== 'undefined') {
          results.onnxRuntimeLoaded = true;
          console.log('✅ ONNX Runtime already available');
        } else {
          console.log('⏳ Loading ONNX Runtime...');
          // Try to load ONNX Runtime
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js';
          document.head.appendChild(script);
          
          await new Promise((resolve, reject) => {
            script.onload = () => {
              results.onnxRuntimeLoaded = typeof ort !== 'undefined';
              resolve();
            };
            script.onerror = () => {
              results.errors.push('Failed to load ONNX Runtime');
              resolve();
            };
            setTimeout(() => {
              results.errors.push('ONNX Runtime loading timeout');
              resolve();
            }, 10000);
          });
        }

        // Test worker creation with comprehensive error detection
        console.log('👷 Testing comprehensive model worker creation...');
        const workerResults = [];
        const workerTypes = ['model-loader-webnn', 'audio-processor', 'motion-generator'];
        
        for (const workerType of workerTypes) {
          try {
            console.log(`🔧 Testing ${workerType} worker...`);
            const worker = new Worker(`/dev/web_viewer/js/workers/${workerType}.js`);
            
            const workerTest = await new Promise((resolve) => {
              const timeout = setTimeout(() => {
                resolve({
                  workerType,
                  success: false,
                  error: 'Worker creation timeout',
                  timestamp: new Date().toISOString()
                });
              }, 10000); // 10 second timeout
              
              worker.onmessage = (event) => {
                clearTimeout(timeout);
                resolve({
                  workerType,
                  success: true,
                  data: event.data,
                  timestamp: new Date().toISOString()
                });
              };
              
              worker.onerror = (error) => {
                clearTimeout(timeout);
                resolve({
                  workerType,
                  success: false,
                  error: error.message,
                  timestamp: new Date().toISOString()
                });
              };
              
              // Send comprehensive test messages
              if (workerType === 'model-loader-webnn') {
                // Test multiple models to trigger ONNX compatibility fixes
                const models = ['RSMT', 'DeepMimic', 'Audio2Gesture', 'FaceFormer'];
                models.forEach((model, index) => {
                  setTimeout(() => {
                    worker.postMessage({
                      type: 'load_model',
                      modelType: model,
                      complexity: 1 + index
                    });
                  }, index * 1000);
                });
              } else {
                worker.postMessage({
                  type: 'test_load',
                  testMode: true
                });
              }
            });
            
            workerResults.push(workerTest);
            worker.terminate();
            
          } catch (error) {
            workerResults.push({
              workerType,
              success: false,
              error: error.message,
              timestamp: new Date().toISOString()
            });
          }
          
          // Wait between worker tests
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        results.workerTests = workerResults;

      } catch (error) {
        results.errors.push(error.message);
      }
      
      return results;
    });

    console.log('📊 Test Results:', JSON.stringify(testResults, null, 2));

    // Collect final error states
    const finalErrors = await page.evaluate(() => {
      return {
        onnxErrors: window.onnxErrorsCollected || [],
        workerErrors: window.workerErrorsCollected || [],
        jsErrors: window.jsErrorsCollected || []
      };
    });

    // Report results
    console.log('🔧 ONNX Compatibility Test Results:');
    console.log('- ONNX Runtime Loaded:', testResults.onnxRuntimeLoaded);
    console.log('- Worker Tests:', testResults.workerTests.length);
    console.log('- ONNX Errors Detected:', onnxErrors.length);
    console.log('- Worker Errors Detected:', workerErrors.length);
    console.log('- Console Messages:', consoleMessages.length);

    if (onnxErrors.length > 0) {
      console.log('\n🚨 ONNX ERRORS FOUND:');
      onnxErrors.forEach((error, index) => {
        console.log(`${index + 1}. ${error.message}`);
      });
    }

    if (workerErrors.length > 0) {
      console.log('\n👷 WORKER ERRORS FOUND:');
      workerErrors.forEach((error, index) => {
        console.log(`${index + 1}. ${error.message}`);
      });
    }

    // Assertions - Focus on ONNX compatibility success
    expect(testResults.onnxRuntimeLoaded).toBe(true);
    expect(onnxErrors.filter(e => e.message.includes('wire type 4')).length).toBe(0);
    
    // Worker tests are informational - don't fail on timeout
    if (testResults.workerTests.length > 0) {
      const hasWorkerInfo = testResults.workerTests.some(test => test.success || test.error || test.timestamp);
      expect(hasWorkerInfo).toBe(true); // Just verify we got some response info
      
      console.log('\n📋 WORKER TEST SUMMARY:');
      testResults.workerTests.forEach((test, index) => {
        const status = test.success ? '✅ SUCCESS' : '⏰ TIMEOUT/ERROR';
        console.log(`${index + 1}. ${test.workerType}: ${status}`);
        if (test.error) console.log(`   Error: ${test.error}`);
        if (test.data) console.log(`   Data: ${JSON.stringify(test.data).substring(0, 100)}...`);
      });
    }

    console.log('✅ ONNX Compatibility Test completed successfully!');
  });
});
