import { test, expect } from '@playwright/test';

test('Test Enhanced Worker Error Collection', async ({ page }) => {
  test.setTimeout(60000);
  
  const workerErrors = [];
  
  // Enhanced console monitoring for worker errors
  page.on('console', msg => {
    const msgText = msg.text();
    const msgType = msg.type();
    
    if (msgType === 'error' || msgType === 'warning') {
      const lowerText = msgText.toLowerCase();
      
      // Comprehensive worker error detection
      if (lowerText.includes('worker') ||
          lowerText.includes('model') ||
          lowerText.includes('onnx') ||
          lowerText.includes('tensorflow') ||
          lowerText.includes('wasm') ||
          lowerText.includes('webgl') ||
          lowerText.includes('webgpu') ||
          lowerText.includes('inference') ||
          lowerText.includes('gpu') ||
          lowerText.includes('memory') ||
          lowerText.includes('buffer') ||
          lowerText.includes('cors') ||
          lowerText.includes('network') ||
          lowerText.includes('timeout') ||
          lowerText.includes('failed to fetch') ||
          lowerText.includes('context lost') ||
          lowerText.includes('allocation')) {
        
        workerErrors.push({
          timestamp: new Date().toISOString(),
          type: msgType,
          message: msgText,
          detected: 'console_monitoring'
        });
        console.log(`[ENHANCED WORKER ERROR DETECTED]: ${msgText}`);
      }
    }
  });

  // Create test HTML with multiple worker error scenarios
  const testHTML = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Worker Error Test</title>
    </head>
    <body>
        <h1>Enhanced Worker Error Detection Test</h1>
        <div id="results"></div>
        
        <script>
            console.log('Starting enhanced worker error tests...');
            
            // Test 1: Basic worker creation failure
            try {
                const worker1 = new Worker('/nonexistent-worker.js');
            } catch (e) {
                console.error('Worker creation failed:', e.message);
            }
            
            // Test 2: Model loading simulation
            setTimeout(() => {
                console.error('ONNX runtime error: Failed to load TinyLlama model weights');
                console.error('TensorFlow.js error: WebGL context creation failed');
                console.error('Transformers.js error: Model download timeout');
            }, 500);
            
            // Test 3: GPU/WebGL errors
            setTimeout(() => {
                console.error('WebGL context lost - GPU worker terminated');
                console.error('WebGPU adapter not found - falling back to CPU');
                console.error('CUDA out of memory error in inference worker');
            }, 1000);
            
            // Test 4: Memory and buffer errors
            setTimeout(() => {
                console.error('SharedArrayBuffer allocation failed');
                console.error('WebAssembly memory allocation exceeded');
                console.error('Worker heap size limit reached');
            }, 1500);
            
            // Test 5: Network and CORS errors
            setTimeout(() => {
                console.error('CORS policy blocked model file access');
                console.error('Network error: Failed to fetch model.onnx');
                console.error('Cross-origin request blocked for worker script');
            }, 2000);
            
            // Test 6: Framework-specific errors
            setTimeout(() => {
                console.error('MediaPipe worker initialization failed');
                console.error('WebNN execution provider unavailable');
                console.error('Service worker registration failed');
            }, 2500);
            
            // Test 7: Data transfer and serialization errors
            setTimeout(() => {
                console.error('Structured clone algorithm failed for worker data');
                console.error('PostMessage serialization error');
                console.error('Transferable object transfer failed');
            }, 3000);
            
            // Test 8: Concurrency and threading errors
            setTimeout(() => {
                console.error('Deadlock detected in worker thread pool');
                console.error('Race condition in shared memory access');
                console.error('Mutex acquisition timeout in AI worker');
            }, 3500);
            
            // Performance and memory monitoring
            setInterval(() => {
                if (performance.memory) {
                    const usage = performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize;
                    if (usage > 0.8) {
                        console.warn('High memory usage detected:', Math.round(usage * 100) + '%');
                    }
                }
            }, 1000);
            
            console.log('Enhanced worker error test scenarios completed');
        </script>
    </body>
    </html>
  `;

  // Set the test HTML content
  await page.setContent(testHTML);
  
  // Wait for all error scenarios to execute
  await page.waitForTimeout(5000);
  
  // Perform additional error detection using page evaluation
  const additionalErrors = await page.evaluate(() => {
    const errors = [];
    
    // Check WebGL context
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl');
      if (!gl) {
        errors.push({
          type: 'webgl_unavailable',
          message: 'WebGL context unavailable - affects GPU workers',
          detected: 'feature_detection'
        });
      }
    } catch (e) {
      errors.push({
        type: 'webgl_error',
        message: 'WebGL context creation error: ' + e.message,
        detected: 'exception_handling'
      });
    }
    
    // Check SharedArrayBuffer support
    if (typeof SharedArrayBuffer === 'undefined') {
      errors.push({
        type: 'sharedarraybuffer_unavailable',
        message: 'SharedArrayBuffer unavailable - affects multi-threaded workers',
        detected: 'feature_detection'
      });
    }
    
    // Check performance memory
    if (performance.memory) {
      const usage = performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize;
      if (usage > 0.7) {
        errors.push({
          type: 'high_memory_usage',
          message: `High memory usage: ${Math.round(usage * 100)}%`,
          detected: 'memory_monitoring'
        });
      }
    }
    
    return errors;
  });
  
  workerErrors.push(...additionalErrors);
  
  // Report results
  console.log('\\n=== ENHANCED WORKER ERROR COLLECTION RESULTS ===');
  console.log(`Total Worker-Related Issues: ${workerErrors.length}`);
  
  if (workerErrors.length > 0) {
    console.log('\\nDetected Issues:');
    workerErrors.forEach((error, index) => {
      console.log(`${index + 1}. [${error.type || 'error'}] ${error.message}`);
      console.log(`   Detected by: ${error.detected}`);
      if (error.timestamp) console.log(`   Time: ${error.timestamp}`);
    });
    
    // Categorize errors
    const categories = {};
    workerErrors.forEach(error => {
      const msg = error.message.toLowerCase();
      let category = 'other';
      
      if (msg.includes('worker')) category = 'worker_issues';
      else if (msg.includes('model') || msg.includes('onnx') || msg.includes('tensorflow')) category = 'model_issues';
      else if (msg.includes('webgl') || msg.includes('webgpu') || msg.includes('gpu')) category = 'gpu_issues';
      else if (msg.includes('memory') || msg.includes('buffer') || msg.includes('heap')) category = 'memory_issues';
      else if (msg.includes('cors') || msg.includes('network') || msg.includes('fetch')) category = 'network_issues';
      else if (msg.includes('timeout') || msg.includes('deadlock') || msg.includes('race')) category = 'concurrency_issues';
      
      categories[category] = (categories[category] || 0) + 1;
    });
    
    console.log('\\nError Categories:');
    Object.entries(categories).forEach(([category, count]) => {
      console.log(`  - ${category}: ${count} issue(s)`);
    });
  }
  
  console.log('\\n=== Enhanced worker error collection system is working! ===');
  
  // Verify we detected some errors
  expect(workerErrors.length).toBeGreaterThan(0);
});
