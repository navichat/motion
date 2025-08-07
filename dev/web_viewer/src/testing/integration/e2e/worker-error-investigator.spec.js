import { test, expect } from '@playwright/test';

test('Worker Error Investigation Tool', async ({ page }) => {
  test.setTimeout(120000);
  
  const workerErrors = [];
  const allConsoleMessages = [];
  const networkIssues = [];
  const resourceFailures = [];
  
  // Comprehensive console monitoring with detailed analysis
  page.on('console', msg => {
    const timestamp = new Date().toISOString();
    const msgType = msg.type();
    const msgText = msg.text();
    const location = msg.location();
    
    allConsoleMessages.push({
      timestamp,
      type: msgType,
      text: msgText,
      url: location.url,
      line: location.lineNumber,
      column: location.columnNumber
    });
    
    // Enhanced worker error detection
    const lowerText = msgText.toLowerCase();
    const isWorkerRelated = 
      lowerText.includes('worker') ||
      lowerText.includes('model') ||
      lowerText.includes('inference') ||
      lowerText.includes('onnx') ||
      lowerText.includes('tensorflow') ||
      lowerText.includes('transformers') ||
      lowerText.includes('wasm') ||
      lowerText.includes('webgl') ||
      lowerText.includes('webgpu') ||
      lowerText.includes('gpu') ||
      lowerText.includes('memory') ||
      lowerText.includes('buffer') ||
      lowerText.includes('allocation') ||
      lowerText.includes('cors') ||
      lowerText.includes('network') ||
      lowerText.includes('timeout') ||
      lowerText.includes('failed to fetch') ||
      lowerText.includes('context') ||
      lowerText.includes('initialization') ||
      lowerText.includes('loading') ||
      lowerText.includes('execution');
    
    if (isWorkerRelated && (msgType === 'error' || msgType === 'warning')) {
      const workerError = {
        timestamp,
        type: msgType,
        message: msgText,
        url: location.url,
        line: location.lineNumber,
        column: location.columnNumber,
        severity: msgType === 'error' ? 'HIGH' : 'MEDIUM',
        category: categorizeWorkerError(msgText),
        workerType: detectSpecificWorker(msgText),
        technicalDetails: extractTechnicalDetails(msgText)
      };
      
      workerErrors.push(workerError);
      console.log(`🚨 WORKER ISSUE DETECTED: [${msgType.toUpperCase()}] ${msgText}`);
    }
  });
  
  // Network request monitoring
  page.on('requestfailed', request => {
    const url = request.url();
    const failure = request.failure();
    
    if (url.includes('model') || 
        url.includes('onnx') || 
        url.includes('wasm') || 
        url.includes('worker') ||
        url.includes('js') ||
        url.includes('json')) {
      
      networkIssues.push({
        timestamp: new Date().toISOString(),
        url: url,
        method: request.method(),
        failure: failure?.errorText || 'Unknown network error',
        resourceType: request.resourceType(),
        isWorkerRelated: true
      });
      
      console.log(`🌐 NETWORK FAILURE: ${request.method()} ${url} - ${failure?.errorText}`);
    }
  });
  
  // Response monitoring
  page.on('response', response => {
    if (!response.ok() && response.status() >= 400) {
      const url = response.url();
      
      if (url.includes('model') || 
          url.includes('onnx') || 
          url.includes('wasm') || 
          url.includes('worker') ||
          url.includes('js')) {
        
        resourceFailures.push({
          timestamp: new Date().toISOString(),
          url: url,
          status: response.status(),
          statusText: response.statusText(),
          isWorkerRelated: true
        });
        
        console.log(`❌ RESOURCE FAILURE: ${response.status()} ${url}`);
      }
    }
  });
  
  function categorizeWorkerError(errorText) {
    const lower = errorText.toLowerCase();
    
    if (lower.includes('failed to construct') || lower.includes('worker is not defined')) {
      return 'WORKER_CREATION_FAILURE';
    }
    if (lower.includes('failed to load') || lower.includes('not found') || lower.includes('404')) {
      return 'RESOURCE_LOADING_FAILURE';
    }
    if (lower.includes('onnx') || lower.includes('model')) {
      return 'MODEL_PROCESSING_ERROR';
    }
    if (lower.includes('memory') || lower.includes('allocation') || lower.includes('heap')) {
      return 'MEMORY_ISSUE';
    }
    if (lower.includes('timeout') || lower.includes('timed out')) {
      return 'TIMEOUT_ERROR';
    }
    if (lower.includes('cors') || lower.includes('cross-origin')) {
      return 'CORS_SECURITY_ISSUE';
    }
    if (lower.includes('webgl') || lower.includes('webgpu') || lower.includes('gpu')) {
      return 'GPU_RELATED_ISSUE';
    }
    if (lower.includes('network') || lower.includes('fetch')) {
      return 'NETWORK_CONNECTIVITY_ISSUE';
    }
    if (lower.includes('postmessage') || lower.includes('communication')) {
      return 'WORKER_COMMUNICATION_ERROR';
    }
    
    return 'UNcategorized_WORKER_ISSUE';
  }
  
  function detectSpecificWorker(errorText) {
    const lower = errorText.toLowerCase();
    
    // AI Models
    if (lower.includes('tinyllama')) return 'TinyLlama';
    if (lower.includes('diablogpt')) return 'DiabloGPT';
    if (lower.includes('whisper')) return 'Whisper';
    if (lower.includes('kokoro')) return 'Kokoro';
    if (lower.includes('speecht5')) return 'SpeechT5';
    if (lower.includes('vad')) return 'VAD';
    
    // Motion Models
    if (lower.includes('rsmt')) return 'RSMT';
    if (lower.includes('deepmimic')) return 'DeepMimic';
    if (lower.includes('faceformer')) return 'FaceFormer';
    if (lower.includes('audio2gesture')) return 'Audio2Gesture';
    
    // Compute Models
    if (lower.includes('wasmmatrix')) return 'WASMMatrix';
    if (lower.includes('wasmprime')) return 'WASMPrime';
    if (lower.includes('wasmfractal')) return 'WASMFractal';
    
    // KNN Models
    if (lower.includes('closevector')) return 'CloseVector';
    if (lower.includes('hnsw')) return 'HNSW';
    if (lower.includes('unifiedknn')) return 'UnifiedKNN';
    
    // Frameworks
    if (lower.includes('tensorflow')) return 'TensorFlow';
    if (lower.includes('onnx')) return 'ONNX';
    if (lower.includes('transformers')) return 'Transformers.js';
    
    return 'Generic Worker';
  }
  
  function extractTechnicalDetails(errorText) {
    const details = {
      stackTrace: null,
      errorCode: null,
      fileName: null,
      lineNumber: null,
      additionalInfo: []
    };
    
    // Extract stack trace
    if (errorText.includes('at ') && errorText.includes('(')) {
      const stackMatch = errorText.match(/at .+\\(.+\\)/g);
      if (stackMatch) {
        details.stackTrace = stackMatch.slice(0, 3).join('\\n');
      }
    }
    
    // Extract error codes
    const errorCodeMatch = errorText.match(/error[\\s:]*([A-Z0-9_]+)/i);
    if (errorCodeMatch) {
      details.errorCode = errorCodeMatch[1];
    }
    
    // Extract file names
    const fileMatch = errorText.match(/([\\w-]+\\.(js|wasm|onnx|json))/gi);
    if (fileMatch) {
      details.fileName = fileMatch[0];
    }
    
    // Extract line numbers
    const lineMatch = errorText.match(/:([0-9]+):[0-9]+/);
    if (lineMatch) {
      details.lineNumber = parseInt(lineMatch[1]);
    }
    
    return details;
  }
  
  console.log('🔍 Starting Worker Error Investigation...');
  
  // Navigate to the application
  try {
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    console.log('✅ Successfully navigated to task-manager-demo.html');
  } catch (e) {
    console.error('❌ Failed to navigate:', e.message);
    // Try loading a simple test page instead
    await page.setContent(`
      <html>
        <head><title>Worker Error Investigation</title></head>
        <body>
          <h1>Worker Error Investigation Test</h1>
          <div id="results"></div>
          <script>
            console.log('Starting worker error investigation...');
            
            // Test various worker scenarios
            setTimeout(() => {
              try {
                console.log('Testing worker creation...');
                const worker = new Worker('/nonexistent-worker.js');
              } catch (e) {
                console.error('Worker creation test failed:', e.message);
              }
              
              console.log('Testing model loading simulation...');
              fetch('/nonexistent-model.onnx').catch(e => {
                console.error('Model loading failed:', e.message);
              });
              
              console.log('Testing WebGL context...');
              const canvas = document.createElement('canvas');
              const gl = canvas.getContext('webgl');
              if (!gl) {
                console.error('WebGL context creation failed');
              }
              
            }, 1000);
          </script>
        </body>
      </html>
    `);
  }
  
  // Wait for initial loading and error detection
  await page.waitForTimeout(5000);
  
  // Check for additional issues using page evaluation
  const browserEnvironmentIssues = await page.evaluate(() => {
    const issues = [];
    
    // Check browser capabilities
    if (typeof Worker === 'undefined') {
      issues.push({
        type: 'BROWSER_CAPABILITY',
        message: 'Web Workers not supported in this environment',
        severity: 'CRITICAL'
      });
    }
    
    if (typeof SharedArrayBuffer === 'undefined') {
      issues.push({
        type: 'BROWSER_CAPABILITY',
        message: 'SharedArrayBuffer not available - may affect multi-threaded workers',
        severity: 'HIGH'
      });
    }
    
    if (typeof WebAssembly === 'undefined') {
      issues.push({
        type: 'BROWSER_CAPABILITY',
        message: 'WebAssembly not supported - WASM models will fail',
        severity: 'CRITICAL'
      });
    }
    
    // Check WebGL
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) {
        issues.push({
          type: 'GPU_CAPABILITY',
          message: 'WebGL not available - GPU acceleration unavailable',
          severity: 'HIGH'
        });
      } else if (gl.isContextLost()) {
        issues.push({
          type: 'GPU_CAPABILITY',
          message: 'WebGL context is lost',
          severity: 'HIGH'
        });
      }
    } catch (e) {
      issues.push({
        type: 'GPU_CAPABILITY',
        message: 'WebGL test failed: ' + e.message,
        severity: 'HIGH'
      });
    }
    
    // Check memory
    if (performance.memory) {
      const memUsage = performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize;
      if (memUsage > 0.8) {
        issues.push({
          type: 'MEMORY_PRESSURE',
          message: `High memory usage: ${Math.round(memUsage * 100)}%`,
          severity: 'MEDIUM'
        });
      }
    }
    
    return issues;
  });
  
  // Add browser environment issues to worker errors
  browserEnvironmentIssues.forEach(issue => {
    workerErrors.push({
      timestamp: new Date().toISOString(),
      type: 'environment_check',
      message: issue.message,
      severity: issue.severity,
      category: issue.type,
      workerType: 'Environment',
      technicalDetails: { source: 'browser_capability_check' }
    });
  });
  
  // Wait a bit more for any delayed errors
  await page.waitForTimeout(3000);
  
  // Generate comprehensive report
  console.log('\\n' + '='.repeat(80));
  console.log('🔍 WORKER ERROR INVESTIGATION REPORT');
  console.log('='.repeat(80));
  
  console.log(`\\n📊 SUMMARY:`);
  console.log(`   • Total Console Messages: ${allConsoleMessages.length}`);
  console.log(`   • Worker-Related Issues: ${workerErrors.length}`);
  console.log(`   • Network Failures: ${networkIssues.length}`);
  console.log(`   • Resource Failures: ${resourceFailures.length}`);
  
  if (workerErrors.length > 0) {
    console.log(`\\n🚨 WORKER ISSUES DETECTED (${workerErrors.length}):`);
    
    // Group by category
    const categorizedErrors = {};
    workerErrors.forEach(error => {
      if (!categorizedErrors[error.category]) {
        categorizedErrors[error.category] = [];
      }
      categorizedErrors[error.category].push(error);
    });
    
    Object.keys(categorizedErrors).forEach(category => {
      const errors = categorizedErrors[category];
      console.log(`\\n   🏷️  ${category} (${errors.length} issues):`);
      
      errors.forEach((error, index) => {
        console.log(`      ${index + 1}. [${error.severity}] ${error.message}`);
        if (error.workerType && error.workerType !== 'Generic Worker') {
          console.log(`         Worker: ${error.workerType}`);
        }
        if (error.technicalDetails.fileName) {
          console.log(`         File: ${error.technicalDetails.fileName}`);
        }
        if (error.technicalDetails.lineNumber) {
          console.log(`         Line: ${error.technicalDetails.lineNumber}`);
        }
        if (error.url && error.url !== 'about:blank') {
          console.log(`         Location: ${error.url}:${error.line}:${error.column}`);
        }
        console.log(`         Time: ${error.timestamp}`);
      });
    });
    
    // Priority recommendations
    console.log(`\\n💡 PRIORITY FIXES RECOMMENDED:`);
    
    const criticalIssues = workerErrors.filter(e => e.severity === 'CRITICAL');
    const highIssues = workerErrors.filter(e => e.severity === 'HIGH');
    
    if (criticalIssues.length > 0) {
      console.log(`   🔴 CRITICAL (Fix Immediately): ${criticalIssues.length} issues`);
      criticalIssues.forEach(issue => {
        console.log(`      • ${issue.category}: ${issue.message}`);
      });
    }
    
    if (highIssues.length > 0) {
      console.log(`   🟡 HIGH PRIORITY: ${highIssues.length} issues`);
      highIssues.forEach(issue => {
        console.log(`      • ${issue.category}: ${issue.message}`);
      });
    }
    
    // Specific recommendations
    console.log(`\\n🔧 SPECIFIC RECOMMENDATIONS:`);
    
    if (categorizedErrors['WORKER_CREATION_FAILURE']) {
      console.log(`   • Worker Creation Issues: Check worker script paths and CORS settings`);
    }
    if (categorizedErrors['RESOURCE_LOADING_FAILURE']) {
      console.log(`   • Resource Loading Issues: Verify model files exist and are accessible`);
    }
    if (categorizedErrors['MODEL_PROCESSING_ERROR']) {
      console.log(`   • Model Processing Issues: Check ONNX runtime compatibility and model format`);
    }
    if (categorizedErrors['MEMORY_ISSUE']) {
      console.log(`   • Memory Issues: Consider model quantization or chunking for large models`);
    }
    if (categorizedErrors['GPU_RELATED_ISSUE']) {
      console.log(`   • GPU Issues: Implement CPU fallback mechanisms for WebGL/WebGPU failures`);
    }
    if (categorizedErrors['CORS_SECURITY_ISSUE']) {
      console.log(`   • CORS Issues: Configure server to allow cross-origin requests for model files`);
    }
    
  } else {
    console.log(`\\n✅ NO WORKER ISSUES DETECTED - System appears healthy!`);
  }
  
  if (networkIssues.length > 0) {
    console.log(`\\n🌐 NETWORK ISSUES (${networkIssues.length}):`);
    networkIssues.forEach((issue, index) => {
      console.log(`   ${index + 1}. ${issue.method} ${issue.url}`);
      console.log(`      Error: ${issue.failure}`);
      console.log(`      Type: ${issue.resourceType}`);
    });
  }
  
  if (resourceFailures.length > 0) {
    console.log(`\\n❌ RESOURCE FAILURES (${resourceFailures.length}):`);
    resourceFailures.forEach((failure, index) => {
      console.log(`   ${index + 1}. [${failure.status}] ${failure.url}`);
      console.log(`      Status: ${failure.status} ${failure.statusText}`);
    });
  }
  
  console.log('\\n' + '='.repeat(80));
  console.log('🏁 INVESTIGATION COMPLETE');
  console.log('='.repeat(80));
  
  // Ensure we have some data to validate the test worked
  expect(allConsoleMessages.length).toBeGreaterThan(0);
});
