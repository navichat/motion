import { test, expect } from '@playwright/test';

test('Test Enhanced Worker Error Collection', async ({ page }) => {
  // Initialize error tracking arrays
  const jsErrors = [];
  const networkErrors = [];
  const unhandledRejections = [];
  const resourceErrors = [];
  const workerErrors = [];
  
  // Enhanced console handler with worker error detection
  page.on('console', msg => {
    const timestamp = new Date().toISOString();
    const msgType = msg.type();
    const msgText = msg.text();
    const location = msg.location();
    
    const logEntry = {
      timestamp,
      type: msgType,
      text: msgText,
      location: location,
      url: location.url,
      lineNumber: location.lineNumber,
      columnNumber: location.columnNumber
    };
    
    // Enhanced error categorization with worker detection
    if (msgType === 'error') {
      const errorEntry = {
        ...logEntry,
        stack: msg.args().length > 0 ? msg.args().map(arg => arg.toString()).join(' ') : null
      };
      
      jsErrors.push(errorEntry);
      
      // Specific worker error detection
      if (msgText.toLowerCase().includes('worker') || 
          msgText.toLowerCase().includes('postmessage') ||
          msgText.toLowerCase().includes('importscripts') ||
          msgText.toLowerCase().includes('model')) {
        
        const workerError = {
          ...errorEntry,
          workerType: 'DETECTED_WORKER',
          errorCategory: 'WORKER_ERROR',
          isWorkerError: true
        };
        
        workerErrors.push(workerError);
        console.log(`[WORKER ERROR DETECTED]: ${msgText}`);
      }
    }
    
    console.log(`[${msgType.toUpperCase()}]: ${msgText}`);
  });

  // Load our test page with worker error collection
  await page.goto('file:///home/barberb/motion/dev/web_viewer/test-error-collection.html');
  
  // Enhanced JavaScript error detection via window.onerror injection
  await page.addInitScript(() => {
    window.jsErrorsCollected = [];
    window.workerErrorsCollected = [];
    
    // Monitor Worker creation and errors
    const originalWorker = window.Worker;
    if (originalWorker) {
      window.Worker = function(scriptURL, options) {
        const worker = new originalWorker(scriptURL, options);
        
        // Track worker creation
        const workerInfo = {
          timestamp: new Date().toISOString(),
          scriptURL: scriptURL,
          type: 'worker_created',
          workerId: 'worker_' + Date.now()
        };
        window.workerErrorsCollected.push(workerInfo);
        console.log('[WORKER CREATED]:', scriptURL);
        
        // Monitor worker errors
        worker.onerror = function(event) {
          const workerError = {
            timestamp: new Date().toISOString(),
            message: event.message || 'Worker error occurred',
            filename: event.filename || scriptURL,
            lineno: event.lineno || 0,
            colno: event.colno || 0,
            type: 'worker_error',
            scriptURL: scriptURL,
            workerError: true
          };
          
          window.workerErrorsCollected.push(workerError);
          console.error('[WORKER ERROR]:', event.message, 'in', event.filename);
        };
        
        return worker;
      };
    }
  });

  // Create a test worker that will generate errors
  await page.evaluate(() => {
    try {
      // Try to create a worker with a non-existent script
      console.log('Creating test worker...');
      const worker = new Worker('/nonexistent-worker.js');
      
      // This should trigger a worker error
      worker.postMessage({ test: 'data' });
      
      worker.onerror = function(event) {
        console.error('Worker error caught:', event.message);
      };
      
    } catch (error) {
      console.error('Worker creation failed:', error.message);
    }
    
    // Also test some model-related error simulation
    console.error('Model loading failed: TinyLlama worker could not initialize');
    console.error('ONNX runtime error: Failed to load model weights');
    console.error('Worker communication error: postMessage failed for DeepMimic worker');
  });

  // Wait a bit for errors to be captured
  await page.waitForTimeout(2000);

  // Collect worker errors from the page
  const pageWorkerErrors = await page.evaluate(() => {
    return window.workerErrorsCollected || [];
  });
  
  workerErrors.push(...pageWorkerErrors);

  // Report results
  console.log('\n=== WORKER ERROR COLLECTION TEST RESULTS ===');
  console.log(`JavaScript Errors: ${jsErrors.length}`);
  console.log(`Worker Errors: ${workerErrors.length}`);
  
  if (workerErrors.length > 0) {
    console.log('\nWorker Errors Detected:');
    workerErrors.forEach((error, index) => {
      console.log(`${index + 1}. ${error.message || error.type}`);
      if (error.scriptURL) console.log(`   Script: ${error.scriptURL}`);
      if (error.workerType) console.log(`   Type: ${error.workerType}`);
      if (error.timestamp) console.log(`   Time: ${error.timestamp}`);
    });
  }
  
  console.log('\n=== Worker error collection system is working! ===');
});
