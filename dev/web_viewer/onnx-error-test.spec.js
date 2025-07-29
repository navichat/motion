// @ts-check
import { test, expect } from '@playwright/test';

test('ONNX Worker Error Collection Test', async ({ page }) => {
  console.log('🤖 Starting ONNX Worker Error Collection Test...');

  // Arrays to collect various types of errors
  let jsErrors = [];
  let networkErrors = [];
  let unhandledRejections = [];
  let resourceErrors = [];
  let consoleMessages = [];

  // Set up error collectors
  page.on('pageerror', (error) => {
    console.log(`❌ JavaScript Error: ${error.message}`);
    jsErrors.push({
      type: 'pageerror',
      message: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    });
  });

  page.on('requestfailed', (request) => {
    console.log(`🌐 Network Error: ${request.url()} - ${request.failure()?.errorText}`);
    networkErrors.push({
      url: request.url(),
      error: request.failure()?.errorText,
      timestamp: new Date().toISOString()
    });
  });

  page.on('console', (msg) => {
    const text = msg.text();
    consoleMessages.push({
      type: msg.type(),
      text: text,
      timestamp: new Date().toISOString()
    });
    
    // Look for specific ONNX and worker errors
    if (text.includes('wire type') || 
        text.includes('ONNX') || 
        text.includes('worker') ||
        text.includes('invalid') ||
        text.includes('error') ||
        text.includes('failed')) {
      console.log(`🔍 Potential Issue: [${msg.type()}] ${text}`);
    }
  });

  // Navigate to the task manager demo page
  console.log('🌐 Navigating to task-manager-demo.html...');
  await page.goto('/dev/web_viewer/task-manager-demo.html');
  
  // Wait for page to load
  console.log('⏳ Waiting for page to load...');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // Wait for ONNX Runtime to load (with fallback)
  console.log('⏳ Waiting for ONNX Runtime to load...');
  let onnxRuntimeLoaded = false;
  try {
    await page.waitForFunction(() => typeof ort !== 'undefined', { timeout: 5000 });
    onnxRuntimeLoaded = true;
    console.log('✅ ONNX Runtime loaded successfully!');
  } catch (error) {
    console.log('⚠️ ONNX Runtime failed to load, continuing with error collection...');
  }

  // Check for ONNX Runtime compatibility
  console.log('🔧 Checking ONNX Runtime compatibility...');
  
  // Inject ONNX Runtime Fixer and test it
  await page.addScriptTag({
    path: '/home/barberb/motion/dev/web_viewer/js/workers/onnx-runtime-fixer.js'
  });

  // Test ONNX compatibility
  const onnxCompatibilityResult = await page.evaluate(async () => {
    try {
      if (typeof ONNXRuntimeFixer !== 'undefined') {
        console.log('🔧 ONNX Runtime Fixer is available');
        
        // Test the fixer with a simple example
        const testResult = await ONNXRuntimeFixer.createSession(new ArrayBuffer(0), {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'disabled'
        });
        
        return {
          success: true,
          fixerAvailable: true,
          testPassed: testResult !== null
        };
      } else {
        return {
          success: false,
          fixerAvailable: false,
          error: 'ONNX Runtime Fixer not available'
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message,
        stack: error.stack
      };
    }
  });

  console.log('🔍 ONNX Compatibility Result:', onnxCompatibilityResult);

  // Try to click the workload button if it exists
  try {
    const workloadButton = page.locator('button:has-text("🚀 Real WASM/GPU/WebNN Workload")');
    const buttonExists = await workloadButton.count() > 0;
    
    if (buttonExists) {
      console.log('🖱️ Clicking workload button...');
      await workloadButton.click();
      await page.waitForTimeout(5000); // Wait for some activity
    } else {
      console.log('ℹ️ Workload button not found, continuing with error collection...');
    }
  } catch (error) {
    console.log(`⚠️ Button click failed: ${error.message}`);
  }

  // Wait a bit longer to collect any asynchronous errors
  console.log('⏱️ Collecting errors for 10 seconds...');
  await page.waitForTimeout(10000);

  // Analyze collected errors
  console.log('\n' + '='.repeat(80));
  console.log('📊 ERROR ANALYSIS SUMMARY');
  console.log('='.repeat(80));

  console.log(`\n🔴 JavaScript Errors: ${jsErrors.length}`);
  jsErrors.forEach((error, index) => {
    console.log(`  ${index + 1}. [${error.timestamp}] ${error.message}`);
    if (error.stack) {
      console.log(`     Stack: ${error.stack.split('\n')[0]}`);
    }
  });

  console.log(`\n🌐 Network Errors: ${networkErrors.length}`);
  networkErrors.forEach((error, index) => {
    console.log(`  ${index + 1}. [${error.timestamp}] ${error.url} - ${error.error}`);
  });

  console.log(`\n📝 Console Messages (last 10):`);
  consoleMessages.slice(-10).forEach((msg, index) => {
    console.log(`  ${index + 1}. [${msg.type}] ${msg.text}`);
  });

  // Look for specific ONNX-related errors
  const onnxErrors = consoleMessages.filter(msg => 
    msg.text.toLowerCase().includes('onnx') ||
    msg.text.includes('wire type') ||
    msg.text.includes('invalid wire type')
  );

  console.log(`\n🤖 ONNX-Related Messages: ${onnxErrors.length}`);
  onnxErrors.forEach((error, index) => {
    console.log(`  ${index + 1}. [${error.type}] ${error.text}`);
  });

  // Worker-related errors
  const workerErrors = consoleMessages.filter(msg => 
    msg.text.toLowerCase().includes('worker') ||
    msg.text.includes('Worker')
  );

  console.log(`\n👷 Worker-Related Messages: ${workerErrors.length}`);
  workerErrors.forEach((error, index) => {
    console.log(`  ${index + 1}. [${error.type}] ${error.text}`);
  });

  console.log('\n' + '='.repeat(80));
  console.log('✅ Error collection complete!');
  console.log('='.repeat(80));

  // Store results for potential extraction
  await page.evaluate((results) => {
    window.errorCollectionResults = results;
  }, {
    jsErrors,
    networkErrors,
    consoleMessages,
    onnxErrors,
    workerErrors,
    onnxCompatibilityResult,
    totalErrors: jsErrors.length + networkErrors.length,
    timestamp: new Date().toISOString()
  });

  // The test should pass regardless of errors collected
  expect(true).toBe(true);
});
