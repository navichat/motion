import { test, expect } from '@playwright/test';

test('Test Enhanced Error Collection', async ({ page }) => {
  // Initialize error tracking arrays
  const jsErrors = [];
  const networkErrors = [];
  const unhandledRejections = [];
  const resourceErrors = [];
  const stackTraces = [];
  const consoleMessages = [];

  // Enhanced console handler with comprehensive error categorization
  page.on('console', msg => {
    const msgText = msg.text();
    const msgType = msg.type();
    const location = msg.location();
    
    consoleMessages.push({
      type: msgType,
      text: msgText,
      location: location,
      timestamp: new Date().toISOString()
    });
    
    console.log(`[${msgType.toUpperCase()}] ${msgText}`);
    
    if (msgType === 'error') {
      jsErrors.push({
        message: msgText,
        location: location,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Page error handler for uncaught exceptions
  page.on('pageerror', exception => {
    const errorInfo = {
      message: exception.message,
      stack: exception.stack,
      name: exception.name,
      timestamp: new Date().toISOString()
    };
    
    jsErrors.push(errorInfo);
    stackTraces.push(errorInfo);
    console.log('Page Error:', exception.message);
  });

  // Network error tracking
  page.on('response', response => {
    if (!response.ok()) {
      networkErrors.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText(),
        timestamp: new Date().toISOString()
      });
    }
  });

  // Request failure tracking
  page.on('requestfailed', request => {
    resourceErrors.push({
      url: request.url(),
      method: request.method(),
      failure: request.failure().errorText,
      timestamp: new Date().toISOString()
    });
  });

  // Client-side error injection for comprehensive error tracking
  await page.addInitScript(() => {
    // Override window.onerror to capture runtime errors
    window.onerror = function(message, source, lineno, colno, error) {
      console.error('Runtime Error:', {
        message: message,
        source: source,
        line: lineno,
        column: colno,
        error: error ? error.stack : 'No stack trace available'
      });
      return false; // Don't prevent default handling
    };

    // Capture unhandled promise rejections
    window.addEventListener('unhandledrejection', event => {
      console.error('Unhandled Promise Rejection:', {
        reason: event.reason,
        promise: event.promise,
        stack: event.reason && event.reason.stack ? event.reason.stack : 'No stack trace available'
      });
    });
  });

  // Navigate to our test page
  const testPagePath = 'file:///home/barberb/motion/dev/web_viewer/test-error-collection.html';
  await page.goto(testPagePath);

  // Wait for the page to load
  await page.waitForTimeout(1000);

  // Trigger different types of errors
  console.log('\n=== Triggering JavaScript Error ===');
  await page.click('#triggerError').catch(e => console.log('Expected error caught:', e.message));
  await page.waitForTimeout(500);

  console.log('\n=== Triggering Promise Rejection ===');
  await page.click('#triggerRejection');
  await page.waitForTimeout(500);

  console.log('\n=== Triggering Network Error ===');
  await page.click('#triggerNetworkError');
  await page.waitForTimeout(500);

  // Wait for delayed errors
  console.log('\n=== Waiting for delayed errors ===');
  await page.waitForTimeout(3000);

  // Generate comprehensive error report
  console.log('\n' + '='.repeat(80));
  console.log('COMPREHENSIVE ERROR COLLECTION REPORT');
  console.log('='.repeat(80));

  console.log(`\n📊 ERROR SUMMARY:`);
  console.log(`   • JavaScript Errors: ${jsErrors.length}`);
  console.log(`   • Network Errors: ${networkErrors.length}`);
  console.log(`   • Unhandled Rejections: ${unhandledRejections.length}`);
  console.log(`   • Resource Errors: ${resourceErrors.length}`);
  console.log(`   • Console Messages: ${consoleMessages.length}`);

  // Console message statistics
  const messageStats = consoleMessages.reduce((stats, msg) => {
    stats[msg.type] = (stats[msg.type] || 0) + 1;
    return stats;
  }, {});

  console.log(`\n📈 CONSOLE MESSAGE STATISTICS:`);
  Object.entries(messageStats).forEach(([type, count]) => {
    console.log(`   • ${type}: ${count}`);
  });

  // Detailed error breakdown
  if (jsErrors.length > 0) {
    console.log(`\n🔴 JAVASCRIPT ERRORS (${jsErrors.length}):`);
    jsErrors.forEach((error, index) => {
      console.log(`   ${index + 1}. ${error.message}`);
      if (error.location) {
        console.log(`      Location: ${error.location.url}:${error.location.lineNumber}:${error.location.columnNumber}`);
      }
      if (error.stack) {
        console.log(`      Stack: ${error.stack.split('\n')[0]}`);
      }
      console.log(`      Time: ${error.timestamp}`);
    });
  }

  if (networkErrors.length > 0) {
    console.log(`\n🌐 NETWORK ERRORS (${networkErrors.length}):`);
    networkErrors.forEach((error, index) => {
      console.log(`   ${index + 1}. HTTP ${error.status} - ${error.url}`);
      console.log(`      Status: ${error.statusText}`);
      console.log(`      Time: ${error.timestamp}`);
    });
  }

  if (resourceErrors.length > 0) {
    console.log(`\n📦 RESOURCE ERRORS (${resourceErrors.length}):`);
    resourceErrors.forEach((error, index) => {
      console.log(`   ${index + 1}. ${error.method} ${error.url}`);
      console.log(`      Failure: ${error.failure}`);
      console.log(`      Time: ${error.timestamp}`);
    });
  }

  console.log(`\n💬 RECENT CONSOLE MESSAGES:`);
  consoleMessages.slice(-10).forEach((msg, index) => {
    console.log(`   ${index + 1}. [${msg.type.toUpperCase()}] ${msg.text}`);
    if (msg.location && msg.location.url) {
      console.log(`      Source: ${msg.location.url}:${msg.location.lineNumber}`);
    }
  });

  console.log('\n' + '='.repeat(80));
  console.log('TEST COMPLETED - Error collection system is working!');
  console.log('='.repeat(80));

  // Verify we captured some errors
  expect(jsErrors.length).toBeGreaterThan(0);
  expect(consoleMessages.length).toBeGreaterThan(0);
});
