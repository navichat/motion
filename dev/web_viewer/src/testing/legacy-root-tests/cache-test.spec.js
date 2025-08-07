const { test, expect } = require('@playwright/test');

test('Force cache refresh and test pool selection', async ({ page }) => {
  console.log('🔍 Testing with cache refresh...');
  
  // Add cache busting parameter
  const timestamp = Date.now();
  await page.goto(`http://localhost:8000/dev/web_viewer/task-manager-demo.html?v=${timestamp}`);
  
  // Force hard refresh
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);
  
  // Check if our TaskManager modification is loaded
  const poolCheckCode = await page.evaluate(() => {
    // Look for the pool selection logic in TaskManager
    if (window.TaskManager && window.TaskManager.prototype._findWorkerForTask) {
      return window.TaskManager.prototype._findWorkerForTask.toString();
    }
    return 'TaskManager not found';
  });
  
  console.log('🔍 Pool selection code includes ONNX check:', poolCheckCode.includes('requirements.onnx'));
  
  // Now run the test
  const messages = [];
  page.on('console', msg => {
    messages.push(msg.text());
  });
  
  await page.evaluate(() => {
    if (window.runRealWorkloadTest) {
      window.runRealWorkloadTest();
    }
  });
  
  await page.waitForTimeout(8000);
  
  // Check potential pools messages
  const poolMessages = messages.filter(msg => msg.includes('Potential pools'));
  console.log('🔍 Pool selection messages:');
  poolMessages.slice(0, 5).forEach(msg => console.log(msg));
});
