const { test, expect } = require('@playwright/test');

test('check scripts loading', async ({ page }) => {
  const consoleMessages = [];
  page.on('console', msg => {
    consoleMessages.push(`${msg.type()}: ${msg.text()}`);
  });
  
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
  await page.waitForTimeout(3000);
  
  const scriptCheck = await page.evaluate(() => {
    try {
      // Try to access the class directly
      const heapTest = new FibonacciHeap();
      return { directFibonacciHeap: 'works' };
    } catch (e) {
      return { directFibonacciHeap: e.message };
    }
  });
  
  console.log('Script check results:', JSON.stringify(scriptCheck, null, 2));
  console.log('Console messages:');
  consoleMessages.forEach(msg => console.log(`  ${msg}`));
});
