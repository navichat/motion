const { test, expect } = require('@playwright/test');

test('check what is available in window', async ({ page }) => {
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
  await page.waitForTimeout(3000);
  
  const windowCheck = await page.evaluate(() => {
    return {
      'typeof window.FibonacciHeap': typeof window.FibonacciHeap,
      'window.FibonacciHeap': window.FibonacciHeap ? 'exists' : 'null/undefined',
      'typeof window.TaskManager': typeof window.TaskManager,
      'typeof window.runRealWorkloadTest': typeof window.runRealWorkloadTest,
      'window.FibonacciHeap constructor': window.FibonacciHeap ? window.FibonacciHeap.toString().substring(0, 100) : 'N/A'
    };
  });
  
  console.log('Window check results:', JSON.stringify(windowCheck, null, 2));
});
