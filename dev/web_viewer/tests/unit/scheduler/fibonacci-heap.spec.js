import { test, expect } from '@playwright/test';

// Load in-browser by injecting the file immediately
test('FibonacciHeap basic ops', async ({ page }) => {
  await page.addScriptTag({ path: 'dev/web_viewer/src/utils/scheduler/FibonacciHeap.js' });
  await page.evaluate(() => {
    if (!window.FibonacciHeap) throw new Error('Heap missing');
  });

  const result = await page.evaluate(() => {
    const h = new window.FibonacciHeap();
    const n1 = h.insert(5, 'a');
    const n2 = h.insert(3, 'b');
    const n3 = h.insert(7, 'c');
    h.decreaseKey(n1, 2);
    const e1 = h.extractMin();
    const e2 = h.extractMin();
    const e3 = h.extractMin();
    return [e1.value, e2.value, e3.value];
  });

  expect(result).toEqual(['a', 'b', 'c']);
});
