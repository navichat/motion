import { test, expect } from '@playwright/test';

test('Run Real WASM/GPU/WebNN Workload Test', async ({ page }) => {
  let testCompleted = false;
  let finalStats = {};

  // Navigate to the demo page
  await page.goto('http://localhost:8000/task-manager-demo.html');

  // Listen for console messages
  page.on('console', msg => {
    const text = msg.text();
    console.log(`[Browser Console] ${text}`); // Log all browser console messages

    // Check for completion message
    if (text.includes('🎉 Real workload test completed in')) {
      testCompleted = true;
      const match = text.match(/completed: (\d+), failed: (\d+)/);
      if (match) {
        finalStats.completed = parseInt(match[1]);
        finalStats.failed = parseInt(match[2]);
      }
      const timeMatch = text.match(/completed in (\d+\.?\d*)s!/);
      if (timeMatch) {
        finalStats.totalTime = parseFloat(timeMatch[1]);
      }
    } else if (text.includes('⏰ Real workload test timeout')) {
      testCompleted = true;
      finalStats.timeout = true;
    }
  });

  // Click the "Real WASM/GPU/WebNN Workload" button
  console.log('Clicking "Real WASM/GPU/WebNN Workload" button...');
  await page.click('button:has-text("Real WASM/GPU/WebNN Workload")');

  // Wait for the test to complete or timeout (max 6 minutes for Playwright)
  await page.waitForFunction(() => window.testCompleted, { timeout: 360000 });

  // Report results
  if (finalStats.timeout) {
    console.log('\n⚠️ Workload test timed out.');
  } else {
    console.log('\n✅ Workload test completed.');
    console.log(`Total tasks completed: ${finalStats.completed}`);
    console.log(`Total tasks failed: ${finalStats.failed}`);
    console.log(`Total execution time: ${finalStats.totalTime}s`);
  }

  // Assert that at least some tasks were completed if not timed out
  if (!finalStats.timeout) {
    expect(finalStats.completed).toBeGreaterThan(0);
  }
});
