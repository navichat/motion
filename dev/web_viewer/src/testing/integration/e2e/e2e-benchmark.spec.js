import { test, expect } from '@playwright/test';

test('Run all AI model tests', async ({ page }) => {
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');

  // Listen for console messages and log them to the terminal
  page.on('console', msg => console.log(msg.text()));

  // Click the "Run All AI Model Tests" button
  await page.click('button:has-text("AI Model Tests")');

  // Wait for the tests to complete
  await page.waitForSelector('text=All AI models completed', { timeout: 60000 });
});