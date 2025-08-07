import { test, expect } from '@playwright/test';

test.describe('AI Model Tests', () => {
  const aiModels = [
    'DeepMimic',
    'FaceFormer',
    'Audio2Gesture',
    'RSMT',
    'Kokoro',
    'Whisper',
    'VAD',
    'TinyLlama',
    'DiabloGPT'
  ];

  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:8000/task-manager-demo.html');
  });

  for (const model of aiModels) {
    test(`should run ${model} test`, async ({ page }) => {
      const consoleLogs = [];
      page.on('console', msg => consoleLogs.push(msg.text()));

      await page.click(`button:has-text('${model}')`);

      // Wait for the completion message
      await page.waitForFunction(
        (modelName) => {
          return document.body.innerText.includes(`${modelName} Test Complete`);
        },
        model,
        { timeout: 120000 }
      );

      const logs = consoleLogs.join('\n');
      expect(logs).toContain(`[TaskManager] Evaluating task`);
      expect(logs).toContain(`[TaskManager] Received message from worker`);
      expect(logs).toContain(`${model} completed`);
    });
  }
});
