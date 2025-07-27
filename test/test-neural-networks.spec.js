import { test, expect } from '@playwright/test';

test.describe('Neural Network Tests', () => {
  const models = [
    'DeepMimic',
    'FaceFormer',
    'Audio2Gesture',
    'RSMT',
    'Kokoro',
    'Whisper',
    'VAD',
    'TinyLlama',
    'DiabloGPT',
  ];

  test('should run all neural network models', async ({ page }) => {
    await page.goto('http://localhost:8000/task-manager-demo.html');

    for (const model of models) {
      console.log(`Testing model: ${model}`);
      await page.click(`button:has-text("${model}")`);
      await page.waitForFunction(() => window.taskManager.getStats().queue.completed > 0);
      const errors = await page.evaluate(() => window.consoleErrors || []);
      expect(errors).toHaveLength(0);
    }
  });
});
