import { test, expect } from '@playwright/test';
import { buildQueryFromEnv } from '../../../src/testing/env_to_query.js';

const pagePath = '/tests/unit/ai/fixtures/ort_speecht5_smoke.html';

test.describe('SpeechT5 ORT web smoke', () => {
  test('runs when configured', async ({ page }) => {
  const q = buildQueryFromEnv(process.env);
  await page.goto(q ? `${pagePath}?${q}` : pagePath);
    const status = await page.evaluate(async () => {
      const g = window;
      if (!g.ModelUrlConfig || !g.ModelUrlConfig.getModelUrl('speecht5') || !g.ort) return 'skipped';
      if (!g.SpeechT5OrtTask) return 'missing_task';
      const { SpeechT5OrtTask } = g.SpeechT5OrtTask || {};
      const task = new SpeechT5OrtTask({ provider: 'wasm' });
      await task.initialize(g);
      const it = task.run({ features: {} });
      const { value } = await it.next();
      const model = value && value.metadata && value.metadata.model;
      return model || 'unknown';
    });
    if (status === 'skipped' || status === 'missing_task') test.skip(true, status);
    expect(status).toMatch(/speecht5_ort|speecht5_fallback/);
  });
});
