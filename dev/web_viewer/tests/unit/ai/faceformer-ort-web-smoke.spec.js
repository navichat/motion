import { test, expect } from '@playwright/test';
import { buildQueryFromEnv } from '../../../src/testing/env_to_query.js';

const pagePath = '/tests/unit/ai/fixtures/ort_faceformer_smoke.html';

test.describe('Faceformer ORT web smoke', () => {
  test('runs when configured', async ({ page }) => {
  const q = buildQueryFromEnv(process.env);
  await page.goto(q ? `${pagePath}?${q}` : pagePath);
    const status = await page.evaluate(async () => {
      const g = window;
      if (!g.ModelUrlConfig || !g.ModelUrlConfig.getModelUrl('faceformer') || !g.ort) return 'skipped';
      if (!g.FaceformerOrtTask) return 'missing_task';
      const { FaceformerOrtTask } = g.FaceformerOrtTask || {};
      const task = new FaceformerOrtTask({ provider: 'wasm' });
      await task.initialize(g);
      const it = task.run({ inputs: { /* viseme inputs */ } });
      const { value } = await it.next();
      const model = value && value.metadata && value.metadata.model;
      return model || 'unknown';
    });
    if (status === 'skipped' || status === 'missing_task') test.skip(true, status);
    expect(status).toMatch(/faceformer_ort|faceformer_fallback/);
  });
});
