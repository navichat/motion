import { test, expect } from '@playwright/test';
import { buildQueryFromEnv } from '../../../src/testing/env_to_query.js';

const pagePath = '/tests/unit/ai/fixtures/ort_silero_vad_smoke.html';

test.describe('Silero VAD ORT web smoke', () => {
  test('runs when configured', async ({ page }) => {
  const q = buildQueryFromEnv(process.env);
  await page.goto(q ? `${pagePath}?${q}` : pagePath);
    const status = await page.evaluate(async () => {
      const g = window;
      if (!g.ModelUrlConfig || !g.ModelUrlConfig.getModelUrl('sileroVad') || !g.ort) return 'skipped';
      if (!g.SileroVadOrtTask) return 'missing_task';
      const { SileroVadOrtTask } = g.SileroVadOrtTask || {};
      const task = new SileroVadOrtTask({ provider: 'wasm' });
      await task.initialize(g);
      const it = task.run({ features: { /* features or frames */ } });
      const { value } = await it.next();
      const model = value && value.metadata && value.metadata.model;
      return model || 'unknown';
    });
    if (status === 'skipped' || status === 'missing_task') test.skip(true, status);
    expect(status).toMatch(/silero_vad_ort|silero_vad_fallback/);
  });
});
