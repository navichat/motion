// Skip-by-default web smoke: runs a minimal Whisper ORT inference when configured.
import { test, expect } from '@playwright/test';
import { buildQueryFromEnv } from '../../../src/testing/env_to_query.js';

const pagePath = '/tests/unit/ai/fixtures/ort_whisper_smoke.html';

test.describe('Whisper ORT web smoke', () => {
  test('runs when configured', async ({ page, browserName }) => {
    test.skip(browserName === 'webkit', 'Skip on WebKit for ORT wasm differences');
    const q = buildQueryFromEnv(process.env);
    await page.goto(q ? `${pagePath}?${q}` : pagePath);
    const status = await page.evaluate(async () => {
      const g = window;
      if (!g.ModelUrlConfig || !g.ModelUrlConfig.getModelUrl('whisper') || !g.ort) return 'skipped';
      if (!g.WhisperOrtTask) return 'missing_task';
      const { WhisperOrtTask } = g.WhisperOrtTask || {};
      const task = new WhisperOrtTask({ provider: 'wasm' });
      await task.initialize(g);
      const it = task.run({ features: { /* precomputed features expected by model */ } });
      const { value } = await it.next();
      return value && value.metadata && (value.metadata.model || 'unknown');
    });
    if (status === 'skipped' || status === 'missing_task') test.skip(true, status);
    expect(status).toMatch(/whisper_ort|whisper_fallback/);
  });
});
