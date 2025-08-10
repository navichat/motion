import { test, expect } from '@playwright/test';
import { buildQueryFromEnv } from '../../../src/testing/env_to_query.js';

const pagePath = '/tests/unit/inference/fixtures/legacy_exposure.html';

test.describe('Legacy module exposure (unit-web)', () => {
  test('exposes ResourceManager and optional legacy modules when LEGACY=1', async ({ page }) => {
    const q = buildQueryFromEnv({ LEGACY: '1' });
    await page.goto(`${pagePath}?${q}`);
    const summary = await page.evaluate(() => {
      // Wait a tick for the inline module to import and attach
      return new Promise((resolve) => setTimeout(() => resolve({
        ResourceManager: !!window.ResourceManager,
        WhisperModule: !!window.WhisperModule,
        KokoroModule: !!window.KokoroModule,
        LlamaModule: !!window.LlamaModule,
      }), 50));
    });
    expect(summary.ResourceManager).toBeTruthy();
    // At least one of the legacy modules should be available in this repo
    expect(summary.WhisperModule || summary.KokoroModule || summary.LlamaModule).toBeTruthy();
  });
});
