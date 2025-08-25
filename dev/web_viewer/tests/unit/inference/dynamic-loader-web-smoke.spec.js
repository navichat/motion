import { test, expect } from '@playwright/test';
import { buildQueryFromEnv } from '../../../src/testing/env_to_query.js';

const pagePath = '/tests/unit/inference/fixtures/legacy_exposure.html';

test.describe('Dynamic loader (unit-web)', () => {
  test('inserts transformers and kokoro-js when configured', async ({ page }) => {
    const q = buildQueryFromEnv({
      TRANSFORMERS_URL: 'https://example.com/transformers.min.js',
      KOKORO_JS_URL: 'https://example.com/kokoro.min.js',
    });
  await page.goto(`${pagePath}?${q}`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('script[src="https://example.com/transformers.min.js"]', { state: 'attached', timeout: 10000 });
  await page.waitForSelector('script[src="https://example.com/kokoro.min.js"]', { state: 'attached', timeout: 10000 });
    const out = await page.evaluate(() => ({
      tf: !!document.querySelector('script[src="https://example.com/transformers.min.js"]'),
      kok: !!document.querySelector('script[src="https://example.com/kokoro.min.js"]'),
    }));
    expect(out.tf).toBe(true);
    expect(out.kok).toBe(true);
  });
});
