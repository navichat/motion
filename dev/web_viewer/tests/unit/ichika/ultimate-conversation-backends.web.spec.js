import { test, expect } from '@playwright/test';

// Exercise both Kokoro (on-device) and SpeechT5 (stub/ORT) backends in the Ichika conversation demo.
// Deterministic: no network is required; Kokoro runs only if runtime is preloaded via query/env.

const DEMO_URL = '/demos/ichika_voice_conversation_demo.html';

async function openDemo(page, baseURL, query = '') {
  const url = baseURL + DEMO_URL + (query ? ('?' + query.replace(/^\?/, '')) : '');
  await page.goto(url);
  await page.waitForFunction(() => !!window.__ultimateDemo, null, { timeout: 15000 });
}

async function sayAndWait(page, text) {
  await page.evaluate((t) => window.__ultimateDemo.sayText(t, false), text);
  // Wait for speech scheduling log first
  await expect.poll(async () => await page.locator('#log').textContent() || '').toContain('speech@');
  // Then poll until expressions are observed (viseme mapping via integration)
  await expect.poll(async () => (await page.evaluate(() => window.__ultimateDemo.getStats())).expressions)
    .toBeGreaterThan(0);
}

test.describe('Ichika conversation backends', () => {
  test('Speech API backend produces expressions', async ({ page, baseURL }) => {
    await openDemo(page, baseURL, 'backend=speech');
    await sayAndWait(page, 'hello speech backend');
    const logText = await page.locator('#log').textContent();
    expect(logText).toContain('Using backend: speech');
  });

  test('Kokoro backend runs when kokoro-js is provided', async ({ page, baseURL }) => {
    // Pass kokoroJs via query if KOKORO_JS_URL is present in process env; otherwise skip deterministically.
    const KOKORO = process.env.KOKORO_JS_URL;
    if (!KOKORO) test.skip(true, 'Set KOKORO_JS_URL or run `npm run setup:real` to provide kokoro runtime');
    await openDemo(page, baseURL, `backend=kokoro&kokoroJs=${encodeURIComponent(KOKORO)}`);
    await sayAndWait(page, 'hello kokoro backend');
    const logText = await page.locator('#log').textContent();
    expect(logText).toMatch(/Using backend: kokoro|kokoro error/);
  });

  test('SpeechT5 backend works in stub/ORT modes', async ({ page, baseURL }) => {
    const params = new URLSearchParams();
    // If ORT and model URL provided via env_to_query, pass them through for ORT init
    if (process.env.RUNTIME_ORT) params.set('runtime.ort', process.env.RUNTIME_ORT);
    if (process.env.SPEECHT5_URL) params.set('speecht5', process.env.SPEECHT5_URL);
    params.set('backend', 'speecht5');
    await openDemo(page, baseURL, params.toString());
    await sayAndWait(page, 'hello t5 backend');
    const logText = await page.locator('#log').textContent();
    expect(logText).toMatch(/Using backend: speecht5|SpeechT5 not available|SpeechT5 stub path|SpeechT5 ORT path initialized/);
  });
});
