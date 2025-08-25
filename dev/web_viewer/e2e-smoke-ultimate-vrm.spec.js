import { test, expect } from '@playwright/test';

// E2E Smoke: attempt to load VRM (if available) and ensure scheduling/expressions still work

test.describe('Ultimate VRM attempt smoke', () => {
  test('Loads demo with vrm=1 and schedules animation [E2E][smoke]', async ({ page, baseURL }) => {
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0';
    await page.goto(url);

    // Drive via UI: set text and click Say
    await page.fill('#text', 'VRM smoke test');
    await page.getByRole('button', { name: /Say/i }).click();

    const log = page.locator('#log');
    // Accept any scheduling marker variant used in the demo
    await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 30000 });

    // Expressions should increase regardless of VRM presence (stub binder records too)
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
    }, { timeout: 15000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);

    // Optional: log indicates VRM status; do not fail if unavailable
    // await expect(log).toContainText(/VRM loaded\.|VRM not found locally; running in stub mode\.|VRM load failed:/);
  });
});
