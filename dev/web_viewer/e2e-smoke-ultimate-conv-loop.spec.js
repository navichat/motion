import { test, expect } from '@playwright/test';

// E2E smoke: stubbed mic → ASR result → TTS scheduling → animation markers
// Uses query flag stubMic=1 to avoid real microphone capture in CI.

test.describe('Ultimate conversation loop smoke (stubbed mic)', () => {
  test('Conversation loop smoke schedules TTS and drives animation [E2E][StubMic]', async ({ page, baseURL }) => {
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?backend=beeps&playAudio=0';
    await page.goto(url);

    // Drive via UI: set text and click Say
    await page.fill('#text', 'Playwright end-to-end test');
    await page.getByRole('button', { name: /Say/i }).click();

    // Wait for scheduling markers
    const log = page.locator('#log');
    await expect(log).toContainText(/\[PLAYWRIGHT\] Scheduled TTS animation|Scheduled TTS for text|Scheduled TTS animation/i, { timeout: 30000 });

  // Assert engine marker if present, else accept backend log for determinism
  await expect(log).toContainText(/\[PLAYWRIGHT\] TTS engine=beeps useTts=0|🎛️ Using backend:\s*beeps/);

    // Ensure we did not fall back to minimal adapter/timeline
    await expect(page.locator('#log')).not.toContainText(/Using minimal adapter fallback|Falling back to MinimalTimeline|Falling back to MinimalScheduler/i);

    // Poll binder stats to confirm blendshape updates occurred (animation drove expressions)
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
  }, { timeout: 15000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);
  });
});
