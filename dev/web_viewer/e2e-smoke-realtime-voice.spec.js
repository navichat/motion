import { test, expect } from '@playwright/test';

// This test uses Chrome flags to provide a fake mic stream in CI
// It verifies the voice conversation page emits viseme and gesture energy signals.

test('[E2E][VRM][Realtime][Voice] Mic + TTS streaming produces metadata signals', async ({ page, browserName }) => {
  // Navigate to the voice page
  await page.goto('/tests/e2e/html/realtime-voice-conversation.html');

  // Wait for the page to set its stats
  await page.waitForFunction(() => !!window.__voiceSmoke, null, { timeout: 10_000 });
  const stats = await page.evaluate(() => window.__voiceSmoke);
  if (stats && stats.error) {
    throw new Error('voice smoke error: ' + stats.error);
  }

  expect(stats.samples).toBeGreaterThanOrEqual(5);
  expect(stats.sawEnergy).toBeTruthy();
  expect(stats.sawViseme).toBeTruthy();
});

test('[E2E][VRM][Realtime][Voice][Mic] Uses MediaStream when fake mic flags enabled', async ({ page, browserName, context }) => {
  // Only meaningful on Chromium where we can pass mic flags
  test.skip(browserName !== 'chromium', 'Fake mic flags only supported on Chromium in this setup');

  await page.goto('/tests/e2e/html/realtime-voice-conversation.html');
  await page.waitForFunction(() => !!window.__voiceSmoke, null, { timeout: 10_000 });
  const stats = await page.evaluate(() => window.__voiceSmoke);
  if (stats && stats.error) throw new Error('voice smoke error: ' + stats.error);

  // micUsed is true if getUserMedia succeeded; CI passes fake device/ui flags so this should be true
  expect(stats.micUsed).toBeTruthy();
});
