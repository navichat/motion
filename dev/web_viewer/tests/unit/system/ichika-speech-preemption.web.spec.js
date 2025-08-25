import { test, expect } from '@playwright/test';

// Skip when no web server is running
test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Keep per-file timeout modest; we just verify logging
test.setTimeout(30_000);

test('Ichika demo: speech preempts face/audio and appends chunks (unit-web)', async ({ page }) => {
  const url = '/demos/ichika_classroom_demo.html';
  await page.goto(url);

  // Wait for demo to initialize orchestrator before interacting
  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));

  const log = page.locator('#log');

  // Start baseline
  await page.getByRole('button', { name: 'Start Idle' }).click();
  await expect(log).toContainText('Manifest entries:', { timeout: 5000 });
  await expect(log).toContainText('Started base clip:', { timeout: 5000 });

  // Click Speech and check logs for clearFrom on face/audio and appendChunk events
  await page.getByRole('button', { name: 'Speak (visemes + gestures)' }).click();

  // We expect at least one clearFrom for face/audio and at least one appendChunk afterwards
  // Being flexible on order but they should appear within a short window
  await expect(log).toContainText('Speech scheduled:', { timeout: 5000 });

  // Look for signatures; allow either exact track names used by orchestrator or generic ones
  // Common tracks: 'face', 'audio' (adapter/orchestrator convention in this repo)
  const text = await log.textContent();
  expect(text).toMatch(/clearFrom: track=(face|audio)/);
  expect(text).toMatch(/appendChunk: track=(face|audio)/);
});
