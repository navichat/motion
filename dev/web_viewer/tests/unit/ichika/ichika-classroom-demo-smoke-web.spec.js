import { test, expect } from '@playwright/test';

// Smoke: Loads the Ichika classroom demo, starts base clip, triggers wave and speech scheduling, and verifies adapter logs.

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Ichika classroom demo schedules base, wave, and speech [VRM][classroom]', async ({ page }) => {
  await page.goto('/demos/ichika_classroom_demo.html');

  // Ensure demo initialized and handlers are attached
  await page.waitForFunction(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));
  await page.waitForFunction(() => !!document.getElementById('start') && typeof document.getElementById('start').onclick === 'function');

  // Start base
  await page.click('#start');
  // Allow inline wiring and any fetch to settle
  await page.waitForFunction(() => (document.getElementById('log')?.textContent || '').includes('Manifest entries:'), { timeout: 5000 });

  const logAfterStart = await page.textContent('#log');
  expect(logAfterStart || '').toMatch(/Manifest entries:/);
  expect(logAfterStart || '').toMatch(/Started base clip:/);

  // Trigger wave
  await page.click('#wave');
  await page.waitForTimeout(50);
  const logAfterWave = await page.textContent('#log');
  expect((logAfterWave || '').toLowerCase()).toMatch(/wave/);
  // Should have appended a chunk (either BVH or manifest)
  expect(logAfterWave || '').toMatch(/appendChunk: track=/);

  // Trigger speech
  await page.click('#speech');
  await page.waitForTimeout(100);
  const logAfterSpeech = await page.textContent('#log');
  expect(logAfterSpeech || '').toMatch(/Speech scheduled: /);
  // Verify adapter recorded at least one append for speech-related tracks
  expect(logAfterSpeech || '').toMatch(/appendChunk: track=/);

  // Basic sanity: demo exposes orchestrator
  const hasOrch = await page.evaluate(() => !!(window.__ichikaDemo && window.__ichikaDemo.orch));
  expect(hasOrch).toBeTruthy();
});
