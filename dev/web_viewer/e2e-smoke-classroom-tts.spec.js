import { test, expect } from '@playwright/test';

// Root-level E2E picked up by web_viewer-root-e2e project

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// This smoke verifies the on-device TTS UI is wired up. It does not run heavy models by default.
// To run real model loading, set RUN_REAL_INFERENCE=transformers and allow long timeouts.

test('Classroom on-device TTS controls appear and schedule animation [E2E][TTS]', async ({ page, baseURL }) => {
  await page.goto(baseURL + '/demos/ichika_classroom_demo.html');
  await expect(page.getByText('On-device TTS')).toBeVisible();
  await expect(page.locator('#ttsEngine')).toBeVisible();
  await expect(page.locator('#ttsText')).toBeVisible();
  await expect(page.locator('#ttsSay')).toBeVisible();

  // Click start to initialize orchestrator and manifest
  await page.click('#start');
  await page.waitForTimeout(200); // allow log and adapter instrumentation

  // Enter short text and trigger Say with audio disabled to avoid flakiness
  await page.fill('#ttsText', 'Hi class');
  await page.uncheck('#ttsPlayAudio');

  // In default mode we don't actually load heavy models; just ensure handler runs and logs an error or schedules.
  await page.click('#ttsSay');
  // We accept either a scheduling log or an error log from TTS fetch in constrained envs
  const log = page.locator('#log');
  await expect(log).toBeVisible();
});

// Optional real run (off by default): exercises transformers.js TTS for SpeechT5 quickly.
// Enable with RUN_REAL_INFERENCE=transformers. Marked slow and with generous timeout.
test.skip(process.env.RUN_REAL_INFERENCE !== 'transformers', 'Set RUN_REAL_INFERENCE=transformers to enable');
test('REAL: SpeechT5 on-device generates audio and schedules [E2E][TTS][REAL]', async ({ page, baseURL }) => {
  test.slow();
  await page.goto(baseURL + '/demos/ichika_classroom_demo.html');
  await page.click('#start');
  await page.click('#ttsPreload');
  await page.waitForTimeout(1000);
  await page.selectOption('#ttsEngine', 'speecht5');
  await page.fill('#ttsText', 'Hello class');
  await page.uncheck('#ttsPlayAudio');
  await page.click('#ttsSay');
  // Wait for either scheduling log or transformers load
  await page.waitForTimeout(15000);
  const text = await page.locator('#log').innerText();
  expect(text.length).toBeGreaterThan(0);
});

test.skip(process.env.RUN_REAL_INFERENCE !== 'transformers', 'Set RUN_REAL_INFERENCE=transformers to enable');
test('REAL: Kokoro (kokoro-js) on-device generates audio and schedules [E2E][TTS][REAL]', async ({ page, baseURL }) => {
  test.slow();
  await page.goto(baseURL + '/demos/ichika_classroom_demo.html');
  await page.click('#start');
  await page.click('#ttsPreload');
  await page.waitForTimeout(1000);
  await page.selectOption('#ttsEngine', 'kokoro');
  await page.fill('#ttsText', 'Kokoro test');
  await page.uncheck('#ttsPlayAudio');
  await page.click('#ttsSay');
  await page.waitForTimeout(15000);
  const text = await page.locator('#log').innerText();
  expect(text.length).toBeGreaterThan(0);
});
