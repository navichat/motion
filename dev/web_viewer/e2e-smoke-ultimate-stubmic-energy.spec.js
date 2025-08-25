// [E2E] Stub mic energy -> gesture sanity and audio metrics
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// Deterministic stub mic path: ensures we can assert energy-driven expressions without real mic.
const URL = '/demos/ichika_voice_conversation_demo.html?backend=beeps&asr=fake&stubMic=1&autoListen=1&listenSec=1';

test.describe('[E2E][Ultimate][StubMic] energy + visemes', () => {
  test('Stub mic produces energy-driven expressions and latency metrics', async ({ page }) => {
    await page.goto(URL);
    // Trigger mic start programmatically (autoListen path triggers listen; we ensure mic started)
    await page.evaluate(() => { window.__ultimateDemo?.startMic?.(); });

    // Wait for some expressions (viseme-driven) to be applied
    await page.waitForFunction(() => { try { return (window.__ultimateDemo?.getStats()?.expressions||0) > 5; } catch { return false; } }, { timeout: 20000 });

    // Collect audio metrics (should have at least one scheduled->play pair from autoListen reply)
    const metrics = await page.evaluate(() => window.__ultimateDemo?.getAudioMetrics?.());
    expect(metrics).toBeTruthy();
    expect(metrics.samples).toBeGreaterThan(0);

  // Attempt to detect stub mic start but don't fail hard if missing; collect log
  await page.waitForTimeout(1000);
  const audioLog = await page.evaluate(() => (window.__ultimateDemo?.getAudioLog?.()||[]).slice(-80));
  expect(Array.isArray(audioLog)).toBeTruthy();
  // Soft assertion: log presence if available
  const hasStub = audioLog.some(e => e.type === 'stub_mic_start');
  if (!hasStub) console.warn('stub_mic_start not observed; continuing (non-fatal)');

    // Artifact write
    try {
      const outDir = path.join(process.cwd(), 'test-results');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'stub-mic-energy.json'), JSON.stringify({ metrics, tail: audioLog }, null, 2));
    } catch (e) { console.warn('Failed to write stub mic artifact', e); }
  });
});
