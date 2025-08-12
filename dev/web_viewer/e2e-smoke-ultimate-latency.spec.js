// [E2E] Ultimate latency guard: ensure TTS schedule->play latency stays under a generous threshold
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// Allow override via env; default generous to avoid flakiness but still catch regressions
const MAX_LATENCY_MS = parseFloat(process.env.LATENCY_MAX_MS || '5000');
const MAX_P50_MS = parseFloat(process.env.LATENCY_P50_MAX_MS || '2500');
const MAX_P95_MS = parseFloat(process.env.LATENCY_P95_MAX_MS || process.env.LATENCY_MAX_MS || '5000');
const MAX_P99_MS = parseFloat(process.env.LATENCY_P99_MAX_MS || process.env.LATENCY_MAX_MS || '5000');

// We use speech backend + fake ASR for determinism.
const url = '/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&autoListen=1&listenSec=1&turns=1&debugAudio=1';

test.describe('[E2E][Ultimate][Latency][FakeASR]', () => {
  test('Average and last latency below threshold', async ({ page }) => {
    await page.goto(url);
    // Wait for audio log array
    await page.waitForFunction(() => Array.isArray(window.__ultimateAudioLog), { timeout: 15000 });
    // Wait until we have at least one schedule + playback pair producing latency samples
    await page.waitForFunction(() => {
      const m = window.__ultimateDemo?.getAudioMetrics?.();
      return m && m.samples > 0 && m.avgLatencyMs >= 0;
    }, { timeout: 30000 });

  const metrics = await page.evaluate(() => window.__ultimateDemo.getAudioMetrics());

    // Basic sanity
    expect(metrics.samples).toBeGreaterThan(0);
    expect(metrics.avgLatencyMs).toBeGreaterThanOrEqual(0);
    expect(metrics.lastLatencyMs).toBeGreaterThanOrEqual(0);

    // Percentiles presence
    for (const k of ['p50','p90','p95','p99']) {
      expect(metrics[k]).toBeGreaterThanOrEqual(0);
    }

    // Write metrics artifact (always) for CI analysis
    try {
      const outDir = path.join(process.cwd(), 'test-results');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'perf-latency.json'), JSON.stringify({
        timestamp: new Date().toISOString(),
        thresholds: { MAX_LATENCY_MS, MAX_P50_MS, MAX_P95_MS, MAX_P99_MS },
        metrics
      }, null, 2));
    } catch (e) { console.warn('Failed to write perf-latency.json', e); }

    // Guard thresholds (apply to key percentiles too)
  if (!(metrics.avgLatencyMs < MAX_LATENCY_MS && metrics.lastLatencyMs < MAX_LATENCY_MS && metrics.p50 < MAX_P50_MS && metrics.p95 < MAX_P95_MS && metrics.p99 < MAX_P99_MS)) {
      const log = await page.evaluate(() => window.__ultimateAudioLog.slice(-25));
      console.log('LATENCY_FAIL_LAST_EVENTS', log);
    }
    expect(metrics.avgLatencyMs).toBeLessThan(MAX_LATENCY_MS);
    expect(metrics.lastLatencyMs).toBeLessThan(MAX_LATENCY_MS);
  expect(metrics.p50).toBeLessThan(MAX_P50_MS);
    expect(metrics.p95).toBeLessThan(MAX_P95_MS);
    expect(metrics.p99).toBeLessThan(MAX_P99_MS);
  });
});
