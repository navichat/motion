// [E2E] Ultimate audio log diagnostic: ensure scheduling + playback events occur
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const scenarios = [
  {
    name: 'speech-backend (fake ASR)',
    backend: 'speech',
    url: '/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&autoListen=1&listenSec=2&debugAudio=1'
  },
  {
    name: 'speecht5 on-device fake (fake ASR)',
    backend: 'speecht5',
    url: '/demos/ichika_voice_conversation_demo.html?backend=speecht5&speecht5OnDevice=1&speecht5OnDeviceFake=1&speecht5Spk=random&asr=fake&autoListen=1&listenSec=2&debugAudio=1'
  },
  {
    name: 'kokoro backend (fake ASR)',
    backend: 'kokoro',
    url: '/demos/ichika_voice_conversation_demo.html?backend=kokoro&asr=fake&autoListen=1&listenSec=2&debugAudio=1'
  }
];

for (const sc of scenarios) {
  test.describe(`[E2E][Ultimate][AudioLog] ${sc.name}`, () => {
    test(`Audio events logged for ${sc.name}`, async ({ page }) => {
      await page.goto(sc.url);
      await page.waitForFunction(() => Array.isArray(window.__ultimateAudioLog), { timeout: 15000 });
      // Wait for both a schedule and a playback style event
      try {
        await page.waitForFunction(() => {
          const log = window.__ultimateAudioLog;
          if (!Array.isArray(log)) return false;
          const types = log.map(e => e.type);
          const hasSchedule = types.includes('tts_scheduled') || types.some(t => /_scheduled$/.test(t));
          const hasPlayback = types.some(t => /(buffer_play|beeps_play|pcm_play|_play$|speech_play_error)/.test(t));
          return hasSchedule && hasPlayback;
        }, { timeout: 30000 });
      } catch (e) {
        const dump = await page.evaluate(() => window.__ultimateAudioLog || []);
        console.log('AUDIO_LOG_DUMP', dump);
        throw e;
      }

      const { types, count, sampleLog, metrics, hasLatencyEntry, firstScheduleIdx, firstPlaybackIdx } = await page.evaluate(() => {
        const log = window.__ultimateAudioLog || [];
        const metrics = window.__ultimateDemo?.getAudioMetrics?.() || {};
        const hasLatencyEntry = log.some(e => typeof e.latencyMs === 'number');
        let firstScheduleIdx = -1; let firstPlaybackIdx = -1;
        for (let i=0;i<log.length;i++) {
          const t = log[i].type;
          if (firstScheduleIdx === -1 && (t === 'tts_scheduled' || /_scheduled$/.test(t))) firstScheduleIdx = i;
          if (firstPlaybackIdx === -1 && /(buffer_play|beeps_play|pcm_play|_play$|speech_play_error)/.test(t)) firstPlaybackIdx = i;
        }
        return { types: [...new Set(log.map(e => e.type))], count: log.length, sampleLog: log.slice(-50), metrics, hasLatencyEntry, firstScheduleIdx, firstPlaybackIdx };
      });
      expect(count).toBeGreaterThan(0);
  expect(types.some(t => t.endsWith('_play') || t === 'buffer_play' || t === 'beeps_play' || t === 'pcm_play' || t === 'speech_play_error')).toBeTruthy();
      expect(types.some(t => t === 'tts_scheduled' || t.endsWith('_scheduled'))).toBeTruthy();
  // Basic latency metrics sanity (we expect at least one schedule->play pair to have produced latency stats)
  if (metrics && metrics.samples !== undefined) {
    expect(metrics.samples).toBeGreaterThan(0);
    expect(metrics.avgLatencyMs).toBeGreaterThanOrEqual(0);
    // Percentiles should be numbers and non-negative
    for (const k of ['p50','p90','p95','p99']) { if (metrics[k] !== undefined) expect(metrics[k]).toBeGreaterThanOrEqual(0); }
  }
  // Ensure we captured at least one concrete latency-bearing playback entry (helps catch regression where schedule ts not stored)
  expect(hasLatencyEntry).toBeTruthy();

  // Ordering: first schedule must occur before first playback (if both found)
  if (firstScheduleIdx !== -1 && firstPlaybackIdx !== -1) {
    expect(firstScheduleIdx).toBeLessThan(firstPlaybackIdx);
  }

  // Percentile monotonicity sanity when we have enough samples (>3 to avoid degenerate repeats)
  if ((metrics.samples||0) > 3) {
    const { p50=0,p90=0,p95=0,p99=0 } = metrics;
    expect(p50).toBeLessThanOrEqual(p90);
    expect(p90).toBeLessThanOrEqual(p95);
    expect(p95).toBeLessThanOrEqual(p99);
  }
  // Also ensure some facial expressions (viseme-driven) were applied
  await page.waitForFunction(() => { try { return (window.__ultimateDemo?.getStats()?.expressions || 0) > 0; } catch { return false; } }, { timeout: 20000 });

      // Write artifact for CI
      try {
        const outDir = path.join(process.cwd(), 'test-results');
        if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
        const safeName = sc.name.replace(/[^a-z0-9]+/ig,'_').replace(/_+/g,'_').toLowerCase();
  fs.writeFileSync(path.join(outDir, `audio-log-${safeName}.json`), JSON.stringify({
          scenario: sc.name,
          backend: sc.backend,
          timestamp: new Date().toISOString(),
            uniqueTypes: types,
          count,
          metrics,
          latencyEntry: hasLatencyEntry,
          tail: sampleLog
        }, null, 2));
      } catch (e) { console.warn('Failed to write audio log artifact', e); }
    });
  });
}
