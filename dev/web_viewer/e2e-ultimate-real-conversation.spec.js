import { test, expect } from '@playwright/test';

// Ultimate REAL multi-turn: Mic -> Whisper (on-device ASR) -> Kokoro (on-device TTS) -> Audio-driven gestures (+ optional VRM)
// Gated by RUN_REAL_INFERENCE=1 and KOKORO_JS env (URL to kokoro runtime script)
// This is heavier than smokes; keep it optional so regular CI stays fast/stable.

const RUN_REAL = (process.env.RUN_REAL_INFERENCE === 'true' || process.env.RUN_REAL_INFERENCE === '1');
const KOKORO_JS = process.env.KOKORO_JS || '';
const DESCRIBE = (RUN_REAL && KOKORO_JS) ? test.describe : test.describe.skip;

DESCRIBE('Ultimate REAL multi-turn mic→Whisper→Kokoro conversation', () => {
  test('Mic hears, TTS replies with real audio energy driving animation (2 turns)', async ({ page, baseURL }) => {
    test.slow(); // allow model downloads
    const params = new URLSearchParams({
      backend: 'kokoro',
      asr: 'whisper',
      asrModel: 'Xenova/whisper-tiny.en',
      listenSec: '1',
      playAudio: '1', // need actual playback to log buffer_play latency
      autoListen: '1',
      turns: '2',
      kokoroJs: KOKORO_JS,
      vrm: '1', // attempt VRM if asset present (non-fatal if missing)
      debugAudio: '1'
    });
    const url = baseURL + '/demos/ichika_voice_conversation_demo.html?' + params.toString();
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    // Kick off conversation
    await page.getByRole('button', { name: /Listen & Reply/i }).click();

    const log = page.locator('#log');
    // Expect we heard something (whisper path) and kokoro produced audio at least once
    await expect(log).toContainText(/\u{1F4DD}\s*Heard:/u, { timeout: 180000 }); // 📝 Heard:
    await expect(log).toContainText('kokoro audio ok', { timeout: 180000 });

    // Wait for second turn (replies>=2) or timeout
    await expect.poll(async () => {
      const stats = await page.evaluate(() => window.__ultimateDemo?.getConversationStats?.() || { replies: 0 });
      return stats.replies || 0;
    }, { timeout: 180000, message: 'Expected at least 2 replies (multi-turn)' }).toBeGreaterThanOrEqual(2);

    // Scheduling markers present
    await expect(log).toContainText(/Scheduled TTS for text|Scheduled TTS animation/, { timeout: 60000 });

    // Validate animation expressions increased (driven by real visemes/energy)
    await expect.poll(async () => {
      const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
      return st.expressions || 0;
    }, { timeout: 40000, intervals: [250, 500, 1000, 1500, 2500] }).toBeGreaterThan(15);

    // Audio metrics sanity: we should have at least one latency sample from buffer_play
    const metrics = await page.evaluate(() => window.__ultimateDemo?.getAudioMetrics?.() || {});
    expect(metrics.samples || 0).toBeGreaterThan(0);
    expect(metrics.avgLatencyMs).toBeGreaterThanOrEqual(0);

    // Ensure at least one 'buffer_play' (kokoro playback) entry carried latency
    const latencyPlay = await page.evaluate(() => {
      const log = window.__ultimateDemo?.getAudioLog?.() || []; return log.some(e => e.type === 'buffer_play' && typeof e.latencyMs === 'number');
    });
    expect(latencyPlay).toBeTruthy();

    // Artifact (optional) - capture metrics + last events
    await page.evaluate(() => {
      try {
        const out = {
          scenario: 'ultimate-real-whisper-kokoro-multiturn',
          metrics: window.__ultimateDemo?.getAudioMetrics?.(),
          conversation: window.__ultimateDemo?.getConversationStats?.(),
          tail: (window.__ultimateDemo?.getAudioLog?.() || []).slice(-60)
        };
        // Expose for Playwright trace; CI step will pick up generic audio-log-* already
        window.__ULTIMATE_REAL_CONVO = out;
      } catch {}
    });
  });
});
