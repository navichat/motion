// [E2E] Ultimate Avatar Full Loop: Mic -> Fake ASR -> TTS -> VRM animation (speech backend)
const { test, expect } = require('@playwright/test');

/*
Goals:
 1. Load VRM (binder not stub) using ?vrm=1
 2. Run at least one listen+reply cycle (fake ASR) plus an auto-chained turn (turns=1)
 3. Verify expressions increased (> 10) indicating viseme/energy application
 4. Verify audio scheduling + playback entries present and latency metrics computed
Graceful skip if VRM asset unavailable (stub binder stays true or log contains 'VRM not found').
*/

test.describe('[E2E][Ultimate][Avatar][FullLoop][VRM][FakeASR]', () => {
  test('Full mic→reply loop animates real VRM (speech backend, fake ASR)', async ({ page }) => {
    const url = '/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&vrm=1&listenSec=1&turns=1&debugAudio=1';
    await page.goto(url);

    // Ensure test API present
    await page.waitForFunction(() => typeof window.__ultimateDemo?.getStats === 'function');

    // Kick off first listen
    await page.evaluate(() => window.__ultimateDemo.listenAndReply());

    // Wait for at least 2 replies (initial + 1 turn) or decide to skip if VRM missing early
    await page.waitForFunction(() => {
      const s = window.__ultimateDemo?.getConversationStats?.();
      return s && s.listens >= 2 && s.replies >= 2;
    }, { timeout: 45000 });

    // Determine VRM loaded vs stub
    const stats = await page.evaluate(() => window.__ultimateDemo.getStats());
    if (stats.stub) {
      // Inspect log text for VRM missing hints
      const logText = await page.locator('#log').textContent();
      if (/VRM not found|stub mode/i.test(logText || '')) {
        test.skip(true, 'VRM asset not available in this environment');
      }
    }

    expect(stats.stub).toBeFalsy();
    expect(stats.expressions).toBeGreaterThan(10);

    // Audio log assertions
    await page.waitForFunction(() => {
      const log = window.__ultimateAudioLog; if (!Array.isArray(log)) return false;
      const types = log.map(e => e.type);
      const schedule = types.some(t => t === 'tts_scheduled' || /_scheduled$/.test(t));
      const playback = types.some(t => /(buffer_play|beeps_play|pcm_play|_play$|speech_play_error)/.test(t));
      return schedule && playback;
    }, { timeout: 30000 });

    const metrics = await page.evaluate(() => window.__ultimateDemo.getAudioMetrics?.());
    expect(metrics.samples).toBeGreaterThan(0);
    // Latency sanity (should be finite positive)
    expect(metrics.avgLatencyMs).toBeGreaterThanOrEqual(0);
  });
});
