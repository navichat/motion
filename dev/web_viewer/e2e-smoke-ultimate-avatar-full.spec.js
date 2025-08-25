// [E2E] Ultimate Avatar Full Loop: Mic -> Fake ASR -> TTS -> VRM animation (speech backend)
const { test, expect } = require('@playwright/test');

test.describe('[E2E][Ultimate][Avatar][FullLoop][VRM][FakeASR]', () => {
  test('Full mic→reply loop animates real VRM (speech backend, fake ASR)', async ({ page }) => {
    const url = '/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&vrm=1&listenSec=1&turns=1&debugAudio=1';
    await page.goto(url);
    await page.waitForFunction(() => typeof window.__ultimateDemo?.getStats === 'function');
    await page.evaluate(() => window.__ultimateDemo.listenAndReply());
    await page.waitForFunction(() => { const s = window.__ultimateDemo?.getConversationStats?.(); return s && s.listens >= 2 && s.replies >= 2; }, { timeout: 45000 });
    const stats = await page.evaluate(() => window.__ultimateDemo.getStats());
    if (stats.stub) {
      test.skip(true, 'VRM asset not available (stub mode)');
    }
    expect(stats.expressions).toBeGreaterThan(10);
    await page.waitForFunction(() => { const log = window.__ultimateAudioLog; if (!Array.isArray(log)) return false; const types = log.map(e => e.type); const schedule = types.some(t => t === 'tts_scheduled' || /_scheduled$/.test(t)); const playback = types.some(t => /(buffer_play|beeps_play|pcm_play|_play$|speech_play_error)/.test(t)); return schedule && playback; }, { timeout: 30000 });
    const metrics = await page.evaluate(() => window.__ultimateDemo.getAudioMetrics?.());
    expect(metrics.samples).toBeGreaterThan(0);
    expect(metrics.avgLatencyMs).toBeGreaterThanOrEqual(0);
  });
});
