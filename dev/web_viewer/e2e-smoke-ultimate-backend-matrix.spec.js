import { test, expect } from '@playwright/test';

const backends = [
  { id: 'beeps', url: (baseURL) => baseURL + '/demos/ichika_voice_conversation_demo.html?backend=beeps&playAudio=0' },
  { id: 'speech', url: (baseURL) => baseURL + '/demos/ichika_voice_conversation_demo.html?backend=speech&playAudio=0' },
  { id: 'speecht5', url: (baseURL) => baseURL + '/demos/ichika_voice_conversation_demo.html?backend=speecht5&playAudio=0' },
];

// Simple variations for rate/pitch to exercise UI wiring deterministically
const variants = [
  { rate: '1.0', pitch: '1.0' },
  { rate: '0.9', pitch: '1.1' },
];

test.describe('Ultimate backend matrix smoke (deterministic)', () => {
  for (const b of backends) {
    for (const v of variants) {
      test(`${b.id} | rate=${v.rate} pitch=${v.pitch} schedules + engine marker [E2E][smoke]`, async ({ page, baseURL }) => {
        await page.goto(b.url(baseURL));
        await page.fill('#text', `matrix ${b.id}`);
        await page.fill('#rate', v.rate);
        await page.fill('#pitch', v.pitch);
        await page.getByRole('button', { name: /Say/i }).click();

        const log = page.locator('#log');
        await expect(log).toContainText(/[PLAYWRIGHT] Scheduled TTS animation|Scheduled TTS animation|Scheduled TTS for text/i, { timeout: 20000 });
        // engine marker or backend log
        await expect(log).toContainText(new RegExp(`\\[PLAYWRIGHT\\] TTS engine=${b.id} useTts=0|🎛️ Using backend:\\s*${b.id}`));
        // expressions observed
        await expect.poll(async () => {
          const st = await page.evaluate(() => window.__ultimateDemo?.getStats?.() || { expressions: 0 });
          return st.expressions || 0;
        }, { timeout: 15000, intervals: [200, 400, 800, 1200, 2000] }).toBeGreaterThan(0);
      });
    }
  }
});
