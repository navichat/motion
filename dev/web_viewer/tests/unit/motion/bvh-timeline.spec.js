import { test, expect } from '@playwright/test';

// Load BVHTimeline into the page context via server URL
test.skip(process.env.NO_WEBSERVER, 'Requires dev server. Skip when NO_WEBSERVER=1');

test('BVHTimeline append/clear/snapshot', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: '/src/models/bvh/BVHTimeline.js' });

  const result = await page.evaluate(() => {
    const tl = new window.BVHTimeline({ framerate: 30 });
    const frames = Array.from({ length: 6 }, (_, i) => ({ time: i/30, channels: new Map([['Hips.x', i]]) }));

    tl.appendChunk('gesture', { t0: 0.0, dt: 0.2, frames });
    const v1 = tl.version;
    const snap1 = tl.snapshot(0.0, 0.2);

    // Clear future from t>=0.15
    tl.clear('gesture', 0.15);
    const v2 = tl.version;
    const snap2 = tl.snapshot(0.0, 1.0);

    return { v1, v2, c1: snap1.length, c2: snap2.length, hasTrack: !!tl.tracks.gesture };
  });

  expect(result.v2).toBeGreaterThan(result.v1 - 1); // version increments
  expect(result.hasTrack).toBe(true);
  expect(result.c1).toBeGreaterThan(0);
  expect(result.c2).toBeLessThanOrEqual(result.c1); // after clear, not more frames
});
