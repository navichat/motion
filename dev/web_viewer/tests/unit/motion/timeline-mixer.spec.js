import { test, expect } from '@playwright/test';

// Load BVHTimeline and TimelineMixer into the page context
test.skip(process.env.NO_WEBSERVER, 'Requires dev server. Skip when NO_WEBSERVER=1');

test('TimelineMixer returns a composed pose', async ({ page }) => {
  await page.goto('/index.html');
  await page.addScriptTag({ url: '/src/models/bvh/BVHTimeline.js' });
  await page.addScriptTag({ url: '/src/models/bvh/TimelineMixer.js' });

  const hasPose = await page.evaluate(() => {
    const tl1 = new window.BVHTimeline({ framerate: 30 });
    const tl2 = new window.BVHTimeline({ framerate: 30 });

    const frames1 = [ { time: 0.10, channels: new Map([["Hips.x", 1]]) } ];
    const frames2 = [ { time: 0.15, channels: new Map([["Hips.x", 2]]) } ];

    tl1.appendChunk('a', { t0: 0.0, dt: 0.2, frames: frames1 });
    tl2.appendChunk('b', { t0: 0.0, dt: 0.2, frames: frames2 });

    const mixer = new window.TimelineMixer();
    const pose = mixer.compose([tl1, tl2], 0.16);
    return pose instanceof Map && pose.get('Hips.x') === 2;
  });

  expect(hasPose).toBe(true);
});
