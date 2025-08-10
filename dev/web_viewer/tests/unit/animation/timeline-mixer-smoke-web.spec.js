import { test, expect } from '@playwright/test';

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

// Smoke: TimelineMixer composes poses from multiple timelines using last-writer-wins

test('TimelineMixer compose picks latest frames per channel across timelines', async ({ page }) => {
  await page.goto('/index.html');

  const result = await page.evaluate(async () => {
    const res = await fetch('/src/models/bvh/TimelineMixer.js');
    const code = await res.text();
    const mod = { exports: {} };
    const MixerCtor = (new Function('window','module','exports', code + '; return (module.exports && module.exports.TimelineMixer) || window.TimelineMixer;'))(window, mod, mod.exports);
    if (typeof MixerCtor !== 'function') throw new Error('TimelineMixer ctor missing');

    // Build two mocked timelines with track clips containing frames and channels
    // Timeline A has an earlier frame on bone 'head'
    const tlA = {
      tracks: {
        base: {
          clips: [{ frames: [
            { time: 0.05, channels: new Map([['head', { rotZ: 5 }]]) },
          ] }]
        }
      }
    };

    // Timeline B has a later frame on bone 'head' and an earlier one on 'leftArm'
  const tlB = {
      tracks: {
        audio: {
          clips: [{ frames: [
      { time: 0.04, channels: new Map([['leftArm', { rotX: 10 }]]) },
      { time: 0.06, channels: new Map([['head', { rotZ: 15 }], ['leftArm', { rotX: 10 }]]) },
          ] }]
        }
      }
    };

    const mixer = new MixerCtor();
    const pose = mixer.compose([tlA, tlB], 0.06);

    const head = pose.get('head');
    const leftArm = pose.get('leftArm');
    return { head, leftArm };
  });

  // At t=0.06, 'head' should be taken from tlB (rotZ: 15), 'leftArm' from tlB earlier frame
  expect(result.head && result.head.rotZ).toBe(15);
  expect(result.leftArm && result.leftArm.rotX).toBe(10);
});
