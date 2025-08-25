import { test, expect } from '@playwright/test';

// Serverless test for BVHClipLibrary.parseDurationFromBVHText using inline BVH text.

test.setTimeout(30_000);

test('BVHClipLibrary.parseDurationFromBVHText extracts duration', async () => {
  // Load module via CommonJS so no web server is needed
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const lib = require('../../../src/components/animation/timeline/BVHClipLibrary.js');
  const BVHClipLibrary = lib.BVHClipLibrary || lib;
  const bvh = [
    'HIERARCHY',
    'ROOT Hips',
    'MOTION',
    'Frames: 60',
    'Frame Time: 0.0166667',
  ].join('\n');
  const dur = BVHClipLibrary.parseDurationFromBVHText(bvh, 30);
  expect(dur).toBeCloseTo(1.0, 4);
});
