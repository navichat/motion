import { test, expect } from '@playwright/test';

// Root-level E2E picked up by web_viewer-root-e2e project

test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server');

test('Classroom Start → Speech → Point produces VRM updates [E2E][VRM][Classroom]', async ({ page }) => {
  // Server root is dev/web_viewer/
  await page.goto('/tests/e2e/html/classroom-start-speech-point.html');
  await page.waitForFunction(() => !!window.__classroomFlow, null, { timeout: 15000 });
  const stats = await page.evaluate(() => window.__classroomFlow);
  if (stats && stats.error) {
    throw new Error('classroom flow error: ' + stats.error);
  }
  // Expect at least a couple updates overall
  expect(Math.max(stats.boneCalls, stats.blendCalls)).toBeGreaterThan(1);
  // Ensure we saw visemes and the point action flag
  expect(stats.sawSpeechViseme).toBeTruthy();
  expect(stats.sawActionPoint).toBeTruthy();
  // Sanity: we sampled multiple frames
  expect(stats.samples).toBeGreaterThan(3);
});
