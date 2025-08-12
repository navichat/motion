import { test, expect } from '@playwright/test';

test('[E2E][VRM][Demo] Demo: Ichika Orchestrator page loads without global/ctor errors', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e?.message || String(e)));
  const logs = [];
  page.on('console', (msg) => logs.push(msg.text()));

  await page.goto('/demos/ichika_vrm_orchestrator_demo.html');

  // Give UMD scripts a brief moment to attach to window
  await page.waitForTimeout(200);

  const globals = await page.evaluate(() => {
    const win = window;
    const hasBVH = !!(win.BVHTimeline && (typeof win.BVHTimeline === 'function' || typeof win.BVHTimeline.BVHTimeline === 'function'));
    const hasTS = !!(win.TaskScheduler && typeof win.TaskScheduler === 'function');
    const hasAdapter = !!(win.TimelineChunkAdapter && (typeof win.TimelineChunkAdapter === 'function' || typeof win.TimelineChunkAdapter.TimelineChunkAdapter === 'function'));
    const hasVRI = !!(win.BVHTimelineVRMIntegration && typeof win.BVHTimelineVRMIntegration === 'function');
    const hasOrch = !!(win.IchikaOrchestrator && typeof win.IchikaOrchestrator === 'function');
    return { hasBVH, hasTS, hasAdapter, hasVRI, hasOrch };
  });

  // Assert no fatal script wiring errors typical from before
  const fatalPatterns = [/already been declared/i, /is not a constructor/i];
  const fatal = errors.find(e => fatalPatterns.some(p => p.test(e))) || logs.find(l => fatalPatterns.some(p => p.test(l)));

  expect(fatal).toBeFalsy();
  expect(globals.hasBVH).toBeTruthy();
  expect(globals.hasTS).toBeTruthy();
  expect(globals.hasAdapter).toBeTruthy();
  expect(globals.hasVRI).toBeTruthy();
  expect(globals.hasOrch).toBeTruthy();
});
