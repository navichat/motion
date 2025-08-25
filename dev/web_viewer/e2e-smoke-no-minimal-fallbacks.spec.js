// [E2E] Ensure core pages do not fall back to minimal timeline/adapter/scheduler
const { test, expect } = require('@playwright/test');

async function assertNoFallbacks(page, url) {
  const logs = [];
  page.on('console', (msg) => {
    const t = msg.text();
    logs.push(t);
  });
  await page.goto(url);
  // Basic constructors should be present
  const hasAdapter = await page.evaluate(() => typeof window.TimelineChunkAdapter === 'function');
  const hasTimeline = await page.evaluate(() => {
    const g = window;
    const TL = (g.BVHTimeline && g.BVHTimeline.BVHTimeline) || g.BVHTimeline;
    return typeof TL === 'function';
  });
  expect(hasAdapter).toBeTruthy();
  expect(hasTimeline).toBeTruthy();
  // Give the page a brief moment to initialize orchestrator and emit logs
  await page.waitForTimeout(200);
  const bad = logs.filter(l => /Falling back to MinimalTimeline|Falling back to MinimalScheduler|Using minimal adapter fallback/i.test(l));
  expect(bad, `Found fallback warnings on ${url}:\n${bad.join('\n')}`).toHaveLength(0);
}

test.describe('[E2E][Infra] No minimal fallbacks on core pages', () => {
  test('[Infra] Voice conversation demo loads full timeline/adapter', async ({ page }) => {
    await assertNoFallbacks(page, '/demos/ichika_voice_conversation_demo.html?backend=speech');
  });
  test('[Infra] Classroom conversation page loads full timeline/adapter', async ({ page }) => {
    await assertNoFallbacks(page, '/tests/e2e/html/classroom-conversation-backends.html?backend=speech');
  });
});
