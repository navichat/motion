import { test, expect } from '@playwright/test';

// Keep it fast; rely on shell/global timeouts
// Per-file timeout for extra safety
test.setTimeout(120_000);

// HTTP-only smoke checks to avoid heavy navigation

test('smoke: VRM demo HTTP fetch 200 + content', async ({ page }) => {
  const res = await page.request.get('/demos/ichika_vrm_orchestrator_demo.html');
  expect(res.status()).toBe(200);
  const body = await res.text();
  expect(body.length).toBeGreaterThan(100);
  expect(body).toMatch(/Ichika|VRM|orchestrator/i);
});

test('smoke: Classroom demo HTTP fetch 200 + content', async ({ page }) => {
  const res = await page.request.get('/demos/ichika_classroom_demo.html');
  expect(res.status()).toBe(200);
  const body = await res.text();
  expect(body).toMatch(/Ichika Classroom Demo|Start Idle|Speak/);
});
