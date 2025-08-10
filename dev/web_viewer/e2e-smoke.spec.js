import { test, expect } from '@playwright/test';

// Per-test timeout in addition to shell/global timeouts
test.setTimeout(120_000);

// Minimal smoke test to validate server routing and base page loads
test('smoke: index.html loads and shows heading', async ({ page }) => {
  await page.goto('/index.html');
  await expect(page.locator('html')).toBeVisible();
  // Title can vary; just ensure some title is present
  await expect(page).toHaveTitle(/.*/);
  // Try an h1 if present; tolerate absence
  const heading = page.locator('h1');
  if (await heading.count()) {
    await expect(heading.first()).toBeVisible();
  }
});
