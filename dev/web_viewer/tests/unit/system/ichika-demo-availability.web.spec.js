import { test, expect } from '@playwright/test';

// Lightweight availability check for the classroom demo page.
// Uses the 'unit-web' project (requires web server) and keeps runtime minimal.

test('[demo] ichika classroom demo page loads and shows controls (unit-web)', async ({ page }) => {
  await page.goto('/demos/ichika_classroom_demo.html');
  await expect(page).toHaveTitle(/Ichika Classroom Demo/i);
  await expect(page.getByRole('button', { name: 'Start Idle' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Point at Board' })).toBeVisible();
});
