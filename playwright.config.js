import { defineConfig } from '@playwright/test';

export default defineConfig({
  timeout: 300000, // 5 minutes
  use: {
    headless: true, // Run tests in headless mode
  },
});