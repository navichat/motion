const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: '.',
  timeout: 300000, // 5 minutes for shell timeout compliance
  use: {
    headless: true,
    viewport: { width: 1280, height: 1024 },
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    launchOptions: {
      args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-gl=swiftshader']
    }
  },
});