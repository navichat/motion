import { defineConfig } from '@playwright/test';

export default defineConfig({
  globalTimeout: 600000, // 10 minutes safety net
  timeout: 300000, // 5 minutes shell timeout
  testDir: './dev/web_viewer/tests',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  outputDir: 'test-results/artifacts',
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['line'],
  ],
  use: {
    headless: false, // Show browser for debugging
    baseURL: 'http://localhost:8080',
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        browserName: 'chromium',
        actionTimeout: 45_000,
        navigationTimeout: 45_000,
        expect: { timeout: 12_000 },
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--enable-features=WebGPU,SharedArrayBuffer',
            '--enable-webgl',
            '--disable-web-security',
            '--use-fake-device-for-media-stream',
            '--use-fake-ui-for-media-stream'
          ]
        }
      }
    }
  ],
  webServer: {
    command: 'python3 dev/web_viewer/serve_with_headers.py',
    port: 8080,
    reuseExistingServer: true, // Always reuse existing server
    timeout: 120_000,
  }
});