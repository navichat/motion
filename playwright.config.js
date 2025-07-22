import { defineConfig } from '@playwright/test';

export default defineConfig({
  timeout: 300000, // 5 minutes
  testDir: './dev/web_viewer/js',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }],
    ['line'],
  ],
  use: {
    headless: true, // Run tests in headless mode
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
        // Enable performance APIs for FLOPS measurement
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--enable-blink-features=MemoryMeasurement',
            '--enable-features=WebGPU,UseWebGPUAdapterNameInWebGLExtension'
          ]
        }
      },
    },
    {
      name: 'firefox',
      use: { browserName: 'firefox' },
    },
    {
      name: 'webkit',
      use: { browserName: 'webkit' },
    },
  ],
  webServer: {
    command: 'python3 dev/web_viewer/serve_with_headers.py',
    port: 8080,
    reuseExistingServer: !process.env.CI,
  },
});