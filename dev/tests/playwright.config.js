import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Motion Viewer 3D testing
 * Focus on visual regression and performance testing
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false, // 3D tests can be resource intensive
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : 2,
  reporter: [
    ['html'],
    ['json', { outputFile: 'results/test-results.json' }],
    ['junit', { outputFile: 'results/junit.xml' }]
  ],
  use: {
    baseURL: 'http://localhost:8081',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    // Longer timeouts for 3D rendering
    actionTimeout: 10000,
    navigationTimeout: 15000
  },

  // Test against multiple browsers for compatibility
  projects: [
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Enable WebGL and hardware acceleration
        launchOptions: {
          args: [
            '--enable-webgl',
            '--enable-accelerated-2d-canvas',
            '--enable-gpu-rasterization',
            '--force-gpu-mem-available-mb=1024',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor'
          ]
        }
      },
    },
    {
      name: 'firefox',
      use: { 
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            'webgl.force-enabled': true,
            'layers.acceleration.force-enabled': true
          }
        }
      },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    // Mobile testing for responsive design
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
  ],

  // Global setup and teardown
  globalSetup: require.resolve('./tools/global-setup.js'),
  globalTeardown: require.resolve('./tools/global-teardown.js'),

  // Output directories
  outputDir: 'results/',
  
  // Web server configuration
  webServer: {
    command: 'cd ../server && python enhanced_motion_server.py',
    port: 8081,
    reuseExistingServer: !process.env.CI,
    timeout: 30000
  },

  // Expect configuration for visual comparisons
  expect: {
    // Threshold for visual comparisons (0-1, where 0 is identical)
    toHaveScreenshot: { 
      threshold: 0.1,
      mode: 'strict'
    },
    toMatchSnapshot: { 
      threshold: 0.1
    }
  }
});
