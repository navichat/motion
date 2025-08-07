import { defineConfig } from '@playwright/test';

export default defineConfig({
  timeout: 300000, // 5 minutes
  testDir: './dev/web_viewer/tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: 'test-results/html-report' }],
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
      testDir: './dev/web_viewer/tests/integration/e2e',
      use: { 
        browserName: 'chromium',
        // Enable performance APIs for FLOPS measurement
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--enable-blink-features=MemoryMeasurement,SharedArrayBuffer',
            '--enable-features=WebGPU,UseWebGPUAdapterNameInWebGLExtension,SharedArrayBuffer',
            '--enable-unsafe-swiftshader',
            '--enable-webgl',
            '--enable-accelerated-2d-canvas',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor'
          ]
        }
      },
    },
    {
      name: 'chromium-webgpu',
      testDir: './dev/web_viewer/tests/integration/e2e',
      use: { 
        browserName: 'chromium',
        // Enhanced WebGPU and GPU acceleration for full compatibility testing
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--enable-blink-features=MemoryMeasurement',
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer,WebGPU,UseWebGPUAdapterNameInWebGLExtension',
            '--disable-vulkan-fallback-to-gl-for-testing',
            '--use-vulkan=native',
            '--force-gpu-mem-available-mb=2048',
            '--disable-web-security',
            '--enable-dawn-features=allow_unsafe_apis',
            '--use-gl=angle',
            '--use-angle=vulkan',
            '--enable-webgl',
            '--enable-accelerated-2d-canvas',
            '--enable-gpu-rasterization'
          ]
        }
      },
    },
    {
      name: 'component-tests',
      testDir: './dev/web_viewer/tests/unit',
      use: { 
        browserName: 'chromium',
        // Optimized for individual component testing
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--enable-features=WebGPU,SharedArrayBuffer',
            '--enable-webgl',
            '--disable-web-security'
          ]
        }
      },
    },
    {
      name: 'firefox',
      testDir: './dev/web_viewer/tests/integration/e2e',
      use: { browserName: 'firefox' },
    },
    {
      name: 'webkit',
      testDir: './dev/web_viewer/tests/integration/e2e',
      use: { browserName: 'webkit' },
    },
  ],
  webServer: {
    command: 'python3 dev/web_viewer/serve_with_headers.py',
    port: 8080,
    reuseExistingServer: !process.env.CI,
  },
});