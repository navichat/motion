import { defineConfig } from '@playwright/test';

export default defineConfig({
  timeout: 300000, // 5 minutes
  testDir: '../e2e/playwright',
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
    baseURL: 'http://localhost:8082',
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
      name: 'chromium-webgpu',
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
      name: 'firefox',
      use: { browserName: 'firefox' },
    },
    {
      name: 'webkit',
      use: { browserName: 'webkit' },
    },
  ],
  // No webServer config - assume server is already running
});
