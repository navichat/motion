import { defineConfig } from '@playwright/test';

const webServer = process.env.NO_WEBSERVER ? undefined : {
  command: 'python3 dev/web_viewer/serve_with_headers.py',
  port: 8080,
  reuseExistingServer: !process.env.CI,
  timeout: 120_000, // ensure shell command has a timeout per repo policy
};

// Build projects dynamically so e2e suites only run when a web server is available
const projects = [];

// Serverless component tests always included
projects.push({
  name: 'component-tests',
  testDir: './dev/web_viewer/tests/unit',
  testMatch: '**/*.serverless.spec.js',
  use: {
    browserName: 'chromium',
  actionTimeout: 30_000,
  navigationTimeout: 30_000,
  expect: { timeout: 10_000 },
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
});

// Only include e2e/integration projects when a web server is enabled
if (webServer) {
  // Lightweight smoke tests at repo-root web_viewer to validate server and routing
  projects.push({
    name: 'web_viewer-root-e2e',
    testDir: './dev/web_viewer',
  // Only run lightweight smoke tests at repo root plus ultimate avatar conversation
  testMatch: ['e2e-smoke*.spec.js', 'e2e-ultimate-*.spec.js'],
  testIgnore: ['**/src/**', '**/tests/**', '**/testing/**', '**/legacy-root-tests/**'],
    use: {
      browserName: 'chromium',
  actionTimeout: 45_000,
  navigationTimeout: 45_000,
  expect: { timeout: 12_000 },
      launchOptions: {
        args: [
          '--enable-precise-memory-info',
          '--enable-blink-features=MemoryMeasurement,SharedArrayBuffer',
          '--enable-features=WebGPU,UseWebGPUAdapterNameInWebGLExtension,SharedArrayBuffer',
          '--enable-webgl',
          '--disable-web-security',
          // Provide a fake microphone device and auto-allow UI prompts for CI
          '--use-fake-device-for-media-stream',
          '--use-fake-ui-for-media-stream'
        ]
      }
    }
  });

  // Unit tests that require a running web server (non-serverless)
  projects.push({
    name: 'unit-web',
    testDir: './dev/web_viewer/tests/unit',
    // Avoid duplicating serverless runs here; they have a dedicated project
    testIgnore: '**/*.serverless.spec.js',
    use: {
      browserName: 'chromium',
  actionTimeout: 30_000,
  navigationTimeout: 30_000,
  expect: { timeout: 10_000 },
      launchOptions: {
        args: [
          '--enable-precise-memory-info',
          '--enable-features=WebGPU,SharedArrayBuffer',
          '--enable-webgl',
          '--disable-web-security',
          // Provide a fake microphone device and auto-allow prompts for mic tests
          '--use-fake-device-for-media-stream',
          '--use-fake-ui-for-media-stream'
        ]
      }
    }
  });

  if (process.env.RUN_FULL_E2E) {
    projects.push(
      {
        name: 'chromium',
        testDir: './dev/web_viewer/src/testing/integration/e2e',
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
        testDir: './dev/web_viewer/src/testing/integration/e2e',
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
        name: 'legacy-root',
        testDir: './dev/web_viewer/tests/legacy-root-tests',
        use: {
          browserName: 'chromium',
          launchOptions: {
            args: [
              '--enable-precise-memory-info',
              '--enable-features=WebGPU,SharedArrayBuffer',
              '--enable-webgl',
              '--disable-web-security'
            ]
          }
        }
      },
      {
        name: 'firefox',
        testDir: './dev/web_viewer/src/testing/integration/e2e',
        use: { browserName: 'firefox' },
      },
      {
        name: 'webkit',
        testDir: './dev/web_viewer/src/testing/integration/e2e',
        use: { browserName: 'webkit' },
      },
    );
  }
}

export default defineConfig({
  globalTimeout: 600000, // 10 minutes safety net
  timeout: 300000, // 5 minutes
  testDir: './dev/web_viewer/tests',
  // Allow CI or local runs to target subsets by title with PW_GREP, e.g., PW_GREP="VRM|animation" npx playwright test
  ...(process.env.PW_GREP ? { grep: new RegExp(process.env.PW_GREP) } : {}),
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  // Store test artifacts outside the HTML report folder to avoid clashes
  outputDir: 'test-results/artifacts',
  reporter: [
    // Use a dedicated folder for HTML report separate from artifacts
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['line'],
  ],
  use: {
    headless: true, // Run tests in headless mode
  // Force IPv4 to avoid environments where localhost resolves to ::1 but server binds IPv4 only
  baseURL: 'http://127.0.0.1:8080',
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects,
  ...(webServer && { webServer }),
});