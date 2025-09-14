import { defineConfig } from '@playwright/test';

const USE_VITE = !!process.env.USE_VITE;
const NO_WEBSERVER = !!process.env.NO_WEBSERVER;
const VITE_PORT = parseInt(process.env.USE_VITE_PORT || '5180', 10);
const PY_PORT = 8080;
const HOST = '127.0.0.1';
const PORT = USE_VITE ? VITE_PORT : PY_PORT;
const BASE_URL = `http://${HOST}:${PORT}`;
// Debug log for troubleshooting server selection and base URL
// eslint-disable-next-line no-console
console.log(`[PW CFG] USE_VITE=${USE_VITE} NO_WEBSERVER=${NO_WEBSERVER} HOST=${HOST} PORT=${PORT} BASE_URL=${BASE_URL}`);

export default defineConfig({
  timeout: 300000, // 5 minutes
  testDir: './',
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
    baseURL: BASE_URL,
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
            '--enable-features=WebGPU,UseWebGPUAdapterNameInWebGLExtension',
            '--use-fake-device-for-media-stream',
            '--use-fake-ui-for-media-stream'
          ]
        }
      },
    },
    {
      name: 'web_viewer-root-e2e',
      use: {
        browserName: 'chromium',
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--enable-blink-features=MemoryMeasurement',
            '--enable-features=WebGPU,UseWebGPUAdapterNameInWebGLExtension',
            '--use-fake-device-for-media-stream',
            '--use-fake-ui-for-media-stream'
          ]
        }
      }
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
  ...(NO_WEBSERVER
    ? {}
    : {
        webServer: {
          command: USE_VITE
            ? `npx -y vite@^6 --host ${HOST} --port ${PORT} --strictPort`
            : 'python3 dev/web_viewer/serve_with_headers.py',
          port: PORT,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
      }),
});