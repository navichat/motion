import { test, expect } from '@playwright/test';

// E2E: Exercise classroom conversation page with Speech, Kokoro, and SpeechT5 backends.

const PAGE = '/tests/e2e/html/classroom-conversation-backends.html';

async function open(page, baseURL, query=""){
  const url = baseURL + PAGE + (query ? ('?' + query.replace(/^\?/, '')) : '');
  await page.goto(url);
  await page.waitForFunction(() => !!window.__classroomConv, null, { timeout: 15000 });
}

async function expectExpressions(page){
  await expect.poll(async () => (await page.evaluate(() => window.__classroomConv?.expressions || 0))).toBeGreaterThan(0);
}

test.describe('Classroom conversation backends [E2E][VRM]', () => {
  test('speech backend', async ({ page, baseURL }) => {
    await open(page, baseURL, 'backend=speech&text=hello%20classroom');
    await expectExpressions(page);
    const info = await page.evaluate(() => window.__classroomConv);
    expect(info.log).toContain('Using backend: speech');
  });

  test('kokoro backend (if provided)', async ({ page, baseURL }) => {
    const KOKORO = process.env.KOKORO_JS_URL;
    if (!KOKORO) test.skip(true, 'Set KOKORO_JS_URL or run setup:real to provide kokoro runtime');
    await open(page, baseURL, `backend=kokoro&kokoroJs=${encodeURIComponent(KOKORO)}&text=hello%20kokoro`);
    await expectExpressions(page);
    const info = await page.evaluate(() => window.__classroomConv);
    expect(info.log).toMatch(/Using backend: kokoro|kokoro error/);
  });

  test('speecht5 backend', async ({ page, baseURL }) => {
    const params = new URLSearchParams();
    if (process.env.RUNTIME_ORT) params.set('runtime.ort', process.env.RUNTIME_ORT);
    if (process.env.SPEECHT5_URL) params.set('speecht5', process.env.SPEECHT5_URL);
    params.set('backend', 'speecht5');
    await open(page, baseURL, params.toString());
    await expectExpressions(page);
    const info = await page.evaluate(() => window.__classroomConv);
    expect(info.log).toMatch(/Using backend: speecht5|SpeechT5 stub|SpeechT5 ORT init ok/);
  });
});
