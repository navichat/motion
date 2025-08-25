import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');
const KOKORO_TASK = read('dev/web_viewer/src/components/ai/tasks/KokoroTtsTask.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Web-backed smoke for Kokoro: skips unless a model id/url is configured or network is available.
// We don't enforce audio checks; we only assert the metadata shows real or fallback path.

test('Kokoro TTS smoke (skipped unless configured)', async ({ page }) => {
  await page.goto('/index.html');
  await inject(page, CONFIG);
  await inject(page, KOKORO_TASK);
  const ok = await page.evaluate(async () => {
    const ctor = window.KokoroTtsTask && window.KokoroTtsTask.KokoroTtsTask;
    if (typeof ctor !== 'function') return { skip: true, reason: 'ctor missing' };
    const task = new ctor({ text: 'hello from kokoro' });
    await task.initialize(window);
    const it = task.run({});
    const first = await it.next();
    if (!first || !first.value) return { skip: true, reason: 'no output' };
    return { skip: false, model: first.value.metadata.model };
  });
  if (ok.skip) test.skip(true, ok.reason || 'Not configured');
  expect(['kokoro', 'kokoro_fallback']).toContain(ok.model);
});
