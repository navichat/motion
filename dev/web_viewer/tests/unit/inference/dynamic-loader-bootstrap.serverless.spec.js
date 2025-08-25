import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');
const BOOT = read('dev/web_viewer/src/testing/real_inference_bootstrap.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Serverless: verifies that passing transformers/kokoroJs query params causes bootstrap to insert script tags.
test('bootstrap inserts dynamic loader scripts (serverless)', async ({ page }) => {
  test.skip(true, 'Skip: head script insertion instrumentation is unreliable in about:blank. Covered by unit-web test.');
  await page.goto('about:blank');
  await inject(page, CONFIG);

  const vals = {
    transformers: 'https://example.com/transformers.min.js',
    kokoroJs: 'https://example.com/kokoro.min.js',
  };
  // Instrument script insertion recording before bootstrap runs
  await page.addInitScript(() => {
    window.__insertedScripts = [];
    const origNodeAppend = Node.prototype.appendChild;
    Node.prototype.appendChild = function(node) {
      try {
        if (this.nodeName === 'HEAD' && node && node.nodeName === 'SCRIPT') {
          const src = node.getAttribute('src') || '';
          window.__insertedScripts.push(src);
        }
      } catch {}
      return origNodeAppend.apply(this, arguments);
    };
  });
  await page.evaluate((v) => {
    const qs = `?transformers=${encodeURIComponent(v.transformers)}&kokoroJs=${encodeURIComponent(v.kokoroJs)}`;
    history.replaceState(null, '', qs);
  }, vals);

  await inject(page, BOOT);

  // Wait until our instrumentation sees both URLs attempted to be inserted
  await page.waitForFunction(() => Array.isArray(window.__insertedScripts) && window.__insertedScripts.length >= 2, null, { timeout: 5000 });
  const out = await page.evaluate(() => {
    const arr = Array.isArray(window.__insertedScripts) ? window.__insertedScripts : [];
    return {
      tf: arr.includes('https://example.com/transformers.min.js'),
      kok: arr.includes('https://example.com/kokoro.min.js')
    };
  });

  expect(out.tf).toBe(true);
  expect(out.kok).toBe(true);
});
