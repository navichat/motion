import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const REG = read('dev/web_viewer/src/animation/ClipRegistry.js');
const MANIFEST = read('dev/web_viewer/src/animation/clip_manifest.sample.json');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validate manifest loading registers clips with metadata

test('[clips] ClipRegistry loadFromManifest registers entries (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, REG);

  const res = await page.evaluate((mjson) => {
    const { ClipRegistry } = window.ClipRegistry || {};
    if (!ClipRegistry) throw new Error('ClipRegistry missing');
    const manifest = JSON.parse(mjson);
    const reg = new ClipRegistry();
    reg.loadFromManifest(manifest);
    const list = reg.list();
    const names = list.map(x => x.name);
    const point = reg.get('point');
    return { count: list.length, names, pointPrio: point.meta.priority, pointTrack: point.meta.track };
  }, MANIFEST);

  expect(res.count).toBeGreaterThanOrEqual(3);
  expect(res.names).toEqual(expect.arrayContaining(['idle','point','wave']));
  expect(res.pointPrio).toBe(5);
  expect(res.pointTrack).toBe('override');
});
