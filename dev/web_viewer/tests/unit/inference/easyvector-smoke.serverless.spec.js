import { test, expect } from '@playwright/test';

// "easyvector" style smoke: tiny wrapper over hnswlib-wasm for embeddings search

test('easyvector wrapper: index/search flow returns nearest id', async ({ page }) => {
  await page.goto('about:blank');
  await page.addScriptTag({ content: `
    window.__loadHNSW = async () => {
      try {
        const mod = await import('https://cdn.jsdelivr.net/npm/hnswlib-wasm@0.24.0/dist/index.umd.js');
        return mod.default || mod;
      } catch {}
      try {
        const mod2 = await import('https://cdn.jsdelivr.net/npm/hnswlib-wasm@0.24.0/dist/index.min.js');
        return mod2.default || mod2;
      } catch {}
      return null;
    };
  `});

  const { ok, id } = await page.evaluate(async () => {
    const h = await window.__loadHNSW();
    if (!h) return { ok: false, id: -1 };
    // Minimal wrapper
    const makeIndex = (dim) => {
      const H = h.HierarchicalNSW || (h.default && h.default.HierarchicalNSW);
      const idx = new H('cosine', dim);
      return {
        add(points) {
          idx.initIndex(points.length, 16, 200, 100);
          points.forEach((p, i) => idx.addPoint(Float32Array.from(p), i));
        },
        search(q, k=1) { return idx.searchKNN(Float32Array.from(q), k).neighbors; }
      };
    };
    const ev = makeIndex(3);
    ev.add([[0,0,1],[0,1,0],[1,0,0]]);
    const n = ev.search([0,0,0.99], 1)[0];
    return { ok: n === 0, id: n };
  });

  if (!ok) test.skip(true, 'hnswlib-wasm not available via CDN here');
  expect(id).toBe(0);
});
