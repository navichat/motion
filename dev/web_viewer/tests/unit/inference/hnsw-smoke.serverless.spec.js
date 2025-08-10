import { test, expect } from '@playwright/test';

// HNSW smoke: tiny index of random 2D vectors and a nearest-neighbor query.
// Stays serverless and very fast.

test('hnswlib-wasm: build tiny index and query nearest', async ({ page }) => {
  await page.goto('about:blank');
  // Load module in page context to avoid bundling
  await page.addScriptTag({ content: `
      window.__loadHNSW = async () => {
        try {
          // Try UMD build first
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

    const probe = await page.evaluate(async () => {
      const mod = await window.__loadHNSW();
      return !!mod && (mod.HierarchicalNSW || (mod.default && mod.default.HierarchicalNSW)) ? true : false;
    });
    if (!probe) test.skip(true, 'hnswlib-wasm not available via CDN in this environment');

    const { ok, idx, res } = await page.evaluate(async () => {
      const hnsw = await window.__loadHNSW();
      const H = hnsw.HierarchicalNSW || (hnsw.default && hnsw.default.HierarchicalNSW);
      const space = new H('l2', 2);
      const points = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [10.0, 10.0],
      ];
      space.initIndex(points.length, 16, 200, 100);
      points.forEach((p, i) => space.addPoint(Float32Array.from(p), i));
      const q = Float32Array.from([0.1, 0.1]);
      const result = space.searchKNN(q, 1);
      return { ok: result.neighbors?.[0] === 0, idx: result.neighbors?.[0], res: result };
    });

    expect(ok).toBeTruthy();
    expect(idx).toBe(0);
    expect(res).toBeTruthy();
});
