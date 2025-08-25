import { test, expect } from '@playwright/test';

// Optional real-inference smokes. These only run when RUN_REAL_INFERENCE is set.
// Shell/global timeouts are enforced via scripts and Playwright config; keep a per-file cap too.
// Note: Transformers test may download a small model on first run; keep timeouts reasonable.

test.setTimeout(300_000); // 5 min safety for first-time model/wasm init

const RUN_REAL = !!process.env.RUN_REAL_INFERENCE;
const MODE = (process.env.RUN_REAL_INFERENCE || '').toLowerCase();

test.describe('real inference (env-gated)', () => {
  test.skip(!RUN_REAL, 'Set RUN_REAL_INFERENCE=1 to enable these optional smokes');

  test('real inference: hnswlib-wasm tiny index search (offline, fast)', async () => {
    test.info().annotations.push({ type: 'real-inference', description: 'hnswlib-wasm tiny index' });
    const { HierarchicalNSW } = await import('hnswlib-wasm');

    const dim = 4;
    const space = 'l2';
    const numElements = 6;
    const hnsw = await HierarchicalNSW.fromSpace(space, dim);
    await hnsw.initIndex(numElements, 8, 16);

    // Insert a few simple vectors
    const items = [
      { id: 0, v: [0, 0, 0, 0] },
      { id: 1, v: [1, 0, 0, 0] },
      { id: 2, v: [0, 1, 0, 0] },
      { id: 3, v: [0, 0, 1, 0] },
      { id: 4, v: [0, 0, 0, 1] },
      { id: 5, v: [1, 1, 0, 0] },
    ];
    for (const { id, v } of items) {
      await hnsw.addPoint(new Float32Array(v), id);
    }

    // Query near [1, 0, 0, 0]
    const q = new Float32Array([1, 0, 0, 0]);
    const { neighbors, distances } = await hnsw.searchKNN(q, 3);

    expect(neighbors.length).toBe(3);
    expect(distances.length).toBe(3);
    // Nearest should be the identical vector id=1
    expect(neighbors[0]).toBe(1);
    expect(Number.isFinite(distances[0])).toBe(true);
  });

  test('real inference: transformers.js feature-extraction (optional network)', async () => {
    test.skip(!(MODE.includes('transformers') || MODE.includes('full') || MODE === '1'),
      'Enable with RUN_REAL_INFERENCE=transformers or full');
    test.info().annotations.push({ type: 'real-inference', description: 'transformers.js feature-extraction' });
    // Lazy import to avoid overhead when skipped
    const transformers = await import('@xenova/transformers');

    // Prefer a relatively small model to keep download time low
    const pipe = await transformers.pipeline('feature-extraction', 'Xenova/bert-base-uncased');
    const out = await pipe('hello world');

    // Output is [batch, tokens, dims]
    expect(Array.isArray(out)).toBe(true);
    const dims = out?.[0]?.[0]?.length ?? 0;
    expect(dims).toBeGreaterThan(100);
    // Ensure values are finite numbers
    const sample = out[0][0].slice(0, 4);
    for (const v of sample) expect(Number.isFinite(v)).toBe(true);
  });
});
