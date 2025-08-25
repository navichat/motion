import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
function read(p) { return fs.readFileSync(path.join(ROOT, p), 'utf8'); }

const HEAP_UTILS = read('dev/web_viewer/src/utils/scheduler/FibonacciHeap.js');
const HEAP_CORE = read('dev/web_viewer/src/core/FibonacciHeap.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// This test compares basic behavior between two heap implementations
// without polluting the window namespace by using local module wrappers.

test('FibonacciHeap utils vs core parity (serverless)', async ({ page }) => {
  await page.goto('about:blank');

  const res = await page.evaluate(({ HEAP_UTILS, HEAP_CORE }) => {
    function loadHeapFrom(code) {
      const mod = { exports: {} };
      try { (new Function('module','exports', code + '; return;'))(mod, mod.exports); } catch {}
      const exp = mod.exports || {};
      // Try window fallback in case the module wrote to window
      const HeapA = exp.FibonacciHeap || (typeof window !== 'undefined' ? window.FibonacciHeap : undefined);
      return HeapA;
    }

    const HeapU = loadHeapFrom(HEAP_UTILS);
    const HeapC = loadHeapFrom(HEAP_CORE);

    if (typeof HeapU !== 'function' || typeof HeapC !== 'function') {
      return { ok: false, reason: 'constructors missing' };
    }

    function exercise(HeapCtor) {
      const h = new HeapCtor();
      const nodes = [];
      nodes.push(h.insert(5, 'a'));
      nodes.push(h.insert(3, 'b'));
      nodes.push(h.insert(7, 'c'));
      // decrease key of 'c' to 2
      if (typeof h.decreaseKey === 'function') h.decreaseKey(nodes[2], 2);
      const order = [];
      let m;
      for (let i = 0; i < 3; i++) {
        m = h.extractMin();
        if (!m) break;
        // utils heap returns node; core heap returns { key, value }
        const key = m.key ?? m.key ?? m.key; // stay generic
        const value = m.value ?? m.value ?? m.value;
        order.push({ key: key ?? (m.key ?? 0), value: value ?? (m.value ?? null) });
      }
      return order;
    }

    const oU = exercise(HeapU);
    const oC = exercise(HeapC);

    return { ok: true, utils: oU, core: oC };
  }, { HEAP_UTILS, HEAP_CORE });

  expect(res.ok).toBeTruthy();
  // Expect both to have three results and the same value order (by keys: c -> b -> a)
  expect(res.utils.length).toBe(3);
  expect(res.core.length).toBe(3);
  const utilsVals = res.utils.map(x => x.value).join(',');
  const coreVals = res.core.map(x => x.value).join(',');
  expect(utilsVals).toBe(coreVals);
});
