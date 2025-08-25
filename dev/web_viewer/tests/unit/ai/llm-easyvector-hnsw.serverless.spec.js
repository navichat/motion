// Serverless tests to validate endpoints configs and fallbacks for LLM/EasyVector/HNSW clients.
import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const CONFIG = read('dev/web_viewer/config/models.config.js');
const LLM = read('dev/web_viewer/src/components/ai/tasks/LlmEndpointTask.js');
const EV = read('dev/web_viewer/src/components/ai/tasks/EasyVectorClient.js');
const HNSW = read('dev/web_viewer/src/components/ai/tasks/HnswClient.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

test('LLM/EasyVector/HNSW configs and fallbacks (serverless)', async ({ page }) => {
  // Stay serverless: no web server navigation
  await page.goto('about:blank');
  await inject(page, CONFIG);
  await inject(page, LLM);
  await inject(page, EV);
  await inject(page, HNSW);

  const out = await page.evaluate(async () => {
    const cfg = window.ModelUrlConfig;
    if (!cfg) throw new Error('ModelUrlConfig missing');
    const { LlmEndpointTask } = window.LlmEndpointTask || {};
    const { EasyVectorClient } = window.EasyVectorClient || {};
    const { HnswClient } = window.HnswClient || {};
    if (!LlmEndpointTask || !EasyVectorClient || !HnswClient) throw new Error('clients missing');

    // LLM echo fallback
    const llm = new LlmEndpointTask({ key: 'llama' });
    const it = llm.run({ prompt: 'hello' });
    const a = await it.next();

    // EasyVector in-memory fallback
    const ev = new EasyVectorClient();
    await ev.initialize();
    await ev.upsert([{ id: 'a', vector: [1,0,0] }, { id: 'b', vector: [0,1,0] }]);
    const q = await ev.query([0.9, 0.1, 0]);

    // HNSW fallback instance
    const hnsw = new HnswClient();
    await hnsw.initialize();
    await hnsw.addItem([1,0,0], 'a');
    await hnsw.addItem([0,1,0], 'b');
    const r = await hnsw.searchKnn([0.8,0.2,0], 1);

    return { llmModel: a.value.metadata.model, ev: q.length, hnsw: r.length };
  });

  expect(out.llmModel).toMatch(/llm_endpoint|llm_fallback/);
  expect(out.ev).toBeGreaterThan(0);
  expect(out.hnsw).toBeGreaterThan(0);
});
