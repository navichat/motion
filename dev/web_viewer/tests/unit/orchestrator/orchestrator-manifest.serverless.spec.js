import { test, expect } from '@playwright/test';

// Serverless test: orchestrator loads manifest into ClipRegistry and starts base clip using registry meta.

test.setTimeout(60_000);

test('IchikaOrchestrator.loadClipManifest + startBaseClip (serverless)', async () => {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { IchikaOrchestrator } = require('../../../src/orchestrator/IchikaOrchestrator.js');
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { ClipRegistry } = require('../../../src/animation/ClipRegistry.js');

  const orch = new IchikaOrchestrator({ timeline: { addClip() { return true; }, clearFrom() {} }, adapter: {
    appended: [],
    appendChunk(track, chunk, opts = {}) { this.appended.push({ track, chunk, opts }); return true; },
    clearFrom() {}
  }});
  const reg = new ClipRegistry();
  orch.setClipRegistry(reg);

  // Load manifest JSON file via Node require
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const manifest = require('../../../assets/manifests/ichika_demo_manifest.json');
  const count = orch.loadClipManifest(manifest);
  expect(count).toBeGreaterThanOrEqual(3);

  const entry = orch.startBaseClip('idle');
  expect(entry && entry.name).toBe('idle');
  // Verify adapter.appendChunk was called with track derived from meta
  expect(orch.adapter.appended.length).toBeGreaterThan(0);
  expect(orch.adapter.appended[0].track).toBe('base');
});
