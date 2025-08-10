import { test, expect } from '@playwright/test';

// Serverless manifest ingestion test for ClipRegistry + BVHClipLibrary hooks.
// This avoids network fetches; we only check metadata registration and passthrough objects.

test.setTimeout(60_000);

test('ClipRegistry: loadFromManifest registers entries with meta and urls', async () => {
  const ns = (globalThis.ClipRegistry || (globalThis.ClipRegistryNS));
  let ClipRegistryCtor = ns && (ns.ClipRegistry || ns);
  if (!ClipRegistryCtor || typeof ClipRegistryCtor !== 'function') {
    // Fall back to CommonJS require path used by orchestrator
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('../../../src/animation/ClipRegistry.js');
    ClipRegistryCtor = mod && (mod.ClipRegistry || (mod.default && mod.default.ClipRegistry));
  }
  expect(typeof ClipRegistryCtor).toBe('function');

  const reg = new ClipRegistryCtor();
  const manifest = [
    { name: 'idle', url: '/assets/bvh/idle.bvh', meta: { track: 'base', duration: 3.2, fadeInMs: 150 } },
    { name: 'point', url: '/assets/bvh/point.bvh', meta: { track: 'override', duration: 1.1, priority: 5 } },
    { name: 'wave', data: { t0: 0, dt: 1.0, frames: [] }, meta: { track: 'override', duration: 1.0 } },
  ];

  const before = reg.list();
  expect(Array.isArray(before)).toBe(true);
  const countBefore = before.length;

  reg.loadFromManifest(manifest);
  const entries = reg.list();
  expect(entries.length).toBe(countBefore + 3);

  // Check one URL entry and one data entry
  const idle = reg.get('idle');
  expect(idle.meta.track).toBe('base');
  expect(idle.data && idle.data.url).toBe('/assets/bvh/idle.bvh');
  expect(idle.meta.duration).toBeCloseTo(3.2, 5);

  const wave = reg.get('wave');
  expect(wave.meta.track).toBe('override');
  expect(wave.data && wave.data.dt).toBe(1.0);
});
