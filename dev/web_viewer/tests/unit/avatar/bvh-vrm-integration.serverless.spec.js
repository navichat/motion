import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

// Minimal timeline and integration modules
const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const INTEGRATION = read('dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

test('BVH → VRM integration applies frames via adapter (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, INTEGRATION);

  const res = await page.evaluate(async () => {
    // CommonJS-style module shim
    const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
    const BVHTimelineCtor = mk(document.scripts[document.scripts.length-2].text, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
    const IntegrationCtor = mk(document.scripts[document.scripts.length-1].text, '(module && module.exports) ? module.exports : window.BVHTimelineVRMIntegration');

    if (typeof BVHTimelineCtor !== 'function') throw new Error('BVHTimeline ctor missing');
    if (typeof IntegrationCtor !== 'function') throw new Error('BVHTimelineVRMIntegration ctor missing');

    // Stub VRM adapter that records updates
    const applied = [];
    const vrmAdapter = {
      updateBone(boneName, position, rotation) {
        applied.push({ boneName, position, rotation });
      },
      updateCalls: 0,
      update(time) { this.updateCalls++; }
    };

    const integration = new IntegrationCtor(vrmAdapter, { smoothing: false });

    // Use minimal timeline just to satisfy connectTimeline API
    const timeline = new BVHTimelineCtor({ framerate: 30 });
    integration.connectTimeline(timeline);

    // Build a synthetic timeline frame with motionData indices matching default map
    const motionData = [];
    for (let i = 0; i < 20; i++) {
      // [px, py, pz, rx(deg), ry(deg), rz(deg)]
      motionData.push([i*0.001, i*0.002, i*0.003, i*1.0, i*2.0, i*3.0]);
    }
    const frame = { motionData, metadata: { source: 'test' } };

    // Process a frame
    integration.handleTimelineFrame(frame, 0.0);

    const stats = integration.getStats();
    const updated = applied.length;
    const hasHips = applied.some(a => a.boneName === 'hips');
    const hasLeftArm = applied.some(a => a.boneName === 'leftArm' || a.boneName === 'leftUpperArm');

    return { framesProcessed: stats.framesProcessed, updated, updateCalls: vrmAdapter.updateCalls, hasHips, hasLeftArm };
  });

  expect(res.framesProcessed).toBeGreaterThan(0);
  expect(res.updated).toBeGreaterThan(0);
  expect(res.updateCalls).toBeGreaterThan(0);
  expect(res.hasHips).toBeTruthy();
  expect(res.hasLeftArm).toBeTruthy();
});
