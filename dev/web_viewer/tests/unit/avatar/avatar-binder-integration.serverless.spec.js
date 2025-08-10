import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const INTEGRATION = read('dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js');
const BINDER = read('dev/web_viewer/src/components/animation/vrm/AvatarBinder.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Serverless test: drive BVHTimelineVRMIntegration with AvatarBinder stub
// and verify bones are updated as expected.

test('AvatarBinder collects VRM bone updates via BVHTimeline integration (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, INTEGRATION);
  await inject(page, BINDER);

  const res = await page.evaluate(() => {
    const IntegrationCtor = (function(){
      const m = { exports: {} }; const e = {};
      // last injected script is binder, second to last integration
      const scripts = Array.from(document.scripts);
      const integrationSrc = scripts[scripts.length-2].text;
      const binderSrc = scripts[scripts.length-1].text; // not used here
      // evaluate integration
      (new Function('window','module','exports', integrationSrc))(window, m, e);
      return (m.exports && m.exports.name) ? m.exports : (window.BVHTimelineVRMIntegration || m.exports);
    })();

    const BinderCtor = (function(){
      const m = { exports: {} }; const e = {};
      const binderSrc = document.scripts[document.scripts.length-1].text;
      (new Function('window','module','exports', binderSrc))(window, m, e);
      return (m.exports && m.exports.AvatarBinder) ? m.exports.AvatarBinder : window.AvatarBinder;
    })();

    if (typeof IntegrationCtor !== 'function' && typeof IntegrationCtor?.constructor !== 'function') throw new Error('Integration missing');
    const BVHtoVRM = (typeof IntegrationCtor === 'function') ? IntegrationCtor : IntegrationCtor.default || IntegrationCtor;

    const binder = new BinderCtor(); // stub mode
    const integ = new BVHtoVRM(binder, { smoothing: false, framerate: 30 });

    // Synthetic frame with two bones
    const frame = {
      motionData: [
        [0.0, 1.0, 2.0, 10.0, 0.0, 0.0], // hips
        [0.0, 1.5, 2.5, 0.0, 20.0, 0.0], // spine
      ],
      metadata: { source: 'unit' }
    };

    integ.handleTimelineFrame(frame, 0.0);

    const stats = binder.getStats();
    const appliedHips = binder.records.find(r => r.boneName === 'hips');
    const appliedSpine = binder.records.find(r => r.boneName === 'spine');

    return { applied: stats.applied, updateCalls: stats.updateCalls, hasHips: !!appliedHips, hasSpine: !!appliedSpine };
  });

  expect(res.applied).toBeGreaterThanOrEqual(2);
  expect(res.updateCalls).toBeGreaterThan(0);
  expect(res.hasHips).toBeTruthy();
  expect(res.hasSpine).toBeTruthy();
});
