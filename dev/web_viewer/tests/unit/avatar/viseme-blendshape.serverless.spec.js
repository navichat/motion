import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const INTEGRATION = read('dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js');
const BINDER = read('dev/web_viewer/src/components/animation/vrm/AvatarBinder.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Serverless test: ensure face viseme metadata propagates to blendshape updates via integration.

test('Viseme metadata drives VRM blendshape updates (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, INTEGRATION);
  await inject(page, BINDER);

  const res = await page.evaluate(() => {
    // Load integration ctor
    const IntegrationCtor = (function(){
      const m = { exports: {} }; const e = {};
      const integrationSrc = document.scripts[document.scripts.length-2].text;
      (new Function('window','module','exports', integrationSrc))(window, m, e);
      return (m.exports && m.exports.name) ? m.exports : (window.BVHTimelineVRMIntegration || m.exports);
    })();
    // Load binder
    const BinderCtor = (function(){
      const m = { exports: {} }; const e = {};
      const binderSrc = document.scripts[document.scripts.length-1].text;
      (new Function('window','module','exports', binderSrc))(window, m, e);
      return (m.exports && m.exports.AvatarBinder) ? m.exports.AvatarBinder : window.AvatarBinder;
    })();

    const binder = new BinderCtor(); // stub
    const Integration = (typeof IntegrationCtor === 'function') ? IntegrationCtor : IntegrationCtor.default || IntegrationCtor;
    const integ = new Integration(binder, { smoothing: false, framerate: 30 });

    // Timeline frame with viseme metadata
    const frame = {
      motionData: [],
      metadata: { faceViseme: 'A' }
    };
    // Process one frame
    integ.handleTimelineFrame(frame, 0.0);

    // Verify binder recorded an expression update
    const stats = binder.getStats();
    const exprCount = stats.expressions;
    // Also capture last expression name if available
    const last = binder.blendshapeRecords[binder.blendshapeRecords.length - 1] || { name: null, weight: null };
    return { exprCount, last };
  });

  expect(res.exprCount).toBeGreaterThanOrEqual(1);
  expect(res.last.name).toBeTruthy();
});
