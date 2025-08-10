import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

const ORCH = read('dev/web_viewer/src/orchestrator/IchikaOrchestrator.js');
const TL = read('dev/web_viewer/src/models/bvh/BVHTimeline.js');
const ADAPTER = read('dev/web_viewer/src/components/animation/timeline/TimelineChunkAdapter.js');
const INTEGRATION = read('dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js');
const BINDER = read('dev/web_viewer/src/components/animation/vrm/AvatarBinder.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Validates IchikaOrchestrator.bindAvatar wires AvatarBinder + Integration.

test('IchikaOrchestrator.bindAvatar wires binder + integration (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, TL);
  await inject(page, ADAPTER);
  await inject(page, INTEGRATION);
  await inject(page, BINDER);
  await inject(page, ORCH);

  const result = await page.evaluate(() => {
  // Resolve constructor exposed by prior script tag (UMD global)
  const OrchestratorCtor = window.IchikaOrchestrator;

    const orch = new OrchestratorCtor({ timeline: new window.BVHTimeline({ framerate: 30 }) });

    // Bind with stub binder (no VRM instance)
    const { binder, integration } = orch.bindAvatar({});

    // Simulate a frame reaching timeline: directly invoke integration if present
    if (integration && typeof integration.handleTimelineFrame === 'function') {
      const frame = { motionData: [ [0,0,0, 0,0,0], [0,0,0, 5,0,0] ] , metadata: {} };
      integration.handleTimelineFrame(frame, 0.0);
    }

    const stats = binder ? binder.getStats() : null;
    return { hasBinder: !!binder, hasIntegration: !!integration, applied: stats ? stats.applied : 0 };
  });

  expect(result.hasBinder).toBeTruthy();
  expect(result.hasIntegration).toBeTruthy();
  expect(result.applied).toBeGreaterThan(0);
});
