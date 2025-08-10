# Playwright Testing Guide (Web Viewer)

This doc explains how to run the unit suites for the web viewer, including serverless vs. web-backed modes, timeouts, and helpful scripts.

## Projects
- component-tests (serverless)
  - Runs dev/web_viewer/tests/unit/**/*.serverless.spec.js in Chromium without a dev server.
- unit-web (web-backed)
  - Runs dev/web_viewer/tests/unit (excluding serverless) against http://localhost:8080 using serve_with_headers.py.
- web_viewer-root-e2e (smoke)
  - Runs dev/web_viewer/e2e-smoke.spec.js to validate routing and base page load.

## Timeouts and stability
- Web server startup timeout: 120s
- Global/test timeouts: 10 min/5 min
- Per-project action/navigation/expect timeouts are set to reduce flake.
- Shell/webServer timeout is enforced via playwright.config.js (webServer.timeout = 120_000) per repo policy.
- Tests prefer resolving constructors from module.exports using a small CommonJS-style eval shim; window.* is used only as a fallback.
- Prefer the minimal BVHTimeline (src/models/bvh/BVHTimeline.js) in tests for faster and more stable runs.

### Module-shim pattern for constructors
When injecting sources with page.addScriptTag({ content }), resolve constructors via a CommonJS-style shim to avoid brittle window.* coupling:

```js
// In page.evaluate
const mk = (code, ret) => (new Function('window','module','exports', code + '; return ' + ret))(window, { exports: {} }, {});
const BVHTimelineCtor = mk(timelineCode, '(module.exports && module.exports.BVHTimeline) || window.BVHTimeline');
const TimelineChunkAdapterCtor = mk(adapterCode, '(module.exports && module.exports.TimelineChunkAdapter) || (window.TimelineChunkAdapter && window.TimelineChunkAdapter.TimelineChunkAdapter) || window.TimelineChunkAdapter');
```

### Troubleshooting
- Error: constructor missing in serverless tests
  - Ensure you’re using the module-shim pattern above and prefer the minimal BVHTimeline.
- Port already in use / web server errors
  - The config uses port 8080. You can set `NO_WEBSERVER=1` to run just serverless tests, or free the port and re-run.

## Common commands
- All tests (matrix):
  - npx playwright test --reporter=line
- Unit tests only:
  - npx playwright test dev/web_viewer/tests/unit --reporter=line
- Serverless-only (component-tests project):
  - npx playwright test --project=component-tests dev/web_viewer/tests/unit --reporter=line
- Web-backed unit tests (unit-web project):
  - npx playwright test --project=unit-web dev/web_viewer/tests/unit --reporter=line
- Scheduler-only:
  - npx playwright test dev/web_viewer/tests/unit/scheduler --reporter=line

## ML-related tests
- Serverless ML helpers and config:
  - npx playwright test dev/web_viewer/tests/unit/ai/ml-runtime-config.serverless.spec.js --project=component-tests --reporter=line
  - npx playwright test dev/web_viewer/tests/unit/ai/models-config.serverless.spec.js --project=component-tests --reporter=line
- Optional artifact presence check (skips if missing):
  - npx playwright test dev/web_viewer/tests/unit/ai/audio2gesture-artifact.spec.js --reporter=line
- ORT Web session-init smoke (skips unless configured via ModelUrlConfig setModelUrl('runtime.ort', ...) and setModelUrl('audio2gesture', ...)):
  - npx playwright test dev/web_viewer/tests/unit/ai/onnx-session-init.spec.js --project=unit-web --reporter=line
 - ORT path simulation with a fake window.ort (serverless):
  - npx playwright test dev/web_viewer/tests/unit/scheduler/audio2gesture-ort-task-sim.serverless.spec.js --project=component-tests --reporter=line
 - Auto-resolve A2G model URL from ModelUrlConfig (serverless):
  - npx playwright test dev/web_viewer/tests/unit/ai/ort-model-url-autoresolve.serverless.spec.js --project=component-tests --reporter=line
 - A2G ORT session run smoke (skips unless configured via ModelUrlConfig):
  - npx playwright test dev/web_viewer/tests/unit/ai/a2g-ort-session-run-web.spec.js --project=unit-web --reporter=line

## Motion tasks and runner tests
- Audio2Gesture stub → timeline append (serverless):
  - npx playwright test dev/web_viewer/tests/unit/scheduler/audio2gesture-stub-task.serverless.spec.js --project=component-tests --reporter=line
- ORT task fallback → timeline append (serverless):
  - npx playwright test dev/web_viewer/tests/unit/scheduler/audio2gesture-ort-task-fallback.serverless.spec.js --project=component-tests --reporter=line
- ModelTaskRunner drives tasks (serverless):
  - npx playwright test dev/web_viewer/tests/unit/scheduler/model-task-runner.serverless.spec.js --project=component-tests --reporter=line
 - RSMT/DeepMimic stubs → timeline append (serverless):
  - npx playwright test dev/web_viewer/tests/unit/scheduler/rsmt-deepmimic-stubs.serverless.spec.js --project=component-tests --reporter=line
 - Orchestrator pipeline with stubs (deterministic serverless):
  - npx playwright test dev/web_viewer/tests/unit/orchestrator/pipeline-stubs.serverless.spec.js --project=component-tests --reporter=line

## npm scripts (root)
- npm run test:unit
- npm run test:scheduler
- npm run test:unit:serverless
- npm run test:unit:web
- npm run serve:web_viewer

## Notes
- The dev server lives at dev/web_viewer/serve_with_headers.py (port 8080). COOP/COEP headers are set for WebGPU/WebNN/WASM.
- For development, keep the pinned context handy: docs/readmes/BVH_SCHEDULER_CONTEXT_PIN.md
