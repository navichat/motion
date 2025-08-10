# Web Viewer (Avatar AI) — Reorganized

This is the reorganized WebNN/WebGPU/WASM avatar workspace. For a complete guide, see:

- docs/README.md — System overview and quick start
- docs/PROJECT_STRUCTURE_GUIDE.md — Detailed structure and component map
- docs/TESTING_INFRASTRUCTURE.md — How to run unit/integration/e2e and legacy tests
- docs/COMPONENT_TESTING_GUIDE.md — Test individual components
- docs/readmes/ICHIKA_CLASSROOM_VRM_IMPLEMENTATION_PLAN.md — End-to-end plan to integrate Ichika VRM in the classroom with BVH timelines and real-time conversation

Quick start:

```bash
cd dev/web_viewer
python3 serve_with_headers.py
npx playwright test --project=chromium-webgpu
```

Organized source lives under src/; tests under src/testing/ and dev/web_viewer/tests/* for Playwright projects.

Dev tip: keep this handy in your editor side pane for the BVH scheduler plan and APIs:
- docs/readmes/BVH_SCHEDULER_CONTEXT_PIN.md

## Testing

See docs/readmes/PLAYWRIGHT_TESTING_GUIDE.md for full details.

Common commands:

```bash
# All projects (serverless + unit-web + smoke)
npx playwright test --reporter=line

# Unit tests only
npx playwright test dev/web_viewer/tests/unit --reporter=line

# Scheduler-only
npx playwright test dev/web_viewer/tests/unit/scheduler --reporter=line

# Or use npm scripts from the repo root:
npm run test:unit
npm run test:scheduler
npm run test:unit:serverless
npm run test:unit:web
```

Notes:
- The dev server runs on http://localhost:8080 with COOP/COEP headers. Playwright enforces a webServer shell timeout (120s) by config.
- In unit tests, prefer resolving constructors via a CommonJS-style module shim and use the minimal BVHTimeline for stability; rely on window.* only as a fallback.

