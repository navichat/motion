Optional real inference smokes

- These tests are skipped by default. Enable with:

  RUN_REAL_INFERENCE=1 npx playwright test --project=component-tests -g "real inference"

- To run the optional transformers.js test (may download a small model):

  RUN_REAL_INFERENCE=transformers npx playwright test --project=component-tests -g "transformers"

- CI posture: serverless/component-tests can run without a web server; global and shell timeouts apply.
Serverless, unit-web, and smokes

- Serverless unit tests run in a blank page and inject modules directly (no web server).
  - Project: `component-tests`
  - Specs: `**/*.serverless.spec.js`
  - Run: `npm run test:serverless`

- Unit-web tests require the dev web server (auto-started by Playwright unless `NO_WEBSERVER=1`).
  - Project: `unit-web`
  - Specs: `dev/web_viewer/tests/unit/**/*.web.spec.js`
  - Run examples:
    - Run all unit-web tests: `npx playwright test --project=unit-web --reporter=line`
    - Ichika subset: `npx playwright test --project=unit-web -g "Ichika demo:" --reporter=line`

- E2E smokes are HTTP-only and fast; they do not navigate heavy pages.
  - Project: `web_viewer-root-e2e`
  - Specs:
    - `dev/web_viewer/e2e-smoke.spec.js` (index DOM checks)
    - `dev/web_viewer/e2e-smoke-http.spec.js` (VRM + classroom demos via HTTP fetch)
  - Run: `npm run test:smoke`

Environment flags
- `NO_WEBSERVER=1` → disable projects that require a dev server (keeps CI green)
- `RUN_FULL_E2E=1` → enable additional full-browser projects (WebGPU/WebNN)
- `PW_GREP` → filter tests by title via regex

Timeout posture
- All Playwright invocations are wrapped in shell timeouts (see package.json scripts).
- Per-file test timeouts are set in specs for smokes and unit-web.
- Web server has its own timeout in `playwright.config.js`.

Targeting Ichika/VRM tests
- Filter by title with `PW_GREP` or use exact titles with `-g`:
  - Run Ichika/VRM subset: `PW_GREP="(ichika|classroom|VRM)" npx playwright test --project=unit-web`
  - Just the speech preemption: `npx playwright test --project=unit-web -g "Ichika demo: speech preempts face/audio"`

