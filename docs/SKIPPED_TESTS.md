## Skipped tests: categories and how to enable

This repo intentionally skips some tests by default to keep CI fast and deterministic. Use the knobs below to enable them locally when desired.

- NO_WEBSERVER guards (web UI/unit-web tests)
  - Pattern: `test.skip(!!process.env.NO_WEBSERVER, 'Requires dev web server')`
  - Default: enabled in unit-web project (Playwright starts a local server automatically).
  - To force-skip: set `NO_WEBSERVER=1` when running tests.

- Real inference (models/CDN/network) — opt-in
  - Pattern: `RUN_REAL_INFERENCE` checks and server URL gates.
  - Covered specs include:
    - Serverless: `unit/inference/*serverless.spec.js` (transformers/hnsw/easyvector/etc.).
    - Web: `unit/inference/real-inference-web.spec.js` (CDN quick checks).
    - Optional server endpoints: `llama-server-smoke.serverless.spec.js` (needs `LLAMA_SERVER_URL`), `diablo-server-smoke.serverless.spec.js` (needs `DIABLO_SERVER_URL`).
  - Default: skipped in CI and when `RUN_REAL_INFERENCE` is not set.
  - Enable locally:
    - `RUN_REAL_INFERENCE=1` to allow general real-inference smokes.
    - Add `LLAMA_SERVER_URL` or `DIABLO_SERVER_URL` for server tests.
    - Some tests import from public CDNs; they will skip if offline or the module is unavailable.

- ONNX Runtime Web smokes (ORT) — opt-in via ModelUrlConfig
  - Specs: `unit/ai/onnx-session-init.spec.js`, `unit/ai/a2g-ort-session-run-web.spec.js`.
  - Requires setting model URLs via `dev/web_viewer/config/models.config.js`:
    - `runtime.ort` → ORT script (e.g., `https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.min.js`).
    - `audio2gesture` → a served `.onnx` (copy under `dev/web_viewer/models/` and reference as `/models/<file>.onnx`).
  - Without these, the tests skip with an explanation.

- Intentional instrumentation skip
  - `unit/inference/dynamic-loader-bootstrap.serverless.spec.js` skips due to unreliable `about:blank` head instrumentation and is covered by unit-web flows.

Shortcuts:
- See `dev/web_viewer/tests/REAL_INFERENCE.md` for details.
- Scripts added in `package.json`:
  - `npm run test:real:serverless`
  - `npm run test:real:web`
  - `npm run test:real:all`
