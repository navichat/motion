# ML Runtime Helpers

This folder contains lightweight, framework-free helpers for model runtime setup used in tests and integrations.

## Files
- `ModelRuntimeConfig.js`
  - `detectWebGPU(env?)`: returns boolean based on navigator.gpu presence
  - `selectOrtProvider(options?, env?)`: 'webgpu' if available (default), otherwise 'wasm'
  - `buildOrtSessionOptions(provider)`: returns session options skeleton (provider-specific)
  - `getModelFsPath(name)`: Node file path helper for artifacts (e.g., 'audio2gesture')

## Usage
- In serverless Playwright tests, inject these helpers and choose a provider without importing onnxruntime-web.
- In web-backed tests, combine with `dev/web_viewer/config/models.config.js` to resolve model URLs to serve.

## Notes
- Keep these helpers side-effect free and browser/Node friendly.
- When wiring real sessions, import onnxruntime-web from the viewer code, not from tests.
