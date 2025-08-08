# Web Viewer (Avatar AI) — Reorganized

This is the reorganized WebNN/WebGPU/WASM avatar workspace. For a complete guide, see:

- docs/README.md — System overview and quick start
- docs/PROJECT_STRUCTURE_GUIDE.md — Detailed structure and component map
- docs/TESTING_INFRASTRUCTURE.md — How to run unit/integration/e2e and legacy tests
- docs/COMPONENT_TESTING_GUIDE.md — Test individual components

Quick start:

```bash
cd dev/web_viewer
python3 src/utils/serve_with_headers.py
npx playwright test --project=chromium-webgpu
```

Organized source lives under src/; tests under src/testing/ and dev/web_viewer/tests/* for Playwright projects.

