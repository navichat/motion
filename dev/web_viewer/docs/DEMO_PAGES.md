# Demo Pages

Authoritative list of demo pages and their purpose. These are used by tests and for manual validation.

## AI Inference

- /dev/web_viewer/demos/ai-inference/task-manager-demo.html
  Purpose: Orchestrates AI jobs across backends (WebNN/WebGPU/WASM). Primary target for integration/e2e tests.

## HTML Tests

- /dev/web_viewer/demos/html-tests/debug_onnx_runtime.html
  Purpose: Inspect ONNX Runtime behavior in-browser.

- /dev/web_viewer/demos/html-tests/direct_ai_test.html
  Purpose: Direct AI pipeline invocation for quick sanity checks.

- /dev/web_viewer/demos/html-tests/test-model-fetch.html
  Purpose: Verify model fetch paths and CORS headers.

- /dev/web_viewer/demos/html-tests/test-quantized-models.html
  Purpose: Validate quantized model loading and inference.

- /dev/web_viewer/demos/html-tests/test-wasm-direct.html
  Purpose: WASM-only workflow validation.

## Serving locally

Use the Playwright webServer config or run the utility server:

```bash
# From repository root
python3 dev/web_viewer/src/utils/serve_with_headers.py
# or
python3 -m http.server 8080
```

Access demos at:
- http://localhost:8080/dev/web_viewer/demos/ai-inference/task-manager-demo.html
- http://localhost:8080/dev/web_viewer/demos/html-tests/test-wasm-direct.html
