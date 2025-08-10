# Real Inference Tests (Opt-in)

These checks exercise actual model inference using client-side pipelines. They are disabled by default to keep CI fast and deterministic.

What’s covered:
- Whisper tiny (ASR) via @xenova/transformers (serverless)
- Text generation via @xenova/transformers (gpt2 model) (serverless)
- VAD via @xenova/transformers (Silero) (serverless)
- HNSW and easyvector-style ANN wrappers using hnswlib-wasm (serverless; skips if CDN unavailable)
- Optional web checks (kokoro TTS, transformers VAD) require the web server

Pre-refactoring ORT tasks (already in dev/web_viewer)
- These web smokes exercise the existing onnxruntime-web tasks and HTML fixtures. They are skipped unless model URLs are provided.
  - Whisper ORT: `dev/web_viewer/tests/unit/ai/whisper-ort-web-smoke.spec.js` → fixture `dev/web_viewer/tests/unit/ai/fixtures/ort_whisper_smoke.html`
  - FaceFormer ORT: `dev/web_viewer/tests/unit/ai/faceformer-ort-web-smoke.spec.js` → fixture `dev/web_viewer/tests/unit/ai/fixtures/ort_faceformer_smoke.html`
  - Silero VAD ORT: `dev/web_viewer/tests/unit/ai/silero-vad-ort-web-smoke.spec.js` → fixture `dev/web_viewer/tests/unit/ai/fixtures/ort_silero_vad_smoke.html`

How model URLs are injected
- The fixtures include `config/models.config.js` and `src/testing/real_inference_bootstrap.js`.
- Tests pass environment variables which are converted to query params by `src/testing/env_to_query.js`.
- Supported env → query keys:
  - RUNTIME_ORT → runtime.ort (onnxruntime-web UMD URL)
  - WHISPER_URL → whisper (model URL)
  - FACEFORMER_URL → faceformer (model URL)
  - VAD_URL → sileroVad (model URL)
  - SPEECHT5_URL → speecht5 (model URL)
  - KOKORO_ID → kokoro (model id/url)
  - LLAMA_ENDPOINT → llama (HTTP endpoint)
  - DIABLO_GPT_ENDPOINT → diabloGpt (HTTP endpoint)
  - EASYVECTOR_ENDPOINT → easyvector (HTTP endpoint)

Legacy module exposure (pre-refactor implementations)
- You can expose the legacy pre-refactor modules to window for compatibility/testing via query flags handled in `src/testing/real_inference_bootstrap.js`.
- Environment variables mapped by `src/testing/env_to_query.js`:
  - LEGACY=1 → enables all legacy exposures (ResourceManager, WhisperModule, KokoroModule, LlamaModule)
  - LEGACY_WHISPER=1 → exposes only WhisperModule
  - LEGACY_KOKORO=1 → exposes only KokoroModule
  - LEGACY_LLAMA=1 → exposes only LlamaModule
- Unit-web smoke that validates this wiring: `dev/web_viewer/tests/unit/inference/legacy-exposure-web-smoke.spec.js` with fixture `dev/web_viewer/tests/unit/inference/fixtures/legacy_exposure.html`.

Running the ORT smokes (example)
- Start the dev server (adds COOP/COEP headers required by ORT): `python3 dev/web_viewer/serve_with_headers.py`
- Provide model/runtime URLs via env, e.g.:
  - RUNTIME_ORT=https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.min.js
  - WHISPER_URL=http://localhost:8080/models/whisper-tiny.onnx
- Then run a single test under the unit-web project (web-server required). For example, Whisper:
  - `npx playwright test --project=unit-web dev/web_viewer/tests/unit/ai/whisper-ort-web-smoke.spec.js`
- The tests will skip gracefully if not configured.

Enable locally:
- Serverless real inference:
  - RUN_REAL_INFERENCE=1 npx playwright test --project=component-tests -g "real inference"
- Web real inference (ensure the dev server is running):
  - RUN_REAL_INFERENCE=1 npx playwright test --project=unit-web -g "real inference (web)"

Notes:
- These tests use CDN imports when possible to avoid bundling changes.
- Some Hugging Face models require authorization; we use open models (e.g., Xenova/gpt2) to avoid auth prompts.
- If a CDN module is not accessible in your environment, relevant tests will skip gracefully.
