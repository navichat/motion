# Real Inference Enablement Guide

This guide shows how to run real inference for models in unit-web smokes. By default, smokes skip or fall back to deterministic outputs.

## Prereqs
- Run the dev server (Playwright does this automatically for unit-web project). Server is at http://localhost:8080.
- Host model artifacts and ORT runtime where the browser can fetch them. Use local paths or static server URLs.

## Configure Model URLs at runtime
From the browser console on any test page (e.g., a smoke fixture), set URLs before the task runs:

```js
// onnxruntime-web runtime (wasm build recommended)
ModelUrlConfig.setModelUrl('runtime.ort', '/libs/ort/ort-wasm.min.js');

// Audio/gesture and visemes
ModelUrlConfig.setModelUrl('audio2gesture', '/models/audio2gesture.onnx');
ModelUrlConfig.setModelUrl('faceformer', '/models/faceformer.onnx');

// ASR / VAD
ModelUrlConfig.setModelUrl('whisper', '/models/whisper-base.onnx');
ModelUrlConfig.setModelUrl('sileroVad', '/models/silero_vad.onnx');

// TTS
ModelUrlConfig.setModelUrl('speecht5', '/models/speecht5.onnx');
ModelUrlConfig.setModelUrl('kokoro', 'kokoro-large'); // or a specific URL if self-hosted

// LLM endpoints (generic JSON POST)
ModelUrlConfig.setModelUrl('llama', 'http://localhost:11434/api/generate');
ModelUrlConfig.setModelUrl('diabloGpt', 'http://localhost:8000/v1/chat/completions');

// Vector / ANN
ModelUrlConfig.MODELS.easyvector = { endpoint: 'http://localhost:7070' };
```

If you need to auto-load ORT into the page, add a script tag (some fixtures attempt this automatically):

```js
const ortSrc = ModelUrlConfig.getModelUrl('runtime.ort');
if (ortSrc) { const s = document.createElement('script'); s.src = ortSrc; document.head.appendChild(s); }
```

## Which tests to run
- All unit-web AI smokes (skips become active when URLs are set):

```bash
npx playwright test --project=unit-web dev/web_viewer/tests/unit/ai/*.spec.js --reporter=line
```

- Serverless (deterministic) AI subset:

```bash
npm run -s test:ml:serverless
```

## Expected Behavior
- When both `window.ort` and the specific `ModelUrlConfig` URL are present, the task initializes a real session and reports metadata like `whisper_ort`, `silero_vad_ort`, `faceformer_ort`, `speecht5_ort`, `kokoro` in the first yielded chunk.
- When missing, the tests skip or fall back to deterministic outputs with metadata like `*_fallback`.

## Notes
- WebGPU is optional; WASM builds work and are simpler to host. The server already sets COOP/COEP headers.
- Keep Playwright shell/webServer timeouts as configured (policy requirement).
