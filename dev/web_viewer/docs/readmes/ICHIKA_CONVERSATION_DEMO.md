# Ichika Conversation Demo (Mic → ASR → TTS → Animation)

A lightweight HTML demo that lets you talk to an Ichika VRM avatar and see reply-driven animations via a BVH timeline-of-timelines orchestrator. Designed for deterministic CI with optional real, on-device inference.

Page: `/demos/ichika_voice_conversation_demo.html`

## Quick start

```bash
# From repo root
python3 dev/web_viewer/serve_with_headers.py

# Open in browser (deterministic beeps, no audio playback)
http://127.0.0.1:8080/demos/ichika_voice_conversation_demo.html?backend=beeps&playAudio=0
```

Playwright tests exercise this page:
- Deterministic avatar conversation: `npm run test:e2e:ultimate:avatar`
- Markers validation: `npm run test:e2e:ultimate:markers`

## Query parameters

- `backend`: `beeps` | `speech` | `kokoro` | `speecht5`
- `playAudio`: `0|1` — disable to avoid real playback in CI
- `autoListen`: `0|1` — auto listen after reply
- `vrm`: `0|1` — attempt to load a local VRM (falls back to stub)
- `asr`: `whisper` | `speech` | `fake`
- `asrModel`: model id for whisper (e.g., `Xenova/whisper-tiny.en`)
- SpeechT5:
  - `speecht5OnDevice`: `0|1` — on-device path via transformers.js
  - `speecht5Model`: model id (default: `Xenova/speecht5_tts`)
  - `speecht5Spk`: `random` | `zero` | URL to speaker embeddings (512)
- Kokoro:
  - `kokoroJs`: URL to a kokoro-js runtime (or provide via ModelUrlConfig)

## Playwright markers

The demo logs deterministic markers to `#log` for CI assertions:

TTS:
- Speech synthesis
  - `[PLAYWRIGHT] TTS start engine=speech`
  - `[PLAYWRIGHT] TTS done engine=speech status=ok|error ms=…`
- Kokoro
  - `[PLAYWRIGHT] TTS start engine=kokoro`
  - `[PLAYWRIGHT] TTS done engine=kokoro status=ok|error ms=… durSec=…`
- SpeechT5
  - `[PLAYWRIGHT] TTS start engine=speecht5 path=on-device|ort|stub`
  - `[PLAYWRIGHT] TTS done engine=speecht5 path=… status=… ms=… durSec=…`
- Fallback beeps
  - `[PLAYWRIGHT] TTS engine=beeps useTts=0|1`
  - `[PLAYWRIGHT] TTS done engine=beeps status=ok ms=0 durSec=…`

ASR:
- `[PLAYWRIGHT] ASR done engine=whisper|speech|fake status=ok|error ms=…`

Scheduling:
- `[PLAYWRIGHT] Scheduled TTS animation`

## CI posture

- Deterministic tests disable real audio playback and accept stub/ORT/on-device paths where applicable.
- Shell timeouts wrap every test step; the dev web server has its own timeout in Playwright config.
- Opt-in REAL tests remain gated via env in `.github/workflows/web-viewer-ultimate.yml`.
