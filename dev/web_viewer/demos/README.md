# Ichika Conversation Demo

Mic → ASR → TTS with audio-driven gestures for a 3D avatar (VRM optional).

- File: `ichika_voice_conversation_demo.html`
- Exposes `window.__ultimateDemo`:
  - `startMic()`, `stopAll()`
  - `sayText(text, play)`
  - `listenAndReply()`
  - `conversationLoop(n)`
  - `getStats()` → `{ applied, expressions, stub }`
  - `maybeLoadVRM()`

Run

```bash
# Python server with COOP/COEP headers
PORT=8080 python3 dev/web_viewer/serve_with_headers.py
# Open in browser
# http://127.0.0.1:8080/demos/ichika_voice_conversation_demo.html
```

Internal Vite web server

```bash
# Playwright-managed Vite (from playwright.config.js)
USE_VITE=1 USE_VITE_PORT=5180 timeout 900s npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-ultimate-conversation.spec.js --reporter=line

# Vite self-serve (external server lifecycle)
npm run test:e2e:ultimate:conversation:vite
```

Query params

- `backend=speech|kokoro|speecht5|beeps`
- `asr=fake|whisper` (Whisper via transformers.js)
- `speecht5OnDevice=1` (enable on-device TTS path)
- `playAudio=0|1` (mute/unmute audio playback)
- `vrm=1` (attempt to load VRM at `../../assets/avatars/ichika.vrm`)
- `listenSec=1|2|…` (mic record length for ASR)

Deterministic E2E (CI gate)

```bash
npm run test:e2e:ultimate:conversation
# or with server lifecycle
npm run test:e2e:ultimate:conversation:local
# or Vite self-serve (starts + reuses external server)
npm run test:e2e:ultimate:conversation:vite
```

Real inference (opt-in)

```bash
RUN_REAL_INFERENCE=1 npm run test:e2e:ultimate:conversation:real
# or
RUN_REAL_INFERENCE=1 npm run test:e2e:ultimate:conversation:real:local
```

Conversation loop (opt-in)

```bash
RUN_CONV_LOOP=1 npm run test:e2e:ultimate:conversation:loop
# or
RUN_CONV_LOOP=1 npm run test:e2e:ultimate:conversation:loop:local
```

VS Code tasks

- Run Ultimate Conversation E2E (web server)
- Run Ultimate Conversation E2E (Vite self-serve)
- Run Ultimate Conversation E2E (real, web server)

CI

- Deterministic CI workflow: .github/workflows/e2e-ultimate-conversation.yml
- Manual real-inference workflow (opt-in): .github/workflows/e2e-ultimate-conversation-real.yml

Notes

- Tests include shell timeouts per repo policy.
- Bind servers to 127.0.0.1 to avoid IPv6 (::1) misbinds.
- Vite is always invoked via `npx -y vite@^6` (internal and tasks) to avoid conflicts with any system `vite` binary.
