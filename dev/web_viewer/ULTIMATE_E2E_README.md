# Ultimate Conversation E2E Quick Runs

This repo includes deterministic and audio-driven end-to-end tests for the Ichika voice conversation demo.

- Deterministic (no playback):
  - Local Python server:
    ```bash
    npm run test:e2e:ultimate:conversation:local
    ```
  - Vite self-serve:
    ```bash
    npm run test:e2e:ultimate:conversation:vite
    ```
  - ASR markers:
    - Direct (no server):
      ```bash
      npm run test:e2e:ultimate:markers:asr
      ```
    - Vite self-serve (auto-port):
      ```bash
      npm run test:e2e:ultimate:markers:asr:vite
      ```

- Audio-driven (playback, energy drives expressions):
  - Local Python server:
    ```bash
    npm run test:e2e:ultimate:audio:local
    ```
  - Vite self-serve (auto-picks free port, cleans up):
    ```bash
    npm run test:e2e:ultimate:audio:vite
    ```

Notes
- All Playwright invocations use shell timeouts.
- Vite runs with `--host 127.0.0.1` and `--strictPort` for determinism.
- If you manually run with a fixed port and see a conflict, either kill it (`fuser -k <port>/tcp`) or pick a new `USE_VITE_PORT`.
- Tests synchronize via DOM `#log` markers prefixed with `[PLAYWRIGHT]` for stability in headless runs.
