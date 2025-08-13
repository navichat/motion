# Web Viewer (Avatar AI) — Reorganized

This is the reorganized WebNN/WebGPU/WASM avatar workspace. For a complete guide, see:

- docs/README.md — System overview and quick start
- docs/PROJECT_STRUCTURE_GUIDE.md — Detailed structure and component map
- docs/TESTING_INFRASTRUCTURE.md — How to run unit/integration/e2e and legacy tests
- docs/COMPONENT_TESTING_GUIDE.md — Test individual components
- docs/readmes/ICHIKA_CLASSROOM_VRM_IMPLEMENTATION_PLAN.md — End-to-end plan to integrate Ichika VRM in the classroom with BVH timelines and real-time conversation

Quick start:

```bash
cd dev/web_viewer
python3 serve_with_headers.py
npx playwright test --project=chromium-webgpu
```

Organized source lives under src/; tests under src/testing/ and dev/web_viewer/tests/* for Playwright projects.

Dev tip: keep this handy in your editor side pane for the BVH scheduler plan and APIs:
- docs/readmes/BVH_SCHEDULER_CONTEXT_PIN.md

## Testing

See docs/readmes/PLAYWRIGHT_TESTING_GUIDE.md for full details.

Common commands:

```bash
# All projects (serverless + unit-web + smoke)
npx playwright test --reporter=line

# Unit tests only
npx playwright test dev/web_viewer/tests/unit --reporter=line

# Scheduler-only
npx playwright test dev/web_viewer/tests/unit/scheduler --reporter=line

# Or use npm scripts from the repo root:
npm run test:unit
npm run test:scheduler
npm run test:unit:serverless
npm run test:unit:web
```

Ultimate conversation smoke:

```bash
# With repo-configured web server
npm run test:e2e:ultimate:smoke

# Start local server then run the same smoke
npm run test:e2e:ultimate:smoke:local

# Optional flags via URL params are supported by the demo:
#   backend=beeps|speech|kokoro|speecht5, playAudio=0|1, asr=whisper|speech|fake, stubMic=1
# The smoke uses backend=beeps&playAudio=0 for determinism.
```

Engine markers smoke (verifies backend markers across beeps/speech/speecht5):

```bash
npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-smoke-ultimate-engine-markers.spec.js --reporter=line
```

VRM attempt smoke (works with or without a local VRM asset):

```bash
npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-smoke-ultimate-vrm.spec.js --reporter=line
```

Opt-in REAL e2e (on-device inference; requires network/CDN and is gated by env):

```bash
To exercise only this test locally with caching of models between runs, consider using a persistent playwright cache directory (e.g., export PLAYWRIGHT_BROWSERS_PATH=~/.cache/playwright) and rely on the transformers.js local IndexedDB caching.
# SpeechT5 on-device (transformers.js)
RUN_REAL_INFERENCE=1 SPEECHT5_MODEL=Xenova/speecht5_tts \
	npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-smoke-ultimate-mic-reply-speecht5-real.spec.js --reporter=line

# Kokoro on-device TTS (kokoro-js runtime URL required)
RUN_REAL_INFERENCE=1 KOKORO_JS=https://cdn.example.com/kokoro.min.js \
	npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-smoke-ultimate-mic-reply-kokoro.spec.js --reporter=line

# Whisper (ASR) + Kokoro (TTS) full loop
RUN_REAL_INFERENCE=1 KOKORO_JS=https://cdn.example.com/kokoro.min.js \
	npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-smoke-ultimate-whisper-kokoro.spec.js --reporter=line
```

Audio pipeline diagnostics (verifies scheduling + playback (or playback error) events across backends speech, kokoro, SpeechT5 fake):

```bash
# Run against existing web server configuration
npm run test:e2e:audio:log

# Start local dev server automatically then run diagnostics
npm run test:e2e:audio:log:local
```

These tests assert:
- At least one TTS scheduling event (tts_scheduled or *_scheduled)
- At least one playback or playback-error event (buffer_play | beeps_play | pcm_play | *_play | speech_play_error)
- Animation expressions > 0 (viseme/energy driven)
- Presence of at least one latency-bearing playback entry (an event containing latencyMs), confirming schedule→play latency capture is functioning

Audio log artifact schema (per scenario) written to test-results/audio-log-<scenario>.json:
```
{
	scenario: string,
	backend: string,                // explicit backend key (speech|kokoro|speecht5|beeps|other)
	timestamp: ISO 8601 string,
	uniqueTypes: string[],          // distinct event.type values observed
	count: number,                  // total events captured (capped window)
	metrics: {                      // from getAudioMetrics(); may be zeroed if no samples
		samples, avgLatencyMs, lastLatencyMs,
		p50, p90, p95, p99
	},
	latencyEntry: boolean,          // true if at least one log entry had a latencyMs field
	tail: EventEntry[]              // last up to 50 raw events
}
```
Where each EventEntry minimally includes `{ ts, type, ...detail, [latencyMs] }`.

Aggregate per-scenario audio metrics table: `node dev/web_viewer/scripts/summarize-audio-logs.js` (automatically included in CI summary) lists Scenario | Events | Types | Samples | Avg | P50 | P95 | P99 | Lat? (latency sample presence).

### Latency & Performance Metrics

The conversation demo (`ichika_voice_conversation_demo.html`) instruments schedule→play latency per utterance.

Exposed metrics (`window.__ultimateDemo.getAudioMetrics()`):
- samples
- avgLatencyMs
- lastLatencyMs
- p50, p90, p95, p99

Run smoke latency guard:
```bash
npm run test:e2e:ultimate:latency
```
Override thresholds:
```bash
LATENCY_MAX_MS=5000 \
LATENCY_P50_MAX_MS=2500 \
LATENCY_P95_MAX_MS=5000 \
LATENCY_P99_MAX_MS=5000 \
npm run test:e2e:ultimate:latency
```

Artifacts emitted:
- `test-results/perf-latency.json`
- `test-results/audio-log-*.json` (tail of events + metrics per scenario)
 - (History) `dev/web_viewer/perf-latency-history.csv` (appended on main)

CI summary step (workflow `web-viewer-ultimate.yml`) aggregates these into the job summary.

### Latency Regression Protection

Baseline file: `dev/web_viewer/perf-baseline.json`

Check for regressions:
```bash
npm run test:perf:latency:verify
```
Env overrides:
```bash
MAX_ABS_INCREASE_P50_MS=400 \
MAX_ABS_INCREASE_P95_MS=800 \
MAX_ABS_INCREASE_P99_MS=1200 \
MAX_REL_INCREASE_P95=2.5 \
DISALLOW_BASELINE_WRITE=1 \
npm run test:perf:latency:verify
```

Process:
1. Run latency test → generates perf-latency.json
2. Run verify script → compares vs baseline; fails if bounds exceeded
3. Update baseline only when an intentional systemic change occurs

Diff / severity classification:
```bash
npm run test:perf:latency:diff
```
Emits a markdown diff with relative change emojis:
- ✅ improvement or neutral (no increase)
- ⬆️ mild increase (below warning threshold)
- ⚠️ moderate increase (approaching limits)
- 🛑 severe increase (would typically fail verification if thresholds exceeded)

Planned enhancements: rolling history, auto PR comments, per-backend budgets.

### Per-Backend Latency Budgets

Granular performance expectations can differ by TTS backend (e.g. simple beep/speech synthesis vs. on-device transformers). The script `dev/web_viewer/scripts/verify-audio-backend-latency.js` enforces backend-specific p50/p95 latency ceilings using the per-scenario audio log artifacts (`test-results/audio-log-*.json`).

Run manually after a test suite that produced audio logs (e.g. the audio diagnostics + latency guard):
```bash
npm run test:perf:latency:verify:backends
```

Default budgets (if no env overrides):
- p50: 5000 ms
- p95: 8000 ms

Override globally:
```bash
DEFAULT_MAX_P50_MS=3000 DEFAULT_MAX_P95_MS=6000 \
	npm run test:perf:latency:verify:backends
```

Override per backend (case-insensitive backend keys resolved from scenario names: speech, kokoro, speecht5, beeps, other):
```bash
BACKEND_SPEECH_MAX_P50_MS=1500 BACKEND_SPEECH_MAX_P95_MS=2500 \
BACKEND_KOKORO_MAX_P50_MS=2200 BACKEND_KOKORO_MAX_P95_MS=3500 \
BACKEND_SPEECHT5_MAX_P50_MS=2600 BACKEND_SPEECHT5_MAX_P95_MS=4200 \
	npm run test:perf:latency:verify:backends
```

Failure conditions:
- Missing latency samples (samples==0) for a scenario that should produce audio playback.
- p50 > budget or p95 > budget (per backend or default thresholds).

CI integration: `web-viewer-ultimate.yml` runs this step (Per-backend latency budgets) after the aggregate latency regression check. Adjust env variables in the workflow / repository secrets to tune budgets without code changes.

Machine-readable export:
```bash
PER_BACKEND_LATENCY_JSON=test-results/per-backend-latency.json \
	npm run test:perf:latency:verify:backends
cat test-results/per-backend-latency.json | jq .
```
Schema:
```
Webhook consolidated payload (for Slack/webhooks):
```bash
npm run perf:webhook:payload
cat test-results/perf-webhook-payload.json | jq .
```

Slack posting (optional; set secret SLACK_WEBHOOK_URL):
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... npm run perf:slack:post
```
Adds a summary (status emoji + key latency figures + truncated markdown tables) to a channel. In CI, the workflow step '(Optional) Post Slack perf summary' runs automatically if the secret is configured.

GitHub PR auto-comment:
The workflow adds/updates a PR comment containing the performance summary (marker header '### 🤖 Performance Summary'). You can generate or post manually:
```bash
npm run perf:pr:comment > /tmp/perf.md
GITHUB_REPOSITORY=owner/repo GITHUB_EVENT_PATH=.github/event.json GITHUB_TOKEN=ghp_xxx \
	npm run perf:pr:post
```
It upserts based on the marker to avoid duplicates.
{
	generatedAt: ISO string,
	defaults: { p50: number, p95: number },
	rows: [ { file, scenario, backend, samples, p50, p95, p50Budget, p95Budget } ],
	violations: [ { backend, file, metric, value, limit, reason? } ],
	ok: boolean
}
```

Rationale: Guard against hidden regressions isolated to a single backend that aggregate metrics might dilute. Enables progressive tightening (e.g., start lenient for new backends, then ratchet down). This also provides early signal when enabling heavier real-model paths in PR CI.

Config file (checked automatically): `dev/web_viewer/perf-backend-budgets.json`
Precedence order for budgets: ENV override > config file entry > global default.
Provide BACKEND_BUDGETS_FILE env to point to an alternate config.

### Ultimate REAL Conversation Test (Optional)

Full multi-turn pipeline (Mic → Whisper on-device ASR → Kokoro on-device TTS → audio-driven gesture animation → optional VRM) is covered by `e2e-ultimate-real-conversation.spec.js`.

Run locally:
```bash
export RUN_REAL_INFERENCE=1
export KOKORO_JS="https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.min.js"
npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-ultimate-real-conversation.spec.js --reporter=line
```

Characteristics:
- Multi-turn (turns=2) with autoListen=1 validates chaining and state carry-over.
- playAudio=1 so kokoro buffer playback produces real schedule→play latency samples.
- Uses quantized `Xenova/whisper-tiny.en` for speed.
- Attempts VRM load (vrm=1); test still passes in stub mode when asset missing.
- Gated by `RUN_REAL_INFERENCE` and `KOKORO_JS` to keep default CI fast.
- Emits latency-bearing `buffer_play` events enabling performance tracking on real audio.

CI: Conditional step in `web-viewer-ultimate.yml` (non-blocking by default). Provide secrets/env to enable; tighten to blocking once stable.

Caching tips:
- Leverage Playwright browser cache (e.g. `export PLAYWRIGHT_BROWSERS_PATH=~/.cache/playwright`).
- transformers.js caches model shards in IndexedDB; run headed once if needed to warm.
- Consider a nightly workflow to prefetch and run the real test with stricter thresholds.

Nightly workflow:
- `web-viewer-ultimate-real-nightly.yml` (scheduled 02:30 UTC) runs the real multi-turn test when `KOKORO_JS_URL` secret is present.
- Artifacts: playwright report + audio-log JSONs; summary includes aggregated audio scenario metrics.
- Purpose: observe real-model drift separately from fast PR CI; can later enforce regression gates.

CI auto-baseline update:
- Occurs only on main branch AND when secret/env `LATENCY_BASELINE_UPDATE=1` is set.
- Safe no-op on PRs or when secret disabled.

Latency history:
- Appended to `dev/web_viewer/perf-latency-history.csv` on main after successful runs.
- Columns: isoTimestamp, gitSha, samples, avg, p50, p90, p95, p99, last
- Use this CSV to visualize trends (import into spreadsheet or plot tool).
- CI summary shows the last N rows (default 10 locally, 15 in CI). Override with env `LATENCY_HISTORY_ROWS`.

Notes:
- The dev server runs on http://localhost:8080 with COOP/COEP headers. Playwright enforces a webServer shell timeout (120s) by config.
- In unit tests, prefer resolving constructors via a CommonJS-style module shim and use the minimal BVHTimeline for stability; rely on window.* only as a fallback.

