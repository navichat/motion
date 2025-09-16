#!/usr/bin/env bash
set -euo pipefail

# Ultimate audio-driven conversation E2E via Vite self-serve with auto port selection
# - Picks a free port (defaults from 5196 upward)
# - Starts Vite with IPv4 + strictPort
# - Waits for demo readiness via curl
# - Runs Playwright with a shell timeout
# - Cleans up the dev server on exit

start_port="${USE_VITE_PORT:-5196}"
max_tries=20

is_port_busy() {
  local p="$1"
  (echo >/dev/tcp/127.0.0.1/"$p") >/dev/null 2>&1 && return 0 || return 1
}

pick_port() {
  local p="$start_port"
  for _ in $(seq 1 "$max_tries"); do
    if ! is_port_busy "$p"; then
      echo "$p"
      return 0
    fi
    p=$((p+1))
  done
  echo "No free port found starting at $start_port" >&2
  return 1
}

PORT_CHOSEN=$(pick_port)
export USE_VITE_PORT="$PORT_CHOSEN"

echo "[ultimate:audio:vite:auto] Using port: $USE_VITE_PORT"

cleanup() {
  if [[ -f /tmp/vite_dev_server.pid ]]; then
    kill "$(cat /tmp/vite_dev_server.pid)" 2>/dev/null || true
    rm -f /tmp/vite_dev_server.pid
  fi
}
trap cleanup EXIT

# Start Vite dev server
npx -y vite@^6 --host 127.0.0.1 --port "$USE_VITE_PORT" --strictPort &
echo $! > /tmp/vite_dev_server.pid

# Wait for demo page to respond
for i in $(seq 1 60); do
  if curl -sSf "http://127.0.0.1:$USE_VITE_PORT/demos/ichika_voice_conversation_demo.html" >/dev/null; then
    break
  fi
  sleep 1
done

echo "[ultimate:audio:vite:auto] Running Playwright audio E2E..."
set +e
EXTERNAL_WEBSERVER=1 \
NO_WEBSERVER=1 \
USE_VITE=1 \
USE_VITE_PORT="$USE_VITE_PORT" \
timeout 900s npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-ultimate-conversation-audio.spec.js --reporter=line
status=$?
set -e

exit "$status"
