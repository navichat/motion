#!/usr/bin/env bash
set -euo pipefail

# Ultimate conversation E2E via Vite self-serve with auto port selection

start_port="${USE_VITE_PORT:-5198}"
max_tries=20

is_port_busy() { (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1; }

pick_port() {
  local p="$start_port"
  for _ in $(seq 1 "$max_tries"); do
    if ! is_port_busy "$p"; then echo "$p"; return 0; fi
    p=$((p+1))
  done
  echo "No free port found starting at $start_port" >&2; return 1
}

PORT_CHOSEN=$(pick_port)
export USE_VITE_PORT="$PORT_CHOSEN"
echo "[ultimate:conversation:vite:auto] Using port: $USE_VITE_PORT"

cleanup() {
  [[ -f /tmp/vite_dev_server.pid ]] && kill "$(cat /tmp/vite_dev_server.pid)" 2>/dev/null || true
  rm -f /tmp/vite_dev_server.pid || true
}
trap cleanup EXIT

npx -y vite@^6 --host 127.0.0.1 --port "$USE_VITE_PORT" --strictPort &
echo $! > /tmp/vite_dev_server.pid

for i in $(seq 1 60); do
  curl -sSf "http://127.0.0.1:$USE_VITE_PORT/demos/ichika_voice_conversation_demo.html" >/dev/null && break || sleep 1
done

echo "[ultimate:conversation:vite:auto] Running Playwright conversation E2E..."
EXTERNAL_WEBSERVER=1 \
NO_WEBSERVER=1 \
USE_VITE=1 \
USE_VITE_PORT="$USE_VITE_PORT" \
timeout 900s npx playwright test --project=web_viewer-root-e2e dev/web_viewer/e2e-ultimate-conversation.spec.js --reporter=line
