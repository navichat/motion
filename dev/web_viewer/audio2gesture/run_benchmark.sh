#!/bin/bash

# Ensure the script exits if any command fails
set -e

echo "Starting Audio2Gesture Playwright Benchmark..."

# Define the path to the directory to serve
SERVE_DIR="/home/barberb/motion"
HTTP_SERVER_PORT=8080

# Check if port is already in use and kill the process
echo "Checking if port ${HTTP_SERVER_PORT} is in use..."
if lsof -i :${HTTP_SERVER_PORT} >/dev/null; then
    echo "Port ${HTTP_SERVER_PORT} is already in use. Attempting to kill the process."
    kill -9 $(lsof -t -i :${HTTP_SERVER_PORT})
    sleep 2
fi

# Start http-server in the background
echo "Attempting to start http-server serving ${SERVE_DIR} on port ${HTTP_SERVER_PORT}..."
nohup http-server "${SERVE_DIR}" -p ${HTTP_SERVER_PORT} > /tmp/http-server.log 2>&1 &
HTTP_SERVER_PID=$!
echo "http-server command executed. PID: $HTTP_SERVER_PID. Log: /tmp/http-server.log"

# Wait for the server to be ready by polling the benchmark URL
echo "Waiting for http-server to respond..."
RETRY_COUNT=0
MAX_RETRIES=12 # Total wait time: 12 * 5s = 60s

# Check if http-server is still running before we start polling
if ! kill -0 $HTTP_SERVER_PID 2>/dev/null; then
    echo "Error: http-server (PID $HTTP_SERVER_PID) failed to start. Check /tmp/http-server.log for details."
    cat /tmp/http-server.log
    exit 1
fi

until curl -s --head "http://localhost:${HTTP_SERVER_PORT}/dev/web_viewer/audio2gesture/audio2gesture_optimization_demo.html" | head -n 1 | grep "HTTP/1.1 200 OK" > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: http-server did not respond with 200 OK after ${MAX_RETRIES} attempts."
        echo "URL checked: http://localhost:${HTTP_SERVER_PORT}/dev/web_viewer/audio2gesture/audio2gesture_optimization_demo.html"
        echo "Please check the server log at /tmp/http-server.log"
        cat /tmp/http-server.log
        kill "$HTTP_SERVER_PID" || true
        exit 1
    fi
    echo "Server not ready yet. Retrying in 5 seconds... (Attempt ${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 5
done

echo "http-server is up and running."

# Run the Playwright benchmark
echo "Running Playwright benchmark..."
node /home/barberb/motion/dev/web_viewer/audio2gesture/benchmark.js

echo "Audio2Gesture Playwright Benchmark Finished."

# Kill the http-server process
echo "Stopping http-server (PID: $HTTP_SERVER_PID)..."
kill "$HTTP_SERVER_PID" || true # Use || true to prevent script from exiting if process is already gone
echo "http-server stopped."
