#!/bin/bash

# Test Runner for Unified Animation System
# This script sets up a local development server and opens the comprehensive animation tests

echo "🚀 Starting Unified Animation System Test Runner"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "unified_animation_test.html" ]; then
    echo "❌ Error: unified_animation_test.html not found in current directory"
    echo "Please run this script from the directory containing the test files"
    exit 1
fi

# Check for Python (for simple HTTP server)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found. Please install Python to run local server"
    exit 1
fi

# Find an available port
PORT=8080
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    PORT=$((PORT + 1))
done

echo "📡 Starting local server on port $PORT"
echo "Files being served:"
echo "  - unified_animation_test.html (Main test interface)"
echo "  - unified_animation_model_test.js (Model verification)"
echo "  - bvh-timeline.js (BVH Timeline implementation)"
echo ""

# Start server in background
$PYTHON_CMD -m http.server $PORT > /dev/null 2>&1 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Failed to start server"
    exit 1
fi

echo "✅ Server started successfully (PID: $SERVER_PID)"
echo ""
echo "🌐 Opening test interface..."
echo "URL: http://localhost:$PORT/unified_animation_test.html"
echo ""
echo "📝 Test Interface Features:"
echo "  - RSMT System Testing (DeepPhase, StyleVAE, TransitionNet)"
echo "  - FaceFormer Audio-to-Face Animation"
echo "  - AudioGesture Enhanced Generator"
echo "  - DeepMimic Policy Networks (Actor/Critic)"
echo "  - BVH Timeline Compositing"
echo "  - Real-time Performance HUD"
echo "  - Model Loading Verification"
echo ""
echo "🔧 Usage Instructions:"
echo "  1. Wait for all models to load (check status indicators)"
echo "  2. Upload audio files for FaceFormer and AudioGesture"
echo "  3. Configure motion parameters for RSMT"
echo "  4. Set policy parameters for DeepMimic"
echo "  5. Use BVH Timeline controls to composite animations"
echo "  6. Monitor performance via HUD overlay"
echo ""

# Try to open browser
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:$PORT/unified_animation_test.html" &
elif command -v open &> /dev/null; then
    open "http://localhost:$PORT/unified_animation_test.html" &
elif command -v start &> /dev/null; then
    start "http://localhost:$PORT/unified_animation_test.html" &
else
    echo "⚠️  Please manually open: http://localhost:$PORT/unified_animation_test.html"
fi

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down server..."
    kill $SERVER_PID 2>/dev/null
    echo "✅ Server stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "⌨️  Press Ctrl+C to stop the server"
echo ""

# Keep script running
wait $SERVER_PID
