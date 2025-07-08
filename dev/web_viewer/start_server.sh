#!/bin/bash

# RSMT Server Startup Script
# This script starts the RSMT neural network server with model warmup support

echo "🚀 Starting RSMT Neural Network Server..."

# Navigate to the web viewer directory
cd "$(dirname "$0")"

# Check if required Python modules are available
echo "📦 Checking dependencies..."

if ! python3 -c "import http.server" 2>/dev/null; then
    echo "❌ Python http.server not available"
    exit 1
fi

# Kill any existing servers on port 8000
echo "🔄 Checking for existing servers on port 8000..."
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is in use. Attempting to stop existing server..."
    pkill -f "python.*8000" 2>/dev/null || true
    sleep 2
fi

# Try to start the advanced server with API support
if python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "🧠 FastAPI available - starting advanced server with neural network support..."
    if [ -f "simple_warmup_server.py" ]; then
        python3 simple_warmup_server.py &
        SERVER_PID=$!
        echo "🎯 Advanced server started with PID $SERVER_PID"
    else
        echo "⚠️  Advanced server script not found, falling back to simple server..."
        python3 -m http.server 8000 &
        SERVER_PID=$!
    fi
else
    echo "📁 Starting simple HTTP server (neural networks will be simulated)..."
    python3 -m http.server 8000 &
    SERVER_PID=$!
    echo "🌐 Simple server started with PID $SERVER_PID"
fi

# Wait a moment for server to start
sleep 3

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server is running on http://localhost:8000"
    echo "🎭 Open http://localhost:8000/rsmt_showcase_modern.html to access the interface"
    echo "🧠 Neural network warmup controls are available in the AI Inference Monitor"
    echo ""
    echo "📊 Available endpoints:"
    echo "   • http://localhost:8000/ - Main interface"
    echo "   • http://localhost:8000/api/status - Server status (if advanced server)"
    echo "   • http://localhost:8000/api/encode_style - StyleVAE warmup (if advanced server)"
    echo "   • http://localhost:8000/api/generate_transition - TransitionNet warmup (if advanced server)"
    echo ""
    echo "🛑 To stop the server: kill $SERVER_PID"
    echo "📝 Server PID saved to server.pid"
    echo $SERVER_PID > server.pid
else
    echo "❌ Failed to start server"
    exit 1
fi
