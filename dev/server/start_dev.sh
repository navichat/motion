#!/bin/bash

# Motion Viewer Development Server - Phase 1 Startup Script

echo "🚀 Starting Motion Viewer Development Environment - Phase 1"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "enhanced_motion_server.py" ]; then
    echo "❌ Error: Please run this script from the dev/server directory"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Check for Node.js (for building the frontend)
if ! command -v node &> /dev/null; then
    echo "⚠️  Warning: Node.js not found. Frontend building may not work."
    echo "   You can still run the server, but you'll need to build the frontend separately."
fi

echo ""
echo "📦 Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install basic dependencies for Phase 1
pip install -q fastapi uvicorn[standard] pydantic

echo "✅ Python dependencies installed"

echo ""
echo "🏗️  Building frontend (if Node.js is available)..."

# Try to build the frontend
cd ../viewer
if command -v npm &> /dev/null; then
    if [ ! -d "node_modules" ]; then
        echo "Installing npm dependencies..."
        npm install --silent
    fi
    
    echo "Building frontend..."
    npm run build 2>/dev/null || echo "⚠️  Frontend build failed or not configured yet"
    echo "✅ Frontend build attempted"
else
    echo "⚠️  Skipping frontend build (Node.js not available)"
fi

cd ../server

echo ""
echo "🌐 Starting development server..."
echo ""
echo "Phase 1 Features Available:"
echo "  ✅ 3D Avatar Loading (VRM support)" 
echo "  ✅ Chat Animation Playback (JSON format)"
echo "  ✅ Classroom Environment"
echo "  ✅ Basic Viewer Controls"
echo "  ✅ REST API Endpoints"
echo ""
echo "Phase 2 Features (Coming Soon):"
echo "  🔄 RSMT Neural Network Integration"
echo "  🔄 100STYLE Dataset Support"
echo "  🔄 Real-time Style Transfer"
echo "  🔄 Motion Transition Generation"
echo ""
echo "🔗 Server will be available at:"
echo "   http://localhost:8081"
echo ""
echo "📚 API Documentation:"
echo "   http://localhost:8081/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================="
echo ""

# Start the server
python3 enhanced_motion_server.py
