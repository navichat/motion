#!/bin/bash

# Motion Platform Development Startup Script
# Integrates chat features with RSMT and 3D viewer

set -e

echo "🚀 Starting Motion Platform Development Environment..."

# Check if Node.js and Python are available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."

# Install main dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing main project dependencies..."
    npm install
fi

# Install psyche dependencies
if [ ! -d "psyche/node_modules" ]; then
    echo "Installing psyche dependencies..."
    cd psyche && npm install && cd ..
fi

# Install webapp dependencies  
if [ ! -d "webapp/node_modules" ]; then
    echo "Installing webapp dependencies..."
    cd webapp && npm install && cd ..
fi

# Install viewer dependencies
if [ ! -d "viewer/node_modules" ]; then
    echo "Installing viewer dependencies..."
    cd viewer && npm install && cd ..
fi

# Install Python dependencies
if [ ! -d "server/venv" ]; then
    echo "Creating Python virtual environment..."
    cd server && python3 -m venv venv && cd ..
fi

echo "Installing Python dependencies..."
cd server
source venv/bin/activate
pip install -r requirements_server.txt
cd ..

# Ensure assets are properly linked
echo "🔗 Setting up asset links..."

# Create symlinks in webapp for assets
if [ ! -L "webapp/assets" ]; then
    ln -sf ../assets webapp/assets
fi

# Create symlinks in viewer for assets
if [ ! -L "viewer/assets" ]; then
    ln -sf ../assets viewer/assets
fi

# Set environment variables
export NODE_ENV=development
export MOTION_ASSETS_PATH="$(pwd)/assets"
export MOTION_SERVER_PORT=8080
export MOTION_WEBAPP_PORT=3000
export MOTION_VIEWER_PORT=3001
export MOTION_PYTHON_PORT=8081

echo "🌟 Starting all services..."

# Create log directory
mkdir -p logs

# Start Python RSMT server in background
echo "Starting Python RSMT server on port $MOTION_PYTHON_PORT..."
cd server
source venv/bin/activate
python enhanced_motion_server.py --port=$MOTION_PYTHON_PORT > ../logs/python-server.log 2>&1 &
PYTHON_PID=$!
cd ..

# Start Node.js chat server in background
echo "Starting Node.js chat server on port $MOTION_SERVER_PORT..."
cd server
PORT=$MOTION_SERVER_PORT node server.js > ../logs/node-server.log 2>&1 &
NODE_PID=$!
cd ..

# Start webapp development server in background
echo "Starting webapp on port $MOTION_WEBAPP_PORT..."
cd webapp
PORT=$MOTION_WEBAPP_PORT npm run dev > ../logs/webapp.log 2>&1 &
WEBAPP_PID=$!
cd ..

# Start viewer development server
echo "Starting viewer on port $MOTION_VIEWER_PORT..."
cd viewer
PORT=$MOTION_VIEWER_PORT npm run dev > ../logs/viewer.log 2>&1 &
VIEWER_PID=$!
cd ..

# Save PIDs for cleanup
echo $PYTHON_PID > logs/python.pid
echo $NODE_PID > logs/node.pid
echo $WEBAPP_PID > logs/webapp.pid
echo $VIEWER_PID > logs/viewer.pid

echo ""
echo "✅ Motion Platform is starting up!"
echo ""
echo "🔗 Available services:"
echo "   • Chat Interface:  http://localhost:$MOTION_WEBAPP_PORT"
echo "   • 3D Viewer:       http://localhost:$MOTION_VIEWER_PORT"
echo "   • Node.js Server:  http://localhost:$MOTION_SERVER_PORT"
echo "   • Python Server:   http://localhost:$MOTION_PYTHON_PORT"
echo ""
echo "📁 Asset directory:   $MOTION_ASSETS_PATH"
echo "📋 Logs directory:    $(pwd)/logs"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ -f logs/python.pid ]; then
        kill $(cat logs/python.pid) 2>/dev/null || true
        rm -f logs/python.pid
    fi
    
    if [ -f logs/node.pid ]; then
        kill $(cat logs/node.pid) 2>/dev/null || true
        rm -f logs/node.pid
    fi
    
    if [ -f logs/webapp.pid ]; then
        kill $(cat logs/webapp.pid) 2>/dev/null || true
        rm -f logs/webapp.pid
    fi
    
    if [ -f logs/viewer.pid ]; then
        kill $(cat logs/viewer.pid) 2>/dev/null || true
        rm -f logs/viewer.pid
    fi
    
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to press Ctrl+C
while true; do
    sleep 1
done
