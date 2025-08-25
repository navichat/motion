#!/bin/bash

# Quick test script for the unified animation system with DeepMimic
echo "🧪 Testing Unified Animation System with DeepMimic Integration"
echo "============================================================"

cd /home/barberb/motion/dev/web_viewer/web_porting_poc

# Check if all required files exist
echo "📁 Checking required files..."

files=(
    "unified_animation_test.html"
    "unified_animation_model_test.js"
    "bvh-timeline.js"
    "deepmimic_onnx/deepmimic_actor.onnx"
    "deepmimic_onnx/deepmimic_critic.onnx"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    echo ""
    echo "🚀 All files exist. Starting integrated test..."
    echo ""
    
    # Start a simple Python HTTP server to serve the files
    echo "📡 Starting local server..."
    python3 -m http.server 8082 > /dev/null 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 2
    
    echo "✅ Server started on http://localhost:8082"
    echo ""
    echo "🌐 Test URLs:"
    echo "  - Unified Animation Test: http://localhost:8082/unified_animation_test.html"
    echo "  - Model Verification: Run 'node unified_animation_model_test.js' in browser console"
    echo ""
    echo "🔧 Features to test:"
    echo "  ✅ RSMT System (DeepPhase, StyleVAE, TransitionNet)"
    echo "  ✅ FaceFormer Audio-to-Face Animation"
    echo "  ✅ AudioGesture Enhanced Generator"
    echo "  ✅ DeepMimic Policy Networks (NEW!)"
    echo "  ✅ BVH Timeline Compositing"
    echo "  ✅ Real-time Performance HUD"
    echo ""
    echo "📝 Testing Instructions:"
    echo "  1. Open http://localhost:8082/unified_animation_test.html"
    echo "  2. Initialize all four systems (RSMT, FaceFormer, AudioGesture, DeepMimic)"
    echo "  3. Configure DeepMimic policy type (Walk, Run, Jump, Custom)"
    echo "  4. Set state input mode (Random, Reference, Manual, Interactive)"
    echo "  5. Execute DeepMimic policy and observe BVH frame generation"
    echo "  6. Test frame compositing with multiple animation sources"
    echo "  7. Monitor HUD for performance metrics and system status"
    echo ""
    echo "⌨️  Press Ctrl+C to stop the server when done testing"
    
    # Keep server running until interrupted
    trap "echo ''; echo '🛑 Stopping server...'; kill $SERVER_PID 2>/dev/null; echo '✅ Server stopped'; exit 0" SIGINT
    
    # Open browser if available
    if command -v xdg-open &> /dev/null; then
        echo "🌐 Opening browser..."
        xdg-open "http://localhost:8082/unified_animation_test.html" &
    elif command -v open &> /dev/null; then
        echo "🌐 Opening browser..."
        open "http://localhost:8082/unified_animation_test.html" &
    fi
    
    # Wait for server
    wait $SERVER_PID
    
else
    echo ""
    echo "❌ Missing required files. Please ensure all components are properly set up."
    exit 1
fi
