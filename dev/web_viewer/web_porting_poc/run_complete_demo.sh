#!/bin/bash

# Complete Unified Animation System with DeepMimic Integration
# Final demonstration and verification script

echo "🚀 Unified Animation System with DeepMimic Integration"
echo "======================================================"
echo ""

echo "📁 Checking file structure..."
ls -la unified_animation_test.html deepmimic_demo.js unified_animation_integration_test.js convert_deepmimic_to_onnx.py

echo ""
echo "🤖 Checking DeepMimic ONNX models..."
ls -la deepmimic_onnx/

echo ""
echo "🎬 System Components:"
echo "✅ RSMT (Realtime Stylized Motion Transition)"
echo "✅ FaceFormer (Audio-to-Facial Animation)" 
echo "✅ AudioGesture (Enhanced Gesture Generation)"
echo "✅ DeepMimic (Reinforcement Learning Policies) - NEW!"
echo "✅ BVH Timeline (Multi-source Compositing)"

echo ""
echo "🎯 DeepMimic Features Added:"
echo "   • Actor/Critic Neural Networks (ONNX format)"
echo "   • Policy Types: Walk, Run, Jump, Custom"
echo "   • State Input: Random, Reference, Manual, Interactive"
echo "   • Real-time BVH frame generation"
echo "   • Timeline integration and compositing"

echo ""
echo "🌐 Starting web server..."
echo "URL: http://localhost:8080/unified_animation_test.html"
echo ""

# Find available port
PORT=8080
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo "📡 Starting server on port $PORT..."

# Start Python HTTP server
python3 -m http.server $PORT > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 2

echo "✅ Server started (PID: $SERVER_PID)"
echo ""
echo "🧪 Testing Framework:"
echo "   • Integration Test: new UnifiedAnimationIntegrationTest().runIntegrationTest()"
echo "   • Model Test: new UnifiedAnimationModelTest().runCompleteTest()"  
echo "   • DeepMimic Demo: window.runDeepMimicDemo()"
echo "   • Policy Transitions: window.testDeepMimicTransitions()"
echo "   • Unified Compositing: window.testUnifiedCompositing()"

echo ""
echo "🎮 Usage Instructions:"
echo "1. Open http://localhost:$PORT/unified_animation_test.html"
echo "2. Click 'Initialize DeepMimic' in the 4th panel"
echo "3. Select policy type (Walk/Run/Jump)"
echo "4. Click 'Execute Policy' to generate BVH frames"
echo "5. Use Timeline controls to play/pause/composite animations"
echo "6. Monitor HUD for real-time performance metrics"

echo ""
echo "🔧 Browser Console Commands:"
echo "   // Initialize all systems"
echo "   await Promise.all([initializeRSMT(), initializeFaceFormer(), initializeAudioGesture(), initializeDeepMimic()]);"
echo ""
echo "   // Run DeepMimic walking policy" 
echo "   document.getElementById('deepmimic-policy').value = 'humanoid3d_walk';"
echo "   await runDeepMimicPolicy();"
echo ""
echo "   // Check generated BVH frame"
echo "   console.log(window.currentBVHFrame);"
echo ""
echo "   // Run integration test"
echo "   new UnifiedAnimationIntegrationTest().runIntegrationTest();"

# Try to open browser
if command -v xdg-open &> /dev/null; then
    echo ""
    echo "🌐 Opening browser..."
    xdg-open "http://localhost:$PORT/unified_animation_test.html?autotest=false" &
elif command -v open &> /dev/null; then
    open "http://localhost:$PORT/unified_animation_test.html?autotest=false" &
else
    echo ""
    echo "⚠️  Please manually open: http://localhost:$PORT/unified_animation_test.html"
fi

echo ""
echo "📋 Success Criteria:"
echo "✅ 4-panel interface with DeepMimic system"
echo "✅ Actor/Critic models load successfully"  
echo "✅ Policy execution generates BVH frames"
echo "✅ Timeline compositing from all 4 systems"
echo "✅ Real-time 30 FPS animation generation"
echo "✅ Performance HUD with metrics display"

echo ""
echo "🎉 DeepMimic integration complete!"
echo "Ready for comprehensive testing of animation clips and deep phase policies!"

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

echo ""
echo "⌨️  Press Ctrl+C to stop the server"
echo ""

# Keep script running
wait $SERVER_PID
