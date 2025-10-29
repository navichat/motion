#!/bin/bash

# VRM System Screenshot Capture Script
# This script tests the VRM system and captures screenshots using headless Chrome

echo "🎭 VRM System Screenshot Capture Starting..."

# Create screenshot directory
mkdir -p /home/runner/work/motion/motion/test-results/vrm-final-screenshots

# Function to capture screenshot
capture_screenshot() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    echo "📸 Capturing: $description"
    echo "🌐 URL: $url"
    echo "📁 Output: $output"
    
    # Use Chrome headless to capture screenshot
    google-chrome \
        --headless \
        --disable-gpu \
        --disable-web-security \
        --enable-features=WebGPU,SharedArrayBuffer \
        --enable-webgl \
        --window-size=1280,720 \
        --screenshot="$output" \
        "$url" 2>/dev/null
    
    if [ -f "$output" ]; then
        local size=$(du -h "$output" | cut -f1)
        echo "✅ Screenshot captured: $size"
        return 0
    else
        echo "❌ Screenshot failed"
        return 1
    fi
}

# Test VRM loading test page
echo ""
echo "🧪 Testing VRM Loading Test Page..."
capture_screenshot \
    "http://localhost:8080/demos/vrm_loading_test.html" \
    "/home/runner/work/motion/motion/test-results/vrm-final-screenshots/01-vrm-loading-test.png" \
    "VRM Loading Test Page"

# Wait for modules to load and test
sleep 3

# Test working voice conversation demo with VRM
echo ""
echo "🎤 Testing Working Voice Conversation Demo with VRM..."
capture_screenshot \
    "http://localhost:8080/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0" \
    "/home/runner/work/motion/motion/test-results/vrm-final-screenshots/02-working-voice-demo-vrm.png" \
    "Working Voice Demo with VRM Enabled"

# Test complete conversation system
echo ""
echo "🏫 Testing Complete Conversation System..."
capture_screenshot \
    "http://localhost:8080/demos/complete_ichika_conversation_system.html" \
    "/home/runner/work/motion/motion/test-results/vrm-final-screenshots/03-complete-system-initial.png" \
    "Complete Conversation System Initial"

# Test working real VRM BVH demo
echo ""
echo "🎭 Testing Real VRM BVH Demo..."
capture_screenshot \
    "http://localhost:8080/demos/real_vrm_bvh_demo.html" \
    "/home/runner/work/motion/motion/test-results/vrm-final-screenshots/04-real-vrm-bvh-demo.png" \
    "Real VRM BVH Demo"

# Test VRM orchestrator demo
echo ""
echo "🎼 Testing VRM Orchestrator Demo..."
capture_screenshot \
    "http://localhost:8080/demos/ichika_vrm_orchestrator_demo.html" \
    "/home/runner/work/motion/motion/test-results/vrm-final-screenshots/05-vrm-orchestrator-demo.png" \
    "VRM Orchestrator Demo"

# Test our new VRM screenshot demo
echo ""
echo "🆕 Testing New VRM Screenshot Demo..."
capture_screenshot \
    "http://localhost:8080/demos/vrm_screenshot_demo.html" \
    "/home/runner/work/motion/motion/test-results/vrm-final-screenshots/06-new-vrm-screenshot-demo.png" \
    "New VRM Screenshot Demo"

echo ""
echo "📊 Screenshot Capture Summary"
echo "=============================="

# List all captured screenshots with sizes
cd /home/runner/work/motion/motion/test-results/vrm-final-screenshots
total_size=0
file_count=0

for file in *.png; do
    if [ -f "$file" ]; then
        size=$(du -b "$file" | cut -f1)
        size_kb=$((size / 1024))
        total_size=$((total_size + size))
        file_count=$((file_count + 1))
        echo "📸 $file - ${size_kb} KB"
    fi
done

total_kb=$((total_size / 1024))
echo ""
echo "📋 Total: $file_count screenshots, ${total_kb} KB"

if [ $file_count -gt 0 ]; then
    echo "✅ Screenshot capture completed successfully!"
    echo "📁 Screenshots saved in: test-results/vrm-final-screenshots/"
else
    echo "❌ No screenshots were captured"
fi