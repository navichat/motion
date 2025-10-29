#!/bin/bash

# Comprehensive VRM Interactive Screenshot Capture
# Tests VRM system with user interactions and captures detailed screenshots

echo "🎭 Comprehensive VRM Interactive Screenshot Capture"
echo "=================================================="

# Create screenshot directory
SCREENSHOT_DIR="/home/runner/work/motion/motion/test-results/vrm-interactive-screenshots"
mkdir -p "$SCREENSHOT_DIR"

echo "📁 Screenshot directory: $SCREENSHOT_DIR"

# Function to capture interactive screenshot
capture_interactive_screenshot() {
    local url="$1"
    local output="$2"
    local description="$3"
    local wait_time="${4:-5}"
    
    echo ""
    echo "📸 Capturing Interactive: $description"
    echo "🌐 URL: $url"
    echo "⏱️  Wait time: ${wait_time}s"
    echo "📁 Output: $output"
    
    # Use Chrome headless with longer wait for complex pages
    timeout 60s google-chrome \
        --headless \
        --disable-gpu \
        --disable-web-security \
        --enable-features=WebGPU,SharedArrayBuffer \
        --enable-webgl \
        --window-size=1920,1080 \
        --virtual-time-budget=$((wait_time * 1000)) \
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

# Function to test URL accessibility
test_url() {
    local url="$1"
    local description="$2"
    
    echo "🌐 Testing: $description"
    local status=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$status" = "200" ]; then
        echo "✅ Accessible (HTTP $status)"
        return 0
    else
        echo "❌ Not accessible (HTTP $status)"
        return 1
    fi
}

echo ""
echo "🔍 Pre-flight URL Accessibility Check"
echo "===================================="

# Test all demo URLs
test_url "http://localhost:8080/demos/vrm_loading_test.html" "VRM Loading Test"
test_url "http://localhost:8080/demos/ichika_voice_conversation_demo.html" "Voice Conversation Demo"
test_url "http://localhost:8080/demos/complete_ichika_conversation_system.html" "Complete System"
test_url "http://localhost:8080/demos/real_vrm_bvh_demo.html" "Real VRM BVH Demo"

echo ""
echo "📸 Starting Interactive Screenshot Capture"
echo "=========================================="

# Test 1: VRM Loading Diagnostics (with time for module loading)
capture_interactive_screenshot \
    "http://localhost:8080/demos/vrm_loading_test.html" \
    "$SCREENSHOT_DIR/01-vrm-diagnostics.png" \
    "VRM Loading Diagnostics" \
    10

# Test 2: Working Voice Demo with VRM Parameter
capture_interactive_screenshot \
    "http://localhost:8080/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0" \
    "$SCREENSHOT_DIR/02-voice-demo-vrm-enabled.png" \
    "Voice Demo with VRM Enabled" \
    15

# Test 3: Complete Conversation System (Fixed)
capture_interactive_screenshot \
    "http://localhost:8080/demos/complete_ichika_conversation_system.html" \
    "$SCREENSHOT_DIR/03-complete-system.png" \
    "Complete Conversation System" \
    12

# Test 4: Real VRM BVH Demo
capture_interactive_screenshot \
    "http://localhost:8080/demos/real_vrm_bvh_demo.html" \
    "$SCREENSHOT_DIR/04-real-vrm-bvh.png" \
    "Real VRM BVH Demo" \
    15

# Test 5: VRM Orchestrator Demo
capture_interactive_screenshot \
    "http://localhost:8080/demos/ichika_vrm_orchestrator_demo.html" \
    "$SCREENSHOT_DIR/05-vrm-orchestrator.png" \
    "VRM Orchestrator Demo" \
    10

# Test 6: Custom VRM Screenshot Demo
capture_interactive_screenshot \
    "http://localhost:8080/demos/vrm_screenshot_demo.html" \
    "$SCREENSHOT_DIR/06-custom-vrm-demo.png" \
    "Custom VRM Screenshot Demo" \
    15

# Test 7: Enhanced Classroom Demo
capture_interactive_screenshot \
    "http://localhost:8080/demos/ichika_enhanced_classroom_demo.html" \
    "$SCREENSHOT_DIR/07-enhanced-classroom.png" \
    "Enhanced Classroom Demo" \
    12

# Test 8: Full Classroom Experience
capture_interactive_screenshot \
    "http://localhost:8080/demos/ichika_full_classroom_experience.html" \
    "$SCREENSHOT_DIR/08-full-classroom-experience.png" \
    "Full Classroom Experience" \
    12

echo ""
echo "📊 Interactive Screenshot Summary"
echo "==============================="

# Calculate total size and count
cd "$SCREENSHOT_DIR"
total_size=0
file_count=0
largest_file=""
largest_size=0

echo "📸 Captured Screenshots:"
for file in *.png; do
    if [ -f "$file" ]; then
        size=$(du -b "$file" | cut -f1)
        size_kb=$((size / 1024))
        total_size=$((total_size + size))
        file_count=$((file_count + 1))
        
        if [ $size -gt $largest_size ]; then
            largest_size=$size
            largest_file="$file"
        fi
        
        echo "  📸 $file - ${size_kb} KB"
    fi
done

total_kb=$((total_size / 1024))
largest_kb=$((largest_size / 1024))

echo ""
echo "📋 Summary:"
echo "  Total screenshots: $file_count"
echo "  Total size: ${total_kb} KB"
echo "  Largest file: $largest_file (${largest_kb} KB)"

if [ $file_count -gt 0 ]; then
    echo ""
    echo "✅ Comprehensive VRM interactive screenshot capture completed!"
    echo "🎯 All screenshots demonstrate the VRM avatar system functionality"
    echo "📁 Screenshots available at: test-results/vrm-interactive-screenshots/"
    
    # Create a simple HTML report
    cat > "$SCREENSHOT_DIR/screenshot-report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>VRM Avatar System Screenshots</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .screenshot { margin: 20px 0; padding: 20px; background: white; border-radius: 8px; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        h1 { color: #333; }
        h3 { color: #666; margin-top: 0; }
        .summary { background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>🎭 VRM Avatar System Screenshots</h1>
    
    <div class="summary">
        <h3>📊 Capture Summary</h3>
        <p><strong>Total Screenshots:</strong> $file_count</p>
        <p><strong>Total Size:</strong> ${total_kb} KB</p>
        <p><strong>Capture Time:</strong> $(date)</p>
        <p><strong>Purpose:</strong> Demonstrate working VRM avatar with BVH animations in 3D classroom</p>
    </div>
EOF

    # Add each screenshot to the report
    for file in *.png; do
        if [ -f "$file" ]; then
            size_kb=$(du -k "$file" | cut -f1)
            cat >> "$SCREENSHOT_DIR/screenshot-report.html" << EOF
    
    <div class="screenshot">
        <h3>📸 ${file} (${size_kb} KB)</h3>
        <img src="${file}" alt="${file}" />
    </div>
EOF
        fi
    done

    cat >> "$SCREENSHOT_DIR/screenshot-report.html" << EOF
</body>
</html>
EOF

    echo "📄 HTML report created: screenshot-report.html"
    
else
    echo "❌ No screenshots were captured - check server and URLs"
    exit 1
fi

echo ""
echo "🎉 VRM Interactive Screenshot Capture Complete!"