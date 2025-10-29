#!/bin/bash

echo "🎭 Enhanced VRM System Screenshot Capture with Initialization..."

# Create screenshot directory
mkdir -p test-results/working-vrm-system

# Function to capture screenshot with JavaScript execution
capture_initialized_screenshot() {
    local url="$1"
    local output="$2"
    local description="$3"
    local wait_time="${4:-8000}"
    
    echo ""
    echo "📸 Capturing: $description"
    echo "🌐 URL: $url"
    echo "⏱️  Wait time: ${wait_time}ms"
    echo "📁 Output: $output"
    
    # Create a temporary HTML file that will initialize the system
    local temp_html="/tmp/vrm_auto_init.html"
    cat > "$temp_html" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script>
        window.onload = function() {
            console.log('🔄 Auto-initializing system...');
            
            // Wait for page to load
            setTimeout(function() {
                // Try to find and click init button
                const initButton = document.getElementById('init-button');
                if (initButton && initButton.style.display !== 'none') {
                    console.log('🚀 Clicking init button...');
                    initButton.click();
                } else {
                    console.log('⚠️ Init button not found or hidden');
                }
                
                // Monitor system status
                let attempts = 0;
                const checkStatus = setInterval(function() {
                    attempts++;
                    const sceneStatus = document.getElementById('status-3d-text');
                    const avatarStatus = document.getElementById('status-avatar-text');
                    
                    if (sceneStatus && avatarStatus) {
                        console.log('📊 Status check ' + attempts + ':');
                        console.log('   🌐 Scene: ' + sceneStatus.textContent);
                        console.log('   👤 Avatar: ' + avatarStatus.textContent);
                    }
                    
                    if (attempts > 10) {
                        clearInterval(checkStatus);
                        console.log('✅ Monitoring complete');
                    }
                }, 1000);
                
            }, 2000);
        };
    </script>
</head>
<body>
    <script>
        // Redirect to the actual demo page
        window.location.href = '$url';
    </script>
    <p>Redirecting to VRM system...</p>
</body>
</html>
EOF
    
    # Use Chrome headless to capture screenshot with longer wait time
    google-chrome \
        --headless=new \
        --disable-gpu \
        --no-sandbox \
        --disable-web-security \
        --enable-features=WebGPU,SharedArrayBuffer \
        --enable-webgl \
        --window-size=1920,1080 \
        --screenshot="$output" \
        --virtual-time-budget="$wait_time" \
        --run-all-compositor-stages-before-draw \
        "$url" 2>/dev/null
    
    # Clean up temp file
    rm -f "$temp_html"
    
    if [ -f "$output" ]; then
        local size=$(du -h "$output" | cut -f1)
        echo "✅ Screenshot captured: $size"
        return 0
    else
        echo "❌ Screenshot failed"
        return 1
    fi
}

echo ""
echo "🏫 Capturing Complete Ichika Conversation System (Extended Wait)..."
capture_initialized_screenshot \
    "http://localhost:8080/demos/complete_ichika_conversation_system.html" \
    "test-results/working-vrm-system/01-complete-system-extended.png" \
    "Complete System - Extended Wait for Initialization" \
    15000

echo ""
echo "🎭 Capturing Real VRM BVH Demo (Extended Wait)..."
capture_initialized_screenshot \
    "http://localhost:8080/demos/real_vrm_bvh_demo.html" \
    "test-results/working-vrm-system/02-vrm-bvh-extended.png" \
    "VRM BVH Demo - Extended Wait" \
    12000

echo ""
echo "🎤 Capturing Voice Conversation Demo (Extended Wait)..."
capture_initialized_screenshot \
    "http://localhost:8080/demos/ichika_voice_conversation_demo.html" \
    "test-results/working-vrm-system/03-voice-demo-extended.png" \
    "Voice Conversation Demo - Extended Wait" \
    10000

echo ""
echo "🎼 Capturing VRM Orchestrator Demo (Extended Wait)..."
capture_initialized_screenshot \
    "http://localhost:8080/demos/ichika_vrm_orchestrator_demo.html" \
    "test-results/working-vrm-system/04-vrm-orchestrator-extended.png" \
    "VRM Orchestrator Demo - Extended Wait" \
    10000

echo ""
echo "🏫 Capturing Enhanced Classroom Demo..."
capture_initialized_screenshot \
    "http://localhost:8080/demos/ichika_enhanced_classroom_demo.html" \
    "test-results/working-vrm-system/05-enhanced-classroom.png" \
    "Enhanced Classroom Demo" \
    12000

echo ""
echo "📊 Working VRM System Screenshot Summary"
echo "========================================"

# List all captured screenshots with sizes
cd test-results/working-vrm-system 2>/dev/null || mkdir -p test-results/working-vrm-system && cd test-results/working-vrm-system
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
    echo "✅ Enhanced VRM screenshot capture completed successfully!"
    echo "📁 Screenshots saved in: test-results/working-vrm-system/"
    echo ""
    echo "🎯 These screenshots show the VRM system with extended initialization time"
    echo "   to capture the system in a more complete state."
else
    echo "❌ No screenshots were captured"
fi