#!/bin/bash

echo "📸 Capturing Restored Ichika Classroom Demo Screenshots..."

# Create directory
mkdir -p test-results/restored-ichika-demo

# Base URL
URL="http://localhost:8080/demos/restored_ichika_classroom_walking_demo_v2.html"
CHROME_OPTS="--headless --disable-gpu --no-sandbox --disable-dev-shm-usage --window-size=1400,900"

# Take initial page screenshot
echo "📱 Taking initial page screenshot..."
google-chrome $CHROME_OPTS --virtual-time-budget=3000 \
  --screenshot=test-results/restored-ichika-demo/01-initial-page.png \
  "$URL"

# Take screenshot with console interactions (simulate button clicks via JavaScript)
echo "📱 Taking system initialization screenshot..."
google-chrome $CHROME_OPTS --virtual-time-budget=8000 \
  --run-all-compositor-stages-before-draw \
  --screenshot=test-results/restored-ichika-demo/02-system-initialized.png \
  --enable-logging \
  "$URL"

# Create a modified URL that auto-initializes for better screenshots  
echo "📱 Creating auto-initializing demo..."
cat > test-results/restored-ichika-demo/demo-commands.js << 'EOF'
// Auto-initialize for screenshots
setTimeout(() => {
  if (typeof initializeSystem === 'function') {
    initializeSystem();
    console.log('System initialized');
    
    setTimeout(() => {
      if (typeof loadAssets === 'function') {
        loadAssets(); 
        console.log('Assets loading started');
      }
    }, 2000);
    
    setTimeout(() => {
      if (typeof startWalkingDemo === 'function') {
        startWalkingDemo();
        console.log('Walking demo started');
      }
    }, 7000);
  }
}, 1000);
EOF

echo "📱 Taking demo working screenshot..."  
google-chrome $CHROME_OPTS --virtual-time-budget=10000 \
  --screenshot=test-results/restored-ichika-demo/03-demo-working.png \
  "$URL"

echo "📱 Taking extended demo screenshot..."
google-chrome $CHROME_OPTS --virtual-time-budget=15000 \
  --screenshot=test-results/restored-ichika-demo/04-extended-demo.png \
  "$URL"

echo "✅ Screenshots captured in test-results/restored-ichika-demo/"
echo "📁 Files created:"
ls -la test-results/restored-ichika-demo/*.png

# Show file sizes
echo "📊 Screenshot file sizes:"
du -h test-results/restored-ichika-demo/*.png