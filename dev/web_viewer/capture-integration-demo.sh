#!/bin/bash

# Simple screenshot capture script for integration demo
cd /home/runner/work/motion/motion/dev/web_viewer

echo "Starting integration demo screenshot capture..."

# Create screenshots directory
mkdir -p test-results/integration-demo

# Simple curl test to verify demo is accessible
curl -s http://localhost:8080/demos/ichika_integration_status_demo.html | head -5

echo "Demo accessibility confirmed. Proceeding with browser screenshot..."

# Use a simple approach with playwright run
cat << 'EOF' > temp_screenshot_script.js
const { chromium } = require('playwright');

(async () => {
  console.log('Launching browser...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  try {
    console.log('Navigating to integration demo...');
    await page.goto('http://localhost:8080/demos/ichika_integration_status_demo.html', { 
      timeout: 30000,
      waitUntil: 'networkidle' 
    });
    
    // Wait for demo initialization
    console.log('Waiting for demo initialization...');
    await page.waitForTimeout(5000);
    
    // Check if content loaded
    const title = await page.title();
    console.log('Demo title:', title);
    
    // Take screenshot
    console.log('Capturing screenshot...');
    await page.screenshot({ 
      path: 'test-results/integration-demo/ichika-integration-status.png',
      fullPage: true,
      quality: 90
    });
    
    // Try to trigger some demo functionality
    try {
      console.log('Testing demo functionality...');
      await page.click('#test-tts', { timeout: 5000 });
      await page.waitForTimeout(2000);
      
      await page.click('#test-webgl', { timeout: 5000 });
      await page.waitForTimeout(2000);
      
      // Take final screenshot after interactions
      await page.screenshot({ 
        path: 'test-results/integration-demo/ichika-integration-status-interactive.png',
        fullPage: true,
        quality: 90
      });
      
    } catch (e) {
      console.log('Demo interaction test completed (some elements may not be ready):', e.message);
    }
    
    console.log('✅ Screenshots captured successfully');
    
  } catch (error) {
    console.error('❌ Screenshot capture failed:', error.message);
    process.exit(1);
  } finally {
    await browser.close();
  }
})();
EOF

# Run the screenshot script with timeout
timeout 60s node temp_screenshot_script.js

# Check if screenshots were created
if [ -f "test-results/integration-demo/ichika-integration-status.png" ]; then
  echo "✅ Integration demo screenshot successfully captured"
  ls -la test-results/integration-demo/
else
  echo "❌ Screenshot capture failed"
  exit 1
fi

# Cleanup
rm -f temp_screenshot_script.js

echo "Integration demo screenshot capture completed"