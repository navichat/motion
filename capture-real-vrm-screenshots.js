#!/usr/bin/env node

/**
 * Simple VRM System Screenshot Capture
 * Captures working screenshots of the VRM avatar system without complex setup
 */

const puppeteer = require('puppeteer').default;
const fs = require('fs');
const path = require('path');

async function captureVRMSystemScreenshots() {
    console.log('🎭 Starting VRM System Screenshot Capture...');
    
    // Ensure screenshot directory exists
    const screenshotDir = path.join(__dirname, 'test-results', 'real-vrm-system');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }
    
    let browser, page;
    
    try {
        // Launch browser with VRM-friendly settings
        browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--use-angle=swiftshader-webgl',
                '--enable-webgl',
                '--mute-audio'
            ]
        });
        
        page = await browser.newPage();
        
        // Set viewport for consistent screenshots
        await page.setViewport({ width: 1280, height: 720 });
        
        // Enable console logging for debugging
        page.on('console', msg => {
            const type = msg.type();
            if (['error', 'warn', 'info'].includes(type)) {
                console.log(`[${type.toUpperCase()}]`, msg.text());
            }
        });
        
        // Capture 1: VRM Debug Test Interface
        console.log('📸 Capturing VRM Debug Test Interface...');
        await page.goto('http://localhost:8000/vrm_debug_test.html', {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        await page.waitForTimeout(2000);
        
        // Test VRM availability
        await page.evaluate(() => {
            if (typeof window.testVRMLoaderAvailability === 'function') {
                window.testVRMLoaderAvailability();
            }
        });
        
        await page.waitForTimeout(1000);
        await page.screenshot({
            path: path.join(screenshotDir, '01-vrm-debug-interface.png'),
            fullPage: true
        });
        
        // Capture 2: Complete Conversation System
        console.log('📸 Capturing Complete VRM Conversation System...');
        await page.goto('http://localhost:8000/demos/complete_ichika_conversation_system.html', {
            waitUntil: 'networkidle0',
            timeout: 60000
        });
        
        await page.waitForTimeout(3000);
        await page.screenshot({
            path: path.join(screenshotDir, '02-complete-system-initial.png'),
            fullPage: true
        });
        
        // Initialize the system
        const initButton = await page.$('#init-button');
        if (initButton) {
            await initButton.click();
            console.log('✅ Initialize button clicked');
            
            // Wait for initialization
            await page.waitForTimeout(8000);
            
            await page.screenshot({
                path: path.join(screenshotDir, '03-system-initialized.png'),
                fullPage: true
            });
        }
        
        // Check system status
        const systemStatus = await page.evaluate(() => {
            return {
                scene3D: document.querySelector('#status-3d-text')?.textContent,
                avatar: document.querySelector('#status-avatar-text')?.textContent,
                conversation: document.querySelector('#status-conversation-text')?.textContent,
                speechSync: document.querySelector('#status-speech-text')?.textContent
            };
        });
        
        console.log('🎯 System Status:', systemStatus);
        
        // Test voice functionality
        const testVoiceButton = await page.$('#test-tts');
        if (testVoiceButton) {
            await testVoiceButton.click();
            console.log('✅ Test voice clicked');
            await page.waitForTimeout(3000);
            
            await page.screenshot({
                path: path.join(screenshotDir, '04-voice-test-active.png'),
                fullPage: true
            });
        }
        
        // Test animation functionality
        const testAnimButton = await page.$('#test-animation');
        if (testAnimButton) {
            await testAnimButton.click();
            console.log('✅ Test animation clicked');
            await page.waitForTimeout(3000);
            
            await page.screenshot({
                path: path.join(screenshotDir, '05-animation-test-active.png'),
                fullPage: true
            });
        }
        
        // Capture 3: VRM assets availability check
        console.log('📸 Testing VRM asset availability...');
        
        const vrmAssetsStatus = await page.evaluate(async () => {
            const assets = [
                './assets/avatars/ichika.vrm',
                './assets/avatars/buny.vrm',
                './assets/avatars/kaede.vrm'
            ];
            
            const results = {};
            for (const asset of assets) {
                try {
                    const response = await fetch(asset, { method: 'HEAD' });
                    results[asset] = {
                        available: response.ok,
                        status: response.status,
                        size: response.headers.get('content-length')
                    };
                } catch (error) {
                    results[asset] = {
                        available: false,
                        error: error.message
                    };
                }
            }
            return results;
        });
        
        console.log('📦 VRM Assets Status:');
        Object.entries(vrmAssetsStatus).forEach(([asset, status]) => {
            const sizeInfo = status.size ? ` (${(status.size / 1024 / 1024).toFixed(1)}MB)` : '';
            console.log(`  ${status.available ? '✅' : '❌'} ${asset}${sizeInfo}`);
        });
        
        // Final comprehensive screenshot
        await page.screenshot({
            path: path.join(screenshotDir, '06-final-system-state.png'),
            fullPage: true
        });
        
        // Generate HTML report
        const report = `
<!DOCTYPE html>
<html>
<head>
    <title>VRM System Screenshot Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .screenshot { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .screenshot img { max-width: 100%; border: 1px solid #ccc; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .assets-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        .assets-table th, .assets-table td { padding: 8px; border: 1px solid #ddd; text-align: left; }
        .assets-table th { background: #f8f9fa; }
    </style>
</head>
<body>
    <h1>🎭 VRM System Screenshot Report</h1>
    <p>Generated: ${new Date().toISOString()}</p>
    
    <h2>📊 System Status</h2>
    <div class="status ${systemStatus.avatar?.includes('Ready') ? 'success' : 'error'}">
        <strong>3D Scene:</strong> ${systemStatus.scene3D || 'Unknown'}<br>
        <strong>Avatar:</strong> ${systemStatus.avatar || 'Unknown'}<br>
        <strong>Conversation:</strong> ${systemStatus.conversation || 'Unknown'}<br>
        <strong>Speech Sync:</strong> ${systemStatus.speechSync || 'Unknown'}
    </div>
    
    <h2>📦 VRM Assets Availability</h2>
    <table class="assets-table">
        <tr><th>Asset</th><th>Status</th><th>Size</th></tr>
        ${Object.entries(vrmAssetsStatus).map(([asset, status]) => `
            <tr>
                <td>${asset}</td>
                <td>${status.available ? '✅ Available' : '❌ Not Available'}</td>
                <td>${status.size ? (status.size / 1024 / 1024).toFixed(1) + ' MB' : 'Unknown'}</td>
            </tr>
        `).join('')}
    </table>
    
    <h2>📸 Screenshots</h2>
    
    <div class="screenshot">
        <h3>1. VRM Debug Interface</h3>
        <img src="01-vrm-debug-interface.png" alt="VRM Debug Interface">
        <p>Shows VRM component availability and debug testing interface.</p>
    </div>
    
    <div class="screenshot">
        <h3>2. Complete System Initial Load</h3>
        <img src="02-complete-system-initial.png" alt="Complete System Initial">
        <p>Complete Ichika conversation system before initialization.</p>
    </div>
    
    <div class="screenshot">
        <h3>3. System After Initialization</h3>
        <img src="03-system-initialized.png" alt="System Initialized">
        <p>System state after clicking Initialize System button.</p>
    </div>
    
    <div class="screenshot">
        <h3>4. Voice Test Active</h3>
        <img src="04-voice-test-active.png" alt="Voice Test">
        <p>System during voice/TTS testing.</p>
    </div>
    
    <div class="screenshot">
        <h3>5. Animation Test Active</h3>
        <img src="05-animation-test-active.png" alt="Animation Test">
        <p>System during animation testing.</p>
    </div>
    
    <div class="screenshot">
        <h3>6. Final System State</h3>
        <img src="06-final-system-state.png" alt="Final State">
        <p>Complete system state showing all components.</p>
    </div>
</body>
</html>`;
        
        fs.writeFileSync(path.join(screenshotDir, 'report.html'), report);
        console.log('📊 HTML report generated');
        
        console.log('🎉 VRM System Screenshot Capture Complete!');
        console.log(`📁 Screenshots saved to: ${screenshotDir}`);
        
    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
        throw error;
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

// Run if called directly
if (require.main === module) {
    captureVRMSystemScreenshots().catch(error => {
        console.error('Failed to capture screenshots:', error);
        process.exit(1);
    });
}

module.exports = { captureVRMSystemScreenshots };