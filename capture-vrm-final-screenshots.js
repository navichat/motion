#!/usr/bin/env node

/**
 * Simple VRM Screenshot Capture using Playwright
 * Downloads and installs Playwright browser on-demand
 */

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

async function captureVRMScreenshotsWithPlaywright() {
    console.log('🎭 Starting VRM Screenshot Capture with Playwright...');
    
    const screenshotDir = path.join(__dirname, 'test-results', 'final-vrm-system');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }
    
    try {
        // Install Playwright if not available
        console.log('📦 Installing Playwright browser...');
        
        try {
            await execAsync('npx playwright install chromium --force', {
                cwd: __dirname,
                timeout: 300000 // 5 minutes
            });
            console.log('✅ Playwright browser installed');
        } catch (error) {
            console.warn('⚠️ Playwright install warning:', error.message);
            // Continue anyway - might already be installed
        }
        
        // Create the actual Playwright test
        const playwrightTest = `
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function runVRMScreenshots() {
    console.log('🚀 Launching browser for VRM screenshot capture...');
    
    const browser = await chromium.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--use-angle=swiftshader-webgl',
            '--enable-webgl',
            '--mute-audio'
        ]
    });
    
    const context = await browser.newContext({
        viewport: { width: 1280, height: 720 }
    });
    
    const page = await context.newPage();
    
    // Enable console logging
    page.on('console', msg => {
        const type = msg.type();
        if (['error', 'warn', 'log'].includes(type)) {
            console.log(\`[BROWSER \${type.toUpperCase()}]\`, msg.text());
        }
    });
    
    try {
        // Screenshot 1: Real VRM System Test
        console.log('📸 Capturing Real VRM System Test...');
        await page.goto('http://localhost:8000/real_vrm_system_test.html', {
            waitUntil: 'networkidle',
            timeout: 30000
        });
        
        await page.waitForTimeout(3000);
        
        await page.screenshot({
            path: '${screenshotDir}/01-real-vrm-system-test.png',
            fullPage: true
        });
        
        // Test VRM components
        await page.evaluate(() => {
            if (typeof window.testVRMComponents === 'function') {
                window.testVRMComponents();
            }
        });
        
        await page.waitForTimeout(2000);
        
        await page.screenshot({
            path: '${screenshotDir}/02-vrm-components-tested.png',
            fullPage: true
        });
        
        // Load real VRM
        console.log('🎭 Testing VRM loading...');
        await page.evaluate(() => {
            if (typeof window.loadRealVRM === 'function') {
                window.loadRealVRM();
            }
        });
        
        await page.waitForTimeout(10000); // Wait for VRM to load
        
        await page.screenshot({
            path: '${screenshotDir}/03-vrm-loading-attempt.png',
            fullPage: true
        });
        
        // Screenshot 2: Complete Conversation System
        console.log('📸 Capturing Complete Conversation System...');
        await page.goto('http://localhost:8000/demos/complete_ichika_conversation_system.html', {
            waitUntil: 'networkidle',
            timeout: 30000
        });
        
        await page.waitForTimeout(3000);
        
        await page.screenshot({
            path: '${screenshotDir}/04-complete-system-initial.png',
            fullPage: true
        });
        
        // Initialize the system
        const initButton = await page.$('#init-button');
        if (initButton) {
            console.log('🔄 Initializing conversation system...');
            await initButton.click();
            
            // Wait for initialization
            await page.waitForTimeout(12000);
            
            await page.screenshot({
                path: '${screenshotDir}/05-system-initialized.png',
                fullPage: true
            });
            
            // Check system status
            const status = await page.evaluate(() => {
                return {
                    scene3D: document.querySelector('#status-3d-text')?.textContent,
                    avatar: document.querySelector('#status-avatar-text')?.textContent,
                    conversation: document.querySelector('#status-conversation-text')?.textContent,
                    speechSync: document.querySelector('#status-speech-text')?.textContent
                };
            });
            
            console.log('🎯 System Status:', JSON.stringify(status, null, 2));
        }
        
        // Screenshot 3: VRM Debug Test
        console.log('📸 Capturing VRM Debug Test...');
        await page.goto('http://localhost:8000/vrm_debug_test.html', {
            waitUntil: 'networkidle',
            timeout: 30000
        });
        
        await page.waitForTimeout(2000);
        
        // Test VRM loader availability
        const testButtons = [
            'button:has-text("Test VRMLoader Availability")',
            'button:has-text("Test Direct VRM Load")',
            'button:has-text("Test 3D Scene")'
        ];
        
        for (const buttonSelector of testButtons) {
            try {
                const button = await page.$(buttonSelector);
                if (button) {
                    await button.click();
                    await page.waitForTimeout(2000);
                }
            } catch (error) {
                console.warn(\`Could not click button \${buttonSelector}:, error.message\`);
            }
        }
        
        await page.screenshot({
            path: '${screenshotDir}/06-vrm-debug-complete.png',
            fullPage: true
        });
        
        console.log('✅ All screenshots captured successfully!');
        
    } catch (error) {
        console.error('❌ Screenshot capture error:', error);
        await page.screenshot({
            path: '${screenshotDir}/error-screenshot.png',
            fullPage: true
        });
    } finally {
        await browser.close();
    }
}

runVRMScreenshots().catch(console.error);
`;
        
        // Write and execute the Playwright test
        const testPath = path.join(__dirname, 'vrm-screenshot-test.js');
        fs.writeFileSync(testPath, playwrightTest);
        
        console.log('🎬 Executing VRM screenshot capture...');
        
        // Run the Playwright test
        await execAsync(`node vrm-screenshot-test.js`, {
            cwd: __dirname,
            timeout: 120000, // 2 minutes
            maxBuffer: 1024 * 1024 * 10 // 10MB buffer for output
        });
        
        // Check if screenshots were created
        const screenshots = fs.readdirSync(screenshotDir).filter(file => file.endsWith('.png'));
        
        console.log('📊 SCREENSHOT CAPTURE SUMMARY:');
        console.log(`   Screenshots Created: ${screenshots.length}`);
        
        screenshots.forEach(screenshot => {
            const filePath = path.join(screenshotDir, screenshot);
            const stats = fs.statSync(filePath);
            const sizeKB = (stats.size / 1024).toFixed(1);
            console.log(`   ✅ ${screenshot} (${sizeKB}KB)`);
        });
        
        if (screenshots.length > 0) {
            console.log('🎉 VRM Screenshot Capture Complete!');
            console.log(`📁 Screenshots saved to: ${screenshotDir}`);
            return true;
        } else {
            console.log('❌ No screenshots were created');
            return false;
        }
        
    } catch (error) {
        console.error('❌ Playwright screenshot capture failed:', error);
        return false;
    }
}

// Run if called directly
if (require.main === module) {
    captureVRMScreenshotsWithPlaywright().then(success => {
        process.exit(success ? 0 : 1);
    }).catch(error => {
        console.error('Failed to capture screenshots:', error);
        process.exit(1);
    });
}

module.exports = { captureVRMScreenshotsWithPlaywright };