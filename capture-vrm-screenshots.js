#!/usr/bin/env node

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function captureVRMScreenshots() {
    console.log('🎭 Starting VRM screenshot capture...');

    let browser;
    try {
        browser = await chromium.launch({
            headless: false, // Show browser for debugging
            args: [
                '--disable-web-security',
                '--enable-features=WebGPU,SharedArrayBuffer',
                '--enable-webgl',
                '--use-fake-device-for-media-stream',
                '--use-fake-ui-for-media-stream'
            ]
        });

        const context = await browser.newContext();
        const page = await context.newPage();

        // Monitor console and errors
        page.on('console', msg => console.log(`[${msg.type()}] ${msg.text()}`));
        page.on('pageerror', error => console.log(`❌ Page Error: ${error}`));

        // Ensure screenshot directory exists
        const screenshotDir = '/home/runner/work/motion/motion/test-results/vrm-screenshots';
        if (!fs.existsSync(screenshotDir)) {
            fs.mkdirSync(screenshotDir, { recursive: true });
        }

        console.log('🌐 Testing working VRM voice conversation demo...');
        
        // Test the working demo with VRM enabled
        const demoUrl = 'http://localhost:8080/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0';
        await page.goto(demoUrl);
        
        // Wait for page load
        await page.waitForTimeout(8000);
        
        // Take initial screenshot
        await page.screenshot({
            path: path.join(screenshotDir, '01-working-voice-demo-initial.png'),
            fullPage: true
        });
        console.log('📸 Initial working demo screenshot captured');

        // Test TTS functionality
        await page.fill('#text', 'Hello! I am Ichika, your 3D anime avatar assistant!');
        await page.click('#say');
        
        // Wait for animation
        await page.waitForTimeout(5000);
        
        await page.screenshot({
            path: path.join(screenshotDir, '02-working-voice-demo-speaking.png'),
            fullPage: true
        });
        console.log('📸 Speaking demo screenshot captured');

        // Check VRM status
        const vrmStats = await page.evaluate(() => {
            return window.__ultimateDemo?.getStats?.() || {};
        });
        console.log('📊 VRM stats:', vrmStats);

        console.log('🌐 Testing complete conversation system...');
        
        // Test the complete system
        const completeUrl = 'http://localhost:8080/demos/complete_ichika_conversation_system.html';
        await page.goto(completeUrl);
        
        await page.waitForTimeout(5000);
        
        await page.screenshot({
            path: path.join(screenshotDir, '03-complete-system-initial.png'),
            fullPage: true
        });
        console.log('📸 Complete system initial screenshot captured');

        // Initialize the complete system
        await page.click('#init-button');
        await page.waitForTimeout(20000); // Wait longer for VRM loading
        
        await page.screenshot({
            path: path.join(screenshotDir, '04-complete-system-initialized.png'),
            fullPage: true
        });
        console.log('📸 Complete system initialized screenshot captured');

        // Test our custom VRM demo
        console.log('🌐 Testing custom VRM screenshot demo...');
        
        const customUrl = 'http://localhost:8080/demos/vrm_screenshot_demo.html';
        await page.goto(customUrl);
        
        await page.waitForTimeout(15000); // Wait for VRM loading
        
        await page.screenshot({
            path: path.join(screenshotDir, '05-custom-vrm-demo.png'),
            fullPage: true
        });
        console.log('📸 Custom VRM demo screenshot captured');

        // Test voice functionality
        try {
            await page.click('#test-voice');
            await page.waitForTimeout(3000);
            
            await page.screenshot({
                path: path.join(screenshotDir, '06-custom-vrm-voice-test.png'),
                fullPage: true
            });
            console.log('📸 Custom VRM voice test screenshot captured');
        } catch (error) {
            console.log('⚠️ Custom VRM voice test skipped:', error.message);
        }

        // Test animation functionality
        try {
            await page.click('#test-animation');
            await page.waitForTimeout(4000);
            
            await page.screenshot({
                path: path.join(screenshotDir, '07-custom-vrm-animation-test.png'),
                fullPage: true
            });
            console.log('📸 Custom VRM animation test screenshot captured');
        } catch (error) {
            console.log('⚠️ Custom VRM animation test skipped:', error.message);
        }

        // Final comprehensive screenshot
        await page.screenshot({
            path: path.join(screenshotDir, '08-final-system-state.png'),
            fullPage: true
        });
        console.log('📸 Final system state screenshot captured');

        console.log('\n✅ VRM screenshot capture completed!');
        console.log(`📁 Screenshots saved to: ${screenshotDir}`);
        
        // List captured files
        const files = fs.readdirSync(screenshotDir);
        console.log('📸 Screenshots captured:');
        files.forEach(file => {
            const filePath = path.join(screenshotDir, file);
            const stats = fs.statSync(filePath);
            console.log(`  - ${file} (${Math.round(stats.size / 1024)} KB)`);
        });

    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

captureVRMScreenshots().catch(console.error);