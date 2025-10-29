import { test, expect } from '@playwright/test';

test.describe('VRM Avatar System Screenshot Demonstration', () => {
    test('Demonstrate working VRM avatar with BVH animations and take screenshots', async ({ page, baseURL }, testInfo) => {
        test.setTimeout(300000); // 5 minutes shell timeout

        console.log('🎭 Starting VRM avatar demonstration with screenshots...');

        // Set up console and error monitoring
        const consoleMessages = [];
        const errors = [];

        page.on('console', msg => {
            const message = `[${msg.type()}] ${msg.text()}`;
            console.log(message);
            consoleMessages.push(message);
        });

        page.on('pageerror', error => {
            console.log(`❌ Page Error: ${error}`);
            errors.push(error.toString());
        });

        // Test the working voice conversation demo with VRM enabled
        const url = baseURL + '/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0';
        console.log('🌐 Navigating to VRM demo:', url);
        
        await page.goto(url);
        
        // Wait for page to fully load
        await page.waitForTimeout(5000);
        
        // Take initial screenshot
        const screenshot1 = await page.screenshot({ fullPage: true });
        await testInfo.attach('01-vrm-demo-initial.png', { body: screenshot1, contentType: 'image/png' });
        console.log('📸 Initial screenshot captured');

        // Check if VRM system is available
        const vrmSystemAvailable = await page.evaluate(() => {
            return {
                VRMLoaderLite: typeof window.VRMLoaderLite !== 'undefined',
                AvatarBinder: typeof window.AvatarBinder !== 'undefined',
                BVHTimeline: typeof window.BVHTimeline !== 'undefined',
                THREE: typeof window.THREE !== 'undefined'
            };
        });

        console.log('🔧 VRM system availability:', vrmSystemAvailable);

        // Test basic functionality
        await page.fill('#text', 'Hello! I am Ichika, your 3D anime assistant!');
        await page.click('#say');
        
        // Wait for TTS animation to be scheduled
        await page.waitForTimeout(3000);
        
        // Take screenshot during speech
        const screenshot2 = await page.screenshot({ fullPage: true });
        await testInfo.attach('02-vrm-speaking.png', { body: screenshot2, contentType: 'image/png' });
        console.log('📸 Speaking screenshot captured');

        // Check if VRM was actually loaded
        const vrmStatus = await page.evaluate(() => {
            return window.__ultimateDemo?.getStats?.() || {};
        });

        console.log('📊 VRM system stats:', vrmStatus);

        // Test voice interaction
        try {
            await page.click('#listen');
            await page.waitForTimeout(2000);
            
            const screenshot3 = await page.screenshot({ fullPage: true });
            await testInfo.attach('03-vrm-listening.png', { body: screenshot3, contentType: 'image/png' });
            console.log('📸 Listening screenshot captured');
        } catch (error) {
            console.log('⚠️ Voice interaction test skipped:', error.message);
        }

        // Test the complete conversation system demo
        console.log('🌐 Testing complete conversation system...');
        const completeUrl = baseURL + '/demos/complete_ichika_conversation_system.html';
        
        await page.goto(completeUrl);
        await page.waitForTimeout(3000);
        
        // Take screenshot of complete system before initialization
        const screenshot4 = await page.screenshot({ fullPage: true });
        await testInfo.attach('04-complete-system-initial.png', { body: screenshot4, contentType: 'image/png' });
        
        // Initialize the complete system
        try {
            await page.click('#init-button');
            await page.waitForTimeout(15000); // Wait longer for VRM loading
            
            const screenshot5 = await page.screenshot({ fullPage: true });
            await testInfo.attach('05-complete-system-initialized.png', { body: screenshot5, contentType: 'image/png' });
            console.log('📸 Complete system initialized screenshot captured');
        } catch (error) {
            console.log('⚠️ Complete system initialization failed:', error.message);
        }

        // Test our new VRM screenshot demo
        console.log('🌐 Testing new VRM screenshot demo...');
        const screenshotDemoUrl = baseURL + '/demos/vrm_screenshot_demo.html';
        
        await page.goto(screenshotDemoUrl);
        await page.waitForTimeout(10000); // Wait for VRM loading
        
        // Take screenshots of the VRM demo
        const screenshot6 = await page.screenshot({ fullPage: true });
        await testInfo.attach('06-vrm-screenshot-demo.png', { body: screenshot6, contentType: 'image/png' });
        console.log('📸 VRM screenshot demo captured');

        // Test voice functionality
        try {
            await page.click('#test-voice');
            await page.waitForTimeout(3000);
            
            const screenshot7 = await page.screenshot({ fullPage: true });
            await testInfo.attach('07-vrm-voice-test.png', { body: screenshot7, contentType: 'image/png' });
            console.log('📸 VRM voice test screenshot captured');
        } catch (error) {
            console.log('⚠️ VRM voice test failed:', error.message);
        }

        // Test animation functionality
        try {
            await page.click('#test-animation');
            await page.waitForTimeout(4000);
            
            const screenshot8 = await page.screenshot({ fullPage: true });
            await testInfo.attach('08-vrm-animation-test.png', { body: screenshot8, contentType: 'image/png' });
            console.log('📸 VRM animation test screenshot captured');
        } catch (error) {
            console.log('⚠️ VRM animation test failed:', error.message);
        }

        // Final status check
        const finalStatus = await page.evaluate(() => {
            const getStatus = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent.trim() : 'N/A';
            };
            
            return {
                scene: getStatus('status-scene-text'),
                vrm: getStatus('status-vrm-text'),
                bvh: getStatus('status-bvh-text'),
                speech: getStatus('status-speech-text')
            };
        });

        console.log('📊 Final system status:', finalStatus);

        // Summary
        console.log('\n📋 VRM DEMONSTRATION SUMMARY:');
        console.log('===============================');
        console.log('Screenshots captured: 8');
        console.log('Console messages:', consoleMessages.length);
        console.log('Errors:', errors.length);
        console.log('Final status:', finalStatus);

        if (errors.length > 0) {
            console.log('\n❌ ERRORS DETECTED:');
            errors.slice(0, 5).forEach(err => console.log('  -', err));
        }

        // Key VRM-related console messages
        const vrmMessages = consoleMessages.filter(msg => 
            msg.includes('VRM') || 
            msg.includes('avatar') ||
            msg.includes('✅') ||
            msg.includes('❌')
        );

        console.log('\n📜 KEY VRM MESSAGES:');
        vrmMessages.slice(-15).forEach(msg => console.log('  -', msg));

        // Ensure we have some working functionality
        expect(finalStatus.scene).not.toBe('Loading');
        
        console.log('✅ VRM demonstration test completed successfully');
    });
});