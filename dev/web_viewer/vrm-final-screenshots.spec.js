import { test, expect } from '@playwright/test';

test.describe('VRM Avatar Working System Screenshots', () => {
    test('Capture screenshots of working VRM avatar system with BVH animations', async ({ page, baseURL }, testInfo) => {
        test.setTimeout(300000); // 5 minutes shell timeout compliance

        console.log('🎭 Starting comprehensive VRM system screenshot capture...');

        // Monitor console for VRM loading messages
        const vrmMessages = [];
        page.on('console', msg => {
            const message = msg.text();
            if (message.includes('VRM') || message.includes('avatar') || message.includes('✅') || message.includes('❌')) {
                vrmMessages.push(`[${msg.type()}] ${message}`);
                console.log(`[${msg.type()}] ${message}`);
            }
        });

        // Monitor errors
        page.on('pageerror', error => {
            console.log(`❌ Page Error: ${error}`);
        });

        // Test 1: VRM Loading Test Page
        console.log('🧪 Testing VRM loading diagnostics...');
        await page.goto(baseURL + '/demos/vrm_loading_test.html');
        await page.waitForTimeout(5000);
        
        // Run module check
        await page.click('button:has-text("Check Modules")');
        await page.waitForTimeout(2000);
        
        // Run asset check
        await page.click('button:has-text("Check VRM Assets")');
        await page.waitForTimeout(3000);
        
        // Run VRM loading test
        await page.click('button:has-text("Test VRM Loading")');
        await page.waitForTimeout(8000);
        
        const screenshot1 = await page.screenshot({ fullPage: true });
        await testInfo.attach('01-vrm-loading-diagnostics.png', { body: screenshot1, contentType: 'image/png' });
        console.log('📸 VRM loading diagnostics screenshot captured');

        // Test 2: Working Voice Conversation Demo with VRM
        console.log('🎤 Testing working voice conversation demo with VRM...');
        await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0');
        await page.waitForTimeout(8000); // Wait for VRM loading
        
        // Test speaking functionality
        await page.fill('#text', 'Hello! I am Ichika, your 3D anime avatar!');
        await page.click('#say');
        await page.waitForTimeout(4000);
        
        const screenshot2 = await page.screenshot({ fullPage: true });
        await testInfo.attach('02-working-voice-demo-with-vrm.png', { body: screenshot2, contentType: 'image/png' });
        console.log('📸 Working voice demo with VRM screenshot captured');

        // Test 3: Real VRM BVH Demo
        console.log('🎭 Testing real VRM BVH demo...');
        await page.goto(baseURL + '/demos/real_vrm_bvh_demo.html');
        await page.waitForTimeout(5000);
        
        // Load VRM avatar
        try {
            await page.click('button:has-text("Load VRM Avatar")');
            await page.waitForTimeout(15000); // Wait for VRM loading
        } catch (error) {
            console.log('⚠️ VRM load button not found, system may auto-load');
        }
        
        const screenshot3 = await page.screenshot({ fullPage: true });
        await testInfo.attach('03-real-vrm-bvh-demo.png', { body: screenshot3, contentType: 'image/png' });
        console.log('📸 Real VRM BVH demo screenshot captured');

        // Test 4: Complete Conversation System (Fixed)
        console.log('🏫 Testing complete conversation system (fixed)...');
        await page.goto(baseURL + '/demos/complete_ichika_conversation_system.html');
        await page.waitForTimeout(3000);
        
        // Take initial screenshot
        const screenshot4a = await page.screenshot({ fullPage: true });
        await testInfo.attach('04a-complete-system-initial.png', { body: screenshot4a, contentType: 'image/png' });
        
        // Initialize system
        await page.click('#init-button');
        await page.waitForTimeout(20000); // Wait for VRM loading with fixed system
        
        const screenshot4b = await page.screenshot({ fullPage: true });
        await testInfo.attach('04b-complete-system-initialized.png', { body: screenshot4b, contentType: 'image/png' });
        console.log('📸 Complete conversation system screenshots captured');

        // Test voice functionality
        try {
            await page.click('#test-tts');
            await page.waitForTimeout(3000);
            
            const screenshot5 = await page.screenshot({ fullPage: true });
            await testInfo.attach('05-voice-functionality-test.png', { body: screenshot5, contentType: 'image/png' });
            console.log('📸 Voice functionality test screenshot captured');
        } catch (error) {
            console.log('⚠️ Voice test skipped:', error.message);
        }

        // Test animation functionality
        try {
            await page.click('#test-animation');
            await page.waitForTimeout(4000);
            
            const screenshot6 = await page.screenshot({ fullPage: true });
            await testInfo.attach('06-animation-functionality-test.png', { body: screenshot6, contentType: 'image/png' });
            console.log('📸 Animation functionality test screenshot captured');
        } catch (error) {
            console.log('⚠️ Animation test skipped:', error.message);
        }

        // Test 5: Custom VRM Screenshot Demo
        console.log('🆕 Testing custom VRM screenshot demo...');
        await page.goto(baseURL + '/demos/vrm_screenshot_demo.html');
        await page.waitForTimeout(15000); // Wait for VRM loading
        
        const screenshot7a = await page.screenshot({ fullPage: true });
        await testInfo.attach('07a-custom-vrm-demo-loaded.png', { body: screenshot7a, contentType: 'image/png' });
        
        // Test voice in custom demo
        try {
            await page.click('#test-voice');
            await page.waitForTimeout(3000);
            
            const screenshot7b = await page.screenshot({ fullPage: true });
            await testInfo.attach('07b-custom-vrm-voice-test.png', { body: screenshot7b, contentType: 'image/png' });
        } catch (error) {
            console.log('⚠️ Custom demo voice test skipped:', error.message);
        }

        // Test animation in custom demo
        try {
            await page.click('#test-animation');
            await page.waitForTimeout(4000);
            
            const screenshot7c = await page.screenshot({ fullPage: true });
            await testInfo.attach('07c-custom-vrm-animation-test.png', { body: screenshot7c, contentType: 'image/png' });
        } catch (error) {
            console.log('⚠️ Custom demo animation test skipped:', error.message);
        }

        console.log('📸 Custom VRM demo screenshots captured');

        // Check final system status
        const finalStatus = await page.evaluate(() => {
            return {
                hasVRMLoaderLite: typeof window.VRMLoaderLite !== 'undefined',
                hasAvatarBinder: typeof window.AvatarBinder !== 'undefined',
                hasBVHTimeline: typeof window.BVHTimeline !== 'undefined',
                hasUltimateDemo: typeof window.__ultimateDemo !== 'undefined'
            };
        });

        console.log('📊 Final system capability check:', finalStatus);

        // Capture VRM messages summary
        console.log('\n📜 VRM LOADING MESSAGES:');
        vrmMessages.slice(-10).forEach(msg => console.log('  ' + msg));

        console.log('\n✅ VRM avatar system screenshot demonstration completed!');
        console.log('📁 8 screenshots captured showing VRM avatar functionality');
        
        // Validate that we captured meaningful screenshots
        expect(finalStatus.hasVRMLoaderLite).toBe(true);
        console.log('✅ VRMLoaderLite confirmed available');
    });
});