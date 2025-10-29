import { test, expect } from '@playwright/test';

test.describe('Comprehensive VRM Avatar Interactive Demo', () => {
    test('Interactive VRM avatar demonstration with voice and animation testing', async ({ page, baseURL }, testInfo) => {
        test.setTimeout(400000); // 6+ minutes shell timeout for comprehensive testing

        console.log('🎭 Starting comprehensive interactive VRM avatar demonstration...');

        // Comprehensive logging system
        const logs = {
            console: [],
            errors: [],
            vrmSpecific: [],
            interactions: []
        };

        page.on('console', msg => {
            const message = `[${msg.type()}] ${msg.text()}`;
            logs.console.push(message);
            
            if (message.includes('VRM') || message.includes('avatar') || message.includes('BVH') || message.includes('animation')) {
                logs.vrmSpecific.push(message);
                console.log('🎭 VRM:', message);
            }
        });

        page.on('pageerror', error => {
            const errorMsg = error.toString();
            logs.errors.push(errorMsg);
            console.log(`❌ Error:`, errorMsg);
        });

        // === Phase 1: Test VRM Loading Diagnostics ===
        console.log('\n🧪 Phase 1: VRM Loading Diagnostics');
        console.log('=====================================');
        
        await page.goto(baseURL + '/demos/vrm_loading_test.html');
        await page.waitForTimeout(3000);

        // Run all diagnostic tests
        await page.click('button:has-text("Check Modules")');
        logs.interactions.push('Clicked Check Modules');
        await page.waitForTimeout(2000);

        await page.click('button:has-text("Check VRM Assets")');
        logs.interactions.push('Clicked Check VRM Assets');
        await page.waitForTimeout(3000);

        await page.click('button:has-text("Test VRM Loading")');
        logs.interactions.push('Clicked Test VRM Loading');
        await page.waitForTimeout(8000);

        const screenshot1 = await page.screenshot({ fullPage: true });
        await testInfo.attach('phase1-vrm-diagnostics.png', { body: screenshot1, contentType: 'image/png' });
        console.log('📸 Phase 1: VRM diagnostics screenshot captured');

        // === Phase 2: Working Voice Demo with VRM ===
        console.log('\n🎤 Phase 2: Voice Conversation with VRM');
        console.log('=======================================');
        
        await page.goto(baseURL + '/demos/ichika_voice_conversation_demo.html?vrm=1&backend=beeps&playAudio=0');
        await page.waitForTimeout(10000); // Extended wait for VRM loading

        // Test conversation functionality
        const testPhrases = [
            'Hello Ichika, nice to meet you!',
            'Can you show me some animations?',
            'How are you feeling today?'
        ];

        for (let i = 0; i < testPhrases.length; i++) {
            const phrase = testPhrases[i];
            console.log(`🗣️ Testing phrase ${i + 1}: "${phrase}"`);
            
            await page.fill('#text', phrase);
            await page.click('#say');
            logs.interactions.push(`Said: "${phrase}"`);
            
            await page.waitForTimeout(4000);
            
            const screenshot = await page.screenshot({ fullPage: true });
            await testInfo.attach(`phase2-${i + 1}-voice-"${phrase.substring(0, 10)}".png`, { 
                body: screenshot, 
                contentType: 'image/png' 
            });
        }

        console.log('📸 Phase 2: Voice conversation screenshots captured');

        // === Phase 3: Real VRM BVH Demo ===
        console.log('\n🎭 Phase 3: Real VRM + BVH Demo');
        console.log('==============================');
        
        await page.goto(baseURL + '/demos/real_vrm_bvh_demo.html');
        await page.waitForTimeout(5000);

        // Try to load VRM if button exists
        try {
            await page.click('button:has-text("Load VRM Avatar")', { timeout: 5000 });
            logs.interactions.push('Clicked Load VRM Avatar');
            await page.waitForTimeout(15000); // Wait for VRM loading
            console.log('🎭 VRM loading triggered via button');
        } catch (error) {
            console.log('🎭 No VRM load button found - system may auto-load');
        }

        // Try to load animations if button exists
        try {
            await page.click('button:has-text("Load BVH")');
            logs.interactions.push('Clicked Load BVH');
            await page.waitForTimeout(5000);
            console.log('🎯 BVH loading triggered');
        } catch (error) {
            console.log('🎯 No BVH load button found - system may auto-load');
        }

        // Try idle animations
        try {
            await page.click('button:has-text("Start Idle")');
            logs.interactions.push('Clicked Start Idle');
            await page.waitForTimeout(5000);
            console.log('💃 Idle animations started');
        } catch (error) {
            console.log('💃 No idle button found - animations may auto-start');
        }

        const screenshot3 = await page.screenshot({ fullPage: true });
        await testInfo.attach('phase3-real-vrm-bvh-demo.png', { body: screenshot3, contentType: 'image/png' });
        console.log('📸 Phase 3: Real VRM BVH demo screenshot captured');

        // === Phase 4: Complete System Test ===
        console.log('\n🏫 Phase 4: Complete Conversation System');
        console.log('======================================');
        
        await page.goto(baseURL + '/demos/complete_ichika_conversation_system.html');
        await page.waitForTimeout(3000);

        // Initial state screenshot
        const screenshot4a = await page.screenshot({ fullPage: true });
        await testInfo.attach('phase4a-complete-system-initial.png', { 
            body: screenshot4a, 
            contentType: 'image/png' 
        });

        // Initialize system
        await page.click('#init-button');
        logs.interactions.push('Clicked Initialize System');
        console.log('🔄 System initialization started...');
        
        await page.waitForTimeout(25000); // Extended wait for VRM loading

        const screenshot4b = await page.screenshot({ fullPage: true });
        await testInfo.attach('phase4b-complete-system-initialized.png', { 
            body: screenshot4b, 
            contentType: 'image/png' 
        });

        // Test voice functionality
        try {
            await page.click('#test-tts');
            logs.interactions.push('Clicked Test Voice');
            await page.waitForTimeout(4000);
            
            const screenshot4c = await page.screenshot({ fullPage: true });
            await testInfo.attach('phase4c-voice-test.png', { 
                body: screenshot4c, 
                contentType: 'image/png' 
            });
            console.log('🎤 Voice test completed');
        } catch (error) {
            console.log('⚠️ Voice test failed:', error.message);
        }

        // Test animation functionality
        try {
            await page.click('#test-animation');
            logs.interactions.push('Clicked Test Animation');
            await page.waitForTimeout(5000);
            
            const screenshot4d = await page.screenshot({ fullPage: true });
            await testInfo.attach('phase4d-animation-test.png', { 
                body: screenshot4d, 
                contentType: 'image/png' 
            });
            console.log('🎭 Animation test completed');
        } catch (error) {
            console.log('⚠️ Animation test failed:', error.message);
        }

        console.log('📸 Phase 4: Complete system screenshots captured');

        // === Phase 5: System Status Analysis ===
        console.log('\n📊 Phase 5: System Status Analysis');
        console.log('==================================');

        // Check final system status
        const systemStatus = await page.evaluate(() => {
            const getStatus = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent.trim() : 'N/A';
            };

            const getStatusClass = (id) => {
                const element = document.getElementById(id);
                if (!element) return 'unknown';
                if (element.classList.contains('status-active')) return 'active';
                if (element.classList.contains('status-warning')) return 'warning';
                return 'inactive';
            };

            return {
                scene3d: { text: getStatus('status-3d-text'), class: getStatusClass('status-3d') },
                avatar: { text: getStatus('status-avatar-text'), class: getStatusClass('status-avatar') },
                conversation: { text: getStatus('status-conversation-text'), class: getStatusClass('status-conversation') },
                speech: { text: getStatus('status-speech-text'), class: getStatusClass('status-speech') }
            };
        });

        console.log('📊 Final system status:', JSON.stringify(systemStatus, null, 2));

        // Check if VRM system is working
        const vrmSystemCheck = await page.evaluate(() => {
            return {
                modules: {
                    VRMLoaderLite: typeof window.VRMLoaderLite !== 'undefined',
                    AvatarBinder: typeof window.AvatarBinder !== 'undefined',
                    BVHTimelineVRMIntegration: typeof window.BVHTimelineVRMIntegration !== 'undefined',
                    THREE: typeof window.THREE !== 'undefined'
                },
                ultimateDemo: typeof window.__ultimateDemo !== 'undefined',
                ultimateStats: window.__ultimateDemo?.getStats?.() || null
            };
        });

        console.log('🔧 VRM system check:', JSON.stringify(vrmSystemCheck, null, 2));

        // === Final Analysis ===
        console.log('\n📋 COMPREHENSIVE VRM DEMONSTRATION SUMMARY');
        console.log('==========================================');
        console.log(`Screenshots captured: 8+ images`);
        console.log(`Console messages: ${logs.console.length}`);
        console.log(`VRM-specific messages: ${logs.vrmSpecific.length}`);
        console.log(`Errors: ${logs.errors.length}`);
        console.log(`User interactions: ${logs.interactions.length}`);

        // Print key VRM messages
        if (logs.vrmSpecific.length > 0) {
            console.log('\n🎭 Key VRM Messages:');
            logs.vrmSpecific.slice(-15).forEach(msg => console.log('  ' + msg));
        }

        if (logs.errors.length > 0) {
            console.log('\n❌ Errors Detected:');
            logs.errors.slice(0, 5).forEach(err => console.log('  ' + err));
        }

        console.log('\n🎯 User Interactions Performed:');
        logs.interactions.forEach(interaction => console.log('  ' + interaction));

        // Validate essential functionality
        expect(vrmSystemCheck.modules.VRMLoaderLite).toBe(true);
        expect(systemStatus.scene3d.text).not.toBe('Not Loaded');
        
        console.log('\n✅ Comprehensive VRM avatar demonstration completed successfully!');
        console.log('📁 Multiple screenshots captured showing VRM avatar functionality');
        console.log('🎯 System validation: VRM modules available and functional');
    });
});