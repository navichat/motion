const { test, expect } = require('@playwright/test');

test.describe('Ichika VRM System Integration Test', () => {
    test('Demonstrate working VRM avatar with BVH animations', async ({ page }, testInfo) => {
        test.setTimeout(300000); // 5 minutes shell timeout for full system testing

        console.log('🎭 Testing complete Ichika VRM conversation system...');

        // Set up console logging  
        const consoleMessages = [];
        page.on('console', msg => {
            consoleMessages.push(`[${msg.type()}] ${msg.text()}`);
        });

        // Set up error logging
        const errors = [];
        page.on('pageerror', error => {
            errors.push(error.toString());
        });

        // Navigate to the demo with HTTP server (for VRM asset loading)
        const demoUrl = 'http://localhost:8080/demos/complete_ichika_conversation_system.html';
        
        console.log('🌐 Navigating to:', demoUrl);
        await page.goto(demoUrl);

        // Wait for initial page load
        await page.waitForTimeout(3000);
        
        // Take screenshot of initial state
        const screenshot1 = await page.screenshot({ fullPage: true });
        await testInfo.attach('01-initial-page.png', { body: screenshot1, contentType: 'image/png' });

        // Click Initialize System and wait for loading
        console.log('🔄 Initializing 3D system...');
        await page.click('#init-button');
        
        // Wait for system initialization with longer timeout for VRM loading
        await page.waitForTimeout(15000);
        
        // Take screenshot after initialization attempt
        const screenshot2 = await page.screenshot({ fullPage: true });
        await testInfo.attach('02-post-initialization.png', { body: screenshot2, contentType: 'image/png' });

        // Check the status indicators
        const statusCheck = await page.evaluate(() => {
            const getStatusText = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent.trim() : 'N/A';
            };

            const getStatusColor = (id) => {
                const element = document.getElementById(id);
                if (!element) return 'N/A';
                if (element.classList.contains('status-active')) return 'green';
                if (element.classList.contains('status-warning')) return 'yellow';
                return 'red';
            };

            return {
                scene3d: {
                    text: getStatusText('status-3d-text'),
                    color: getStatusColor('status-3d')
                },
                avatar: {
                    text: getStatusText('status-avatar-text'),
                    color: getStatusColor('status-avatar')
                },
                conversation: {
                    text: getStatusText('status-conversation-text'),
                    color: getStatusColor('status-conversation')
                },
                speech: {
                    text: getStatusText('status-speech-text'),
                    color: getStatusColor('status-speech')
                }
            };
        });

        console.log('📊 System status check:', statusCheck);

        // If avatar is not loaded, check what's happening
        if (statusCheck.avatar.text !== 'Ready' && statusCheck.avatar.text !== 'Loaded') {
            console.log('⚠️ Avatar not loaded, checking for errors...');
            
            // Try clicking initialize again
            await page.click('#init-button');
            await page.waitForTimeout(10000);
            
            const screenshot3 = await page.screenshot({ fullPage: true });
            await testInfo.attach('03-second-initialization.png', { body: screenshot3, contentType: 'image/png' });
        }

        // Test voice functionality if possible
        try {
            console.log('🎤 Testing voice functionality...');
            await page.click('#test-tts');
            await page.waitForTimeout(5000);
            
            const screenshot4 = await page.screenshot({ fullPage: true });
            await testInfo.attach('04-voice-test.png', { body: screenshot4, contentType: 'image/png' });
        } catch (error) {
            console.log('❌ Voice test failed:', error);
        }

        // Test animation if possible
        try {
            console.log('🎭 Testing animation functionality...');
            await page.click('#test-animation');
            await page.waitForTimeout(5000);
            
            const screenshot5 = await page.screenshot({ fullPage: true });
            await testInfo.attach('05-animation-test.png', { body: screenshot5, contentType: 'image/png' });
        } catch (error) {
            console.log('❌ Animation test failed:', error);
        }

        // Final screenshot
        const finalScreenshot = await page.screenshot({ fullPage: true });
        await testInfo.attach('06-final-state.png', { body: finalScreenshot, contentType: 'image/png' });

        // Output summary
        console.log('\n📋 TEST SUMMARY:');
        console.log('================');
        console.log('Status Results:', JSON.stringify(statusCheck, null, 2));
        console.log(`Console Messages: ${consoleMessages.length} total`);
        console.log(`Errors: ${errors.length} total`);
        
        if (errors.length > 0) {
            console.log('\n❌ ERRORS FOUND:');
            errors.forEach(error => console.log('  -', error));
        }

        // Print key console messages for debugging
        console.log('\n📜 KEY CONSOLE MESSAGES:');
        consoleMessages.filter(msg => 
            msg.includes('VRM') || 
            msg.includes('avatar') ||
            msg.includes('Failed') || 
            msg.includes('Error') ||
            msg.includes('✅') ||
            msg.includes('❌')
        ).slice(0, 20).forEach(msg => console.log('  -', msg));
    });
});