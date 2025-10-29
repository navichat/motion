const { test, expect } = require('@playwright/test');

test('Debug current Ichika conversation system state', async ({ page }, testInfo) => {
    test.setTimeout(180000); // 3 minutes for debugging

    console.log('🔍 Starting debug test to understand current system state...');

    // Navigate to the complete demo
    const demoUrl = 'file:///home/runner/work/motion/motion/dev/web_viewer/demos/complete_ichika_conversation_system.html';
    
    await page.goto(demoUrl);
    console.log('✅ Navigated to demo page');

    // Wait for page to load
    await page.waitForTimeout(5000);

    // Take initial screenshot
    const screenshot1 = await page.screenshot({ fullPage: true });
    await testInfo.attach('01-initial-load.png', { body: screenshot1, contentType: 'image/png' });
    
    // Check console messages for errors
    const messages = [];
    page.on('console', msg => {
        messages.push(`${msg.type()}: ${msg.text()}`);
    });

    // Click Initialize System
    try {
        await page.click('#init-button');
        console.log('✅ Clicked Initialize System button');
    } catch (error) {
        console.log('❌ Failed to click Initialize System button:', error);
    }

    // Wait for initialization
    await page.waitForTimeout(10000);

    // Take screenshot after initialization
    const screenshot2 = await page.screenshot({ fullPage: true });
    await testInfo.attach('02-after-initialization.png', { body: screenshot2, contentType: 'image/png' });

    // Check status indicators
    const statuses = await page.evaluate(() => {
        const getStatus = (id) => {
            const element = document.getElementById(id);
            return element ? element.textContent : 'Element not found';
        };

        return {
            scene3d: getStatus('status-3d-text'),
            avatar: getStatus('status-avatar-text'),
            conversation: getStatus('status-conversation-text'),
            speech: getStatus('status-speech-text')
        };
    });

    console.log('📊 Current system statuses:', statuses);

    // Log console messages
    console.log('📜 Console messages (first 10):');
    messages.slice(0, 10).forEach(msg => console.log(msg));

    // Check if VRM files are accessible
    const vrmAccessible = await page.evaluate(async () => {
        try {
            const response = await fetch('../assets/avatars/ichika.vrm');
            return { accessible: response.ok, status: response.status };
        } catch (error) {
            return { accessible: false, error: error.message };
        }
    });

    console.log('🎭 VRM file accessibility:', vrmAccessible);

    // Check Three.js availability
    const threeJsAvailable = await page.evaluate(() => {
        return {
            THREE: typeof THREE !== 'undefined',
            GLTFLoader: typeof THREE?.GLTFLoader !== 'undefined',
            VRMLoaderPlugin: typeof THREE?.VRMLoaderPlugin !== 'undefined',
            windowTHREE: typeof window.THREE !== 'undefined',
            windowLoaders: typeof window.THREELoaders !== 'undefined'
        };
    });

    console.log('🔧 Three.js availability:', threeJsAvailable);
});