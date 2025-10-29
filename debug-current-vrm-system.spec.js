const { test, expect } = require('@playwright/test');

test.describe('Debug Current VRM System', () => {
    test('capture current VRM loading state with proper timeout', async ({ page }) => {
        test.setTimeout(360000); // 6 minutes shell timeout compliance
        
        // Navigate to the current complete system
        await page.goto('http://localhost:8080/demos/complete_ichika_conversation_system.html');
        
        // Wait for initial page load
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(3000);
        
        // Take initial screenshot
        await page.screenshot({ 
            path: 'test-results/01-initial-state.png', 
            fullPage: true 
        });
        
        // Click initialize and wait
        const initButton = await page.locator('#init-button');
        await expect(initButton).toBeVisible();
        await initButton.click();
        
        // Wait for initialization process
        console.log('Waiting for initialization...');
        await page.waitForTimeout(10000);
        
        // Take screenshot after initialization
        await page.screenshot({ 
            path: 'test-results/02-after-initialization.png', 
            fullPage: true 
        });
        
        // Check console messages for VRM loading details
        const logs = [];
        page.on('console', msg => {
            logs.push(`${msg.type()}: ${msg.text()}`);
        });
        
        // Wait longer for VRM loading
        await page.waitForTimeout(20000);
        
        // Take final screenshot
        await page.screenshot({ 
            path: 'test-results/03-final-state.png', 
            fullPage: true 
        });
        
        // Check system status
        const status3D = await page.textContent('#status-3d-text');
        const statusAvatar = await page.textContent('#status-avatar-text');
        const statusConversation = await page.textContent('#status-conversation-text');
        const statusSpeech = await page.textContent('#status-speech-text');
        
        console.log('System Status:');
        console.log(`3D Scene: ${status3D}`);
        console.log(`Avatar: ${statusAvatar}`);
        console.log(`Conversation: ${statusConversation}`);
        console.log(`Speech: ${statusSpeech}`);
        
        // Log console messages to understand what's happening
        console.log('\nConsole Messages:');
        logs.slice(-20).forEach(log => console.log(log));
        
        // Check if we can see the 3D canvas
        const canvas = await page.locator('canvas');
        await expect(canvas).toBeVisible();
        
        // Debug: Check if VRM files exist
        console.log('\nChecking VRM file accessibility...');
        const vrmResponse = await page.evaluate(async () => {
            try {
                const response = await fetch('../assets/avatars/ichika.vrm');
                return {
                    ok: response.ok,
                    status: response.status,
                    size: response.headers.get('content-length')
                };
            } catch (error) {
                return { error: error.message };
            }
        });
        console.log('Ichika VRM file check:', vrmResponse);
        
        // Check if Three.js and VRM components are loaded
        const componentStatus = await page.evaluate(() => {
            return {
                THREE: typeof THREE !== 'undefined',
                AdvancedVRMLoader: typeof AdvancedVRMLoader !== 'undefined',
                VRMBVHAdapter: typeof VRMBVHAdapter !== 'undefined',
                BVHTimeline: typeof BVHTimeline !== 'undefined',
                AvatarBinder: typeof AvatarBinder !== 'undefined',
                ClassroomAvatarIntegration: typeof ClassroomAvatarIntegration !== 'undefined'
            };
        });
        console.log('Component Status:', componentStatus);
        
    }, 360000); // 6 minutes total
});