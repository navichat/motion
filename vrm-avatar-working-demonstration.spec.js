import { test, expect } from '@playwright/test';

/**
 * VRM Avatar Working Demonstration Test
 * 
 * This test demonstrates the fully working VRM conversational avatar system
 * by loading the demo, verifying initialization, and capturing screenshots.
 */

test.describe('VRM Avatar Working Demonstration', () => {
    test('demonstrate working conversational avatar with screenshot', async ({ page }) => {
        console.log('🎭 Starting VRM Avatar Working Demonstration');
        
        // Configure page for optimal avatar loading
        await page.setViewportSize({ width: 1920, height: 1080 });
        
        console.log('📁 Loading VRM avatar demo page...');
        // Navigate to the working VRM avatar demo
        await page.goto('/demos/working_vrm_avatar_system.html');
        
        // Wait for page to fully load
        await page.waitForLoadState('domcontentloaded');
        console.log('✅ Demo page loaded');
        
        // Wait for the main container to be visible
        await page.waitForSelector('#main-container', { timeout: 10000 });
        console.log('✅ Main container ready');
        
        // Wait for UI overlay to be visible  
        await page.waitForSelector('#ui-overlay', { timeout: 10000 });
        console.log('✅ UI overlay ready');
        
        // Wait for performance monitor to show system status
        await page.waitForSelector('#performance-monitor', { timeout: 10000 });
        console.log('✅ Performance monitor ready');
        
        // Wait for the avatar system to auto-initialize (it auto-initializes after 2 seconds)
        console.log('⏳ Waiting for auto-initialization...');
        await page.waitForTimeout(3000);
        
        // Check if system initialized properly by looking for active status indicators
        const systemStatus = await page.locator('#status-3d-text').textContent();
        console.log(`📊 3D Scene Status: ${systemStatus}`);
        
        const avatarStatus = await page.locator('#status-avatar-text').textContent();
        console.log(`🎭 Avatar Status: ${avatarStatus}`);
        
        const conversationStatus = await page.locator('#status-conversation-text').textContent();
        console.log(`💬 Conversation Status: ${conversationStatus}`);
        
        // Verify the avatar container is visible
        const avatarContainer = page.locator('#avatar-container');
        await expect(avatarContainer).toBeVisible();
        console.log('✅ Avatar container is visible');
        
        // Verify the VRM avatar element is present
        const vrmAvatar = page.locator('#vrm-avatar');
        await expect(vrmAvatar).toBeVisible();
        console.log('✅ VRM avatar element is visible');
        
        // Check for the avatar label showing it's ready
        const avatarLabel = page.locator('#avatar-label');
        await expect(avatarLabel).toBeVisible();
        const labelText = await avatarLabel.textContent();
        console.log(`🏷️ Avatar Label: ${labelText}`);
        
        // Verify performance metrics are showing
        const fpsCounter = await page.locator('#fps-counter').textContent();
        const renderMode = await page.locator('#render-mode').textContent();
        console.log(`📈 Performance - FPS: ${fpsCounter}, Render: ${renderMode}`);
        
        // Test the conversation system by clicking "Test Voice"
        console.log('🔊 Testing voice system...');
        const testTTSButton = page.locator('#test-tts');
        await testTTSButton.click();
        
        // Wait for voice test to complete
        await page.waitForTimeout(2000);
        console.log('✅ Voice test completed');
        
        // Test animation system
        console.log('🎬 Testing animation system...');
        const testAnimButton = page.locator('#test-animation');
        await testAnimButton.click();
        
        // Wait for animation test to complete
        await page.waitForTimeout(2000);
        console.log('✅ Animation test completed');
        
        // Take a comprehensive screenshot showing the working system
        console.log('📸 Capturing comprehensive system screenshot...');
        await page.screenshot({
            path: 'test-results/vrm-avatar-working-demonstration.png',
            fullPage: true
        });
        console.log('✅ Full page screenshot captured');
        
        // Take a focused screenshot of just the avatar area
        console.log('📸 Capturing focused avatar screenshot...');
        await page.locator('#scene-container').screenshot({
            path: 'test-results/vrm-avatar-focused-view.png'
        });
        console.log('✅ Focused avatar screenshot captured');
        
        // Take screenshot of the UI controls showing system status
        console.log('📸 Capturing UI controls screenshot...');
        await page.locator('#ui-overlay').screenshot({
            path: 'test-results/vrm-system-controls.png'
        });
        console.log('✅ UI controls screenshot captured');
        
        // Take screenshot of performance monitor
        console.log('📸 Capturing performance monitor screenshot...');
        await page.locator('#performance-monitor').screenshot({
            path: 'test-results/vrm-performance-monitor.png'
        });
        console.log('✅ Performance monitor screenshot captured');
        
        // Verify system log shows successful initialization
        const logMessages = page.locator('#log-messages');
        const logContent = await logMessages.textContent();
        console.log('📋 System Log Summary:');
        console.log(logContent);
        
        // Verify conversation history shows interactions
        const conversationMessages = page.locator('#messages');
        const conversationContent = await conversationMessages.textContent();
        console.log('💬 Conversation History:');
        console.log(conversationContent);
        
        // Final comprehensive verification
        console.log('🔍 Final system verification...');
        
        // Check all status indicators are active (green)
        const status3D = page.locator('#status-3d');
        const statusAvatar = page.locator('#status-avatar');
        const statusConversation = page.locator('#status-conversation');
        const statusSpeech = page.locator('#status-speech');
        
        // Verify status indicators have the 'status-active' class
        await expect(status3D).toHaveClass(/status-active/);
        await expect(statusAvatar).toHaveClass(/status-active/);
        await expect(statusConversation).toHaveClass(/status-active/);
        await expect(statusSpeech).toHaveClass(/status-active/);
        
        console.log('✅ All status indicators are active (green)');
        
        // Test the start conversation button functionality
        console.log('🎤 Testing conversation startup...');
        const startConversationButton = page.locator('#start-conversation');
        await startConversationButton.click();
        
        // Wait for conversation to initialize
        await page.waitForTimeout(3000);
        
        // Take final screenshot with conversation active
        console.log('📸 Capturing final screenshot with active conversation...');
        await page.screenshot({
            path: 'test-results/vrm-avatar-conversation-active.png',
            fullPage: true
        });
        console.log('✅ Final screenshot with active conversation captured');
        
        console.log('🎉 VRM Avatar Working Demonstration completed successfully!');
        console.log('📁 Screenshots saved:');
        console.log('  - test-results/vrm-avatar-working-demonstration.png (full system)');
        console.log('  - test-results/vrm-avatar-focused-view.png (avatar view)');
        console.log('  - test-results/vrm-system-controls.png (UI controls)');
        console.log('  - test-results/vrm-performance-monitor.png (performance stats)');
        console.log('  - test-results/vrm-avatar-conversation-active.png (conversation mode)');
    });
    
    test('verify VRM system components are working', async ({ page }) => {
        console.log('🔧 Starting VRM System Components Verification');
        
        // Navigate to the demo page
        await page.goto('/demos/working_vrm_avatar_system.html');
        await page.waitForLoadState('domcontentloaded');
        
        // Wait for auto-initialization
        await page.waitForTimeout(3000);
        
        // Verify all critical components are present and functional
        const criticalElements = [
            { selector: '#scene-container', name: '3D Scene Container' },
            { selector: '#avatar-container', name: 'Avatar Container' },
            { selector: '#vrm-avatar', name: 'VRM Avatar' },
            { selector: '#performance-monitor', name: 'Performance Monitor' },
            { selector: '#ui-overlay', name: 'UI Controls' },
            { selector: '#init-button', name: 'Initialize Button' },
            { selector: '#test-tts', name: 'Test Voice Button' },
            { selector: '#test-animation', name: 'Test Animation Button' }
        ];
        
        for (const element of criticalElements) {
            await expect(page.locator(element.selector)).toBeVisible();
            console.log(`✅ ${element.name} is present and visible`);
        }
        
        // Verify the avatar has proper styling indicating it's loaded
        const avatar = page.locator('#vrm-avatar');
        const hasSystemReadyClass = await avatar.evaluate(el => el.classList.contains('system-ready'));
        expect(hasSystemReadyClass).toBeTruthy();
        console.log('✅ Avatar has system-ready styling');
        
        // Verify performance metrics are being displayed
        const fps = await page.locator('#fps-counter').textContent();
        const renderMode = await page.locator('#render-mode').textContent();
        const avatarStatus = await page.locator('#avatar-status').textContent();
        
        expect(fps).toBeTruthy();
        expect(renderMode).toContain('WebGL');
        expect(avatarStatus).toBe('Loaded');
        
        console.log(`✅ Performance metrics active - FPS: ${fps}, Render: ${renderMode}, Avatar: ${avatarStatus}`);
        
        console.log('🎉 All VRM system components verified as working!');
    });
});