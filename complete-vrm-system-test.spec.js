const { test, expect } = require('@playwright/test');

test.describe('Complete Working VRM Avatar System', () => {
    test.setTimeout(360000); // 6 minutes shell timeout compliance
    
    test('demonstrate complete VRM avatar conversation system', async ({ page }) => {
        console.log('🎯 Testing complete VRM avatar system with real Ichika...');
        
        // Navigate to working VRM system
        await page.goto('http://localhost:8080/demos/working_vrm_avatar_system.html');
        await page.waitForLoadState('networkidle');
        
        // Wait for auto-initialization
        console.log('⏳ Waiting for system auto-initialization...');
        await page.waitForTimeout(3000);
        
        // Verify system status indicators
        const status3D = await page.textContent('#status-3d-text');
        const statusAvatar = await page.textContent('#status-avatar-text');
        const statusConversation = await page.textContent('#status-conversation-text');
        const statusSpeech = await page.textContent('#status-speech-text');
        
        expect(status3D).toBe('Loaded');
        expect(statusAvatar).toBe('Ready');
        expect(statusConversation).toBe('Ready');
        expect(statusSpeech).toBe('Ready');
        
        // Take initial screenshot
        await page.screenshot({ 
            path: 'test-results/01-vrm-system-ready.png', 
            fullPage: true 
        });
        
        console.log('✅ System Status Verified:');
        console.log(`  - 3D Scene: ${status3D}`);
        console.log(`  - Avatar: ${statusAvatar}`);
        console.log(`  - Conversation: ${statusConversation}`);
        console.log(`  - Speech: ${statusSpeech}`);
        
        // Verify performance monitor shows VRM details
        const vrmFile = await page.textContent('#performance-monitor');
        expect(vrmFile).toContain('ichika.vrm');
        expect(vrmFile).toContain('WebGL+VRM');
        expect(vrmFile).toContain('Active');
        
        console.log('✅ VRM Performance Monitor Verified');
        
        // Test voice system
        console.log('🔊 Testing VRM voice system...');
        await page.click('#test-tts');
        await page.waitForTimeout(2000);
        
        // Verify conversation history shows voice test
        const conversationHistory = await page.textContent('#messages');
        expect(conversationHistory).toContain('This is a test of my voice system');
        
        await page.screenshot({ 
            path: 'test-results/02-vrm-voice-test.png', 
            fullPage: true 
        });
        
        console.log('✅ Voice Test Completed');
        
        // Test animation system
        console.log('🎬 Testing VRM animation system...');
        await page.click('#test-animation');
        await page.waitForTimeout(2000);
        
        // Verify animation was logged
        const systemLog = await page.textContent('#log-messages');
        expect(systemLog).toContain('Testing VRM animations');
        expect(systemLog).toContain('Playing animation:');
        
        await page.screenshot({ 
            path: 'test-results/03-vrm-animation-test.png', 
            fullPage: true 
        });
        
        console.log('✅ Animation Test Completed');
        
        // Test conversation system
        console.log('💬 Testing VRM conversation system...');
        await page.click('#start-conversation');
        await page.waitForTimeout(3000);
        
        // Verify conversation mode is active
        const stopButton = await page.locator('#stop-conversation');
        await expect(stopButton).not.toBeDisabled();
        
        // Verify conversation started message
        const finalConversationHistory = await page.textContent('#messages');
        expect(finalConversationHistory).toContain('listening! What would you like to talk about');
        
        await page.screenshot({ 
            path: 'test-results/04-vrm-conversation-active.png', 
            fullPage: true 
        });
        
        console.log('✅ Conversation System Active');
        
        // Stop conversation
        await page.click('#stop-conversation');
        await page.waitForTimeout(2000);
        
        await page.screenshot({ 
            path: 'test-results/05-vrm-conversation-complete.png', 
            fullPage: true 
        });
        
        // Verify avatar is visible and interactive
        const avatarContainer = await page.locator('#avatar-container');
        await expect(avatarContainer).toBeVisible();
        
        const avatarLabel = await page.textContent('#avatar-label');
        expect(avatarLabel).toBe('Ichika Avatar Ready');
        
        console.log('✅ VRM Avatar Verified as Visible and Ready');
        
        // Verify system performance
        const fpsCounter = await page.textContent('#fps-counter');
        const renderMode = await page.textContent('#render-mode');
        const avatarStatus = await page.textContent('#avatar-status');
        
        expect(fpsCounter).toBe('60');
        expect(renderMode).toBe('WebGL+VRM');
        expect(avatarStatus).toBe('Active');
        
        console.log('✅ Performance Metrics Verified:');
        console.log(`  - FPS: ${fpsCounter}`);
        console.log(`  - Render: ${renderMode}`);
        console.log(`  - Avatar: ${avatarStatus}`);
        
        // Final comprehensive screenshot
        await page.screenshot({ 
            path: 'test-results/06-vrm-system-complete.png', 
            fullPage: true 
        });
        
        console.log('🎉 VRM Avatar System Test Complete!');
        console.log('📸 Screenshots captured showing working Ichika VRM system');
        console.log('✅ All features functional: 3D rendering, voice, animation, conversation');
        
    });
    
    test('verify VRM system performance and features', async ({ page }) => {
        console.log('⚡ Testing VRM system performance and feature verification...');
        
        await page.goto('http://localhost:8080/demos/working_vrm_avatar_system.html');
        await page.waitForLoadState('networkidle');
        
        // Wait for full initialization
        await page.waitForTimeout(5000);
        
        // Verify all UI elements are present and functional
        const initButton = await page.locator('#init-button');
        const startButton = await page.locator('#start-conversation');
        const testVoiceButton = await page.locator('#test-tts');
        const testAnimButton = await page.locator('#test-animation');
        
        await expect(initButton).toBeVisible();
        await expect(startButton).toBeVisible();
        await expect(testVoiceButton).toBeVisible();
        await expect(testAnimButton).toBeVisible();
        
        // Test multiple voice and animation cycles
        for (let i = 0; i < 3; i++) {
            console.log(`🔄 Testing cycle ${i + 1}/3...`);
            
            await testVoiceButton.click();
            await page.waitForTimeout(1000);
            
            await testAnimButton.click();
            await page.waitForTimeout(1000);
        }
        
        // Verify system remains stable
        const finalSystemLog = await page.textContent('#log-messages');
        expect(finalSystemLog).toContain('VRM avatar system fully initialized');
        expect(finalSystemLog).toContain('Testing VRM voice synthesis');
        expect(finalSystemLog).toContain('Testing VRM animations');
        
        await page.screenshot({ 
            path: 'test-results/07-vrm-performance-test.png', 
            fullPage: true 
        });
        
        console.log('✅ Performance and stability test completed');
    });
});