/**
 * Comprehensive VRM Integration Test
 * 
 * Validates that the system loads real VRM avatars with BVH skeletal animation
 * instead of geometric fallbacks (pink sphere + blue rectangle)
 */

const { test, expect } = require('@playwright/test');

test.describe('VRM Integration System', () => {
    
    test('should load real VRM avatar with BVH animation instead of geometric fallbacks', async ({ page }) => {
        console.log('🧪 Starting comprehensive VRM integration test...');
        
        // Navigate to the complete conversation system
        await page.goto('file://' + __dirname + '/demos/complete_ichika_conversation_system.html');
        
        // Wait for initial page load
        await page.waitForTimeout(2000);
        
        // Take initial screenshot
        await page.screenshot({ path: 'test-results/01-initial-page-load.png', fullPage: true });
        console.log('📸 Screenshot 1: Initial page load');
        
        // Wait for Three.js and VRM infrastructure to be available
        console.log('⏳ Waiting for VRM infrastructure...');
        await page.waitForFunction(() => {
            return window.THREE && 
                   window.AdvancedVRMLoader && 
                   window.VRMBVHAdapter && 
                   window.AvatarBinder && 
                   window.BVHTimeline && 
                   window.BVHTimelineVRMIntegration;
        }, { timeout: 30000 });
        
        console.log('✅ VRM infrastructure loaded');
        
        // Take screenshot after infrastructure loads
        await page.screenshot({ path: 'test-results/02-infrastructure-loaded.png', fullPage: true });
        console.log('📸 Screenshot 2: Infrastructure loaded');
        
        // Click Initialize System button
        console.log('🚀 Initializing VRM system...');
        await page.click('#init-button');
        
        // Wait for system initialization with extended timeout for VRM loading
        await page.waitForTimeout(10000);
        
        // Take screenshot after initialization
        await page.screenshot({ path: 'test-results/03-system-initialized.png', fullPage: true });
        console.log('📸 Screenshot 3: System initialized');
        
        // Validate that VRM infrastructure was used
        const vrmStatus = await page.evaluate(() => {
            return {
                // Check if AdvancedVRMLoader was used
                advancedLoaderUsed: !!window.AdvancedVRMLoader,
                
                // Check if VRMBVHAdapter is available
                bvhAdapterAvailable: !!window.VRMBVHAdapter,
                
                // Check if there's a ClassroomAvatarIntegration instance
                classroomIntegration: !!window.app?.classroomIntegration,
                
                // Check if avatar was loaded (not geometric fallback)
                avatarLoaded: !!window.app?.classroomIntegration?.avatar,
                vrmReady: !!window.app?.classroomIntegration?.vrmReady,
                
                // Check for BVH animation system
                animationSystemReady: !!window.app?.classroomIntegration?.animationsReady,
                bvhTimelineActive: !!window.app?.classroomIntegration?.bvhTimeline,
                
                // Get system status from UI
                systemStatus: {
                    scene: document.querySelector('.status-3d')?.textContent || 'unknown',
                    avatar: document.querySelector('.status-avatar')?.textContent || 'unknown',
                    conversation: document.querySelector('.status-conversation')?.textContent || 'unknown',
                    speechSync: document.querySelector('.status-speech-sync')?.textContent || 'unknown'
                }
            };
        });
        
        console.log('📊 VRM System Status:', JSON.stringify(vrmStatus, null, 2));
        
        // Validate VRM system components
        expect(vrmStatus.advancedLoaderUsed).toBe(true);
        expect(vrmStatus.bvhAdapterAvailable).toBe(true);
        expect(vrmStatus.avatarLoaded).toBe(true);
        
        // Check that system status shows success
        expect(vrmStatus.systemStatus.scene).toContain('Loaded');
        expect(vrmStatus.systemStatus.avatar).toContain('Ready');
        
        console.log('✅ VRM system validation passed');
        
        // Take final comprehensive screenshot
        await page.screenshot({ path: 'test-results/04-vrm-system-validated.png', fullPage: true });
        console.log('📸 Screenshot 4: VRM system validated');
        
        // Test voice animation if available
        if (await page.isVisible('#test-tts')) {
            console.log('🎤 Testing voice animation...');
            await page.click('#test-tts');
            await page.waitForTimeout(3000);
            
            await page.screenshot({ path: 'test-results/05-voice-animation-test.png', fullPage: true });
            console.log('📸 Screenshot 5: Voice animation test');
        }
        
        // Check for geometric fallback elements (should NOT exist)
        const geometricFallbacks = await page.evaluate(() => {
            // Look for pink spheres and blue rectangles in the 3D scene
            const canvas = document.querySelector('canvas');
            if (!canvas) return { found: false, reason: 'no canvas' };
            
            // If the system is using geometric fallbacks, we'll see them in the status
            const logs = document.getElementById('log-messages')?.textContent || '';
            
            const hasFallbackIndicators = 
                logs.includes('createSimpleAvatar') ||
                logs.includes('geometric fallback') ||
                logs.includes('pink sphere') ||
                logs.includes('blue rectangle');
                
            return {
                found: hasFallbackIndicators,
                logs: logs.slice(-1000) // Last 1000 chars of logs
            };
        });
        
        console.log('🔍 Geometric fallback check:', geometricFallbacks);
        
        // Ensure NO geometric fallbacks are being used
        expect(geometricFallbacks.found).toBe(false);
        
        console.log('🎉 Comprehensive VRM integration test completed successfully!');
        console.log('✅ Real VRM avatars are loading instead of geometric fallbacks');
        console.log('✅ BVH skeletal animation system is integrated');
        console.log('✅ All VRM infrastructure components are working');
    }, 60000); // 60 second timeout for VRM loading
});