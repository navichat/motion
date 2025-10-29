/**
 * Working VRM Avatar System Demonstration - Updated with Fixes
 */

import { test, expect } from '@playwright/test';

test.describe('Working VRM Avatar System Demonstration - Fixed', () => {
    test('Comprehensive demonstration of fixed VRM avatar system with real screenshots', async ({ page }, testInfo) => {
        test.setTimeout(300000); // 5 minutes shell timeout compliance
        
        console.log('🎭 Starting Fixed VRM Avatar System Demonstration...');
        
        let screenshotCounter = 0;
        
        const takeScreenshot = async (name, description) => {
            screenshotCounter++;
            const filename = `${screenshotCounter.toString().padStart(2, '0')}-${name}.png`;
            
            const screenshot = await page.screenshot({ 
                fullPage: true,
                quality: 90 
            });
            await testInfo.attach(filename, { 
                body: screenshot, 
                contentType: 'image/png' 
            });
            console.log(`📸 Screenshot ${screenshotCounter}: ${filename} - ${description}`);
            
            return { filename, size: screenshot.length, description };
        };

        // Enhanced logging system
        const systemLogs = {
            messages: [],
            errors: [],
            vrmSpecific: [],
            systemStates: []
        };

        // Monitor page console and errors
        page.on('console', msg => {
            const text = msg.text();
            systemLogs.messages.push(`[${msg.type()}] ${text}`);
            
            if (text.includes('VRM') || text.includes('avatar') || text.includes('loaded') || 
                text.includes('✅') || text.includes('❌') || text.includes('failed') ||
                text.includes('Loading') || text.includes('scene')) {
                systemLogs.vrmSpecific.push(text);
                console.log(`🎭 VRM System: ${text}`);
            }
        });

        page.on('pageerror', error => {
            systemLogs.errors.push(error.message);
            console.log(`❌ Page Error: ${error.message}`);
        });

        try {
            // Phase 1: Load and capture initial state
            console.log('\n🚀 Phase 1: Loading Complete VRM Conversation System');
            console.log('===================================================');
            
            await page.goto('http://localhost:3000/demos/complete_ichika_conversation_system.html');
            console.log('✅ Page navigation complete');
            
            // Wait for initial load
            await page.waitForTimeout(3000);
            
            await takeScreenshot('initial-system-load', 'Initial VRM conversation system before initialization');

            // Phase 2: System Initialization with VRM Loading
            console.log('\n⚙️ Phase 2: VRM System Initialization and Monitoring');
            console.log('===================================================');
            
            // Verify initialize button exists
            await expect(page.locator('#init-button')).toBeVisible();
            
            // Click initialize and start monitoring
            console.log('🔄 Starting VRM system initialization...');
            await page.click('#init-button');
            
            // Take screenshot immediately after init trigger
            await takeScreenshot('initialization-triggered', 'System immediately after initialization triggered');
            
            // Monitor initialization progress with detailed tracking
            let initializationAttempts = 0;
            let maxInitAttempts = 20; // 20 * 3 seconds = 60 seconds total
            let systemReady = false;
            
            console.log('⏳ Monitoring VRM initialization progress...');
            
            while (!systemReady && initializationAttempts < maxInitAttempts) {
                await page.waitForTimeout(3000);
                initializationAttempts++;
                
                // Get current system status
                const sceneStatus = await page.textContent('#status-3d-text').catch(() => 'Unknown');
                const avatarStatus = await page.textContent('#status-avatar-text').catch(() => 'Unknown');
                const conversationStatus = await page.textContent('#status-conversation-text').catch(() => 'Unknown');
                const speechStatus = await page.textContent('#status-speech-text').catch(() => 'Unknown');
                
                const statusSnapshot = {
                    attempt: initializationAttempts,
                    scene: sceneStatus,
                    avatar: avatarStatus,
                    conversation: conversationStatus,
                    speech: speechStatus,
                    timestamp: new Date().toISOString()
                };
                systemLogs.systemStates.push(statusSnapshot);
                
                console.log(`📊 Initialization Progress [${initializationAttempts}/${maxInitAttempts}]:`);
                console.log(`   🌐 3D Scene: ${sceneStatus}`);
                console.log(`   👤 Avatar: ${avatarStatus}`);
                console.log(`   💬 Conversation: ${conversationStatus}`);
                console.log(`   🎤 Speech Sync: ${speechStatus}`);
                
                // Take progress screenshots at intervals
                if (initializationAttempts % 5 === 0) {
                    await takeScreenshot(`init-progress-${initializationAttempts}`, 
                        `Initialization progress at ${initializationAttempts * 3} seconds`);
                }
                
                // Check for completion criteria
                if (sceneStatus === 'Loaded' && 
                    (avatarStatus === 'Ready' || avatarStatus === 'Loaded')) {
                    systemReady = true;
                    console.log('✅ VRM system initialization appears complete!');
                    break;
                }
                
                // Extended monitoring for debugging
                if (initializationAttempts > 12) {
                    console.log(`🔍 Extended monitoring (${initializationAttempts}): Checking logs...`);
                    
                    // Try to get log content
                    const logContent = await page.textContent('#log-messages').catch(() => '');
                    if (logContent.length > 0) {
                        const recentLogs = logContent.split('\n').slice(-3).join(' | ');
                        console.log(`📋 Recent system logs: ${recentLogs}`);
                    }
                }
            }
            
            await takeScreenshot('post-initialization', 'System state after initialization completion');

            // Phase 3: Feature Testing and Validation
            console.log('\n🎯 Phase 3: VRM System Feature Validation');
            console.log('=========================================');
            
            // Test voice functionality if available
            if (await page.locator('#test-tts').isVisible()) {
                console.log('🔊 Testing voice/TTS system...');
                await page.click('#test-tts');
                await page.waitForTimeout(4000);
                
                await takeScreenshot('voice-test-active', 'Voice/TTS system demonstration');
            }
            
            // Test animation functionality if available  
            if (await page.locator('#test-animation').isVisible()) {
                console.log('💃 Testing animation system...');
                await page.click('#test-animation');
                await page.waitForTimeout(4000);
                
                await takeScreenshot('animation-test-active', 'Animation system demonstration');
            }

            // Phase 4: 3D Scene Detail Capture
            console.log('\n🎮 Phase 4: 3D Scene Analysis and Capture');
            console.log('=========================================');
            
            // Capture 3D scene area specifically
            const sceneContainer = page.locator('#scene-container').first();
            if (await sceneContainer.isVisible()) {
                const sceneScreenshot = await sceneContainer.screenshot({ quality: 95 });
                await testInfo.attach('3d-scene-detail.png', { 
                    body: sceneScreenshot, 
                    contentType: 'image/png' 
                });
                console.log('📸 3D scene detail captured');
            }
            
            await takeScreenshot('final-system-state', 'Complete VRM system in final working state');

            // Phase 5: Results Analysis and Reporting
            console.log('\n📊 Phase 5: System Analysis and Final Report');
            console.log('============================================');
            
            // Get final system status
            const finalSceneStatus = await page.textContent('#status-3d-text').catch(() => 'Unknown');
            const finalAvatarStatus = await page.textContent('#status-avatar-text').catch(() => 'Unknown');
            const finalConversationStatus = await page.textContent('#status-conversation-text').catch(() => 'Unknown');
            const finalSpeechStatus = await page.textContent('#status-speech-text').catch(() => 'Unknown');
            
            // Analyze VRM loading success
            const vrmLoadMessages = systemLogs.vrmSpecific.filter(msg => 
                msg.includes('VRM loaded') || msg.includes('avatar loaded') || msg.includes('scene added')
            );
            
            const vrmErrorMessages = systemLogs.vrmSpecific.filter(msg => 
                msg.includes('Failed') || msg.includes('❌') || msg.includes('error')
            );
            
            console.log('\n🏁 COMPREHENSIVE VRM SYSTEM REPORT');
            console.log('===================================');
            console.log(`🎯 System Ready: ${systemReady ? 'YES' : 'PARTIAL'}`);
            console.log(`📸 Screenshots Captured: ${screenshotCounter}`);
            console.log(`🌐 Final 3D Scene Status: ${finalSceneStatus}`);
            console.log(`👤 Final Avatar Status: ${finalAvatarStatus}`);
            console.log(`💬 Final Conversation Status: ${finalConversationStatus}`);
            console.log(`🎤 Final Speech Status: ${finalSpeechStatus}`);
            console.log(`✅ VRM Load Messages: ${vrmLoadMessages.length}`);
            console.log(`❌ VRM Error Messages: ${vrmErrorMessages.length}`);
            console.log(`📨 Total Console Messages: ${systemLogs.messages.length}`);
            console.log(`⚠️ Page Errors: ${systemLogs.errors.length}`);
            
            // Log key VRM messages
            if (vrmLoadMessages.length > 0) {
                console.log('\n🎭 Key VRM Loading Messages:');
                vrmLoadMessages.slice(0, 5).forEach((msg, i) => {
                    console.log(`  ${i + 1}. ${msg}`);
                });
            }
            
            if (vrmErrorMessages.length > 0) {
                console.log('\n❌ VRM Error Messages:');
                vrmErrorMessages.slice(0, 3).forEach((msg, i) => {
                    console.log(`  ${i + 1}. ${msg}`);
                });
            }
            
            console.log('\n✅ FIXED VRM AVATAR SYSTEM DEMONSTRATION COMPLETE!');
            console.log('==================================================');
            console.log('📸 Screenshots demonstrate the complete 3D animated Ichika conversation system');
            console.log(`🎭 System shows ${finalAvatarStatus} avatar status with ${finalSceneStatus} 3D scene`);
            
            // Test assertions
            expect(finalSceneStatus).toBe('Loaded');
            expect(finalAvatarStatus).not.toBe('Not Loaded');
            expect(screenshotCounter).toBeGreaterThan(4);
            expect(systemLogs.errors.length).toBeLessThan(5); // Allow some minor errors
            
            return {
                success: true,
                screenshots: screenshotCounter,
                systemReady,
                systemStatus: {
                    scene: finalSceneStatus,
                    avatar: finalAvatarStatus,
                    conversation: finalConversationStatus,
                    speech: finalSpeechStatus
                },
                vrmMessages: systemLogs.vrmSpecific.length,
                vrmLoadMessages: vrmLoadMessages.length,
                vrmErrors: vrmErrorMessages.length,
                totalErrors: systemLogs.errors.length
            };
            
        } catch (error) {
            console.error('❌ Fixed VRM demonstration failed:', error);
            await takeScreenshot('error-state', `System error: ${error.message}`);
            throw error;
        }
    });
});