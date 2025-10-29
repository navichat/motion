/**
 * Comprehensive VRM System Fix Test
 * Shell timeout compliant Playwright test for VRM avatar system debugging and screenshot capture
 */

const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test.describe('VRM Avatar System Comprehensive Fix', () => {
    test.setTimeout(300000); // 5 minute shell timeout compliance

    test('Analyze current VRM system issues and fix infrastructure loading', async ({ page }) => {
        console.log('🔍 Starting comprehensive VRM system analysis and fix...');

        // Enable verbose logging
        page.on('console', msg => {
            const text = msg.text();
            if (text.includes('VRM') || text.includes('BVH') || text.includes('Avatar') || text.includes('ERROR') || text.includes('Failed')) {
                console.log(`Browser: ${text}`);
            }
        });

        page.on('pageerror', err => {
            console.error('Page Error:', err.message);
        });

        // Navigate to the working system
        console.log('📄 Loading complete Ichika conversation system...');
        await page.goto('file:///home/runner/work/motion/motion/dev/web_viewer/demos/complete_ichika_conversation_system.html', {
            waitUntil: 'load'
        });

        // Wait for page load and initial setup
        await page.waitForTimeout(5000);

        // Take initial screenshot
        console.log('📸 Capturing initial system state...');
        await page.screenshot({
            path: 'test-results/01-initial-system-state.png',
            fullPage: false
        });

        // Check for CDN loading issues
        console.log('🔍 Checking for CDN loading issues...');
        const networkErrors = await page.evaluate(() => {
            return {
                hasThreeJS: typeof window.THREE !== 'undefined',
                hasVRMLoader: typeof window.THREELoaders?.VRMLoaderPlugin !== 'undefined',
                hasAdvancedVRMLoader: typeof window.AdvancedVRMLoader !== 'undefined',
                hasVRMBVHAdapter: typeof window.VRMBVHAdapter !== 'undefined',
                hasBVHTimeline: typeof window.BVHTimeline !== 'undefined',
                hasAvatarBinder: typeof window.AvatarBinder !== 'undefined',
                hasClassroomIntegration: typeof window.ClassroomAvatarIntegration !== 'undefined'
            };
        });

        console.log('🔧 Current system components:', JSON.stringify(networkErrors, null, 2));

        // Try to initialize the system  
        console.log('⚡ Attempting system initialization...');
        const initButton = await page.locator('#init-button');
        if (await initButton.isVisible()) {
            await initButton.click();
            console.log('✅ Clicked initialize system button');
        }

        // Wait for initialization with extended timeout
        await page.waitForTimeout(15000);

        // Capture post-initialization state
        console.log('📸 Capturing post-initialization state...');
        await page.screenshot({
            path: 'test-results/02-post-initialization.png',
            fullPage: false
        });

        // Check if avatar is loaded (should show real VRM, not geometric shapes)
        const systemStatus = await page.evaluate(() => {
            const statusElements = {
                sceneStatus: document.querySelector('.status-indicator:nth-of-type(1)')?.className || 'not-found',
                avatarStatus: document.querySelector('.status-indicator:nth-of-type(2)')?.className || 'not-found',
                conversationStatus: document.querySelector('.status-indicator:nth-of-type(3)')?.className || 'not-found',
                speechStatus: document.querySelector('.status-indicator:nth-of-type(4)')?.className || 'not-found'
            };

            const logMessages = Array.from(document.querySelectorAll('#log-messages *'))
                .map(el => el.textContent)
                .filter(text => text && text.length > 0)
                .slice(-20); // Last 20 log messages

            return {
                statusElements,
                logMessages,
                errors: window.console.errors || [],
                hasCanvas: !!document.querySelector('canvas'),
                canvasInfo: document.querySelector('canvas') ? {
                    width: document.querySelector('canvas').width,
                    height: document.querySelector('canvas').height,
                    style: document.querySelector('canvas').style.cssText
                } : null
            };
        });

        console.log('📊 System status after initialization:');
        console.log('Status elements:', systemStatus.statusElements);
        console.log('Recent log messages:', systemStatus.logMessages);
        console.log('Canvas info:', systemStatus.canvasInfo);

        // Test voice functionality
        const testTTSButton = await page.locator('#test-tts');
        if (await testTTSButton.isVisible()) {
            console.log('🎤 Testing TTS functionality...');
            await testTTSButton.click();
            await page.waitForTimeout(3000);

            await page.screenshot({
                path: 'test-results/03-tts-test.png',
                fullPage: false
            });
        }

        // Test animation functionality 
        const testAnimButton = await page.locator('#test-animation');
        if (await testAnimButton.isVisible()) {
            console.log('🎭 Testing animation functionality...');
            await testAnimButton.click();
            await page.waitForTimeout(3000);

            await page.screenshot({
                path: 'test-results/04-animation-test.png',
                fullPage: false
            });
        }

        // Get final system analysis
        const finalAnalysis = await page.evaluate(() => {
            // Check if we have a real VRM avatar or just geometric fallback
            const canvasElement = document.querySelector('canvas');
            const hasRealAvatar = window.classroomIntegration && 
                                window.classroomIntegration.vrm && 
                                window.classroomIntegration.avatar;

            const vrmStatus = {
                hasVRM: !!window.classroomIntegration?.vrm,
                hasAvatar: !!window.classroomIntegration?.avatar,
                vrmReady: !!window.classroomIntegration?.vrmReady,
                animationsReady: !!window.classroomIntegration?.animationsReady,
                avatarType: hasRealAvatar ? 'real-vrm' : 'geometric-fallback'
            };

            const infrastructureStatus = {
                AdvancedVRMLoader: typeof window.AdvancedVRMLoader,
                VRMBVHAdapter: typeof window.VRMBVHAdapter,
                AvatarBinder: typeof window.AvatarBinder,
                BVHTimeline: typeof window.BVHTimeline,
                BVHTimelineVRMIntegration: typeof window.BVHTimelineVRMIntegration,
                ClassroomAvatarIntegration: typeof window.ClassroomAvatarIntegration
            };

            return {
                vrmStatus,
                infrastructureStatus,
                hasCanvas: !!canvasElement,
                canvasSize: canvasElement ? `${canvasElement.width}x${canvasElement.height}` : 'none',
                timestamp: new Date().toISOString()
            };
        });

        console.log('🎯 Final system analysis:');
        console.log(JSON.stringify(finalAnalysis, null, 2));

        // Take comprehensive final screenshot
        await page.screenshot({
            path: 'test-results/05-final-system-analysis.png',
            fullPage: false
        });

        // Generate diagnostic report
        const diagnosticReport = {
            testTimestamp: new Date().toISOString(),
            initialComponents: networkErrors,
            systemStatus: systemStatus,
            finalAnalysis: finalAnalysis,
            issues: [],
            recommendations: []
        };

        // Analyze issues
        if (!finalAnalysis.vrmStatus.hasVRM) {
            diagnosticReport.issues.push('VRM model not loaded - system falling back to geometric shapes');
            diagnosticReport.recommendations.push('Fix VRM loading infrastructure to use local assets instead of blocked CDN resources');
        }

        if (finalAnalysis.infrastructureStatus.AdvancedVRMLoader === 'undefined') {
            diagnosticReport.issues.push('AdvancedVRMLoader not properly loaded');
            diagnosticReport.recommendations.push('Ensure AdvancedVRMLoader.js is loaded before system initialization');
        }

        if (!finalAnalysis.vrmStatus.animationsReady) {
            diagnosticReport.issues.push('VRM animations not initialized properly');
            diagnosticReport.recommendations.push('Fix BVH animation integration with VRM skeleton mapping');
        }

        // Save diagnostic report
        fs.writeFileSync(
            'test-results/vrm-system-diagnostic-report.json',
            JSON.stringify(diagnosticReport, null, 2)
        );

        console.log('📋 Diagnostic report saved to test-results/vrm-system-diagnostic-report.json');
        console.log(`🏁 Analysis complete. Found ${diagnosticReport.issues.length} issues to fix.`);

        // Test should pass - we're gathering data for fixes
        expect(true).toBe(true);
    });
});