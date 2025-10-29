const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test('Capture Working Ichika VRM Neural BVH Integration Demo', async ({ page, browserName }) => {
    console.log('🎭 Starting Working Ichika VRM Neural BVH Integration Demo Screenshot Capture...');
    
    // Set viewport for consistent screenshots
    await page.setViewportSize({ width: 1400, height: 900 });
    
    // Create screenshots directory
    const screenshotDir = path.join(__dirname, 'test-results', 'working-vrm-demo-screenshots');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }

    // Enable console logging to see what's happening
    page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        if (type === 'error') {
            console.log('❌ Browser console error:', text);
        } else if (type === 'warning') {
            console.log('⚠️ Browser console warning:', text);  
        } else if (text.includes('VRM') || text.includes('BVH') || text.includes('Neural')) {
            console.log(`📋 Browser console [${type}]:`, text);
        }
    });

    try {
        // Navigate to the working demo
        const demoPath = path.resolve(__dirname, 'dev/web_viewer/demos/working_ichika_neural_bvh_integration.html');
        const demoUrl = `file://${demoPath}`;
        
        console.log(`🌐 Navigating to: ${demoUrl}`);
        await page.goto(demoUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });

        // Wait for system initialization
        console.log('⏳ Waiting for system to initialize...');
        await page.waitForTimeout(6000);

        // Take initial loading screenshot
        await page.screenshot({
            path: path.join(screenshotDir, '01-loading-state.png'),
            fullPage: false
        });

        // Wait for loading to complete (check if overlay is hidden)
        try {
            await page.waitForFunction(() => {
                const overlay = document.querySelector('#loading-overlay');
                return !overlay || overlay.style.display === 'none' || overlay.style.opacity === '0';
            }, { timeout: 20000 });
            console.log('✅ Loading overlay hidden');
        } catch (error) {
            console.log('⚠️ Loading overlay timeout, continuing...');
        }

        // Wait a bit more for VRM loading
        await page.waitForTimeout(3000);

        console.log('📸 Taking system ready screenshot...');
        await page.screenshot({
            path: path.join(screenshotDir, '02-system-ready.png'),
            fullPage: false
        });

        // Test neural BVH generation
        try {
            console.log('🧠 Testing neural BVH generation...');
            const testButton = await page.locator('#test-neural-bvh');
            if (await testButton.isEnabled()) {
                await testButton.click();
                await page.waitForTimeout(4000);
                
                console.log('📸 Taking neural BVH test screenshot...');
                await page.screenshot({
                    path: path.join(screenshotDir, '03-neural-bvh-test.png'),
                    fullPage: false
                });
            }
        } catch (error) {
            console.log('⚠️ Neural BVH test error:', error.message);
        }

        // Test gesture animation if available
        try {
            const gestureButton = await page.locator('#play-gesture');
            if (await gestureButton.isEnabled()) {
                await gestureButton.click();
                await page.waitForTimeout(2000);
                
                console.log('📸 Taking gesture animation screenshot...');
                await page.screenshot({
                    path: path.join(screenshotDir, '04-gesture-animation.png'),
                    fullPage: false
                });
            }
        } catch (error) {
            console.log('⚠️ Gesture animation not available');
        }

        // Get detailed system status
        const systemStatus = await page.evaluate(() => {
            const getStatus = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent : 'UNKNOWN';
            };
            
            const getLogText = () => {
                const logElement = document.getElementById('log-output');
                return logElement ? logElement.textContent : 'No log available';
            };
            
            return {
                vrmStatus: getStatus('vrm-status'),
                bvhAdapterStatus: getStatus('bvh-adapter-status'),
                sceneStatus: getStatus('scene-status'),
                audio2gestureStatus: getStatus('audio2gesture-status'),
                deepmimicStatus: getStatus('deepmimic-status'),
                framesApplied: getStatus('frames-applied'),
                activeBones: getStatus('active-bones'),
                animationFps: getStatus('animation-fps'),
                inferenceTime: getStatus('inference-time'),
                logText: getLogText()
            };
        });

        // Take final comprehensive screenshot
        console.log('📸 Taking final comprehensive screenshot...');
        await page.screenshot({
            path: path.join(screenshotDir, '05-final-comprehensive.png'),
            fullPage: true
        });

        // Report system status
        console.log('📊 Working VRM Demo System Status:');
        console.log('  ✅ VRM Model (ichika.vrm):', systemStatus.vrmStatus);
        console.log('  ✅ BVH Adapter:', systemStatus.bvhAdapterStatus);
        console.log('  ✅ 3D Scene:', systemStatus.sceneStatus);
        console.log('  🧠 Audio2Gesture:', systemStatus.audio2gestureStatus);
        console.log('  🧠 DeepMimic:', systemStatus.deepmimicStatus);
        console.log('  📊 BVH Frames Applied:', systemStatus.framesApplied);
        console.log('  🦴 Active Bones:', systemStatus.activeBones);
        console.log('  🎬 Animation FPS:', systemStatus.animationFps);
        console.log('  ⚡ Inference Time:', systemStatus.inferenceTime, 'ms');

        // Show key log entries
        if (systemStatus.logText) {
            const logLines = systemStatus.logText.split('\n').filter(line => 
                line.includes('VRM loaded') || 
                line.includes('BVH frames') || 
                line.includes('Applied BVH') ||
                line.includes('Neural network') ||
                line.includes('Ready')
            );
            
            if (logLines.length > 0) {
                console.log('📝 Key System Log Entries:');
                logLines.slice(-10).forEach(line => console.log('    ', line.trim()));
            }
        }

        console.log('✅ Working VRM Demo screenshot capture complete!');
        console.log(`📁 Screenshots saved to: ${screenshotDir}`);

    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
        
        // Take error screenshot
        try {
            await page.screenshot({
                path: path.join(screenshotDir, 'error-state.png'),
                fullPage: true
            });
        } catch (screenshotError) {
            console.error('❌ Error screenshot also failed:', screenshotError);
        }
        
        throw error;
    }
});