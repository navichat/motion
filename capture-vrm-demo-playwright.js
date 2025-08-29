const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function captureWorkingVRMDemo() {
    console.log('🎭 Starting Working Ichika VRM Neural BVH Integration Demo Screenshot Capture...');
    
    // Create screenshots directory
    const screenshotDir = path.join(__dirname, 'test-results', 'working-vrm-demo-screenshots');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }

    const browser = await chromium.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',
            '--allow-running-insecure-content',
            '--enable-webgl',
            '--enable-accelerated-2d-canvas'
        ]
    });

    try {
        const context = await browser.newContext({
            viewport: { width: 1400, height: 900 }
        });
        
        const page = await context.newPage();
        
        // Enable console logging
        page.on('console', msg => {
            const type = msg.type();
            const text = msg.text();
            if (type === 'error') {
                console.log('❌ Browser console error:', text);
            } else if (type === 'warning') {
                console.log('⚠️ Browser console warning:', text);
            } else if (text.includes('VRM') || text.includes('BVH') || text.includes('Neural') || text.includes('✅') || text.includes('🎭')) {
                console.log(`📋 Browser console [${type}]:`, text);
            }
        });

        // Navigate to the BVH Neural VRM mapping demonstration
        const demoPath = path.resolve(__dirname, 'dev/web_viewer/demos/bvh_neural_vrm_mapping_demonstration.html');
        const demoUrl = `file://${demoPath}`;
        
        console.log(`🌐 Navigating to: ${demoUrl}`);
        await page.goto(demoUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });

        // Wait for system initialization
        console.log('⏳ Waiting for system to initialize...');
        await page.waitForTimeout(8000);

        // Take initial loading screenshot
        console.log('📸 Taking loading state screenshot...');
        await page.screenshot({
            path: path.join(screenshotDir, '01-loading-state.png'),
            fullPage: false
        });

        // Wait for loading overlay to disappear
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
        await page.waitForTimeout(5000);

        console.log('📸 Taking system ready screenshot...');
        await page.screenshot({
            path: path.join(screenshotDir, '02-system-ready.png'),
            fullPage: false
        });

        // Test BVH generation
        try {
            console.log('🧠 Testing BVH neural generation...');
            const generateButton = await page.locator('button:has-text("Generate Neural BVH")');
            const isEnabled = await generateButton.isVisible();
            console.log('Generate BVH button visible:', isEnabled);
            
            if (isEnabled) {
                await generateButton.click();
                console.log('✅ Clicked generate neural BVH button');
                await page.waitForTimeout(4000);
                
                console.log('📸 Taking neural BVH generation screenshot...');
                await page.screenshot({
                    path: path.join(screenshotDir, '03-neural-bvh-generation.png'),
                    fullPage: false
                });
            } else {
                console.log('⚠️ Generate BVH button not visible');
            }
        } catch (error) {
            console.log('⚠️ Neural BVH generation error:', error.message);
        }

        // Test VRM application
        try {
            const applyButton = await page.locator('button:has-text("Apply BVH to VRM")');
            const isVisible = await applyButton.isVisible();
            console.log('Apply BVH to VRM button visible:', isVisible);
            
            if (isVisible) {
                await applyButton.click();
                console.log('✅ Clicked apply BVH to VRM button');
                await page.waitForTimeout(3000);
                
                console.log('📸 Taking BVH to VRM application screenshot...');
                await page.screenshot({
                    path: path.join(screenshotDir, '04-bvh-to-vrm-application.png'),
                    fullPage: false
                });
            }
        } catch (error) {
            console.log('⚠️ BVH to VRM application error:', error.message);
        }

        // Test animation playback
        try {
            const playButton = await page.locator('button:has-text("Play Animation")');
            const isVisible = await playButton.isVisible();
            console.log('Play animation button visible:', isVisible);
            
            if (isVisible) {
                await playButton.click();
                console.log('✅ Clicked play animation button');
                await page.waitForTimeout(3000);
                
                console.log('📸 Taking animation playback screenshot...');
                await page.screenshot({
                    path: path.join(screenshotDir, '05-animation-playback.png'),
                    fullPage: false
                });
            }
        } catch (error) {
            console.log('⚠️ Animation playback error:', error.message);
        }

        // Get detailed system status for the BVH mapping demonstration
        const systemStatus = await page.evaluate(() => {
            const getStatus = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent : 'UNKNOWN';
            };
            
            const getLogText = () => {
                const logElement = document.getElementById('log-output');
                return logElement ? logElement.textContent : 'No log available';
            };
            
            const getMetrics = () => {
                return {
                    framesGenerated: document.getElementById('frames-generated')?.textContent || '0',
                    bonesAnimated: document.getElementById('bones-animated')?.textContent || '0',
                    fpsCounter: document.getElementById('fps-counter')?.textContent || '60',
                    neuralLatency: document.getElementById('neural-latency')?.textContent || '--'
                };
            };
            
            return {
                audio2gestureStatus: getStatus('audio2gesture-status'),
                deepmimicStatus: getStatus('deepmimic-status'),
                rsmt: getStatus('rsmt-status'),
                faceformerStatus: getStatus('faceformer-status'),
                vrmCharacterStatus: getStatus('vrm-character-status'),
                bvhAdapterStatus: getStatus('bvh-adapter-status'),
                boneMappingStatus: getStatus('bone-mapping-status'),
                animationStatus: getStatus('animation-status'),
                metrics: getMetrics(),
                logText: getLogText()
            };
        });

        // Take final comprehensive screenshot showing all components
        console.log('📸 Taking final comprehensive screenshot...');
        await page.screenshot({
            path: path.join(screenshotDir, '06-final-comprehensive-demo.png'),
            fullPage: true
        });

        // Report system status
        console.log('\n📊 BVH Neural VRM Mapping Demonstration Status:');
        console.log('════════════════════════════════════════════════');
        console.log('  🧠 Audio2Gesture Network:', systemStatus.audio2gestureStatus);
        console.log('  🧠 DeepMimic Network:', systemStatus.deepmimicStatus);
        console.log('  🧠 RSMT Network:', systemStatus.rsmt);
        console.log('  🧠 FaceFormer Network:', systemStatus.faceformerStatus);
        console.log('  🎭 VRM Character (ichika.vrm):', systemStatus.vrmCharacterStatus);
        console.log('  🔗 BVH Adapter:', systemStatus.bvhAdapterStatus);
        console.log('  🦴 Bone Mapping:', systemStatus.boneMappingStatus);
        console.log('  🎬 Animation System:', systemStatus.animationStatus);
        console.log('\n📊 Animation Metrics:');
        console.log('  📈 BVH Frames Generated:', systemStatus.metrics.framesGenerated);
        console.log('  🦴 Bones Animated:', systemStatus.metrics.bonesAnimated);
        console.log('  🎬 Animation FPS:', systemStatus.metrics.fpsCounter);
        console.log('  ⚡ Neural Latency:', systemStatus.metrics.neuralLatency, 'ms');

        // Show key log entries
        if (systemStatus.logText && systemStatus.logText !== 'No log available') {
            const logLines = systemStatus.logText.split('\n').filter(line => 
                line.trim() && (
                    line.includes('VRM loaded') || 
                    line.includes('BVH frames') || 
                    line.includes('Applied BVH') ||
                    line.includes('Neural network') ||
                    line.includes('initialized') ||
                    line.includes('Ready') ||
                    line.includes('✅') ||
                    line.includes('❌') ||
                    line.includes('⚠️')
                )
            );
            
            if (logLines.length > 0) {
                console.log('\n📝 Key System Log Entries:');
                console.log('────────────────────────────');
                logLines.slice(-15).forEach(line => {
                    const trimmedLine = line.trim();
                    if (trimmedLine) {
                        console.log('   ', trimmedLine);
                    }
                });
            }
        }

        console.log('\n✅ Working VRM Demo screenshot capture complete!');
        console.log(`📁 Screenshots saved to: ${screenshotDir}`);

        // List created files
        const files = fs.readdirSync(screenshotDir);
        console.log('\n📸 Created Screenshots:');
        files.forEach(file => {
            const filePath = path.join(screenshotDir, file);
            const stats = fs.statSync(filePath);
            console.log(`   ${file} (${(stats.size / 1024).toFixed(1)} KB)`);
        });

    } catch (error) {
        console.error('\n❌ Screenshot capture failed:', error);
        
        // Take error screenshot
        try {
            const page = await browser.newPage();
            await page.goto(`file://${path.resolve(__dirname, 'dev/web_viewer/demos/working_ichika_neural_bvh_integration.html')}`);
            await page.waitForTimeout(5000);
            await page.screenshot({
                path: path.join(screenshotDir, 'error-state.png'),
                fullPage: true
            });
            console.log('📸 Error state screenshot saved');
        } catch (screenshotError) {
            console.error('❌ Error screenshot also failed:', screenshotError.message);
        }
        
        throw error;
    } finally {
        await browser.close();
    }
}

// Run the capture
captureWorkingVRMDemo().catch(console.error);