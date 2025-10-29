const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function captureWorkingVRMDemo() {
    console.log('🎭 Starting Working Ichika VRM Neural BVH Integration Demo Screenshot Capture...');
    
    const browser = await puppeteer.launch({
        headless: false,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',
            '--allow-running-insecure-content',
            '--disable-features=VizDisplayCompositor',
            '--enable-webgl',
            '--use-gl=swiftshader',
            '--enable-accelerated-2d-canvas'
        ],
        defaultViewport: {
            width: 1400,
            height: 900
        }
    });

    try {
        const page = await browser.newPage();
        
        // Enable console logging
        page.on('console', msg => {
            const type = msg.type();
            const text = msg.text();
            if (type === 'error') {
                console.log('❌ Browser console error:', text);
            } else if (type === 'warning') {
                console.log('⚠️ Browser console warning:', text);
            } else {
                console.log(`📋 Browser console [${type}]:`, text);
            }
        });

        // Navigate to the demo
        const demoPath = path.resolve(__dirname, '../dev/web_viewer/demos/working_ichika_neural_bvh_integration.html');
        const demoUrl = `file://${demoPath}`;
        
        console.log(`🌐 Navigating to: ${demoUrl}`);
        await page.goto(demoUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });

        // Wait for system initialization
        console.log('⏳ Waiting for system to initialize...');
        await page.waitForTimeout(8000);

        // Wait for VRM model to load (look for status indicators)
        try {
            await page.waitForFunction(() => {
                const vrmStatus = document.querySelector('#vrm-status');
                return vrmStatus && (vrmStatus.textContent === 'READY' || vrmStatus.textContent === 'ERROR');
            }, { timeout: 20000 });
        } catch (error) {
            console.log('⚠️ VRM status timeout, continuing with screenshot...');
        }

        // Take screenshot after loading overlay is hidden
        await page.waitForFunction(() => {
            const overlay = document.querySelector('#loading-overlay');
            return !overlay || overlay.style.display === 'none' || overlay.style.opacity === '0';
        }, { timeout: 15000 });

        console.log('📸 Taking initial system screenshot...');
        await page.screenshot({
            path: path.join(__dirname, 'test-results/working-vrm-neural-bvh-demo-initialized.png'),
            fullPage: false
        });

        // Test neural BVH generation
        console.log('🧠 Testing neural BVH generation...');
        await page.click('#test-neural-bvh');
        await page.waitForTimeout(3000);

        console.log('📸 Taking neural BVH test screenshot...');
        await page.screenshot({
            path: path.join(__dirname, 'test-results/working-vrm-neural-bvh-demo-bvh-test.png'),
            fullPage: false
        });

        // Test gesture animation if available
        try {
            await page.click('#play-gesture');
            await page.waitForTimeout(2000);
            
            console.log('📸 Taking gesture animation screenshot...');
            await page.screenshot({
                path: path.join(__dirname, 'test-results/working-vrm-neural-bvh-demo-animation.png'),
                fullPage: false
            });
        } catch (error) {
            console.log('⚠️ Gesture animation not available');
        }

        // Capture final state
        console.log('📸 Taking final state screenshot...');
        await page.screenshot({
            path: path.join(__dirname, 'test-results/working-vrm-neural-bvh-demo-final.png'),
            fullPage: false
        });

        // Get system status from the page
        const systemStatus = await page.evaluate(() => {
            const getStatus = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent : 'UNKNOWN';
            };
            
            return {
                vrmStatus: getStatus('vrm-status'),
                bvhAdapterStatus: getStatus('bvh-adapter-status'),
                sceneStatus: getStatus('scene-status'),
                audio2gestureStatus: getStatus('audio2gesture-status'),
                deepmimicStatus: getStatus('deepmimic-status'),
                framesApplied: getStatus('frames-applied'),
                activeBones: getStatus('active-bones')
            };
        });

        console.log('📊 System Status Report:');
        console.log('  VRM Model:', systemStatus.vrmStatus);
        console.log('  BVH Adapter:', systemStatus.bvhAdapterStatus);
        console.log('  3D Scene:', systemStatus.sceneStatus);
        console.log('  Audio2Gesture:', systemStatus.audio2gestureStatus);
        console.log('  DeepMimic:', systemStatus.deepmimicStatus);
        console.log('  BVH Frames Applied:', systemStatus.framesApplied);
        console.log('  Active Bones:', systemStatus.activeBones);

        console.log('✅ Screenshot capture complete!');

    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
    } finally {
        await browser.close();
    }
}

// Create test results directory
const testResultsDir = path.join(__dirname, 'test-results');
if (!fs.existsSync(testResultsDir)) {
    fs.mkdirSync(testResultsDir, { recursive: true });
}

captureWorkingVRMDemo().catch(console.error);