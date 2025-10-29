const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Ensure test-results directory exists
const resultsDir = 'test-results/working-ichika-final';
if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
}

async function captureIchikaDemo() {
    console.log('🎭 Starting Ichika Classroom Demo Screenshot Capture');
    
    const browser = await puppeteer.launch({
        headless: false, // Set to false to see what's happening
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--allow-running-insecure-content',
            '--disable-features=VizDisplayCompositor'
        ]
    });
    
    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 1400, height: 900 });
        
        // Navigate to the demo
        const demoUrl = 'http://localhost:8080/demos/working_ichika_classroom_with_walking.html';
        console.log('🔗 Navigating to:', demoUrl);
        
        await page.goto(demoUrl, { 
            waitUntil: 'networkidle2',
            timeout: 60000
        });
        
        // Wait for page to be ready
        await page.waitForTimeout(3000);
        
        // Take initial screenshot
        await page.screenshot({ 
            path: path.join(resultsDir, '01-initial-interface.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 1: Initial interface');
        
        // Initialize system
        console.log('🚀 Initializing system...');
        await page.click('#initSystem');
        await page.waitForTimeout(5000);
        
        // Wait for Three.js status to be ready
        await page.waitForSelector('#statusThree.ready', { timeout: 15000 });
        console.log('✅ Three.js system ready');
        
        await page.screenshot({ 
            path: path.join(resultsDir, '02-system-initialized.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 2: System initialized');
        
        // Load assets
        console.log('📥 Loading assets...');
        await page.click('#loadAssets');
        await page.waitForTimeout(12000); // Give plenty of time for VRM loading
        
        await page.screenshot({ 
            path: path.join(resultsDir, '03-assets-loading.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 3: Assets loading');
        
        // Wait a bit more for VRM to load
        await page.waitForTimeout(8000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '04-assets-loaded.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 4: Assets loaded');
        
        // Start walking demo
        console.log('🚶‍♀️ Starting walking demo...');
        await page.click('#startDemo');
        await page.waitForTimeout(3000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '05-walking-demo-started.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 5: Walking demo started');
        
        // Test walking controls
        console.log('📍 Testing walk to board...');
        await page.click('#walkToBoard');
        await page.waitForTimeout(4000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '06-walk-to-board.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 6: Walk to board');
        
        // Walk to center
        console.log('📍 Testing walk to center...');
        await page.click('#walkToCenter');
        await page.waitForTimeout(4000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '07-walk-to-center.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 7: Walk to center');
        
        // Test animations
        console.log('🎭 Testing wave animation...');
        await page.click('#wave');
        await page.waitForTimeout(2000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '08-wave-animation.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 8: Wave animation');
        
        // Test camera views
        console.log('📹 Testing front view...');
        await page.click('#viewFront');
        await page.waitForTimeout(1000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '09-front-view.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 9: Front view');
        
        // Side view
        console.log('📹 Testing side view...');
        await page.click('#viewSide');
        await page.waitForTimeout(1000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '10-side-view.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 10: Side view');
        
        // Random walk
        console.log('🎲 Testing random walk...');
        await page.click('#walkRandom');
        await page.waitForTimeout(4000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '11-random-walk.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 11: Random walk');
        
        // Final comprehensive screenshot
        await page.click('#walkToCenter');
        await page.waitForTimeout(3000);
        
        await page.screenshot({ 
            path: path.join(resultsDir, '12-final-system.png'),
            fullPage: true 
        });
        console.log('📸 Screenshot 12: Final system view');
        
        // Get system status
        const systemStatus = await page.evaluate(() => {
            return {
                three: document.querySelector('#statusThree')?.textContent,
                vrm: document.querySelector('#statusVRM')?.textContent,
                classroom: document.querySelector('#statusClassroom')?.textContent,
                animation: document.querySelector('#statusAnimation')?.textContent,
                walking: document.querySelector('#statusWalking')?.textContent,
                avatarStatus: document.querySelector('#avatarStatus')?.textContent,
                fps: document.querySelector('#fpsValue')?.textContent
            };
        });
        
        console.log('📊 Final System Status:');
        console.log(`- Three.js: ${systemStatus.three}`);
        console.log(`- VRM: ${systemStatus.vrm}`);
        console.log(`- Classroom: ${systemStatus.classroom}`);
        console.log(`- Animation: ${systemStatus.animation}`);
        console.log(`- Walking: ${systemStatus.walking}`);
        console.log(`- Avatar: ${systemStatus.avatarStatus}`);
        console.log(`- FPS: ${systemStatus.fps}`);
        
        console.log('🎉 Screenshot capture complete!');
        console.log(`📁 Screenshots saved to: ${resultsDir}`);
        
    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
    } finally {
        await browser.close();
    }
}

// Run the capture
captureIchikaDemo().catch(console.error);