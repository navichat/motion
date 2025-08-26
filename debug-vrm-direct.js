const { chromium } = require('playwright');

async function testVRMSystem() {
    console.log('🎭 Starting VRM system diagnostic test...');
    
    const browser = await chromium.launch({ 
        headless: false,
        args: [
            '--disable-web-security',
            '--enable-features=WebGPU,SharedArrayBuffer',
            '--enable-webgl'
        ]
    });
    
    const context = await browser.newContext();
    const page = await context.newPage();

    // Set up console and error monitoring
    const consoleMessages = [];
    const errors = [];

    page.on('console', msg => {
        const message = `[${msg.type()}] ${msg.text()}`;
        console.log(message);
        consoleMessages.push(message);
    });

    page.on('pageerror', error => {
        console.log(`❌ Page Error: ${error}`);
        errors.push(error.toString());
    });

    try {
        console.log('🌐 Loading demo page...');
        await page.goto('http://localhost:8080/demos/complete_ichika_conversation_system.html');
        
        // Wait for page to load
        await page.waitForTimeout(5000);
        console.log('✅ Page loaded');

        // Take initial screenshot
        await page.screenshot({ path: 'debug-01-initial.png', fullPage: true });
        console.log('📸 Initial screenshot saved');

        // Click Initialize System
        console.log('🔄 Clicking Initialize System...');
        await page.click('#init-button');

        // Wait for longer period for VRM loading
        await page.waitForTimeout(20000);

        // Take screenshot after initialization
        await page.screenshot({ path: 'debug-02-after-init.png', fullPage: true });
        console.log('📸 Post-initialization screenshot saved');

        // Check system status
        const status = await page.evaluate(() => {
            const getStatus = (id) => {
                const element = document.getElementById(id);
                return element ? element.textContent.trim() : 'N/A';
            };
            
            return {
                scene3d: getStatus('status-3d-text'),
                avatar: getStatus('status-avatar-text'),
                conversation: getStatus('status-conversation-text'),
                speech: getStatus('status-speech-text')
            };
        });

        console.log('📊 Final system status:', status);

        // Check if VRM file can be fetched
        const vrmCheck = await page.evaluate(async () => {
            try {
                const response = await fetch('/assets/avatars/ichika.vrm');
                return {
                    accessible: response.ok,
                    status: response.status,
                    contentType: response.headers.get('content-type'),
                    size: response.headers.get('content-length')
                };
            } catch (error) {
                return { accessible: false, error: error.message };
            }
        });

        console.log('🎭 VRM file check:', vrmCheck);

        // Check Three.js modules
        const moduleCheck = await page.evaluate(() => {
            return {
                THREE: typeof THREE !== 'undefined',
                AdvancedVRMLoader: typeof AdvancedVRMLoader !== 'undefined',
                VRMBVHAdapter: typeof VRMBVHAdapter !== 'undefined',
                GLTFLoader: typeof THREE?.GLTFLoader !== 'undefined',
                VRMLoaderPlugin: typeof THREE?.VRMLoaderPlugin !== 'undefined'
            };
        });

        console.log('🔧 Module availability:', moduleCheck);

        // If initialization failed, let's manually debug the VRM loading process
        if (status.avatar === 'Not Loaded') {
            console.log('🔍 Avatar not loaded, debugging VRM loading process...');
            
            const debugResult = await page.evaluate(async () => {
                try {
                    // Try manual VRM loading
                    if (typeof THREE !== 'undefined' && typeof AdvancedVRMLoader !== 'undefined') {
                        console.log('Attempting manual VRM load...');
                        
                        const loader = new AdvancedVRMLoader();
                        const result = await loader.loadVRMCharacter('/assets/avatars/ichika.vrm', window.scene || new THREE.Scene());
                        
                        return { success: true, vrmLoaded: !!result };
                    } else {
                        return { success: false, reason: 'Required classes not available' };
                    }
                } catch (error) {
                    return { success: false, error: error.message };
                }
            });

            console.log('🧪 Manual VRM loading test:', debugResult);
        }

        // Take final screenshot
        await page.screenshot({ path: 'debug-03-final.png', fullPage: true });
        console.log('📸 Final screenshot saved');

        console.log('\n📋 DIAGNOSTIC COMPLETE');
        console.log('======================');
        console.log('Console Messages:', consoleMessages.length);
        console.log('Errors:', errors.length);
        console.log('Final Status:', status);

        if (errors.length > 0) {
            console.log('\n❌ ERRORS DETECTED:');
            errors.forEach(err => console.log('  -', err));
        }

        // Keep browser open for 10 seconds for manual inspection
        console.log('⏱️  Keeping browser open for 10 seconds for manual inspection...');
        await page.waitForTimeout(10000);

    } catch (error) {
        console.error('❌ Test failed:', error);
    } finally {
        await browser.close();
    }
}

testVRMSystem().catch(console.error);