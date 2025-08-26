const { chromium } = require('playwright');

async function captureVRMScreenshots() {
    console.log('🎭 Starting VRM avatar demonstration screenshot capture...');
    
    const browser = await chromium.launch({ 
        headless: true,
        args: ['--no-sandbox', '--disable-dev-shm-usage']
    });
    
    const context = await browser.newContext({
        viewport: { width: 1280, height: 720 }
    });
    
    const page = await context.newPage();
    
    try {
        console.log('📖 Loading conversation system...');
        await page.goto('file://' + __dirname + '/demos/complete_ichika_conversation_system.html');
        
        // Wait for page load
        await page.waitForTimeout(3000);
        
        console.log('📸 Capturing initial page screenshot...');
        await page.screenshot({ 
            path: 'test-results/01-initial-page-load.png', 
            fullPage: true 
        });
        
        // Wait for VRM infrastructure
        console.log('⏳ Waiting for VRM infrastructure...');
        try {
            await page.waitForFunction(() => {
                return window.THREE && 
                       window.AdvancedVRMLoader && 
                       window.VRMBVHAdapter;
            }, { timeout: 20000 });
            
            console.log('✅ VRM infrastructure detected');
        } catch (e) {
            console.log('⚠️ VRM infrastructure timeout, continuing...');
        }
        
        await page.screenshot({ 
            path: 'test-results/02-infrastructure-ready.png', 
            fullPage: true 
        });
        
        // Click initialize button
        console.log('🚀 Initializing VRM system...');
        await page.click('#init-button');
        
        // Wait for initialization with longer timeout for VRM loading
        await page.waitForTimeout(15000);
        
        console.log('📸 Capturing initialized system...');
        await page.screenshot({ 
            path: 'test-results/03-vrm-system-initialized.png', 
            fullPage: true 
        });
        
        // Get system status
        const status = await page.evaluate(() => {
            return {
                hasCanvas: !!document.querySelector('canvas'),
                logs: document.getElementById('log-messages')?.textContent?.slice(-500) || 'no logs',
                statusElements: {
                    scene: document.querySelector('.status-3d')?.textContent || 'unknown',
                    avatar: document.querySelector('.status-avatar')?.textContent || 'unknown',
                    conversation: document.querySelector('.status-conversation')?.textContent || 'unknown'
                },
                vrmInfrastructure: {
                    AdvancedVRMLoader: !!window.AdvancedVRMLoader,
                    VRMBVHAdapter: !!window.VRMBVHAdapter,
                    AvatarBinder: !!window.AvatarBinder,
                    BVHTimeline: !!window.BVHTimeline
                }
            };
        });
        
        console.log('📊 System Status:', JSON.stringify(status, null, 2));
        
        // Test voice if available
        if (await page.isVisible('#test-tts')) {
            console.log('🎤 Testing voice system...');
            await page.click('#test-tts');
            await page.waitForTimeout(3000);
            
            await page.screenshot({ 
                path: 'test-results/04-voice-test.png', 
                fullPage: true 
            });
        }
        
        console.log('✅ VRM system screenshot capture completed successfully!');
        console.log('📂 Screenshots saved to test-results/');
        
    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
        
        // Take error screenshot
        await page.screenshot({ 
            path: 'test-results/error-screenshot.png', 
            fullPage: true 
        });
        
        throw error;
    } finally {
        await browser.close();
    }
}

// Run with proper error handling
captureVRMScreenshots()
    .then(() => {
        console.log('🎉 Screenshot capture completed successfully');
        process.exit(0);
    })
    .catch(error => {
        console.error('💥 Screenshot capture failed:', error);
        process.exit(1);
    });
