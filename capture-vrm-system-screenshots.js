const { chromium } = require('playwright');

async function captureVRMSystemScreenshots() {
    console.log('🎭 Starting VRM System Screenshot Capture...');
    
    const browser = await chromium.launch({ 
        headless: false,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const context = await browser.newContext();
    const page = await context.newPage();
    
    // Enhanced console monitoring
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('VRM') || text.includes('avatar') || text.includes('loaded') || 
            text.includes('✅') || text.includes('❌') || text.includes('system')) {
            console.log(`🎭 System: ${text}`);
        }
    });

    page.on('pageerror', error => {
        console.log(`❌ Error: ${error.message}`);
    });

    try {
        console.log('📱 Loading Complete Ichika Conversation System...');
        await page.goto('http://localhost:8080/demos/complete_ichika_conversation_system.html');
        await page.waitForTimeout(5000);
        
        // Take initial screenshot
        console.log('📸 Capturing initial system state...');
        await page.screenshot({ 
            path: 'test-results/01-initial-system-load.png',
            fullPage: true,
            quality: 90
        });
        
        // Initialize the system
        console.log('🔄 Initializing VRM system...');
        await page.click('#init-button');
        
        // Wait for system initialization with monitoring
        let attempts = 0;
        const maxAttempts = 15;
        
        while (attempts < maxAttempts) {
            await page.waitForTimeout(4000);
            attempts++;
            
            // Check system status
            const sceneStatus = await page.textContent('#status-3d-text').catch(() => 'Unknown');
            const avatarStatus = await page.textContent('#status-avatar-text').catch(() => 'Unknown');
            const conversationStatus = await page.textContent('#status-conversation-text').catch(() => 'Unknown');
            const speechStatus = await page.textContent('#status-speech-text').catch(() => 'Unknown');
            
            console.log(`📊 System Status (${attempts}/${maxAttempts}):`);
            console.log(`   🌐 3D Scene: ${sceneStatus}`);
            console.log(`   👤 Avatar: ${avatarStatus}`);
            console.log(`   💬 Conversation: ${conversationStatus}`);
            console.log(`   🎤 Speech Sync: ${speechStatus}`);
            
            // Take progress screenshot
            if (attempts % 3 === 0) {
                const filename = `test-results/02-progress-${attempts}.png`;
                await page.screenshot({ 
                    path: filename,
                    fullPage: true,
                    quality: 90
                });
                console.log(`📸 Progress screenshot: ${filename}`);
            }
            
            // Check if we have good enough state to proceed
            if (sceneStatus === 'Loaded') {
                console.log('✅ 3D Scene loaded successfully!');
                break;
            }
        }
        
        // Take post-initialization screenshot
        console.log('📸 Capturing post-initialization state...');
        await page.screenshot({ 
            path: 'test-results/03-post-initialization.png',
            fullPage: true,
            quality: 95
        });
        
        // Test voice system
        console.log('🔊 Testing voice system...');
        try {
            await page.click('#test-tts');
            await page.waitForTimeout(3000);
            await page.screenshot({ 
                path: 'test-results/04-voice-test.png',
                fullPage: true,
                quality: 95
            });
            console.log('📸 Voice test screenshot captured');
        } catch (e) {
            console.log('⚠️ Voice test not available');
        }
        
        // Test animation system
        console.log('💃 Testing animation system...');
        try {
            await page.click('#test-animation');
            await page.waitForTimeout(3000);
            await page.screenshot({ 
                path: 'test-results/05-animation-test.png',
                fullPage: true,
                quality: 95
            });
            console.log('📸 Animation test screenshot captured');
        } catch (e) {
            console.log('⚠️ Animation test not available');
        }
        
        // Capture 3D scene specifically
        console.log('🎮 Capturing 3D scene detail...');
        try {
            const sceneElement = await page.locator('#scene-container').first();
            await sceneElement.screenshot({ 
                path: 'test-results/06-3d-scene-detail.png',
                quality: 95
            });
            console.log('📸 3D scene detail captured');
        } catch (e) {
            console.log('⚠️ 3D scene capture not available');
        }
        
        // Final comprehensive screenshot
        console.log('📸 Capturing final comprehensive view...');
        await page.screenshot({ 
            path: 'test-results/07-final-working-system.png',
            fullPage: true,
            quality: 95
        });
        
        console.log('✅ VRM System Screenshot Capture Complete!');
        console.log('Screenshots saved to test-results/ directory');
        
    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
    } finally {
        await browser.close();
    }
}

// Run the screenshot capture
if (require.main === module) {
    captureVRMSystemScreenshots().catch(console.error);
}

module.exports = { captureVRMSystemScreenshots };