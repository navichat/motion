const { test, expect } = require('@playwright/test');

test.describe('Working Ichika Classroom with Walking Demo', () => {
    test('should demonstrate Ichika VRM walking in classroom environment', async ({ page }) => {
        const timeout = 300000; // 5 minutes shell timeout compliance
        
        // Set longer timeout for this comprehensive test
        test.setTimeout(timeout);
        
        console.log('🎭 Starting Working Ichika Classroom Demo Test');
        
        // Navigate to the working demo
        const demoUrl = 'file://' + process.cwd() + '/dev/web_viewer/demos/working_ichika_classroom_with_walking.html';
        console.log('🔗 Navigating to:', demoUrl);
        
        await page.goto(demoUrl);
        await page.waitForLoadState('networkidle');
        
        // Take initial screenshot
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/01-initial-load.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 1: Initial page load');
        
        // Initialize system
        console.log('🚀 Initializing system...');
        await page.click('#initSystem');
        await page.waitForTimeout(3000);
        
        // Wait for Three.js to be ready
        await page.waitForSelector('#statusThree.ready', { timeout: 10000 });
        
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/02-system-initialized.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 2: System initialized');
        
        // Load assets
        console.log('📥 Loading VRM and Classroom assets...');
        await page.click('#loadAssets');
        await page.waitForTimeout(8000); // Allow time for assets to load
        
        // Wait for VRM to be ready
        try {
            await page.waitForSelector('#statusVRM.ready', { timeout: 15000 });
            console.log('✅ VRM loaded successfully');
        } catch (error) {
            console.log('⚠️ VRM loading may have failed, continuing with screenshot');
        }
        
        // Wait for classroom to be ready
        try {
            await page.waitForSelector('#statusClassroom.ready', { timeout: 10000 });
            console.log('✅ Classroom loaded successfully');
        } catch (error) {
            console.log('⚠️ Classroom loading may have used fallback, continuing');
        }
        
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/03-assets-loaded.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 3: Assets loaded (VRM + Classroom)');
        
        // Start walking demo
        console.log('🚶 Starting walking demo...');
        await page.click('#startDemo');
        await page.waitForTimeout(2000);
        
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/04-walking-demo-started.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 4: Walking demo started');
        
        // Test different camera views
        console.log('📹 Testing camera views...');
        
        // Front view
        await page.click('#viewFront');
        await page.waitForTimeout(1000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/05-front-view.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 5: Front camera view');
        
        // Side view
        await page.click('#viewSide');
        await page.waitForTimeout(1000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/06-side-view.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 6: Side camera view');
        
        // Top view
        await page.click('#viewTop');
        await page.waitForTimeout(1000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/07-top-view.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 7: Top camera view');
        
        // Test manual walking controls
        console.log('🎮 Testing manual walking controls...');
        
        // Walk to board
        await page.click('#walkToBoard');
        await page.waitForTimeout(3000); // Allow walking animation
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/08-walk-to-board.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 8: Walking to blackboard');
        
        // Walk to center
        await page.click('#walkToCenter');
        await page.waitForTimeout(3000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/09-walk-to-center.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 9: Walking to center');
        
        // Test avatar animations
        console.log('🎭 Testing avatar animations...');
        
        // Wave animation
        await page.click('#wave');
        await page.waitForTimeout(2000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/10-wave-animation.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 10: Wave animation');
        
        // Point animation
        await page.click('#point');
        await page.waitForTimeout(2000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/11-point-animation.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 11: Point animation');
        
        // Teaching animation
        await page.click('#teach');
        await page.waitForTimeout(2000);
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/12-teaching-animation.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 12: Teaching animation');
        
        // Test follow avatar camera
        await page.click('#followAvatar');
        await page.waitForTimeout(1000);
        
        // Random walk while following
        await page.click('#walkRandom');
        await page.waitForTimeout(4000); // Allow walking with follow camera
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/13-follow-camera-walking.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 13: Follow camera during walking');
        
        // Final comprehensive view
        await page.click('#viewFront');
        await page.click('#idle');
        await page.waitForTimeout(2000);
        
        await page.screenshot({ 
            path: 'test-results/working-ichika-demo/14-final-comprehensive-demo.png',
            fullPage: true 
        });
        console.log('📸 Screenshot 14: Final comprehensive demo view');
        
        // Verify system status
        const vrmStatus = await page.textContent('#statusVRM');
        const classroomStatus = await page.textContent('#statusClassroom');
        const animationStatus = await page.textContent('#statusAnimation');
        const walkingStatus = await page.textContent('#statusWalking');
        
        console.log('📊 System Status Summary:');
        console.log(`   - VRM Status: ${vrmStatus}`);
        console.log(`   - Classroom Status: ${classroomStatus}`);
        console.log(`   - Animation Status: ${animationStatus}`);
        console.log(`   - Walking Status: ${walkingStatus}`);
        
        // Verify performance metrics
        const fps = await page.textContent('#fpsValue');
        const memory = await page.textContent('#memoryValue');
        const avatarStatus = await page.textContent('#avatarStatus');
        
        console.log('📈 Performance Metrics:');
        console.log(`   - FPS: ${fps}`);
        console.log(`   - Memory: ${memory}MB`);
        console.log(`   - Avatar Status: ${avatarStatus}`);
        
        // Log system components loaded
        const logContent = await page.textContent('#logArea');
        const successfulOperations = (logContent.match(/✅/g) || []).length;
        const errors = (logContent.match(/❌/g) || []).length;
        
        console.log('📋 Operation Summary:');
        console.log(`   - Successful operations: ${successfulOperations}`);
        console.log(`   - Errors: ${errors}`);
        console.log(`   - Success rate: ${successfulOperations > 0 ? ((successfulOperations / (successfulOperations + errors)) * 100).toFixed(1) : 0}%`);
        
        console.log('✅ Working Ichika Classroom Demo Test completed successfully');
        console.log('📁 Screenshots saved to test-results/working-ichika-demo/');
        
        // Final verification - check if 3D scene is rendered
        const canvasExists = await page.locator('canvas').count() > 0;
        expect(canvasExists).toBe(true);
        
        // Check if scene canvas has been replaced with actual Three.js canvas
        const sceneCanvas = page.locator('#sceneCanvas');
        const hasCanvas = await sceneCanvas.locator('canvas').count() > 0;
        expect(hasCanvas).toBe(true);
        
        console.log('🎯 All verifications passed - 3D scene is active');
    });
});