const playwright = require('playwright');
const fs = require('fs');
const path = require('path');

async function captureRestoredDemo() {
    console.log('📸 Starting Restored Ichika VRM Classroom demo capture...');
    
    let browser;
    try {
        // Launch browser
        browser = await playwright.chromium.launch({ headless: true });
        const page = await browser.newPage();
        
        // Set viewport
        await page.setViewportSize({ width: 1400, height: 900 });
        
        console.log('🌐 Loading restored VRM classroom demo...');
        
        // Navigate to the demo
        await page.goto('http://localhost:8080/demos/restored_working_ichika_classroom.html', {
            waitUntil: 'networkidle',
            timeout: 30000
        });
        
        console.log('⏳ Waiting for initial loading...');
        await page.waitForTimeout(5000);
        
        // Take initial screenshot
        console.log('📷 Capturing initial screenshot...');
        await page.screenshot({
            path: 'test-results/restored-ichika-demo/01-initial-page.png',
            fullPage: true
        });
        
        // Wait for loading screen to disappear if it exists
        try {
            await page.waitForSelector('#loading', { state: 'hidden', timeout: 20000 });
            console.log('✅ Loading completed');
        } catch {
            console.log('⚠️ Loading screen timeout or not found, continuing...');
        }
        
        // Take screenshot after loading
        console.log('📷 Capturing loaded screenshot...');
        await page.screenshot({
            path: 'test-results/restored-ichika-demo/02-system-loaded.png',
            fullPage: true
        });
        
        // Wait a bit more for 3D scene initialization
        await page.waitForTimeout(5000);
        
        // Try to interact with controls
        try {
            console.log('🎭 Clicking Load Ichika VRM...');
            await page.click('button:has-text("Load Ichika VRM")');
            await page.waitForTimeout(5000);
            
            console.log('📷 Capturing VRM load attempt...');
            await page.screenshot({
                path: 'test-results/restored-ichika-demo/03-vrm-load-attempt.png',
                fullPage: true
            });
        } catch (error) {
            console.log('⚠️ VRM load button interaction failed:', error.message);
        }
        
        try {
            console.log('🏫 Clicking Load Classroom GLB...');
            await page.click('button:has-text("Load Classroom GLB")');
            await page.waitForTimeout(5000);
            
            console.log('📷 Capturing classroom load attempt...');
            await page.screenshot({
                path: 'test-results/restored-ichika-demo/04-classroom-load-attempt.png',
                fullPage: true
            });
        } catch (error) {
            console.log('⚠️ Classroom load button interaction failed:', error.message);
        }
        
        // Check debug log content
        try {
            const debugContent = await page.$eval('#debug-info', el => el.textContent);
            console.log('🔍 Debug log preview:', debugContent.substring(0, 300));
            
            // Save debug log to file
            fs.writeFileSync('test-results/restored-ichika-demo/debug-log.txt', debugContent);
        } catch (error) {
            console.log('⚠️ Could not read debug log:', error.message);
        }
        
        // Check status indicators
        try {
            const vrmStatus = await page.$eval('#vrm-status', el => el.textContent).catch(() => 'Not found');
            const classroomStatus = await page.$eval('#classroom-status', el => el.textContent).catch(() => 'Not found');
            const animationStatus = await page.$eval('#animation-status', el => el.textContent).catch(() => 'Not found');
            
            console.log('📊 Status Summary:');
            console.log('  VRM:', vrmStatus);
            console.log('  Classroom:', classroomStatus);
            console.log('  Animation:', animationStatus);
            
            // Save status to file
            fs.writeFileSync('test-results/restored-ichika-demo/status-summary.txt', 
                `VRM Status: ${vrmStatus}\nClassroom Status: ${classroomStatus}\nAnimation Status: ${animationStatus}`);
        } catch (error) {
            console.log('⚠️ Could not read status indicators:', error.message);
        }
        
        // Final comprehensive screenshot
        console.log('📷 Capturing final comprehensive screenshot...');
        await page.screenshot({
            path: 'test-results/restored-ichika-demo/05-final-state.png',
            fullPage: true
        });
        
        console.log('✅ Restored demo capture completed successfully!');
        console.log('📁 Screenshots saved to test-results/restored-ichika-demo/');
        
    } catch (error) {
        console.error('❌ Error during capture:', error);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

// Run the capture if this script is executed directly
if (require.main === module) {
    captureRestoredDemo().catch(console.error);
}

module.exports = { captureRestoredDemo };