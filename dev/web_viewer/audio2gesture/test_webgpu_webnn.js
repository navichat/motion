import { chromium } from 'playwright';

(async () => {
    console.log('Launching Chromium with WebGPU and WebNN flags...');
    const browser = await chromium.launch({
        headless: false, // Set to false to see the browser UI
        args: [
            '--enable-features=WebGPU,UnsafeWebGPU',
            '--enable-unsafe-webgpu',
            '--enable-features=WebNN',
            '--enable-features=PartitionWebGPUOnGPUProcess', // Optional: for better isolation
            '--disable-gpu-sandbox' // May be needed on some Linux systems
        ]
    });

    const page = await browser.newPage();

    // Listen for console messages from the page
    page.on('console', msg => console.log(`PAGE CONSOLE: ${msg.text()}`));

    // Listen for page errors
    page.on('pageerror', err => console.error(`PAGE ERROR: ${err.message}`));

    // Navigate to a simple page or your demo page
    const demoPageUrl = 'http://localhost:8080/web_viewer/audio2gesture/audio2gesture_optimization_demo.html';
    console.log(`Navigating to ${demoPageUrl}`);
    try {
        await page.goto(demoPageUrl, { waitUntil: 'domcontentloaded' });
        console.log('Page loaded. Check browser console for WebGPU/WebNN status.');
        // You might want to add a short delay here to allow the page's scripts to run
        await page.waitForTimeout(5000); 
    } catch (error) {
        console.error(`Navigation failed: ${error.message}`);
    } finally {
        await browser.close();
        console.log('Browser closed.');
    }
})();
