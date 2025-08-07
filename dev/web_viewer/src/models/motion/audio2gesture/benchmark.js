import { chromium } from 'playwright';

(async () => {
    const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-gl=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--use-angle=vulkan'] }); // Enable WebGPU in headless mode
    const page = await browser.newPage();

    // Listen for console messages
    page.on('console', msg => console.log(`PAGE CONSOLE: ${msg.text()}`));

    // Listen for page errors
    page.on('pageerror', err => console.error(`PAGE ERROR: ${err.message}`));

    const demoPageUrl = 'http://localhost:8080/dev/web_viewer/audio2gesture/audio2gesture_optimization_demo.html';
    console.log(`Navigating to ${demoPageUrl}`);
    await page.goto(demoPageUrl, { waitUntil: 'networkidle' });

    console.log('Page loaded. Initializing generator...');

    // Wait for the generator to initialize
    // Wait for the testOutput element to exist in the DOM
    console.log('Waiting for #testOutput element to be visible...');
    await page.waitForSelector('#testOutput', { timeout: 120000 }); // Increased timeout

    console.log('Waiting for generator initialization message in #testOutput...');
    // Now wait for its content to indicate initialization
    await page.waitForFunction(() => {
        const testOutputElement = document.getElementById('testOutput');
        // Ensure the element exists before trying to read its textContent
        if (testOutputElement) {
            const log = testOutputElement.textContent;
            return log.includes('Generator initialized successfully!') || log.includes('Fallback failed');
        }
        return false; // Element not found yet, keep waiting
    }, { timeout: 180000 }); // Increased timeout for model loading to 3 minutes

    console.log('Generator initialized (or fallback attempted).');

    const backends = ['webgpu', 'webnn', 'wasm', 'cpu'];
    const results = {};

    for (const backend of backends) {
        console.log(`
--- Testing Backend: ${backend.toUpperCase()} ---`);

        // Select the backend
        await page.selectOption('#backendSelect', backend);
        console.log(`Selected backend: ${backend}`);

        // Click the performance comparison button
        const compareButton = page.locator('#compareBtn');
        await compareButton.click();
        console.log('Clicked "Run Performance Comparison". Waiting for completion...');

        // Wait for the comparison to complete
        // The button will be disabled during comparison and re-enabled afterwards
        console.log('Waiting for compare button to be enabled...');
        await page.waitForFunction(selector => {
            const button = document.querySelector(selector);
            return button && !button.disabled;
        }, '#compareBtn', { timeout: 120000 }); // Wait for the button to be enabled
        console.log('Compare button is now enabled.');
        console.log('Performance comparison completed.');

        await page.waitForSelector('#comparisonCharts .chart-fill', { timeout: 120000 });

        // Take a screenshot before extracting metrics for debugging
        await page.screenshot({ path: 'benchmark_results.png' });
        console.log('Screenshot saved to benchmark_results.png');

        // Extract metrics
        const standardFps = await page.textContent('#comparisonCharts .chart-bar:nth-child(2) .chart-fill', { timeout: 120000 });
        const optimizedFps = await page.textContent('#comparisonCharts .chart-bar:nth-child(3) .chart-fill', { timeout: 120000 });
        const speedup = await page.textContent('#comparisonCharts .chart-bar:nth-child(4) .chart-fill', { timeout: 120000 });

        results[backend] = {
            standardFps: standardFps ? standardFps.trim() : 'N/A',
            optimizedFps: optimizedFps ? optimizedFps.trim() : 'N/A',
            speedup: speedup ? speedup.trim() : 'N/A',
        };

        console.log(`Results for ${backend.toUpperCase()}:`);
        console.log(`  Standard FPS: ${results[backend].standardFps}`);
        console.log(`  Optimized FPS: ${results[backend].optimizedFps}`);
        console.log(`  Speedup: ${results[backend].speedup}`);
    }

    console.log('\n--- All Benchmarks Complete ---');
    console.log('Summary of Results:');
    console.table(results);

    await browser.close();
})();