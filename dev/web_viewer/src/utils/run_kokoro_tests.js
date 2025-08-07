const { chromium } = require('playwright');
const path = require('path');

(async () => {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    page.setDefaultNavigationTimeout(1200000); // 20 minutes
    page.setDefaultTimeout(1200000); // 20 minutes for all actions

    const url = `http://localhost:8080/automated_kokoro_vad_whisper_test.html`;

    console.log(`Navigating to: ${url}`);
    page.on('console', msg => {
        for (let i = 0; i < msg.args().length; ++i) {
            console.log(`[PAGE CONSOLE]: ${msg.text()}`);
        }
    });
    page.on('pageerror', error => {
        console.error(`[PAGE ERROR]: ${error.message}`);
    });
    page.on('requestfailed', request => {
        console.error(`[REQUEST FAILED]: ${request.url()} ${request.failure().errorText}`);
    });

    await page.goto(url, { waitUntil: 'networkidle' });

    console.log('Clicking "Start Tests" button...');
    await page.click('#startButton');

    console.log('Waiting for tests to complete...');

    // Wait for the test summary to update, indicating all tests have run
    await page.waitForFunction(() => {
        const total = parseInt(document.getElementById('totalTests').textContent);
        const passed = parseInt(document.getElementById('passedTests').textContent);
        const failed = parseInt(document.getElementById('failedTests').textContent);
        return total > 0 && (passed + failed === total);
    }, { timeout: 1200000 }); // Increased timeout to 20 minutes for model loading

    const totalTests = await page.evaluate(() => document.getElementById('totalTests').textContent);
    const passedTests = await page.evaluate(() => document.getElementById('passedTests').textContent);
    const failedTests = await page.evaluate(() => document.getElementById('failedTests').textContent);

    console.log('\n--- Test Summary ---');
    console.log(`Total Tests: ${totalTests}`);
    console.log(`Passed: ${passedTests}`);
    console.log(`Failed: ${failedTests}`);
    console.log('--------------------');

    if (parseInt(failedTests) > 0) {
        console.error('Some tests failed!');
        process.exitCode = 1; // Indicate failure
    } else {
        console.log('All tests passed!');
    }

    // Optionally, capture console logs from the page
    page.on('console', msg => console.log(`[PAGE CONSOLE]: ${msg.text()}`));

    await browser.close();
})();
