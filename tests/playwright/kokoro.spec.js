import { test, expect } from '@playwright/test';

test('Kokoro TTS, VAD, and Whisper integration test', async ({ page }) => {
    page.on('console', msg => {
        console.log(`Browser console [${msg.type()}]: ${msg.text()}`);
    });

    // Navigate to the test HTML file
    await page.goto('http://localhost:8080/dev/web_viewer/automated_kokoro_vad_whisper_test.html');

    // Wait for a fixed time to allow the page to process
    await page.waitForTimeout(5000); // Wait for 5 seconds, as no heavy loading is expected

    const testStatusText = await page.locator('#test-status').textContent();
    const transcriptionOutputText = await page.locator('#transcription-output').textContent();

    console.log(`Test Status: ${testStatusText}`);
    console.log(`Transcription Output: ${transcriptionOutputText}`);

    // Expect the test to pass based on the text content
    expect(testStatusText).toContain('PASS');
});
