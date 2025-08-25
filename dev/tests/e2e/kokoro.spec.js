import { test, expect } from '@playwright/test';

test('Kokoro TTS Headless Test - should load Kokoro TTS and synthesize text', async ({ page }) => {
        // Navigate to the test page
        await page.goto('/web_viewer/test_kokoro.html');

        // Wait for the KokoroModule to be initialized on the page
        await page.waitForFunction(() => window.kokoroModule !== undefined);

        // Load the Kokoro model
        const loadSuccess = await page.evaluate(() => window.loadKokoro());
        expect(loadSuccess).toBe(true);

        // Synthesize text and check if audio data is returned
        const synthesizeSuccess = await page.evaluate((text) => window.synthesizeText(text), 'Hello, this is a test of Kokoro TTS.');
        expect(synthesizeSuccess).toBe(true);
    });