/**
 * Quick test to verify quantization improvements are working
 */

const { test, expect } = require('@playwright/test');

test('Test Quantization Performance Improvements', async ({ page }) => {
    console.log('🧠 Testing quantization performance improvements...');
    
    // Navigate to the demo page
    await page.goto('/dev/web_viewer/task-manager-demo.html');
    
    // Wait for page to load
    await page.waitForTimeout(2000);
    
    // Listen for console messages
    const quantizationResults = [];
    
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('Selected') && text.includes('quantization')) {
            console.log(`🚀 ${text}`);
            quantizationResults.push(text);
        }
        if (text.includes('AVATAR AI COLLECTED') && (text.includes('TinyLlama') || text.includes('DiabloGPT'))) {
            console.log(`✅ Language model result: ${text.substring(0, 200)}...`);
            try {
                const result = JSON.parse(text.replace('AVATAR AI COLLECTED ', ''));
                if (result.modelOutput && result.modelOutput.precision_mode) {
                    console.log(`🔧 Precision mode: ${result.modelOutput.precision_mode}`);
                    console.log(`⚡ Speed improvement: ${result.modelOutput.speed_improvement_percent}%`);
                    console.log(`💾 Memory reduction: ${result.modelOutput.memory_reduction_factor}x`);
                }
            } catch (e) {
                console.log('Could not parse result details');
            }
        }
    });
    
    // Click the button to run language model tests
    try {
        await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
        console.log('✅ Button clicked successfully');
    } catch (error) {
        console.log('❌ Button not found, trying alternative approach');
        // Look for any button and click it
        const buttons = await page.locator('button').all();
        console.log(`Found ${buttons.length} buttons`);
        for (let i = 0; i < buttons.length; i++) {
            const buttonText = await buttons[i].textContent();
            console.log(`Button ${i}: "${buttonText}"`);
            if (buttonText && buttonText.includes('Real')) {
                await buttons[i].click();
                console.log(`✅ Clicked button: "${buttonText}"`);
                break;
            }
        }
    }
    
    // Wait for quantization selection and model execution
    console.log('⏰ Waiting for quantized language model execution...');
    await page.waitForTimeout(10000); // Wait 10 seconds
    
    // Verify we got quantization improvements
    expect(quantizationResults.length).toBeGreaterThan(0);
    console.log(`🎉 Test completed! Found ${quantizationResults.length} quantization optimizations`);
});
