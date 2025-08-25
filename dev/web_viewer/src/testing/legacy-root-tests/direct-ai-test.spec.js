import { test, expect } from '@playwright/test';

test('Direct AI Model Test', async ({ page }) => {
  console.log('🚀 Starting direct AI model test...');
  
  const modelOutputs = [];
  
  // Listen for console messages
  page.on('console', msg => {
    const text = msg.text();
    console.log(`[Browser Console] ${text}`);
    
    // Capture AI model results
    if (text.startsWith('AI_MODEL_TEST_RESULTS:')) {
      try {
        const results = JSON.parse(text.replace('AI_MODEL_TEST_RESULTS:', ''));
        modelOutputs.push(...results);
      } catch (e) {
        console.log('Failed to parse AI model results:', e);
      }
    }
    
    // Also capture individual model messages
    if (text.includes('model loaded successfully') || 
        text.includes('inference completed successfully') ||
        text.includes('failed:')) {
      modelOutputs.push({
        type: 'direct_test',
        message: text,
        timestamp: new Date().toISOString()
      });
    }
  });
  
  // Navigate to the direct test page
  await page.goto('http://localhost:8000/direct_ai_test.html');
  
  // Wait for tests to complete (longer timeout for multiple models)
  console.log('⏳ Waiting for AI model tests to complete...');
  await page.waitForTimeout(45000); // 45 seconds for all model tests
  
  // Additional wait for final results
  await page.waitForFunction(() => window.aiModelTestResults && window.aiModelTestResults.length > 0, { timeout: 10000 });
  
  // Check if results were collected
  const results = await page.evaluate(() => window.aiModelTestResults);
  
  console.log(`📊 Direct AI Model Test Results:`);
  console.log(`Total outputs collected: ${modelOutputs.length}`);
  
  if (results && results.length > 0) {
    console.log(`Direct test results: ${results.length} models tested`);
    const successful = results.filter(r => r.success).length;
    console.log(`Successful models: ${successful}/${results.length}`);
    
    results.forEach(result => {
      console.log(`  ${result.modelType}: ${result.success ? '✅' : '❌'} ${result.success ? 'success' : result.error}`);
    });
  } else {
    console.log('No direct test results found in window object');
  }
  
  // Output all collected messages
  if (modelOutputs.length > 0) {
    console.log('\n📝 All collected AI model outputs:');
    modelOutputs.forEach((output, index) => {
      console.log(`${index + 1}. ${JSON.stringify(output)}`);
    });
  }
  
  // Expect at least some outputs were collected
  expect(modelOutputs.length).toBeGreaterThan(0);
  
  console.log('✅ Direct AI model test completed');
});
