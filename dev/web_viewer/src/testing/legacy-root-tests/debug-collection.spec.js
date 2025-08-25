const { test, expect } = require('@playwright/test');

test('Debug avatar results collection', async ({ page }) => {
  console.log('🔍 Debugging avatar results collection...');
  
  // Listen to all console messages
  const messages = [];
  page.on('console', msg => {
    messages.push({
      type: msg.type(),
      text: msg.text()
    });
  });
  
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);
  
  // Click the button to run the workload test
  await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
  
  // Wait for some processing
  await page.waitForTimeout(15000);
  
  // Check if avatarInferenceResults object exists
  const hasResults = await page.evaluate(() => {
    return typeof window.avatarInferenceResults !== 'undefined';
  });
  
  console.log('🔍 avatarInferenceResults exists:', hasResults);
  
  if (hasResults) {
    const results = await page.evaluate(() => {
      return {
        totalResults: window.avatarInferenceResults.metadata.totalResults,
        status: window.avatarInferenceResults.metadata.testStatus,
        completedModels: window.avatarInferenceResults.metadata.completedModels,
        resultsCount: window.avatarInferenceResults.results.length
      };
    });
    console.log('🔍 Results summary:', results);
  }
  
  // Check for AI COLLECTED messages
  const aiMessages = messages.filter(msg => msg.text.includes('AVATAR AI COLLECTED'));
  console.log(`🔍 Found ${aiMessages.length} AVATAR AI COLLECTED messages`);
  
  // Show some sample AI messages
  aiMessages.slice(0, 5).forEach((msg, i) => {
    console.log(`🔍 AI Result ${i+1}:`, msg.text.substring(0, 100) + '...');
  });
  
  // Show collection messages
  const collectionMessages = messages.filter(msg => msg.text.includes('Collected result'));
  console.log(`🔍 Found ${collectionMessages.length} collection tracking messages`);
  collectionMessages.slice(0, 5).forEach(msg => console.log('📊', msg.text));
});
