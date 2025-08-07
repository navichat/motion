const { test, expect } = require('@playwright/test');

test('Debug console output during workload test', async ({ page }) => {
  console.log('🔍 Starting console debug test...');
  
  // Listen to console messages
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
  
  console.log('🔍 Calling runRealWorkloadTest...');
  
  await page.evaluate(() => {
    window.runRealWorkloadTest();
  });
  
  // Wait for tasks to process
  await page.waitForTimeout(15000);
  
  // Log recent console messages
  console.log('🔍 Recent console messages:');
  const recentMessages = messages.slice(-20).map(msg => `[${msg.type}] ${msg.text}`);
  recentMessages.forEach(msg => console.log(msg));
  
  // Check if we have any AI COLLECTED messages
  const aiMessages = messages.filter(msg => msg.text.includes('AVATAR AI COLLECTED'));
  console.log(`🔍 Found ${aiMessages.length} AVATAR AI COLLECTED messages`);
  
  // Check for any errors
  const errorMessages = messages.filter(msg => msg.type === 'error');
  console.log(`🔍 Found ${errorMessages.length} error messages`);
  errorMessages.forEach(msg => console.log(`❌ Error: ${msg.text}`));
});
