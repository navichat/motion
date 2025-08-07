const { test, expect } = require('@playwright/test');

test('Check pool selection logic', async ({ page }) => {
  console.log('🔍 Testing pool selection...');
  
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);
  
  // Listen to console messages
  const messages = [];
  page.on('console', msg => {
    messages.push(msg.text());
  });
  
  // Call the workload test and capture initial messages
  await page.evaluate(() => {
    // Force reload the TaskManager JS by clearing cache and running again
    if (window.runRealWorkloadTest) {
      window.runRealWorkloadTest();
    }
  });
  
  await page.waitForTimeout(8000);
  
  // Look for pool selection messages
  const poolMessages = messages.filter(msg => 
    msg.includes('Potential pools') || 
    msg.includes('webnn') ||
    msg.includes('Checking pool:')
  );
  
  console.log('🔍 Pool-related messages:');
  poolMessages.slice(0, 10).forEach(msg => console.log(msg));
  
  // Check if WebNN pool is being checked
  const webnnChecks = messages.filter(msg => msg.includes('Checking pool: webnn'));
  console.log(`🔍 Found ${webnnChecks.length} WebNN pool checks`);
});
