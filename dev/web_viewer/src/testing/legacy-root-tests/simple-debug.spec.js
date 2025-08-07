const { test, expect } = require('@playwright/test');

test('Debug initialization issue', async ({ page }) => {
  console.log('🔍 Starting debug test...');
  
  await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
  
  // Wait for page to load completely
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(5000); // Give it extra time
  
  // Debug what's available on window
  const windowKeys = await page.evaluate(() => {
    return Object.keys(window).filter(key => 
      key.includes('avatar') || 
      key.includes('inference') || 
      key.includes('task') ||
      key.includes('TaskManager')
    );
  });
  
  console.log('🔍 Window keys with avatar/inference/task:', windowKeys);
  
  // Check if avatarInferenceResults exists
  const hasResults = await page.evaluate(() => {
    return typeof window.avatarInferenceResults !== 'undefined';
  });
  
  console.log('🔍 avatarInferenceResults exists:', hasResults);
  
  // Check what functions are available
  const availableFunctions = await page.evaluate(() => {
    return Object.keys(window).filter(key => typeof window[key] === 'function');
  });
  
  console.log('🔍 Available functions:', availableFunctions.slice(0, 10));
  
  // Check if runRealWorkloadTest function exists
  const hasWorkloadTest = await page.evaluate(() => {
    return typeof window.runRealWorkloadTest === 'function';
  });
  
  console.log('🔍 runRealWorkloadTest exists:', hasWorkloadTest);
  
  if (hasWorkloadTest) {
    console.log('🔍 Trying to call runRealWorkloadTest...');
    await page.evaluate(() => {
      window.runRealWorkloadTest();
    });
    
    // Wait and check again
    await page.waitForTimeout(10000);
    
    const hasResultsAfter = await page.evaluate(() => {
      return typeof window.avatarInferenceResults !== 'undefined';
    });
    
    console.log('🔍 avatarInferenceResults exists after call:', hasResultsAfter);
    
    if (hasResultsAfter) {
      const results = await page.evaluate(() => {
        return window.avatarInferenceResults;
      });
      console.log('🔍 Results structure:', results);
    }
  }
});
