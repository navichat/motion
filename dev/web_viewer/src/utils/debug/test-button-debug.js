const { chromium } = require('playwright');

async function testButtonClick() {
  console.log('🧪 Testing button click and function call...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Listen for ALL console messages
  page.on('console', msg => {
    const text = msg.text();
    console.log(`[CONSOLE]: ${text}`);
  });
  
  page.on('pageerror', error => {
    console.log(`[PAGE ERROR]: ${error.message}`);
  });
  
  try {
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    console.log('✅ Page loaded');
    
    // Wait for page to initialize
    await page.waitForTimeout(5000);
    
    // Check if runRealWorkloadTest exists
    const functionExists = await page.evaluate(() => {
      return typeof window.runRealWorkloadTest === 'function';
    });
    
    console.log(`🔍 runRealWorkloadTest function exists: ${functionExists}`);
    
    if (functionExists) {
      // Click the button using the same selector as the test
      console.log('🖱️ Clicking button...');
      await page.getByRole('button', { name: '🚀 Real WASM/GPU/WebNN Workload' }).click();
      console.log('✅ Button clicked');
      
      // Wait and listen for results
      await page.waitForTimeout(10000);
    } else {
      console.log('❌ Function does not exist');
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
  
  await browser.close();
  console.log('🏁 Test completed');
}

testButtonClick();
