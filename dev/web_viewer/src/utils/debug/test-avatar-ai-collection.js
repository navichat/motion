const { chromium } = require('playwright');

async function testAvatarAICollection() {
  console.log('🧪 Testing AVATAR AI COLLECTED messages...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const collectedResults = [];
  
  // Listen for the specific messages the test expects
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('AVATAR AI COLLECTED')) {
      console.log(`📝 Found: ${text}`);
      collectedResults.push(text);
    }
  });
  
  try {
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    console.log('✅ Page loaded');
    
    // Wait for page to initialize
    await page.waitForTimeout(3000);
    
    // Click the workload button
    await page.click('button[onclick="runRealWorkloadTest()"]');
    console.log('✅ Button clicked');
    
    // Wait for results
    await page.waitForTimeout(10000);
    
    console.log(`\n📊 Summary: Found ${collectedResults.length} AVATAR AI COLLECTED messages`);
    
    if (collectedResults.length > 0) {
      console.log('🎯 Sample messages:');
      collectedResults.slice(0, 3).forEach((msg, i) => {
        try {
          const jsonPart = msg.substring(msg.indexOf('{'));
          const parsed = JSON.parse(jsonPart);
          console.log(`  ${i+1}. ${parsed.jobType}: executionTime=${parsed.executionTime}ms, hasModelOutput=${!!parsed.modelOutput}`);
        } catch (e) {
          console.log(`  ${i+1}. Raw: ${msg.substring(0, 100)}...`);
        }
      });
    } else {
      console.log('❌ No AVATAR AI COLLECTED messages found');
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
  
  await browser.close();
  console.log('🏁 Test completed');
}

testAvatarAICollection();
