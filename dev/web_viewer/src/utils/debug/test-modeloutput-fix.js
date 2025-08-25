const { chromium } = require('playwright');

(async () => {
  console.log('🧪 Testing modelOutput fix...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Listen for console messages and errors
  page.on('console', msg => console.log(`[PAGE]: ${msg.text()}`));
  page.on('pageerror', err => console.log(`[PAGE ERROR]: ${err.message}`));

  // Start a simple server
  const { spawn } = require('child_process');
  const server = spawn('python3', ['-m', 'http.server', '8001'], { 
    cwd: '/home/barberb/motion',
    stdio: 'pipe'
  });
  
  // Wait for server to start
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  try {
    await page.goto('http://localhost:8001/dev/web_viewer/task-manager-demo.html');
    console.log('✅ Page loaded successfully');
    
    // Wait for the page to initialize
    await page.waitForTimeout(5000);
    
    // Check what buttons are available
    const buttons = await page.evaluate(() => {
      const allButtons = Array.from(document.querySelectorAll('button'));
      return allButtons.map(btn => ({
        text: btn.textContent.trim(),
        onclick: btn.getAttribute('onclick')
      })).slice(0, 10);
    });
    
    console.log('� Available buttons:');
    buttons.forEach((btn, i) => {
      console.log(`  ${i+1}. Text: "${btn.text}", onclick: ${btn.onclick}`);
    });
    
    // Try to find and click any real workload button
    const realWorkloadBtn = buttons.find(btn => 
      btn.text.includes('Real') && btn.text.includes('Workload')
    );
    
    if (realWorkloadBtn) {
      console.log(`🖱️ Found workload button: "${realWorkloadBtn.text}"`);
      await page.click(`button[onclick="${realWorkloadBtn.onclick}"]`);
      console.log('✅ Button clicked, waiting for AI inference...');
      
      // Wait for some results
      await page.waitForTimeout(8000);
    } else {
      console.log('❌ No real workload button found');
    }
    
    // Look for AI model results
    const results = await page.evaluate(() => {
      if (window.globalResults && window.globalResults.length > 0) {
        return window.globalResults.slice(0, 10).map(result => ({
          jobType: result.jobType,
          hasModelOutput: !!result.modelOutput,
          usingRealModel: result.usingRealModel,
          workerType: result.workerType,
          executionProvider: result.executionProvider,
          modelOutputType: typeof result.modelOutput
        }));
      }
      return null;
    });
    
    if (results) {
      console.log(`🎯 Found ${results.length} AI model results:`);
      results.forEach((result, i) => {
        console.log(`  ${i+1}. ${result.jobType}: modelOutput=${result.hasModelOutput}(${result.modelOutputType}), realModel=${result.usingRealModel}, worker=${result.workerType}, provider=${result.executionProvider}`);
      });
      
      const hasModelOutput = results.filter(r => r.hasModelOutput).length;
      console.log(`📊 Summary: ${hasModelOutput}/${results.length} results have modelOutput`);
    } else {
      console.log('❌ No results found in globalResults');
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
  
  await browser.close();
  server.kill();
  console.log('🏁 Test completed');
})();
