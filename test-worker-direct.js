// Quick test to verify our modelOutput fixes work
const { chromium } = require('playwright');

async function testModelOutputFix() {
  console.log('🧪 Testing modelOutput fix in workers...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Set up console logging
  const results = [];
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('modelOutput') || text.includes('AI Model') || text.includes('COMPLETED')) {
      console.log(`[CONSOLE]: ${text}`);
      results.push(text);
    }
  });
  
  try {
    // Create a test page that directly tests workers
    await page.setContent(`
      <!DOCTYPE html>
      <html>
      <head><title>Worker Test</title></head>
      <body>
        <h1>Testing Workers with modelOutput</h1>
        <div id="results"></div>
        <script>
          async function testWorkers() {
            console.log('🚀 Starting worker test...');
            
            // Test WebNN worker
            const webnnWorker = new Worker('http://localhost:8002/dev/web_viewer/js/workers/webnn-worker-simple.js');
            webnnWorker.postMessage({
              type: 'task',
              taskId: 'test-1',
              jobType: 'FaceFormer',
              taskData: { complexity: 1, duration: 1000 }
            });
            
            webnnWorker.onmessage = function(e) {
              if (e.data.type === 'completed') {
                console.log('WebNN Worker Result:', JSON.stringify(e.data.result, null, 2));
                console.log('Has modelOutput:', !!e.data.result.modelOutput);
              }
            };
            
            // Test CPU worker
            const cpuWorker = new Worker('http://localhost:8002/dev/web_viewer/js/workers/cpu-worker-simple.js');
            cpuWorker.postMessage({
              type: 'task',
              taskId: 'test-2',
              jobType: 'TinyLlama',
              taskData: { complexity: 1, duration: 1000 }
            });
            
            cpuWorker.onmessage = function(e) {
              if (e.data.type === 'completed') {
                console.log('CPU Worker Result:', JSON.stringify(e.data.result, null, 2));
                console.log('Has modelOutput:', !!e.data.result.modelOutput);
              }
            };
          }
          
          // Start the test
          testWorkers();
        </script>
      </body>
      </html>
    `);
    
    // Wait for workers to complete
    await page.waitForTimeout(5000);
    
    console.log(`✅ Test completed. Found ${results.length} relevant console messages.`);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
  
  await browser.close();
}

testModelOutputFix();
