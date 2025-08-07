/**
 * Simple test to verify WASM worker loading and execution
 */

const { test, expect } = require('@playwright/test');

test('Direct WASM Worker Loading Test', async ({ page }) => {
    // Enable console logging
    page.on('console', msg => console.log(`[${msg.type()}] ${msg.text()}`));
    page.on('pageerror', err => console.error('Page error:', err));

    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html');
    
    // Wait for initial page load
    await page.waitForTimeout(2000);
    
    // Test WASM module loading directly in main thread
    const wasmTestResult = await page.evaluate(async () => {
        try {
            // Try to load the WASM modules script manually
            const script = document.createElement('script');
            script.src = './js/workers/real-wasm-modules.js';
            
            return new Promise((resolve) => {
                script.onload = function() {
                    try {
                        console.log('WASM script loaded, checking RealWasmCompute...');
                        console.log('typeof RealWasmCompute:', typeof RealWasmCompute);
                        
                        if (typeof RealWasmCompute !== 'undefined') {
                            resolve({ success: true, message: 'RealWasmCompute class available' });
                        } else {
                            resolve({ success: false, message: 'RealWasmCompute class not found after script load' });
                        }
                    } catch (error) {
                        resolve({ success: false, message: `Error after script load: ${error.message}` });
                    }
                };
                
                script.onerror = function() {
                    resolve({ success: false, message: 'Failed to load real-wasm-modules.js script' });
                };
                
                document.head.appendChild(script);
            });
        } catch (error) {
            return { success: false, message: `Evaluation error: ${error.message}` };
        }
    });
    
    console.log('WASM Test Result:', wasmTestResult);
    
    // Test creating a worker directly
    const workerTestResult = await page.evaluate(async () => {
        try {
            console.log('Creating WASM worker...');
            const worker = new Worker('./js/workers/wasm-worker-simple-real.js');
            
            return new Promise((resolve) => {
                const timeout = setTimeout(() => {
                    resolve({ success: false, message: 'Worker initialization timeout' });
                }, 10000);
                
                worker.onmessage = function(event) {
                    clearTimeout(timeout);
                    console.log('Worker message:', event.data);
                    
                    if (event.data.type === 'ready') {
                        resolve({ 
                            success: true, 
                            message: 'Worker ready', 
                            capabilities: event.data.capabilities 
                        });
                    } else {
                        resolve({ success: false, message: `Unexpected message: ${event.data.type}` });
                    }
                };
                
                worker.onerror = function(error) {
                    clearTimeout(timeout);
                    resolve({ success: false, message: `Worker error: ${error.message}` });
                };
                
                // Initialize the worker
                worker.postMessage({ type: 'init' });
            });
        } catch (error) {
            return { success: false, message: `Worker creation error: ${error.message}` };
        }
    });
    
    console.log('Worker Test Result:', workerTestResult);
    
    // Summary
    console.log('\n=== WASM LOADING TEST SUMMARY ===');
    console.log('Main Thread WASM Loading:', wasmTestResult.success ? '✅' : '❌', wasmTestResult.message);
    console.log('Worker WASM Loading:', workerTestResult.success ? '✅' : '❌', workerTestResult.message);
    if (workerTestResult.capabilities) {
        console.log('Worker Capabilities:', JSON.stringify(workerTestResult.capabilities, null, 2));
    }
});
