const { test, expect } = require('@playwright/test');

test('Verify Real Backend Usage vs Fake Labels', async ({ page }) => {
    console.log('\n=== BACKEND VERIFICATION TEST ===');
    
    // Navigate to the task manager demo
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html', {
        waitUntil: 'networkidle'
    });

    // Check WebNN availability
    const webnnAvailable = await page.evaluate(async () => {
        return 'ml' in navigator;
    });
    console.log(`WebNN API Available: ${webnnAvailable}`);

    // Check WebGPU availability
    const webgpuAvailable = await page.evaluate(async () => {
        return 'gpu' in navigator;
    });
    console.log(`WebGPU API Available: ${webgpuAvailable}`);

    // Check WebAssembly availability
    const wasmAvailable = await page.evaluate(async () => {
        return 'WebAssembly' in window;
    });
    console.log(`WebAssembly API Available: ${wasmAvailable}`);

    // Try to actually initialize WebGPU
    const webgpuInit = await page.evaluate(async () => {
        if (!navigator.gpu) return { success: false, error: 'WebGPU not available' };
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return { success: false, error: 'No WebGPU adapter' };
            
            const device = await adapter.requestDevice();
            return { 
                success: true, 
                limits: Object.keys(device.limits),
                features: Array.from(device.features)
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });
    console.log(`WebGPU Initialization: ${JSON.stringify(webgpuInit, null, 2)}`);

    // Try to actually initialize WebNN
    const webnnInit = await page.evaluate(async () => {
        if (!navigator.ml) return { success: false, error: 'WebNN not available' };
        
        try {
            const context = await navigator.ml.createContext();
            return { success: true, context: !!context };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });
    console.log(`WebNN Initialization: ${JSON.stringify(webnnInit, null, 2)}`);

    // Try to create a simple WASM module
    const wasmInit = await page.evaluate(async () => {
        if (!window.WebAssembly) return { success: false, error: 'WebAssembly not available' };
        
        try {
            // Simple WASM module that adds two numbers
            const wasmCode = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01,
                0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07,
                0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, 0x0a, 0x09, 0x01,
                0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b
            ]);
            
            const wasmModule = await WebAssembly.instantiate(wasmCode);
            const result = wasmModule.instance.exports.add(5, 3);
            
            return { success: true, result: result, supportsWasm: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });
    console.log(`WebAssembly Initialization: ${JSON.stringify(wasmInit, null, 2)}`);

    console.log('\n=== WORKER EXECUTION TEST ===');

    // Capture console messages to see actual execution
    const consoleMessages = [];
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('AVATAR AI COLLECTED') || text.includes('Worker') || text.includes('execution')) {
            consoleMessages.push(text);
        }
    });

    // Initialize TaskManager and run a single task of each type
    await page.evaluate(async () => {
        window.taskManager = new TaskManager({
            workerPools: {
                cpu: { count: 1, maxConcurrentTasks: 1 },
                gpu: { count: 1, maxConcurrentTasks: 1 },
                webnn: { count: 1, maxConcurrentTasks: 1 },
                wasm: { count: 1, maxConcurrentTasks: 1 }
            },
            maxConcurrentTasks: 4
        });
        
        await window.taskManager.initialize();
    });

    // Wait for initialization
    await page.waitForTimeout(2000);

    // Submit one task of each type and capture the actual backend usage
    const testTasks = [
        { jobType: 'LanguageModel', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'VoiceActivityDetection', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'WASMMatrix', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'Audio2Gesture', priority: 1, resourceRequirements: { memory: 1 } }
    ];

    for (const task of testTasks) {
        console.log(`\nSubmitting ${task.jobType} task...`);
        
        await page.evaluate(async (taskData) => {
            try {
                const result = await window.taskManager.submitTask(taskData);
                console.log(`Task ${taskData.jobType} completed:`, result);
            } catch (error) {
                console.error(`Task ${taskData.jobType} failed:`, error.message);
            }
        }, task);
        
        await page.waitForTimeout(3000);
    }

    // Wait for all tasks to complete
    await page.waitForTimeout(5000);

    // Log captured console messages
    console.log('\n=== CAPTURED EXECUTION MESSAGES ===');
    consoleMessages.forEach(msg => console.log(msg));

    // Final analysis
    console.log('\n=== BACKEND VERIFICATION SUMMARY ===');
    console.log(`WebNN Available: ${webnnAvailable} | Initialized: ${webnnInit.success}`);
    console.log(`WebGPU Available: ${webgpuAvailable} | Initialized: ${webgpuInit.success}`);
    console.log(`WebAssembly Available: ${wasmAvailable} | Initialized: ${wasmInit.success}`);
    
    if (!webnnInit.success && !webgpuInit.success) {
        console.log('\n❌ CRITICAL: No hardware acceleration backends available!');
        console.log('All "executionProvider" labels are FAKE - everything runs on CPU fallback');
    } else {
        console.log('\n✅ Real hardware acceleration backends detected');
    }
});
