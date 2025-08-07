const { test, expect } = require('@playwright/test');

test('Test REAL Backend Implementation vs Fake Labels', async ({ page }) => {
    console.log('\n=== REAL BACKEND IMPLEMENTATION TEST ===');
    console.log('Testing honest execution provider reporting with real WebGPU/WASM');
    
    // Start HTTP server
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html', {
        waitUntil: 'networkidle'
    });

    // Capture console messages to see actual execution
    const consoleMessages = [];
    const executionProviders = [];
    
    page.on('console', msg => {
        const text = msg.text();
        consoleMessages.push(text);
        
        // Extract execution provider information
        if (text.includes('AVATAR AI COLLECTED')) {
            try {
                const match = text.match(/AVATAR AI COLLECTED (.+)/);
                if (match) {
                    const data = JSON.parse(match[1]);
                    if (data.modelOutput && data.modelOutput.executionProvider) {
                        executionProviders.push({
                            jobType: data.jobType,
                            provider: data.modelOutput.executionProvider,
                            actualWasmUsed: data.modelOutput.actualWasmUsed,
                            actualWebGPUUsed: data.modelOutput.actualWebGPUUsed
                        });
                    }
                }
            } catch (e) {
                // Ignore parsing errors
            }
        }
    });

    // Wait for page to fully load
    await page.waitForFunction(() => window.TaskManager !== undefined, { timeout: 10000 });
    
    // Debug what's available
    const taskManagerInfo = await page.evaluate(() => {
        const TaskManager = window.TaskManager;
        if (!TaskManager) return { error: 'TaskManager not found' };
        
        const instance = new TaskManager();
        return {
            constructor: !!TaskManager,
            instanceCreated: !!instance,
            hasInitialize: typeof instance.initialize === 'function',
            methods: Object.getOwnPropertyNames(TaskManager.prototype),
            instanceMethods: Object.getOwnPropertyNames(instance)
        };
    });
    
    console.log('TaskManager debug info:', JSON.stringify(taskManagerInfo, null, 2));
    
    // Initialize TaskManager with real backend workers
    await page.evaluate(async () => {
        console.log('[Test] Creating TaskManager instance...');
        
        window.taskManager = new window.TaskManager({
            workerPools: {
                cpu: { count: 2, maxConcurrentTasks: 2 },
                gpu: { count: 2, maxConcurrentTasks: 2 },
                webnn: { count: 1, maxConcurrentTasks: 1 },
                wasm: { count: 2, maxConcurrentTasks: 2 }
            },
            maxConcurrentTasks: 8
        });
        
        console.log('[Test] TaskManager created');
        console.log('[Test] TaskManager methods:', Object.getOwnPropertyNames(window.taskManager));
        console.log('[Test] TaskManager has initialize:', typeof window.taskManager.initialize);
        
        if (typeof window.taskManager.initialize === 'function') {
            console.log('[Test] Calling initialize...');
            await window.taskManager.initialize();
            console.log('[Test] Initialize completed');
        } else {
            console.log('[Test] No initialize method, starting directly...');
            window.taskManager.start();
        }
    });

    // Wait for initialization
    await page.waitForTimeout(3000);

    console.log('\n=== SUBMITTING TEST TASKS ===');

    // Test different worker types with honest backend detection
    const testTasks = [
        // CPU tasks (should report honest CPU execution)
        { jobType: 'LanguageModel', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'VoiceActivityDetection', priority: 1, resourceRequirements: { memory: 1 } },
        
        // WASM tasks (should report real WASM if available, honest fallback if not)
        { jobType: 'WASMMatrix', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'WASMPrime', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'WASMFractal', priority: 1, resourceRequirements: { memory: 1 } },
        
        // GPU tasks (should report real WebGPU if available, honest fallback if not)  
        { jobType: 'Audio2Gesture', priority: 1, resourceRequirements: { memory: 1 } },
        { jobType: 'GPUMatrixMultiply', priority: 1, resourceRequirements: { memory: 1 } }
    ];

    // Submit tasks sequentially to avoid overwhelming
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
        
        await page.waitForTimeout(2000);
    }

    // Wait for all tasks to complete
    await page.waitForTimeout(10000);

    // Analyze results
    console.log('\n=== EXECUTION PROVIDER ANALYSIS ===');
    console.log(`Total tasks with execution provider data: ${executionProviders.length}`);

    let realWasmCount = 0;
    let realWebGPUCount = 0;
    let honestFallbackCount = 0;
    let suspiciousLabelCount = 0;

    executionProviders.forEach((item, index) => {
        console.log(`\n${index + 1}. ${item.jobType}:`);
        console.log(`   Execution Provider: ${JSON.stringify(item.provider)}`);
        
        if (item.actualWasmUsed === true) {
            console.log(`   ✅ REAL WASM: Actually using WebAssembly`);
            realWasmCount++;
        } else if (item.actualWebGPUUsed === true) {
            console.log(`   ✅ REAL WebGPU: Actually using GPU compute`);
            realWebGPUCount++;
        } else if (Array.isArray(item.provider) && item.provider.includes('cpu')) {
            console.log(`   ✅ HONEST FALLBACK: Correctly reports CPU execution`);
            honestFallbackCount++;
        } else if (
            (Array.isArray(item.provider) && (item.provider.includes('wasm') || item.provider.includes('webgpu') || item.provider.includes('webnn'))) ||
            (typeof item.provider === 'string' && (item.provider.includes('wasm') || item.provider.includes('webgpu') || item.provider.includes('webnn')))
        ) {
            console.log(`   ❌ SUSPICIOUS: Claims advanced backend but actualWasm/WebGPU = false`);
            suspiciousLabelCount++;
        }
    });

    console.log('\n=== FINAL VERIFICATION SUMMARY ===');
    console.log(`Real WASM executions: ${realWasmCount}`);
    console.log(`Real WebGPU executions: ${realWebGPUCount}`);
    console.log(`Honest CPU fallbacks: ${honestFallbackCount}`);
    console.log(`Suspicious fake labels: ${suspiciousLabelCount}`);

    if (suspiciousLabelCount > 0) {
        console.log('\n❌ CRITICAL: Still found fake execution provider labels!');
        console.log('Some tasks claim to use WebGPU/WASM/WebNN but actualUsage = false');
    } else if (realWasmCount > 0 || realWebGPUCount > 0) {
        console.log('\n✅ SUCCESS: Real hardware acceleration detected!');
        console.log('Tasks successfully using actual WebGPU/WASM backends');
    } else {
        console.log('\n✅ HONEST SYSTEM: No fake labels, honest CPU fallback reporting');
        console.log('All tasks correctly report CPU JavaScript execution when no real backends available');
    }

    // Test should pass if we have either real backends OR honest fallback reporting
    const hasRealBackends = realWasmCount > 0 || realWebGPUCount > 0;
    const hasHonestFallbacks = honestFallbackCount > 0 && suspiciousLabelCount === 0;
    
    expect(hasRealBackends || hasHonestFallbacks).toBe(true);
    expect(suspiciousLabelCount).toBe(0); // No fake labels allowed

    console.log('\n✅ TEST PASSED: Real backend implementation verified');
});
