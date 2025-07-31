const { test, expect } = require('@playwright/test');

test('Debug Real WASM/WebGPU Workers', async ({ page }) => {
    console.log('\n=== DEBUGGING REAL BACKEND WORKERS ===');
    
    // Capture all console messages including errors
    const consoleMessages = [];
    const errors = [];
    
    page.on('console', msg => {
        const text = msg.text();
        consoleMessages.push(`[${msg.type()}] ${text}`);
        if (msg.type() === 'error') {
            errors.push(text);
        }
    });

    page.on('pageerror', error => {
        errors.push(`Page Error: ${error.message}`);
    });

    // Navigate to the task manager demo
    await page.goto('http://localhost:8000/dev/web_viewer/task-manager-demo.html', {
        waitUntil: 'networkidle'
    });

    // Wait a bit for workers to load
    await page.waitForTimeout(3000);

    // Try to run specific WASM and WebGPU tasks
    const result = await page.evaluate(async () => {
        try {
            console.log('[Test] Testing Real Backend Workers...');
            
            if (!window.TaskManager) {
                return { error: 'TaskManager not found' };
            }

            const taskManager = new window.TaskManager({
                cpuWorkers: 1,
                gpuWorkers: 1,
                wasmWorkers: 1,
                webnnWorkers: 1,
                maxConcurrentTasks: 4
            });

            console.log('[Test] TaskManager created, starting...');
            taskManager.start();

            // Wait for workers to be ready
            await new Promise(resolve => {
                if (taskManager.readyWorkers >= 2) { // At least WASM and GPU workers
                    resolve();
                } else {
                    taskManager.on('allWorkersReady', resolve);
                    setTimeout(resolve, 5000); // Fallback timeout
                }
            });

            console.log('[Test] Workers ready, scheduling real WASM tasks...');

            // Schedule real WASM tasks
            const wasmMatrixTask = {
                jobType: 'WASMMatrix',
                type: 'real-wasm',
                action: 'matrixMultiply',
                dimensions: [4, 4],
                priority: 1,
                resourceRequirements: { cpu: 0, wasm: 1, memory: 10 }
            };

            const wasmPrimeTask = {
                jobType: 'WASMPrime', 
                type: 'real-wasm',
                action: 'primeComputation',
                limit: 100,
                priority: 1,
                resourceRequirements: { cpu: 0, wasm: 1, memory: 10 }
            };

            const gpuTask = {
                jobType: 'WebGPUCompute',
                type: 'real-webgpu', 
                action: 'matrixMultiply',
                dimensions: [256, 256],
                priority: 1,
                resourceRequirements: { cpu: 0, gpu: 1, memory: 100 }
            };

            console.log('[Test] Scheduling tasks...');
            const wasmMatrixId = taskManager.scheduleTask(wasmMatrixTask);
            const wasmPrimeId = taskManager.scheduleTask(wasmPrimeTask);
            const gpuId = taskManager.scheduleTask(gpuTask);

            console.log('[Test] Tasks scheduled:', { wasmMatrixId, wasmPrimeId, gpuId });

            // Wait for tasks to complete or timeout
            const results = await new Promise((resolve) => {
                const taskResults = {};
                let completedCount = 0;
                const totalTasks = 3;

                const checkComplete = () => {
                    if (completedCount >= totalTasks) {
                        resolve(taskResults);
                    }
                };

                // Monitor task completion
                const checkTasks = () => {
                    // Check WASM Matrix task
                    if (wasmMatrixId && !taskResults.wasmMatrix) {
                        const task = taskManager.completedTasks.get(wasmMatrixId) || taskManager.failedTasks.get(wasmMatrixId);
                        if (task) {
                            taskResults.wasmMatrix = { status: task.status, result: task.result, error: task.error };
                            completedCount++;
                        }
                    }

                    // Check WASM Prime task  
                    if (wasmPrimeId && !taskResults.wasmPrime) {
                        const task = taskManager.completedTasks.get(wasmPrimeId) || taskManager.failedTasks.get(wasmPrimeId);
                        if (task) {
                            taskResults.wasmPrime = { status: task.status, result: task.result, error: task.error };
                            completedCount++;
                        }
                    }

                    // Check GPU task
                    if (gpuId && !taskResults.gpu) {
                        const task = taskManager.completedTasks.get(gpuId) || taskManager.failedTasks.get(gpuId);
                        if (task) {
                            taskResults.gpu = { status: task.status, result: task.result, error: task.error };
                            completedCount++;
                        }
                    }

                    checkComplete();
                };

                // Check every 500ms
                const interval = setInterval(checkTasks, 500);
                
                // Timeout after 15 seconds
                setTimeout(() => {
                    clearInterval(interval);
                    console.log('[Test] Task execution timeout reached');
                    resolve(taskResults);
                }, 15000);
            });

            return { success: true, results, 
                    queueSize: taskManager.heap.size(),
                    runningTasks: taskManager.runningTasks.size,
                    completedTasks: taskManager.completedTasks.size,
                    failedTasks: taskManager.failedTasks.size,
                    stats: taskManager.stats
                   };

        } catch (error) {
            console.error('[Test] Error:', error);
            return { error: error.message, stack: error.stack };
        }
    });

    console.log('\n=== REAL WORKER TEST RESULT ===');
    console.log(JSON.stringify(result, null, 2));

    console.log('\n=== CONSOLE MESSAGES ===');
    consoleMessages.forEach(msg => console.log(msg));

    console.log('\n=== ERRORS ===');
    if (errors.length > 0) {
        console.log(`❌ Found ${errors.length} errors`);
        errors.forEach((error, i) => {
            console.log(`${i + 1}. ${error}`);
        });
    } else {
        console.log('✅ No errors found');
    }

    // Check if we got actual results
    if (result.success && result.results) {
        console.log('\n=== TASK EXECUTION ANALYSIS ===');
        const { wasmMatrix, wasmPrime, gpu } = result.results;
        
        if (wasmMatrix) {
            console.log(`🔧 WASM Matrix: ${wasmMatrix.status} - ${wasmMatrix.result ? 'HAS RESULT' : 'NO RESULT'}`);
        } else {
            console.log('❌ WASM Matrix: NOT COMPLETED');
        }
        
        if (wasmPrime) {
            console.log(`🔧 WASM Prime: ${wasmPrime.status} - ${wasmPrime.result ? 'HAS RESULT' : 'NO RESULT'}`);
        } else {
            console.log('❌ WASM Prime: NOT COMPLETED');
        }
        
        if (gpu) {
            console.log(`🎮 WebGPU: ${gpu.status} - ${gpu.result ? 'HAS RESULT' : 'NO RESULT'}`);
        } else {
            console.log('❌ WebGPU: NOT COMPLETED');
        }
    }
});
