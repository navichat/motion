const { test, expect } = require('@playwright/test');

test('Debug Missing Models (TinyLlama, DiabloGPT, Whisper)', async ({ page }) => {
    console.log('\n=== DEBUGGING MISSING MODELS ===');
    
    // Capture all console messages including errors
    const consoleMessages = [];
    const avatarResults = [];
    const errors = [];
    
    page.on('console', msg => {
        const text = msg.text();
        consoleMessages.push(`[${msg.type()}] ${text}`);
        
        // Capture AVATAR AI COLLECTED messages
        if (text.includes('AVATAR AI COLLECTED')) {
            try {
                const jsonString = text.substring(text.indexOf('{'));
                const parsed = JSON.parse(jsonString);
                avatarResults.push(parsed);
                console.log(`📊 CAPTURED: ${parsed.jobType}`);
            } catch (e) {
                console.error('Failed to parse AVATAR result:', e);
            }
        }
        
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

    // Create TaskManager and schedule only the missing models
    const result = await page.evaluate(async () => {
        try {
            console.log('[Debug] Creating TaskManager for missing models...');
            
            if (!window.TaskManager) {
                return { error: 'TaskManager not found' };
            }

            const taskManager = new window.TaskManager({
                cpuWorkers: 2,
                gpuWorkers: 1,
                wasmWorkers: 1,
                webnnWorkers: 2,  // Increase WebNN workers since missing models use this backend
                maxConcurrentTasks: 6
            });

            console.log('[Debug] Starting TaskManager...');
            taskManager.start();

            // Wait for workers to be ready
            await new Promise(resolve => {
                if (taskManager.readyWorkers >= 4) {
                    resolve();
                } else {
                    taskManager.on('allWorkersReady', resolve);
                    setTimeout(resolve, 5000); // Fallback timeout
                }
            });

            console.log('[Debug] Workers ready, scheduling missing model tasks...');

            // Schedule the specific missing models with same logic as runRealWorkloadTest
            const missingModels = [
                // TinyLlama - uses webnn backend
                {
                    type: 'TinyLlama',
                    duration: 2000,
                    complexity: 1,
                    resourceRequirements: { memory: 512 },
                    useRealInference: true,
                    backend: 'webnn'
                },
                // DiabloGPT - uses webnn backend  
                {
                    type: 'DiabloGPT',
                    duration: 2000,
                    complexity: 1,
                    resourceRequirements: { memory: 512 },
                    useRealInference: true,
                    backend: 'webnn'
                },
                // Whisper - uses webnn backend
                {
                    type: 'Whisper',
                    duration: 1500,
                    complexity: 1,
                    resourceRequirements: { memory: 256 },
                    useRealInference: true,
                    backend: 'webnn'
                }
            ];

            const taskIds = [];
            for (const job of missingModels) {
                console.log(`[Debug] Scheduling ${job.type} with ${job.backend} backend...`);
                const taskId = taskManager.scheduleTask(job);
                taskIds.push({ taskId, jobType: job.type });
                console.log(`[Debug] ${job.type} scheduled with ID: ${taskId}`);
            }

            // Wait for tasks to complete
            await new Promise((resolve) => {
                let completedCount = 0;
                const totalTasks = missingModels.length;

                const checkComplete = () => {
                    if (completedCount >= totalTasks) {
                        resolve();
                    }
                };

                const checkTasks = () => {
                    for (const { taskId, jobType } of taskIds) {
                        const completed = taskManager.completedTasks.get(taskId);
                        const failed = taskManager.failedTasks.get(taskId);
                        
                        if ((completed || failed) && !completed?.counted) {
                            completedCount++;
                            if (completed) completed.counted = true;
                            if (failed) failed.counted = true;
                            
                            console.log(`[Debug] ${jobType} task ${completed ? 'completed' : 'failed'}: ${taskId}`);
                            if (failed) {
                                console.error(`[Debug] ${jobType} error:`, failed.error);
                            }
                        }
                    }
                    checkComplete();
                };

                // Check every 500ms
                const interval = setInterval(checkTasks, 500);
                
                // Timeout after 15 seconds
                setTimeout(() => {
                    clearInterval(interval);
                    console.log('[Debug] Task execution timeout reached');
                    resolve();
                }, 15000);
            });

            return { 
                success: true, 
                taskIds,
                completedTasks: taskManager.completedTasks.size,
                failedTasks: taskManager.failedTasks.size,
                runningTasks: taskManager.runningTasks.size,
                queueSize: taskManager.heap.size()
            };

        } catch (error) {
            console.error('[Debug] Error:', error);
            return { error: error.message, stack: error.stack };
        }
    });

    console.log('\n=== DEBUG RESULT ===');
    console.log(JSON.stringify(result, null, 2));

    console.log('\n=== CAPTURED AVATAR RESULTS ===');
    console.log(`Found ${avatarResults.length} AVATAR AI COLLECTED messages:`);
    avatarResults.forEach((result, i) => {
        console.log(`${i + 1}. ${result.jobType}: ${result.modelOutput?.type || 'unknown'}`);
    });

    console.log('\n=== SAMPLE CONSOLE MESSAGES ===');
    // Show last 50 messages to see what's happening
    consoleMessages.slice(-50).forEach(msg => console.log(msg));

    console.log('\n=== ERRORS ===');
    if (errors.length > 0) {
        console.log(`❌ Found ${errors.length} errors`);
        errors.forEach((error, i) => {
            console.log(`${i + 1}. ${error}`);
        });
    } else {
        console.log('✅ No errors found');
    }

    // Verify we got the missing models
    const tinyLlama = avatarResults.find(r => r.jobType === 'TinyLlama');
    const diabloGPT = avatarResults.find(r => r.jobType === 'DiabloGPT');
    const whisper = avatarResults.find(r => r.jobType === 'Whisper');

    console.log('\n=== MISSING MODELS STATUS ===');
    console.log(`TinyLlama: ${tinyLlama ? '✅ FOUND' : '❌ MISSING'}`);
    console.log(`DiabloGPT: ${diabloGPT ? '✅ FOUND' : '❌ MISSING'}`);  
    console.log(`Whisper: ${whisper ? '✅ FOUND' : '❌ MISSING'}`);
});
