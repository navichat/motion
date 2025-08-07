const { test, expect } = require('@playwright/test');

test('Debug Worker Loading Issues', async ({ page }) => {
    console.log('\n=== DEBUGGING WORKER LOADING ===');
    
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
    await page.waitForTimeout(5000);

    // Try to create TaskManager and see what happens
    const result = await page.evaluate(async () => {
        try {
            console.log('[Test] Creating TaskManager...');
            
            if (!window.TaskManager) {
                return { error: 'TaskManager not found' };
            }

            const taskManager = new window.TaskManager({
                workerPools: {
                    cpu: { count: 1, maxConcurrentTasks: 1 },
                    gpu: { count: 1, maxConcurrentTasks: 1 },
                    webnn: { count: 1, maxConcurrentTasks: 1 },
                    wasm: { count: 1, maxConcurrentTasks: 1 }
                },
                maxConcurrentTasks: 4
            });

            console.log('[Test] TaskManager created');

            // Try to submit a simple task
            const task = {
                jobType: 'WASMMatrix',
                priority: 1,
                resourceRequirements: { memory: 1 }
            };

            console.log('[Test] Submitting test task...');
            taskManager.start();
            
            const taskId = taskManager.scheduleTask(task);
            console.log('[Test] Task scheduled with ID:', taskId);

            return { success: true, taskId };

        } catch (error) {
            console.error('[Test] Error:', error);
            return { error: error.message, stack: error.stack };
        }
    });

    console.log('\n=== EXECUTION RESULT ===');
    console.log(JSON.stringify(result, null, 2));

    console.log('\n=== CONSOLE MESSAGES ===');
    consoleMessages.forEach(msg => console.log(msg));

    console.log('\n=== ERRORS ===');
    errors.forEach(error => console.log(`ERROR: ${error}`));

    if (errors.length > 0) {
        console.log(`\n❌ Found ${errors.length} errors`);
        errors.forEach((error, i) => {
            console.log(`${i + 1}. ${error}`);
        });
    } else {
        console.log('\n✅ No errors found');
    }
});
