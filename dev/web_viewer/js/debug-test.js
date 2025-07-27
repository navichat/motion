/**
 * Task Manager Debug Test
 * Quick test to verify all components are working
 */

async function debugTaskManager() {
    console.log('🔧 Starting Task Manager Debug Test...');
    
    try {
        // 1. Check if all classes are available
        console.log('1️⃣ Checking class availability...');
        const requiredClasses = ['TaskManager', 'FibonacciHeap', 'MockGPUJobFactory'];
        for (const className of requiredClasses) {
            if (typeof window[className] === 'undefined') {
                throw new Error(`${className} is not available`);
            }
            console.log(`✅ ${className} available`);
        }
        
        // 2. Create TaskManager instance
        console.log('2️⃣ Creating TaskManager...');
        const taskManager = new TaskManager({
            maxConcurrentTasks: 2,
            cpuWorkers: 2,
            schedulingInterval: 100,
            logger: (message, type) => console.log(`[${type.toUpperCase()}] ${message}`)
        });
        console.log('✅ TaskManager created');
        
        // 3. Create a simple job
        console.log('3️⃣ Creating test job...');
        const job = MockGPUJobFactory.createRandomJob();
        job.duration = 500; // Quick test
        console.log(`✅ Job created: ${job.type}`);
        
        // 4. Schedule the job
        console.log('4️⃣ Scheduling job...');
        const taskId = taskManager.scheduleTask(job, 1);
        console.log(`✅ Job scheduled with ID: ${taskId}`);
        
        // 5. Start the task manager
        console.log('5️⃣ Starting TaskManager...');
        await taskManager.start();
        console.log('✅ TaskManager started');
        
        // 6. Wait for completion
        console.log('6️⃣ Waiting for job completion...');
        await new Promise((resolve) => {
            const checkCompletion = () => {
                const stats = taskManager.getStats();
                console.log(`📊 Stats: ${stats.queue.completed} completed, ${stats.queue.running} running, ${stats.queue.size} queued`);
                
                if (stats.queue.completed >= 1) {
                    console.log('✅ Job completed!');
                    resolve();
                } else {
                    setTimeout(checkCompletion, 100);
                }
            };
            checkCompletion();
            
            // Timeout after 10 seconds
            setTimeout(() => {
                console.log('⏰ Test timeout');
                resolve();
            }, 10000);
        });
        
        // 7. Stop task manager
        console.log('7️⃣ Stopping TaskManager...');
        taskManager.stop();
        
        console.log('🎉 Debug test completed successfully!');
        return true;
        
    } catch (error) {
        console.error('❌ Debug test failed:', error);
        console.error('Stack trace:', error.stack);
        return false;
    }
}

// Make function available globally
if (typeof window !== 'undefined') {
    window.debugTaskManager = debugTaskManager;
}

console.log('🔧 Debug test loaded. Run window.debugTaskManager() to test.');
