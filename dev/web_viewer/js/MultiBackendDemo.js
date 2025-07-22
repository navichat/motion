/**
 * Multi-Backend Task Distribution Demo
 * Demonstrates tasks being properly assigned to CPU, GPU, and WebNN workers
 */

async function runMultiBackendDemo() {
    console.log('🎯 Multi-Backend Task Distribution Demo');
    
    try {
        // Initialize TaskManager
        const manager = new TaskManager();
        await manager.initialize();
        
        console.log('\n✅ TaskManager initialized with all worker types');
        
        // Create a mix of tasks for different backends
        const tasks = [];
        
        // CPU Tasks (WASM)
        if (typeof WASMMatrixJob !== 'undefined') {
            tasks.push({
                job: new WASMMatrixJob('cpu-matrix-1', 128, 1),
                priority: 5,
                expectedWorker: 'CPU'
            });
            tasks.push({
                job: new WASMPrimeJob('cpu-prime-1', 50000, 1),
                priority: 6,
                expectedWorker: 'CPU'
            });
        }
        
        // GPU Tasks (WebGPU)
        if (typeof WebGPUMatrixJob !== 'undefined') {
            tasks.push({
                job: new WebGPUMatrixJob('gpu-matrix-1', 128, 1),
                priority: 7,
                expectedWorker: 'GPU'
            });
            tasks.push({
                job: new WebGPUImageJob('gpu-image-1', 256, 256, 1),
                priority: 8,
                expectedWorker: 'GPU'
            });
        }
        
        // WebNN Tasks
        if (typeof WebNNImageClassificationJob !== 'undefined') {
            tasks.push({
                job: new WebNNImageClassificationJob('webnn-image-1', 16, 224, 1),
                priority: 9,
                expectedWorker: 'WebNN'
            });
            tasks.push({
                job: new WebNNTextProcessingJob('webnn-text-1', 128, 8, 1),
                priority: 10,
                expectedWorker: 'WebNN'
            });
        }
        
        console.log(`\n📝 Scheduling ${tasks.length} tasks across multiple backends...`);
        
        // Schedule all tasks and track assignments
        const scheduledTasks = [];
        for (const taskInfo of tasks) {
            console.log(`\nTask: ${taskInfo.job.id} (${taskInfo.job.type})`);
            console.log(`Resource Requirements:`, taskInfo.job.resourceRequirements);
            console.log(`Expected Worker Type: ${taskInfo.expectedWorker}`);
            
            const taskId = manager.scheduleTask(taskInfo.job, taskInfo.priority);
            scheduledTasks.push({ ...taskInfo, taskId });
        }
        
        // Monitor execution
        console.log('\n⏱️  Monitoring task execution across workers...');
        
        let completedCount = 0;
        const targetCount = scheduledTasks.length;
        
        const completionPromise = new Promise((resolve) => {
            manager.on('taskCompleted', (task) => {
                completedCount++;
                console.log(`✅ Task ${task.id} completed on ${task.result?.workerType || 'unknown'} worker (${completedCount}/${targetCount})`);
                
                if (completedCount >= targetCount) {
                    resolve();
                }
            });
            
            manager.on('taskFailed', (task) => {
                completedCount++;
                console.log(`❌ Task ${task.id} failed (${completedCount}/${targetCount})`);
                
                if (completedCount >= targetCount) {
                    resolve();
                }
            });
        });
        
        // Show real-time stats
        const statsInterval = setInterval(() => {
            const stats = manager.getStatistics();
            console.log(`📊 Workers: CPU ${stats.workers.cpu.busy}/${stats.workers.cpu.total}, GPU ${stats.workers.gpu.busy}/${stats.workers.gpu.total}, WebNN ${stats.workers.webnn.busy}/${stats.workers.webnn.total}`);
        }, 1000);
        
        // Wait for completion
        await completionPromise;
        clearInterval(statsInterval);
        
        // Final statistics
        const finalStats = manager.getStatistics();
        console.log('\n📊 Final Results:');
        console.log(`Tasks Completed: ${finalStats.performance.tasksCompleted}`);
        console.log(`Tasks Failed: ${finalStats.performance.tasksFailed}`);
        console.log(`Total Execution Time: ${finalStats.performance.totalExecutionTime}ms`);
        
        console.log('\n🏆 Worker Utilization:');
        console.log(`CPU Workers: Processed ${finalStats.workers.cpu.tasksCompleted || 0} tasks`);
        console.log(`GPU Workers: Processed ${finalStats.workers.gpu.tasksCompleted || 0} tasks`);
        console.log(`WebNN Workers: Processed ${finalStats.workers.webnn.tasksCompleted || 0} tasks`);
        
        manager.shutdown();
        console.log('\n🎉 Multi-backend demo completed successfully!');
        
    } catch (error) {
        console.error('❌ Multi-backend demo failed:', error);
    }
}

// Make it available globally
window.runMultiBackendDemo = runMultiBackendDemo;
