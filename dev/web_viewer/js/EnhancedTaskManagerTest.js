/**
 * Enhanced TaskManager Test - Testing integration with real workers
 */

import { TaskManager } from './TaskManager.js';
import { JobA, JobB, JobC } from './MockGPUJobs.js';

// Test enhanced TaskManager functionality
async function testEnhancedTaskManager() {
    console.log('🧪 Testing Enhanced TaskManager with Real Workers');
    
    const manager = new TaskManager({
        maxConcurrentTasks: 3,
        preemptionEnabled: true,
        schedulingInterval: 100,
        workerPools: {
            cpu: { size: 2 },
            gpu: { size: 1 },
            webnn: { size: 1 }
        }
    });

    // Set up event listeners
    manager.on('taskStarted', (task) => {
        console.log(`✅ Task ${task.id} started on ${task.worker.type} worker`);
    });

    manager.on('taskCompleted', (task) => {
        console.log(`✅ Task ${task.id} completed in ${task.endTime - task.startTime}ms`);
    });

    manager.on('taskProgress', (data) => {
        console.log(`📊 Task ${data.task.id} progress: ${data.progress}%`);
    });

    manager.on('taskFailed', (task) => {
        console.log(`❌ Task ${task.id} failed: ${task.error}`);
    });

    // Start the manager
    await manager.start();

    console.log('📝 Scheduling test tasks...');

    // Schedule mixed workload
    const tasks = [
        manager.scheduleTask(new JobA('cpu-intensive-1', 2000), 5),
        manager.scheduleTask(new JobB('neural-inference-1', 3000), 8),
        manager.scheduleTask(new JobC('media-processing-1', 1500), 3),
        manager.scheduleTask(new JobA('cpu-intensive-2', 1000), 7),
        manager.scheduleTask(new JobB('neural-inference-2', 2500), 9),
        manager.scheduleTask(new JobC('media-processing-2', 1800), 4)
    ];

    console.log(`📦 Scheduled ${tasks.length} tasks`);

    // Monitor execution for 15 seconds
    const startTime = Date.now();
    const monitorInterval = setInterval(() => {
        const stats = manager.getStats();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        console.log(`⏱️  [${elapsed}s] Queue: ${stats.queue.pending} pending, ${stats.queue.running} running, ${stats.tasksCompleted} completed`);
        
        // Show worker utilization
        const cpuBusy = stats.workers.cpu.filter(w => w.busy).length;
        const gpuBusy = stats.workers.gpu.filter(w => w.busy).length;
        const webnnBusy = stats.workers.webnn.filter(w => w.busy).length;
        
        console.log(`👥 Workers: CPU ${cpuBusy}/${stats.workers.cpu.length}, GPU ${gpuBusy}/${stats.workers.gpu.length}, WebNN ${webnnBusy}/${stats.workers.webnn.length}`);
        
        // Check if all tasks are complete
        if (stats.tasksCompleted >= tasks.length) {
            clearInterval(monitorInterval);
            
            console.log('\n🎉 All tasks completed!');
            console.log('📊 Final Statistics:');
            console.log(`   Total execution time: ${stats.totalExecutionTime}ms`);
            console.log(`   Average task time: ${(stats.totalExecutionTime / stats.tasksCompleted).toFixed(1)}ms`);
            console.log(`   Tasks completed: ${stats.tasksCompleted}`);
            console.log(`   Tasks failed: ${stats.tasksFailed}`);
            
            manager.stop();
        }
    }, 1000);

    // Stop monitoring after 15 seconds if not done
    setTimeout(() => {
        if (monitorInterval) {
            clearInterval(monitorInterval);
            console.log('\n⏰ Test timeout reached');
            manager.stop();
        }
    }, 15000);
}

// Test worker communication
async function testWorkerCommunication() {
    console.log('\n🔗 Testing Worker Communication');
    
    const manager = new TaskManager({
        workerPools: {
            cpu: { size: 1 },
            gpu: { size: 1 }
        }
    });

    await manager.start();

    // Test if workers are properly initialized
    const stats = manager.getStats();
    console.log(`CPU workers: ${stats.workers.cpu.length} (real: ${stats.workers.cpu.filter(w => w.actualWorker).length})`);
    console.log(`GPU workers: ${stats.workers.gpu.length} (real: ${stats.workers.gpu.filter(w => w.actualWorker).length})`);

    // Schedule a simple task to test communication
    const task = manager.scheduleTask(new JobA('communication-test', 1000), 10);
    
    // Wait for task completion
    return new Promise((resolve) => {
        manager.on('taskCompleted', (completedTask) => {
            if (completedTask.id === task.id) {
                console.log(`✅ Worker communication test passed`);
                console.log(`   Task executed: ${completedTask.result ? 'Yes' : 'No'}`);
                console.log(`   Execution time: ${completedTask.endTime - completedTask.startTime}ms`);
                manager.stop();
                resolve();
            }
        });

        manager.on('taskFailed', (failedTask) => {
            if (failedTask.id === task.id) {
                console.log(`❌ Worker communication test failed: ${failedTask.error}`);
                manager.stop();
                resolve();
            }
        });

        // Timeout after 5 seconds
        setTimeout(() => {
            console.log(`⏰ Worker communication test timeout`);
            manager.stop();
            resolve();
        }, 5000);
    });
}

// Main test runner
async function runTests() {
    try {
        await testWorkerCommunication();
        await testEnhancedTaskManager();
        console.log('\n🎯 All tests completed!');
    } catch (error) {
        console.error('❌ Test error:', error);
    }
}

// Export for use
export { testEnhancedTaskManager, testWorkerCommunication, runTests };

// Auto-run if this file is loaded directly
if (typeof window !== 'undefined') {
    window.testEnhancedTaskManager = testEnhancedTaskManager;
    window.testWorkerCommunication = testWorkerCommunication;
    window.runEnhancedTests = runTests;
    
    // Auto-run tests after page load
    document.addEventListener('DOMContentLoaded', () => {
        console.log('🚀 Enhanced TaskManager tests loaded. Run with: runEnhancedTests()');
    });
}
