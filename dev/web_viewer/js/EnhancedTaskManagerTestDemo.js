/**
 * Enhanced TaskManager Test - Non-module version for demo page
 */

// Test job classes if not already defined
if (typeof EnhancedJobA === 'undefined') {
    class EnhancedJobA {
        constructor(id, duration = 500) {
            this.id = id;
            this.duration = duration;
            this.type = 'computational';
        }

        async execute(worker) {
            // Simulate computational work
            return new Promise((resolve) => {
                setTimeout(() => {
                    resolve({
                        jobId: this.id,
                        result: `JobA ${this.id} completed`,
                        executionTime: this.duration
                    });
                }, this.duration);
            });
        }
    }

    class JobB {
        constructor(id, duration = 700) {
            this.id = id;
            this.duration = duration;
            this.type = 'gpu';
        }

        async execute(worker) {
            // Simulate GPU work
            return new Promise((resolve) => {
                setTimeout(() => {
                    resolve({
                        jobId: this.id,
                        result: `JobB ${this.id} completed`,
                        executionTime: this.duration
                    });
                }, this.duration);
            });
        }
    }

    // Make classes globally available
    window.EnhancedJobA = EnhancedJobA;
    window.JobB = JobB;
}

// Test enhanced TaskManager functionality with real workers
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

    // Schedule mixed workload - use available job types
    let tasks = [];
    
    if (typeof EnhancedJobA !== 'undefined') {
        // Use enhanced mock jobs if available
        tasks = [
            manager.scheduleTask(new EnhancedJobA('cpu-intensive-1', 2000), 5),
            manager.scheduleTask(new JobB('neural-inference-1', 3000), 8),
            manager.scheduleTask(new JobC('media-processing-1', 1500), 3),
            manager.scheduleTask(new EnhancedJobA('cpu-intensive-2', 1000), 7),
            manager.scheduleTask(new JobB('neural-inference-2', 2500), 9),
            manager.scheduleTask(new JobC('media-processing-2', 1800), 4)
        ];
    } else if (typeof WASMMatrixJob !== 'undefined') {
        // Use real WASM/WebGPU jobs if available
        tasks = [
            manager.scheduleTask(new WASMMatrixJob('wasm-matrix-1', 256, 1), 5),
            manager.scheduleTask(new WebGPUMatrixJob('gpu-matrix-1', 256, 1), 8),
            manager.scheduleTask(new WASMPrimeJob('wasm-prime-1', 50000, 1), 3),
            manager.scheduleTask(new WASMFractalJob('wasm-fractal-1', 256, 50, 1), 7),
            manager.scheduleTask(new WebGPUImageJob('gpu-image-1', 512, 512, 1), 9),
            manager.scheduleTask(new WASMMatrixJob('wasm-matrix-2', 256, 1), 4)
        ];
    } else {
        // Fallback to simple test jobs
        tasks = [
            manager.scheduleTask({
                id: 'test-job-1',
                type: 'TestJob',
                execute: async (progress) => {
                    for (let i = 0; i < 20; i++) {
                        if (progress) progress((i + 1) * 5);
                        await new Promise(r => setTimeout(r, 100));
                    }
                    return { success: true };
                }
            }, 5),
            manager.scheduleTask({
                id: 'test-job-2',
                type: 'TestJob',
                execute: async (progress) => {
                    for (let i = 0; i < 30; i++) {
                        if (progress) progress(Math.round((i + 1) / 30 * 100));
                        await new Promise(r => setTimeout(r, 100));
                    }
                    return { success: true };
                }
            }, 8)
        ];
    }

    console.log(`📦 Scheduled ${tasks.length} tasks`);

    // Monitor execution for 15 seconds
    const startTime = Date.now();
    const monitorInterval = setInterval(() => {
        const stats = manager.getStats();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        console.log(`⏱️  [${elapsed}s] Queue: ${stats.queue.pending} pending, ${stats.queue.running} running, ${stats.tasksCompleted} completed`);
        
        // Show worker utilization
        const cpuBusy = stats.workers.cpu.busy;
        const gpuBusy = stats.workers.gpu.busy;
        const webnnBusy = stats.workers.webnn.busy;
        
        console.log(`👥 Workers: CPU ${cpuBusy}/${stats.workers.cpu.total}, GPU ${gpuBusy}/${stats.workers.gpu.total}, WebNN ${webnnBusy}/${stats.workers.webnn.total}`);
        
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
    
    // Defensive check for stats structure
    if (stats && stats.workers) {
        console.log(`CPU workers: ${stats.workers.cpu?.total || 0} (busy: ${stats.workers.cpu?.busy || 0})`);
        console.log(`GPU workers: ${stats.workers.gpu?.total || 0} (busy: ${stats.workers.gpu?.busy || 0})`);
        
        // Additional validation that stats are in correct format
        if (typeof stats.workers.cpu === 'object' && !Array.isArray(stats.workers.cpu)) {
            console.log('✅ Worker pool stats structure is correct');
        } else {
            console.log('⚠️ Unexpected worker pool stats structure:', typeof stats.workers.cpu);
        }
    } else {
        console.log('❌ Stats object structure is invalid');
        console.log('Stats:', stats);
    }

    // Schedule a simple task to test communication
    let task;
    
    if (typeof EnhancedJobA !== 'undefined') {
        task = manager.scheduleTask(new EnhancedJobA('communication-test', 1000), 10);
    } else {
        // Fallback to simple job
        task = manager.scheduleTask({
            id: 'communication-test',
            type: 'CommunicationTest',
            execute: async (progress, shouldStop) => {
                for (let i = 0; i < 10; i++) {
                    if (shouldStop()) return null;
                    if (progress) progress((i + 1) * 10);
                    await new Promise(r => setTimeout(r, 100));
                }
                return { success: true, duration: 1000 };
            }
        }, 10);
    }
    
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
async function runEnhancedTests() {
    try {
        await testWorkerCommunication();
        await testEnhancedTaskManager();
        console.log('\n🎯 All enhanced tests completed!');
    } catch (error) {
        console.error('❌ Enhanced test error:', error);
    }
}

// Make functions globally available
window.testEnhancedTaskManager = testEnhancedTaskManager;
window.testWorkerCommunication = testWorkerCommunication;
window.runEnhancedTests = runEnhancedTests;
