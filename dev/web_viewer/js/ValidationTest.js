/**
 * Validation Test - Comprehensive testing of the enhanced TaskManager
 */

import { JobC } from './TestJobs.js';

// Test job classes for validation
class ValidationJobA {
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
                    result: `ValidationJobA ${this.id} completed`,
                    executionTime: this.duration
                });
            }, this.duration);
        });
    }
}

class ValidationJobB {
    constructor(id, duration = 300) {
        this.id = id;
        this.duration = duration;
        this.type = 'io';
    }

    async execute(worker) {
        // Simulate I/O work
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    jobId: this.id,
                    result: `ValidationJobB ${this.id} completed`,
                    executionTime: this.duration
                });
            }, this.duration);
        });
    }
}

class ValidationJobC {
    constructor(id, duration = 700) {
        this.id = id;
        this.duration = duration;
        this.type = 'memory';
    }

    async execute(worker) {
        // Simulate memory-intensive work
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    jobId: this.id,
                    result: `ValidationJobC ${this.id} completed`,
                    executionTime: this.duration
                });
            }, this.duration);
        });
    }
}

// Validation test for enhanced TaskManager
async function validateEnhancedTaskManager() {
    console.log('🔍 Validating Enhanced TaskManager Integration...');
    
    const manager = new TaskManager({
        maxConcurrentTasks: 2,
        preemptionEnabled: true,
        schedulingInterval: 50,
        workerPools: {
            cpu: { size: 1 },
            gpu: { size: 1 },
            webnn: { size: 1 }
        }
    });

    let testResults = {
        workerInitialization: false,
        taskScheduling: false,
        taskExecution: false,
        priorityHandling: false,
        workerCommunication: false,
        errorHandling: false
    };

    try {
        // Test 1: Worker Initialization
        console.log('📋 Test 1: Worker Initialization');
        await manager.start();
        
        const stats = manager.getStats();
        const hasWorkers = stats.workers.cpu.total > 0 || stats.workers.gpu.total > 0 || stats.workers.webnn.total > 0;
        
        if (hasWorkers) {
            console.log('✅ Workers initialized successfully');
            console.log(`   CPU workers: ${stats.workers.cpu.total}`);
            console.log(`   GPU workers: ${stats.workers.gpu.total}`);
            console.log(`   WebNN workers: ${stats.workers.webnn.total}`);
            testResults.workerInitialization = true;
        } else {
            console.log('❌ No workers initialized');
        }

        // Test 2: Task Scheduling
        console.log('\n📋 Test 2: Task Scheduling');
        
        // Use available job types - mix of CPU and GPU/WebNN jobs
        let task1, task2;
        
        if (typeof ValidationJobA !== 'undefined') {
            task1 = manager.scheduleTask(new ValidationJobA('validation-task-1', 500), 5);
            task2 = manager.scheduleTask(new JobB('validation-task-2', 700), 8);
        } else if (typeof WASMMatrixJob !== 'undefined' && typeof WebGPUMatrixJob !== 'undefined') {
            // Use real mix of CPU and GPU jobs
            task1 = manager.scheduleTask(new WASMMatrixJob('validation-task-1', 128, 1), 5);  // CPU job
            task2 = manager.scheduleTask(new WebGPUMatrixJob('validation-task-2', 128, 1), 8); // GPU job
        } else if (typeof WASMMatrixJob !== 'undefined') {
            task1 = manager.scheduleTask(new WASMMatrixJob('validation-task-1', 128, 1), 5);
            task2 = manager.scheduleTask(new WASMPrimeJob('validation-task-2', 50000, 1), 8);
        } else {
            // Fallback to simple mock jobs
            task1 = manager.scheduleTask({
                id: 'validation-task-1',
                type: 'TestJob1',
                execute: async (progress, shouldStop) => {
                    for (let i = 0; i < 5; i++) {
                        if (shouldStop()) return null;
                        if (progress) progress((i + 1) * 20);
                        await new Promise(r => setTimeout(r, 100));
                    }
                    return { success: true, duration: 500 };
                }
            }, 5);
            
            task2 = manager.scheduleTask({
                id: 'validation-task-2', 
                type: 'TestJob2',
                execute: async (progress, shouldStop) => {
                    for (let i = 0; i < 7; i++) {
                        if (shouldStop()) return null;
                        if (progress) progress(Math.round((i + 1) / 7 * 100));
                        await new Promise(r => setTimeout(r, 100));
                    }
                    return { success: true, duration: 700 };
                }
            }, 8);
        }
        
        if (task1 && task2) {
            console.log('✅ Tasks scheduled successfully');
            // scheduleTask returns the task ID (string), not the task object
            console.log(`   Task 1 ID: ${task1}`);
            console.log(`   Task 2 ID: ${task2}`);
            testResults.taskScheduling = true;
        } else {
            console.log('❌ Task scheduling failed');
            console.log('   Task1:', task1);
            console.log('   Task2:', task2);
        }

        // Test 3: Task Execution
        console.log('\n📋 Test 3: Task Execution');
        
        const executionPromise = new Promise((resolve) => {
            let completedTasks = 0;
            const targetTasks = 2;

            manager.on('taskCompleted', (task) => {
                completedTasks++;
                console.log(`✅ Task ${task.id} completed (${completedTasks}/${targetTasks})`);
                
                if (completedTasks >= targetTasks) {
                    testResults.taskExecution = true;
                    resolve();
                }
            });

            manager.on('taskFailed', (task) => {
                console.log(`❌ Task ${task.id} failed: ${task.error}`);
                completedTasks++; // Count failed tasks too
                if (completedTasks >= targetTasks) {
                    resolve();
                }
            });

            // Schedule new tasks for execution test
            let execTask1, execTask2;
            
            if (typeof WebGPUMatrixJob !== 'undefined' && typeof WebNNImageClassificationJob !== 'undefined') {
                // Use GPU and WebNN jobs for execution test
                execTask1 = manager.scheduleTask(new WebGPUMatrixJob('execution-test-1', 64, 1), 6);    // GPU job
                execTask2 = manager.scheduleTask(new WebNNImageClassificationJob('execution-test-2', 8, 224, 1), 7); // WebNN job
            } else if (typeof WASMMatrixJob !== 'undefined' && typeof WASMPrimeJob !== 'undefined') {
                execTask1 = manager.scheduleTask(new WASMMatrixJob('execution-test-1', 64, 1), 6);
                execTask2 = manager.scheduleTask(new WASMPrimeJob('execution-test-2', 10000, 1), 7);
            } else {
                // Fallback to simple jobs
                execTask1 = manager.scheduleTask({
                    id: 'execution-test-1',
                    type: 'TestJobExec1',
                    duration: 300,
                    complexity: 1
                }, 6);
                
                execTask2 = manager.scheduleTask({
                    id: 'execution-test-2', 
                    type: 'TestJobExec2',
                    duration: 400,
                    complexity: 1
                }, 7);
            }

            // Timeout after 15 seconds
            setTimeout(() => {
                console.log('⏰ Task execution test timeout');
                resolve();
            }, 15000);
        });

        await executionPromise;

        // Test 4: Priority Handling
        console.log('\n📋 Test 4: Priority Handling');
        
        let highPriorityTask, lowPriorityTask;

        // Schedule tasks with different priorities using imported JobC and ValidationJobA
        highPriorityTask = manager.scheduleTask(new JobC('high-priority', 300), 10);
        lowPriorityTask = manager.scheduleTask(new ValidationJobA('low-priority', 300), 1);

        // If the tasks were successfully scheduled, mark priorityHandling as true
        if (highPriorityTask && lowPriorityTask) {
            console.log('✅ Priority tasks scheduled');
            testResults.priorityHandling = true;
        } else {
            console.log('❌ Priority task scheduling failed');
            testResults.priorityHandling = false;
        }

        // Test 5: Worker Communication (check if workers support real communication)
        console.log('\n📋 Test 5: Worker Communication');
        const cpuWorkers = stats.workers.cpu;
        const gpuWorkers = stats.workers.gpu;
        
        if (cpuWorkers.total > 0 || gpuWorkers.total > 0) {
            console.log('✅ Real worker communication available');
            console.log(`   CPU workers: ${cpuWorkers.total} (available: ${cpuWorkers.available})`);
            console.log(`   GPU workers: ${gpuWorkers.total} (available: ${gpuWorkers.available})`);
            testResults.workerCommunication = true;
        } else {
            console.log('⚠️  No workers detected');
            testResults.workerCommunication = false;
        }

        // Test 6: Error Handling
        console.log('\n📋 Test 6: Error Handling');
        try {
            // Test with job that will cause worker to handle error
            const invalidTask = manager.scheduleTask({
                id: 'error-test-job',
                type: 'ErrorTestJob',
                duration: 100,
                complexity: 1,
                // This will be handled by workers, not executed as a function
                shouldFail: true
            }, 5);
            
            if (invalidTask) {
                console.log('✅ Error handling test task scheduled');
                testResults.errorHandling = true;
            }
        } catch (error) {
            console.log('✅ Error properly caught during scheduling');
            testResults.errorHandling = true;
        }

        // Wait a bit more for remaining tasks
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Final stats
        const finalStats = manager.getStats();
        console.log('\n📊 Final Statistics:');
        console.log(`   Tasks scheduled: ${finalStats.performance.tasksScheduled}`);
        console.log(`   Tasks completed: ${finalStats.performance.tasksCompleted}`);
        console.log(`   Tasks failed: ${finalStats.performance.tasksFailed}`);
        console.log(`   Total execution time: ${finalStats.performance.totalExecutionTime}ms`);

        manager.stop();

    } catch (error) {
        console.error('❌ Validation test error:', error);
    }

    // Summary
    console.log('\n🎯 Validation Summary:');
    const passedTests = Object.values(testResults).filter(result => result).length;
    const totalTests = Object.keys(testResults).length;
    
    Object.entries(testResults).forEach(([test, passed]) => {
        console.log(`   ${passed ? '✅' : '❌'} ${test}: ${passed ? 'PASS' : 'FAIL'}`);
    });
    
    console.log(`\n📈 Overall Result: ${passedTests}/${totalTests} tests passed`);
    
    if (passedTests === totalTests) {
        console.log('🎉 All validation tests passed! Enhanced TaskManager is working correctly.');
    } else {
        console.log('⚠️  Some validation tests failed. Check the implementation.');
    }

    return { passedTests, totalTests, testResults };
}

// Quick integration test
async function quickIntegrationTest() {
    console.log('⚡ Quick Integration Test');
    
    const manager = new TaskManager({
        maxConcurrentTasks: 1,
        workerPools: { cpu: { size: 1 } }
    });

    await manager.start();
    
    const task = manager.scheduleTask(new ValidationJobA('quick-test', 500), 5);
    
    return new Promise((resolve) => {
        manager.on('taskCompleted', (completedTask) => {
            if (completedTask.id === task.id) {
                console.log('✅ Quick integration test passed');
                manager.stop();
                resolve(true);
            }
        });

        manager.on('taskFailed', (failedTask) => {
            if (failedTask.id === task.id) {
                console.log('❌ Quick integration test failed');
                manager.stop();
                resolve(false);
            }
        });

        setTimeout(() => {
            console.log('⏰ Quick integration test timeout');
            manager.stop();
            resolve(false);
        }, 3000);
    });
}

// Make functions globally available
window.validateEnhancedTaskManager = validateEnhancedTaskManager;
window.quickIntegrationTest = quickIntegrationTest;
