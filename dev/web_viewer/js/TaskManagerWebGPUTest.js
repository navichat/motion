/**
 * TaskManager WebGPU Test - Testing integration with WebGPU workers and performance
 */

import { TaskManager } from './TaskManager.js';
import { MockGPUJob } from './MockGPUJobs.js';

// Test TaskManager with WebGPU functionality and performance
async function testTaskManagerWebGPU() {
    console.log('🧪 Testing TaskManager with WebGPU Workers and Performance');

    const manager = new TaskManager({
        maxConcurrentTasks: 1,
        workerPools: {
            gpu: { size: 1 }
        }
    });

    const metrics = {
        schedulingTime: 0,
        executionTime: 0,
        workerAssignmentTime: 0
    };

    return new Promise(async (resolve, reject) => {
        const testTimeout = setTimeout(() => {
            reject(new Error('WebGPU test timed out'));
        }, 10000); // 10-second timeout for the entire test

        manager.on('taskCompleted', (task) => {
            console.log(`✅ Task ${task.id} completed successfully.`);
            metrics.executionTime = task.endTime - task.startTime;
            console.log(`📊 Performance Metrics:
              - Scheduling Time: ${metrics.schedulingTime.toFixed(2)}ms
              - Execution Time: ${metrics.executionTime.toFixed(2)}ms
              - Worker Assignment Time: ${metrics.workerAssignmentTime.toFixed(2)}ms`);
            clearTimeout(testTimeout);
            manager.stop();
            resolve();
        });

        manager.on('taskFailed', (task) => {
            console.error(`❌ Task ${task.id} failed: ${task.error}`);
            clearTimeout(testTimeout);
            manager.stop();
            reject(new Error(`Task ${task.id} failed: ${task.error}`));
        });

        manager.on('taskStarted', (task) => {
            metrics.workerAssignmentTime = performance.now() - task.startTime;
        });

        await manager.start();

        console.log('📝 Scheduling a WebGPU task...');
        const schedulingStartTime = performance.now();
        const job = new MockGPUJob('webgpu-job', 2000, 1, { backend: 'gpu' });
        manager.scheduleTask(job, 1);
        metrics.schedulingTime = performance.now() - schedulingStartTime;
    });
}

// Main test runner
async function runWebGPUTests() {
    try {
        await testTaskManagerWebGPU();
        console.log('All WebGPU tests completed!');
    } catch (error) {
        console.error('❌ WebGPU test error:', error);
    }
}

// Export for use
export { testTaskManagerWebGPU, runWebGPUTests };

// Export functions for ES6 modules
export { testTaskManagerWebGPU, runWebGPUTests };

// Auto-run if this file is loaded directly
if (typeof window !== 'undefined') {
    window.testTaskManagerWebGPU = testTaskManagerWebGPU;
    window.runWebGPUTests = runWebGPUTests;

    document.addEventListener('DOMContentLoaded', () => {
        console.log('🚀 TaskManager WebGPU tests loaded. Run with: runWebGPUTests()');
    });
}