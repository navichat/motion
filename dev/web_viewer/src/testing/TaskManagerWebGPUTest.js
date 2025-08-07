import { MockGPUJob } from '../../testing/MockGPUJobs.js';

/**
 * TaskManager WebGPU Test - Testing integration with WebGPU workers and performance
 */

// Test TaskManager with WebGPU functionality and performance
async function testTaskManagerWebGPU() {
    console.log('🧪 Testing TaskManager with WebGPU Workers and Performance');

    // Check if required classes are available
    if (typeof TaskManager === 'undefined') {
        throw new Error('TaskManager class not available');
    }
    if (typeof MockGPUJob === 'undefined') {
        throw new Error('MockGPUJob class not available');
    }

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

// Make functions available globally
if (typeof window !== 'undefined') {
    window.testTaskManagerWebGPU = testTaskManagerWebGPU;
    window.runWebGPUTests = runWebGPUTests;

    console.log('🚀 TaskManager WebGPU tests loaded. Run with: runWebGPUTests()');
}