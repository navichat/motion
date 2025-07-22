/**
 * Worker Assignment Test - Verify tasks go to correct worker types
 */

async function testWorkerAssignment() {
    console.log('🧪 Testing Worker Assignment...');
    
    try {
        // Initialize TaskManager
        const manager = new TaskManager();
        await manager.initialize();
        
        console.log('\n✅ TaskManager initialized');
        
        // Test 1: CPU Job (WASM)
        console.log('\n📋 Test 1: CPU Job Assignment');
        if (typeof WASMMatrixJob !== 'undefined') {
            const cpuJob = new WASMMatrixJob('test-cpu-job', 64, 1);
            console.log('CPU Job resource requirements:', cpuJob.resourceRequirements);
            const cpuTaskId = manager.scheduleTask(cpuJob, 5);
            console.log(`CPU task scheduled: ${cpuTaskId}`);
        }
        
        // Test 2: GPU Job (WebGPU)
        console.log('\n📋 Test 2: GPU Job Assignment');
        if (typeof WebGPUMatrixJob !== 'undefined') {
            const gpuJob = new WebGPUMatrixJob('test-gpu-job', 64, 1);
            console.log('GPU Job resource requirements:', gpuJob.resourceRequirements);
            const gpuTaskId = manager.scheduleTask(gpuJob, 6);
            console.log(`GPU task scheduled: ${gpuTaskId}`);
        }
        
        // Test 3: WebNN Job
        console.log('\n📋 Test 3: WebNN Job Assignment');
        if (typeof WebNNImageClassificationJob !== 'undefined') {
            const webnnJob = new WebNNImageClassificationJob('test-webnn-job', 8, 224, 1);
            console.log('WebNN Job resource requirements:', webnnJob.resourceRequirements);
            const webnnTaskId = manager.scheduleTask(webnnJob, 7);
            console.log(`WebNN task scheduled: ${webnnTaskId}`);
        }
        
        // Wait to see task assignments
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Get statistics
        const stats = manager.getStatistics();
        console.log('\n📊 Worker Statistics:');
        console.log(`CPU Workers: ${stats.workers.cpu.busy}/${stats.workers.cpu.total} busy`);
        console.log(`GPU Workers: ${stats.workers.gpu.busy}/${stats.workers.gpu.total} busy`);
        console.log(`WebNN Workers: ${stats.workers.webnn.busy}/${stats.workers.webnn.total} busy`);
        
        // Wait for completion
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        manager.shutdown();
        console.log('\n🎉 Worker assignment test completed!');
        
    } catch (error) {
        console.error('❌ Worker assignment test failed:', error);
    }
}

// Make it available globally
window.testWorkerAssignment = testWorkerAssignment;
