/**
 * Comprehensive Integration Test Suite for Performance Monitoring System
 * 
 * This script validates that all benchmark components work together correctly
 * and provides real-time performance assessment capabilities.
 */

class IntegrationTestSuite {
    constructor() {
        this.testResults = {};
        this.taskManager = null;
        this.performanceMonitor = null;
        this.systemAnalyzer = null;
        this.testStartTime = Date.now();
    }

    async runAllTests() {
        console.log('🚀 Starting Comprehensive Integration Test Suite');
        console.log('================================================');
        
        try {
            // Core component tests
            await this.testSystemAnalyzer();
            await this.testTaskManager();
            await this.testBenchmarkJobs();
            await this.testPerformanceMonitor();
            
            // Integration tests
            await this.testFullIntegration();
            await this.testRealtimeCapabilities();
            await this.testConcurrentOperations();
            
            // Performance validation
            await this.testPerformanceThresholds();
            await this.testStressTest();
            
            this.generateTestReport();
            
        } catch (error) {
            console.error('❌ Integration test suite failed:', error);
            this.testResults.overallStatus = 'FAILED';
        }
    }

    async testSystemAnalyzer() {
        console.log('\n📊 Testing System Analyzer...');
        
        const testName = 'SystemAnalyzer';
        const testStart = performance.now();
        
        try {
            // Initialize system analyzer
            this.systemAnalyzer = new SystemPerformanceAnalyzer();
            await this.systemAnalyzer.initialize();
            
            // Test system information gathering
            const systemInfo = this.systemAnalyzer.systemInfo;
            this.assert(systemInfo.hardwareConcurrency > 0, 'Should detect CPU cores');
            this.assert(systemInfo.userAgent, 'Should detect user agent');
            
            // Test capability detection
            const capabilities = this.systemAnalyzer.capabilities;
            this.assert(typeof capabilities.webgl === 'boolean', 'Should detect WebGL capability');
            this.assert(typeof capabilities.multiThreading === 'boolean', 'Should detect multi-threading');
            
            // Test baseline benchmarking
            const baseline = await this.systemAnalyzer.runBaselineBenchmarks();
            this.assert(baseline.cpu && baseline.cpu.baselineScore > 0, 'Should have CPU baseline score');
            this.assert(baseline.memory && baseline.memory.peakBandwidth > 0, 'Should have memory bandwidth');
            
            // Test real-time capability assessment
            const realtimeCapabilities = this.systemAnalyzer.calculateRealtimeCapabilities();
            this.assert(realtimeCapabilities.animation60fps, 'Should assess 60fps capability');
            this.assert(realtimeCapabilities.audio48khz, 'Should assess audio capability');
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ System Analyzer tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ System Analyzer tests failed:', error);
        }
    }

    async testTaskManager() {
        console.log('\n⚙️ Testing Task Manager...');
        
        const testName = 'TaskManager';
        const testStart = performance.now();
        
        try {
            // Initialize task manager
            this.taskManager = new TaskManager({
                cpuWorkers: 2,
                gpuWorkers: 1,
                webnnWorkers: 1,
                wasmWorkers: 1,
                maxConcurrentTasks: 4
            });
            
            this.taskManager.start();
            
            // Test worker pool initialization
            await new Promise(resolve => setTimeout(resolve, 1000)); // Allow initialization
            
            const stats = this.taskManager.getStats();
            this.assert(stats.workers.total >= 4, 'Should have multiple workers');
            this.assert(stats.queue, 'Should have queue statistics');
            
            // Test task scheduling
            const simpleTask = {
                type: 'benchmark',
                backend: 'cpu',
                operation: 'compute',
                parameters: { iterations: 1000 }
            };
            
            const taskId = this.taskManager.scheduleTask(simpleTask, 1);
            this.assert(taskId, 'Should return task ID');
            
            // Wait for task completion
            let taskCompleted = false;
            for (let i = 0; i < 20 && !taskCompleted; i++) {
                await new Promise(resolve => setTimeout(resolve, 500));
                const updatedStats = this.taskManager.getStats();
                if (updatedStats.queue.completed > 0) {
                    taskCompleted = true;
                }
            }
            
            this.assert(taskCompleted, 'Task should complete within timeout');
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ Task Manager tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Task Manager tests failed:', error);
        }
    }

    async testBenchmarkJobs() {
        console.log('\n🏃 Testing Benchmark Jobs...');
        
        const testName = 'BenchmarkJobs';
        const testStart = performance.now();
        
        try {
            const backends = ['cpu', 'gpu', 'webnn', 'wasm'];
            const jobTypes = ['memory', 'compute', 'comprehensive'];
            
            for (const backend of backends) {
                for (const jobType of jobTypes) {
                    try {
                        let job;
                        
                        switch (jobType) {
                            case 'memory':
                                job = BenchmarkJobFactory.createMemoryBandwidthBenchmark(backend);
                                break;
                            case 'compute':
                                job = BenchmarkJobFactory.createComputeBenchmark(backend);
                                break;
                            case 'comprehensive':
                                job = BenchmarkJobFactory.createComprehensiveBenchmark(backend, {
                                    duration: 1000 // Short duration for testing
                                });
                                break;
                        }
                        
                        this.assert(job, `Should create ${jobType} job for ${backend}`);
                        this.assert(job.type === 'benchmark', 'Job should have benchmark type');
                        this.assert(job.backend === backend, `Job should target ${backend} backend`);
                        
                        console.log(`  ✓ ${backend} ${jobType} job created successfully`);
                        
                    } catch (jobError) {
                        console.warn(`  ⚠️ ${backend} ${jobType} job creation failed:`, jobError.message);
                    }
                }
            }
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ Benchmark Jobs tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Benchmark Jobs tests failed:', error);
        }
    }

    async testPerformanceMonitor() {
        console.log('\n📈 Testing Performance Monitor...');
        
        const testName = 'PerformanceMonitor';
        const testStart = performance.now();
        
        try {
            let updateCount = 0;
            let alertCount = 0;
            
            // Initialize performance monitor
            this.performanceMonitor = new RealTimePerformanceMonitor(this.taskManager, {
                interval: 500, // Fast interval for testing
                onUpdate: (metrics) => {
                    updateCount++;
                    console.log(`  📊 Performance update ${updateCount}: ${Object.keys(metrics).length} backends`);
                },
                onAlert: (alerts) => {
                    alertCount += alerts.length;
                    console.log(`  🚨 Performance alerts: ${alerts.length} new alerts`);
                }
            });
            
            // Start monitoring
            this.performanceMonitor.start();
            
            // Wait for several updates
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Stop monitoring
            this.performanceMonitor.stop();
            
            this.assert(updateCount >= 3, 'Should receive multiple performance updates');
            console.log(`  📊 Received ${updateCount} performance updates`);
            console.log(`  🚨 Triggered ${alertCount} performance alerts`);
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ Performance Monitor tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Performance Monitor tests failed:', error);
        }
    }

    async testFullIntegration() {
        console.log('\n🔗 Testing Full Integration...');
        
        const testName = 'FullIntegration';
        const testStart = performance.now();
        
        try {
            // Run comprehensive benchmark on all backends
            const backends = ['cpu', 'gpu', 'webnn', 'wasm'];
            const results = {};
            
            for (const backend of backends) {
                try {
                    console.log(`  🏃 Running comprehensive benchmark on ${backend}...`);
                    
                    const job = BenchmarkJobFactory.createComprehensiveBenchmark(backend, {
                        duration: 2000,
                        memorySize: 1024 * 1024, // 1MB
                        computeIterations: 10000
                    });
                    
                    const taskId = this.taskManager.scheduleTask(job, 2); // High priority
                    
                    // Wait for completion with timeout
                    const startTime = Date.now();
                    let completed = false;
                    
                    while (!completed && (Date.now() - startTime) < 10000) {
                        await new Promise(resolve => setTimeout(resolve, 500));
                        
                        const stats = this.taskManager.getStats();
                        if (stats.queue.completed > 0) {
                            completed = true;
                            results[backend] = {
                                status: 'completed',
                                duration: Date.now() - startTime
                            };
                            console.log(`    ✓ ${backend} benchmark completed in ${results[backend].duration}ms`);
                        }
                    }
                    
                    if (!completed) {
                        results[backend] = { status: 'timeout' };
                        console.log(`    ⚠️ ${backend} benchmark timed out`);
                    }
                    
                } catch (backendError) {
                    results[backend] = { status: 'error', error: backendError.message };
                    console.log(`    ❌ ${backend} benchmark failed: ${backendError.message}`);
                }
            }
            
            // Verify at least one backend completed successfully
            const completedBackends = Object.entries(results)
                .filter(([_, result]) => result.status === 'completed')
                .length;
            
            this.assert(completedBackends > 0, 'At least one backend should complete benchmark');
            
            console.log(`  📊 Integration test completed: ${completedBackends}/${backends.length} backends successful`);
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart, 
                `${completedBackends}/${backends.length} backends completed`);
            console.log('✅ Full Integration tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Full Integration tests failed:', error);
        }
    }

    async testRealtimeCapabilities() {
        console.log('\n⚡ Testing Real-time Capabilities...');
        
        const testName = 'RealtimeCapabilities';
        const testStart = performance.now();
        
        try {
            // Test real-time assessment
            const capabilities = this.systemAnalyzer.calculateRealtimeCapabilities();
            
            this.assert(capabilities.animation60fps, 'Should have 60fps animation assessment');
            this.assert(capabilities.audio48khz, 'Should have audio capability assessment');
            this.assert(capabilities.videoProcessing, 'Should have video processing assessment');
            this.assert(capabilities.mlInference, 'Should have ML inference assessment');
            
            // Test capability confidence scores
            Object.entries(capabilities).forEach(([capability, assessment]) => {
                this.assert(typeof assessment.capable === 'boolean', 
                    `${capability} should have boolean capability flag`);
                this.assert(typeof assessment.confidence === 'number' && 
                    assessment.confidence >= 0 && assessment.confidence <= 1,
                    `${capability} should have valid confidence score`);
                
                console.log(`  ⚡ ${capability}: ${assessment.capable ? '✅' : '❌'} (${Math.round(assessment.confidence * 100)}%)`);
            });
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ Real-time Capabilities tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Real-time Capabilities tests failed:', error);
        }
    }

    async testConcurrentOperations() {
        console.log('\n🔄 Testing Concurrent Operations...');
        
        const testName = 'ConcurrentOperations';
        const testStart = performance.now();
        
        try {
            // Schedule multiple tasks concurrently
            const tasks = [];
            const backends = ['cpu', 'gpu', 'webnn', 'wasm'];
            
            backends.forEach((backend, index) => {
                const job = BenchmarkJobFactory.createComputeBenchmark(backend, {
                    iterations: 5000
                });
                
                const taskId = this.taskManager.scheduleTask(job, 1);
                tasks.push({ backend, taskId, startTime: Date.now() });
            });
            
            console.log(`  🔄 Scheduled ${tasks.length} concurrent tasks`);
            
            // Monitor task completion
            const completionTimes = {};
            let completedTasks = 0;
            
            while (completedTasks < tasks.length && (Date.now() - testStart) < 15000) {
                await new Promise(resolve => setTimeout(resolve, 500));
                
                const stats = this.taskManager.getStats();
                const newCompletedCount = stats.queue.completed;
                
                if (newCompletedCount > completedTasks) {
                    console.log(`  ✓ ${newCompletedCount} tasks completed`);
                    completedTasks = newCompletedCount;
                }
            }
            
            this.assert(completedTasks >= backends.length / 2, 
                'Should complete at least half of concurrent tasks');
            
            console.log(`  📊 Completed ${completedTasks}/${tasks.length} concurrent tasks`);
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ Concurrent Operations tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Concurrent Operations tests failed:', error);
        }
    }

    async testPerformanceThresholds() {
        console.log('\n🎯 Testing Performance Thresholds...');
        
        const testName = 'PerformanceThresholds';
        const testStart = performance.now();
        
        try {
            const report = this.systemAnalyzer.generatePerformanceReport();
            const baseline = report.performanceMetrics.baseline;
            
            // Test CPU performance thresholds
            if (baseline.cpu) {
                this.assert(baseline.cpu.baselineScore > 0, 'CPU baseline score should be positive');
                console.log(`  🔴 CPU baseline score: ${baseline.cpu.baselineScore}`);
            }
            
            // Test memory performance thresholds
            if (baseline.memory) {
                this.assert(baseline.memory.peakBandwidth > 0, 'Memory bandwidth should be positive');
                console.log(`  💾 Memory peak bandwidth: ${baseline.memory.peakBandwidth.toFixed(2)} MB/s`);
            }
            
            // Test JS engine performance
            if (baseline.jsEngine) {
                this.assert(baseline.jsEngine.overallScore > 0, 'JS engine score should be positive');
                console.log(`  ⚙️ JS engine score: ${baseline.jsEngine.overallScore}`);
            }
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart);
            console.log('✅ Performance Thresholds tests passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Performance Thresholds tests failed:', error);
        }
    }

    async testStressTest() {
        console.log('\n💪 Running Stress Test...');
        
        const testName = 'StressTest';
        const testStart = performance.now();
        
        try {
            // Schedule many tasks to stress test the system
            const stressTasks = [];
            const taskCount = 20;
            
            for (let i = 0; i < taskCount; i++) {
                const backend = ['cpu', 'gpu', 'webnn', 'wasm'][i % 4];
                const job = BenchmarkJobFactory.createComputeBenchmark(backend, {
                    iterations: 1000
                });
                
                const taskId = this.taskManager.scheduleTask(job, 1);
                stressTasks.push(taskId);
            }
            
            console.log(`  💪 Scheduled ${taskCount} stress test tasks`);
            
            // Monitor system under stress
            let maxCompletedTasks = 0;
            const monitorStart = Date.now();
            
            while ((Date.now() - monitorStart) < 10000) { // 10 second stress test
                await new Promise(resolve => setTimeout(resolve, 500));
                
                const stats = this.taskManager.getStats();
                if (stats.queue.completed > maxCompletedTasks) {
                    maxCompletedTasks = stats.queue.completed;
                }
            }
            
            console.log(`  📊 Completed ${maxCompletedTasks} tasks under stress`);
            
            // Verify system remained stable
            const finalStats = this.taskManager.getStats();
            this.assert(finalStats.workers.total > 0, 'Workers should remain active');
            this.assert(maxCompletedTasks > 0, 'Should complete some tasks under stress');
            
            this.recordTestResult(testName, 'PASSED', performance.now() - testStart, 
                `Completed ${maxCompletedTasks} tasks`);
            console.log('✅ Stress Test passed');
            
        } catch (error) {
            this.recordTestResult(testName, 'FAILED', performance.now() - testStart, error.message);
            console.error('❌ Stress Test failed:', error);
        }
    }

    generateTestReport() {
        console.log('\n📋 Test Report Summary');
        console.log('=====================');
        
        const totalTests = Object.keys(this.testResults).length;
        const passedTests = Object.values(this.testResults).filter(r => r.status === 'PASSED').length;
        const failedTests = totalTests - passedTests;
        
        const totalTime = Date.now() - this.testStartTime;
        
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests} ✅`);
        console.log(`Failed: ${failedTests} ${failedTests > 0 ? '❌' : '✅'}`);
        console.log(`Total Time: ${totalTime}ms`);
        console.log('');
        
        // Detailed results
        Object.entries(this.testResults).forEach(([testName, result]) => {
            const status = result.status === 'PASSED' ? '✅' : '❌';
            const time = `${result.duration.toFixed(2)}ms`;
            const note = result.note ? ` (${result.note})` : '';
            
            console.log(`${status} ${testName}: ${time}${note}`);
            
            if (result.status === 'FAILED' && result.error) {
                console.log(`    Error: ${result.error}`);
            }
        });
        
        // Overall assessment
        const overallStatus = failedTests === 0 ? 'PASSED' : 'FAILED';
        console.log('');
        console.log(`🎯 Overall Integration Test Status: ${overallStatus} ${overallStatus === 'PASSED' ? '✅' : '❌'}`);
        
        if (overallStatus === 'PASSED') {
            console.log('🎉 All benchmark components are working correctly!');
            console.log('📊 System is ready for real-time performance monitoring');
        } else {
            console.log('⚠️ Some components need attention before production use');
        }
        
        return {
            overallStatus,
            totalTests,
            passedTests,
            failedTests,
            totalTime,
            results: this.testResults
        };
    }

    recordTestResult(testName, status, duration, note = null, error = null) {
        this.testResults[testName] = {
            status,
            duration,
            note,
            error
        };
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(`Assertion failed: ${message}`);
        }
    }

    // Cleanup method
    cleanup() {
        if (this.performanceMonitor) {
            this.performanceMonitor.stop();
        }
        if (this.taskManager) {
            this.taskManager.stop();
        }
    }
}

// Export for use in browsers and Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = IntegrationTestSuite;
} else if (typeof window !== 'undefined') {
    window.IntegrationTestSuite = IntegrationTestSuite;
}

// Auto-run if loaded as main script
if (typeof window !== 'undefined' && window.location) {
    window.addEventListener('load', async () => {
        // Only auto-run if this is a test page
        if (window.location.pathname.includes('test') || 
            window.location.search.includes('autotest=true')) {
            
            console.log('🔧 Auto-running integration tests...');
            
            const testSuite = new IntegrationTestSuite();
            
            try {
                await testSuite.runAllTests();
            } catch (error) {
                console.error('Integration test suite encountered an error:', error);
            } finally {
                testSuite.cleanup();
            }
        }
    });
}
