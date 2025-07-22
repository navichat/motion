/**
 * Playwright Tests for TaskManager Performance and Optimization
 * Tests FLOPS, memory bandwidth, task throughput, and worker efficiency
 */

const { test, expect } = require('@playwright/test');

// Test configuration
const TEST_CONFIG = {
    testPage: '/dev/web_viewer/taskmanager-perf-test.html',
    perfMetrics: {
        maxLatency: 5000, // Max acceptable task latency (ms)
        minThroughput: 10, // Min tasks per second
        maxMemoryUsage: 500, // Max memory usage (MB)
        minFLOPS: 1000000, // Min floating point operations per second
    }
};

// Helper function to inject mock jobs
async function injectMockJobs(page) {
    await page.addInitScript(() => {
        // Mock job classes for testing
        window.MockCPUJob = class {
            constructor(options = {}) {
                this.type = 'MockCPU';
                this.duration = options.duration || 1000;
                this.complexity = options.complexity || 1;
                this.resourceRequirements = { cpu: 1, memory: 50 };
                this.progress = 0;
                this.flops = 0;
                this.memoryBandwidth = 0;
            }

            async execute(progressCallback, shouldStop) {
                const start = performance.now();
                const steps = 100;
                const operations = 1000000 * this.complexity; // 1M ops per complexity
                
                for (let i = 0; i < steps && !shouldStop(); i++) {
                    // Simulate CPU-intensive work
                    let result = 0;
                    for (let j = 0; j < operations / steps; j++) {
                        result += Math.sqrt(j) * Math.sin(j);
                    }
                    
                    this.progress = (i + 1) / steps;
                    this.flops += operations / steps;
                    
                    if (progressCallback) {
                        progressCallback(this.progress * 100, {
                            flops: this.flops,
                            memoryBandwidth: this.memoryBandwidth,
                            operations: operations
                        });
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, this.duration / steps));
                }
                
                const executionTime = performance.now() - start;
                return {
                    success: true,
                    executionTime,
                    flops: this.flops,
                    operations,
                    flopsPerSecond: this.flops / (executionTime / 1000)
                };
            }

            interrupt() {
                // Mock interrupt
            }
        };

        window.MockGPUJob = class {
            constructor(options = {}) {
                this.type = 'MockGPU';
                this.duration = options.duration || 500;
                this.complexity = options.complexity || 1;
                this.resourceRequirements = { gpu: 1, memory: 100 };
                this.backend = 'gpu';
                this.progress = 0;
                this.flops = 0;
                this.memoryBandwidth = 0;
            }

            async execute(progressCallback, shouldStop) {
                const start = performance.now();
                const steps = 50;
                const matrixSize = 1000 * this.complexity;
                const memoryOps = matrixSize * matrixSize * 4; // 4 bytes per float
                
                for (let i = 0; i < steps && !shouldStop(); i++) {
                    // Simulate GPU matrix operations
                    const operations = matrixSize * matrixSize * 2; // Multiply-add
                    this.flops += operations;
                    this.memoryBandwidth += memoryOps;
                    
                    this.progress = (i + 1) / steps;
                    
                    if (progressCallback) {
                        progressCallback(this.progress * 100, {
                            flops: this.flops,
                            memoryBandwidth: this.memoryBandwidth,
                            matrixSize
                        });
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, this.duration / steps));
                }
                
                const executionTime = performance.now() - start;
                return {
                    success: true,
                    executionTime,
                    flops: this.flops,
                    memoryBandwidth: this.memoryBandwidth,
                    flopsPerSecond: this.flops / (executionTime / 1000),
                    bandwidthPerSecond: this.memoryBandwidth / (executionTime / 1000)
                };
            }

            interrupt() {
                // Mock interrupt
            }
        };

        window.MockWebNNJob = class {
            constructor(options = {}) {
                this.type = 'MockWebNN';
                this.duration = options.duration || 800;
                this.complexity = options.complexity || 1;
                this.resourceRequirements = { webnn: 1, memory: 200 };
                this.backend = 'webnn';
                this.progress = 0;
                this.flops = 0;
                this.inferences = 0;
            }

            async execute(progressCallback, shouldStop) {
                const start = performance.now();
                const steps = 20;
                const inferenceOps = 50000000 * this.complexity; // 50M ops per inference
                
                for (let i = 0; i < steps && !shouldStop(); i++) {
                    this.flops += inferenceOps;
                    this.inferences++;
                    this.progress = (i + 1) / steps;
                    
                    if (progressCallback) {
                        progressCallback(this.progress * 100, {
                            flops: this.flops,
                            inferences: this.inferences,
                            inferenceOps
                        });
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, this.duration / steps));
                }
                
                const executionTime = performance.now() - start;
                return {
                    success: true,
                    executionTime,
                    flops: this.flops,
                    inferences: this.inferences,
                    flopsPerSecond: this.flops / (executionTime / 1000),
                    inferencesPerSecond: this.inferences / (executionTime / 1000)
                };
            }

            interrupt() {
                // Mock interrupt
            }
        };

        // Performance measurement utilities
        window.PerfMeasure = class {
            constructor() {
                this.metrics = {
                    totalFlops: 0,
                    totalMemoryBandwidth: 0,
                    totalTasks: 0,
                    totalExecutionTime: 0,
                    peakMemoryUsage: 0,
                    startTime: null,
                    endTime: null
                };
            }

            start() {
                this.metrics.startTime = performance.now();
                // Start memory monitoring
                if (performance.memory) {
                    this.memoryInterval = setInterval(() => {
                        const memUsage = performance.memory.usedJSHeapSize / 1024 / 1024; // MB
                        this.metrics.peakMemoryUsage = Math.max(this.metrics.peakMemoryUsage, memUsage);
                    }, 100);
                }
            }

            recordTask(result) {
                this.metrics.totalTasks++;
                this.metrics.totalExecutionTime += result.executionTime || 0;
                this.metrics.totalFlops += result.flops || 0;
                this.metrics.totalMemoryBandwidth += result.memoryBandwidth || 0;
            }

            finish() {
                this.metrics.endTime = performance.now();
                if (this.memoryInterval) {
                    clearInterval(this.memoryInterval);
                }
                
                const totalTime = (this.metrics.endTime - this.metrics.startTime) / 1000; // seconds
                
                return {
                    ...this.metrics,
                    totalTimeSeconds: totalTime,
                    tasksPerSecond: this.metrics.totalTasks / totalTime,
                    avgFlopsPerSecond: this.metrics.totalFlops / totalTime,
                    avgBandwidthPerSecond: this.metrics.totalMemoryBandwidth / totalTime,
                    avgTaskExecutionTime: this.metrics.totalExecutionTime / this.metrics.totalTasks,
                    peakMemoryUsageMB: this.metrics.peakMemoryUsage
                };
            }
        };
    });
}

test.describe('TaskManager Performance Tests', () => {
    test.beforeEach(async ({ page }) => {
        await injectMockJobs(page);
        await page.goto(TEST_CONFIG.testPage);
        
        // Wait for TaskManager to be available
        await page.waitForFunction(() => window.TaskManager && window.FibonacciHeap);
    });

    test('Basic TaskManager functionality', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const taskManager = new TaskManager({
                cpuWorkers: 2,
                gpuWorkers: 1,
                webnnWorkers: 1
            });
            
            taskManager.start();
            
            // Schedule a simple task
            const job = new MockCPUJob({ duration: 100, complexity: 1 });
            const taskId = taskManager.scheduleTask(job, 1);
            
            // Wait for completion
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    if (task.id === taskId) resolve();
                });
            });
            
            const stats = taskManager.getStats();
            taskManager.stop();
            
            return {
                taskCompleted: stats.performance.tasksCompleted === 1,
                workersInitialized: stats.workers.cpu.total === 2,
                stats
            };
        });
        
        expect(result.taskCompleted).toBe(true);
        expect(result.workersInitialized).toBe(true);
        expect(result.stats.performance.tasksCompleted).toBe(1);
    });

    test('CPU-intensive workload performance', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const perfMeasure = new PerfMeasure();
            const taskManager = new TaskManager({
                cpuWorkers: 4,
                maxConcurrentTasks: 4
            });
            
            taskManager.start();
            perfMeasure.start();
            
            const tasks = [];
            const numTasks = 20;
            
            // Schedule CPU-intensive tasks
            for (let i = 0; i < numTasks; i++) {
                const job = new MockCPUJob({ 
                    duration: 200, 
                    complexity: Math.floor(Math.random() * 3) + 1 
                });
                const taskId = taskManager.scheduleTask(job, Math.floor(Math.random() * 10));
                tasks.push(taskId);
            }
            
            // Wait for all tasks to complete
            let completedTasks = 0;
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    perfMeasure.recordTask(task.result);
                    completedTasks++;
                    if (completedTasks === numTasks) resolve();
                });
            });
            
            const perfResults = perfMeasure.finish();
            const stats = taskManager.getStats();
            taskManager.stop();
            
            return {
                perfResults,
                stats,
                numTasks
            };
        });
        
        expect(result.stats.performance.tasksCompleted).toBe(result.numTasks);
        expect(result.perfResults.tasksPerSecond).toBeGreaterThan(5); // At least 5 tasks/sec
        expect(result.perfResults.avgFlopsPerSecond).toBeGreaterThan(TEST_CONFIG.perfMetrics.minFLOPS);
        expect(result.perfResults.peakMemoryUsageMB).toBeLessThan(TEST_CONFIG.perfMetrics.maxMemoryUsage);
    });

    test('Mixed workload with multiple worker types', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const perfMeasure = new PerfMeasure();
            const taskManager = new TaskManager({
                cpuWorkers: 2,
                gpuWorkers: 2,
                webnnWorkers: 1,
                maxConcurrentTasks: 5
            });
            
            taskManager.start();
            perfMeasure.start();
            
            const tasks = [];
            const workloadMix = [
                { type: 'cpu', count: 10, JobClass: MockCPUJob },
                { type: 'gpu', count: 8, JobClass: MockGPUJob },
                { type: 'webnn', count: 5, JobClass: MockWebNNJob }
            ];
            
            // Schedule mixed workload
            for (const workload of workloadMix) {
                for (let i = 0; i < workload.count; i++) {
                    const job = new workload.JobClass({ 
                        duration: 150 + Math.random() * 200,
                        complexity: Math.floor(Math.random() * 2) + 1
                    });
                    const taskId = taskManager.scheduleTask(job, Math.floor(Math.random() * 10));
                    tasks.push({ id: taskId, type: workload.type });
                }
            }
            
            // Track completion by worker type
            const completionStats = { cpu: 0, gpu: 0, webnn: 0 };
            let totalCompleted = 0;
            const totalTasks = workloadMix.reduce((sum, w) => sum + w.count, 0);
            
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    perfMeasure.recordTask(task.result);
                    
                    // Determine worker type from task
                    if (task.job.type === 'MockCPU') completionStats.cpu++;
                    else if (task.job.type === 'MockGPU') completionStats.gpu++;
                    else if (task.job.type === 'MockWebNN') completionStats.webnn++;
                    
                    totalCompleted++;
                    if (totalCompleted === totalTasks) resolve();
                });
            });
            
            const perfResults = perfMeasure.finish();
            const stats = taskManager.getStats();
            taskManager.stop();
            
            return {
                perfResults,
                stats,
                completionStats,
                totalTasks,
                workloadMix: workloadMix.map(w => ({ type: w.type, count: w.count }))
            };
        });
        
        expect(result.stats.performance.tasksCompleted).toBe(result.totalTasks);
        expect(result.completionStats.cpu).toBe(10);
        expect(result.completionStats.gpu).toBe(8);
        expect(result.completionStats.webnn).toBe(5);
        expect(result.perfResults.tasksPerSecond).toBeGreaterThan(8); // Higher throughput with mixed workload
    });

    test('Priority and preemption performance', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const taskManager = new TaskManager({
                cpuWorkers: 2,
                maxConcurrentTasks: 2,
                preemptionEnabled: true
            });
            
            taskManager.start();
            
            const results = [];
            let taskCounter = 0;
            
            // Schedule low priority tasks first
            const lowPriorityTasks = [];
            for (let i = 0; i < 4; i++) {
                const job = new MockCPUJob({ duration: 2000, complexity: 1 });
                const taskId = taskManager.scheduleTask(job, 10); // Low priority
                lowPriorityTasks.push(taskId);
            }
            
            // Wait a bit for tasks to start
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Schedule high priority tasks
            const highPriorityTasks = [];
            for (let i = 0; i < 2; i++) {
                const job = new MockCPUJob({ duration: 300, complexity: 1 });
                const taskId = taskManager.scheduleTask(job, 1); // High priority
                highPriorityTasks.push(taskId);
            }
            
            const startTime = performance.now();
            
            // Track task completion order
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    results.push({
                        taskId: task.id,
                        priority: task.priority,
                        completionTime: performance.now() - startTime,
                        isHighPriority: highPriorityTasks.includes(task.id)
                    });
                    
                    if (results.length === 6) resolve(); // All tasks completed
                });
            });
            
            const stats = taskManager.getStats();
            taskManager.stop();
            
            // Analyze preemption effectiveness
            const highPriorityCompletions = results.filter(r => r.isHighPriority);
            const avgHighPriorityTime = highPriorityCompletions.reduce((sum, r) => sum + r.completionTime, 0) / highPriorityCompletions.length;
            
            return {
                results,
                stats,
                preemptionsOccurred: stats.performance.tasksPreempted > 0,
                avgHighPriorityCompletionTime: avgHighPriorityTime,
                highPriorityTasksCompletedFirst: highPriorityCompletions.every(hp => 
                    results.filter(r => !r.isHighPriority && r.completionTime < hp.completionTime).length < 2
                )
            };
        });
        
        expect(result.preemptionsOccurred).toBe(true);
        expect(result.avgHighPriorityCompletionTime).toBeLessThan(1000); // High priority tasks complete quickly
        expect(result.stats.performance.tasksCompleted).toBe(6);
    });

    test('Memory bandwidth optimization', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const perfMeasure = new PerfMeasure();
            const taskManager = new TaskManager({
                gpuWorkers: 2,
                maxConcurrentTasks: 4
            });
            
            taskManager.start();
            perfMeasure.start();
            
            // Schedule memory-intensive GPU tasks
            const tasks = [];
            for (let i = 0; i < 10; i++) {
                const job = new MockGPUJob({ 
                    duration: 300,
                    complexity: 2 + i % 3 // Varying complexity
                });
                const taskId = taskManager.scheduleTask(job, Math.floor(Math.random() * 5));
                tasks.push(taskId);
            }
            
            let completedTasks = 0;
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    perfMeasure.recordTask(task.result);
                    completedTasks++;
                    if (completedTasks === tasks.length) resolve();
                });
            });
            
            const perfResults = perfMeasure.finish();
            const stats = taskManager.getStats();
            taskManager.stop();
            
            return {
                perfResults,
                stats,
                memoryEfficiency: perfResults.avgBandwidthPerSecond / perfResults.peakMemoryUsageMB
            };
        });
        
        expect(result.stats.performance.tasksCompleted).toBe(10);
        expect(result.perfResults.avgBandwidthPerSecond).toBeGreaterThan(1000000); // 1MB/s minimum
        expect(result.memoryEfficiency).toBeGreaterThan(1000); // Good memory efficiency ratio
    });

    test('Scalability stress test', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const perfMeasure = new PerfMeasure();
            const taskManager = new TaskManager({
                cpuWorkers: 4,
                gpuWorkers: 2,
                webnnWorkers: 2,
                maxConcurrentTasks: 8
            });
            
            taskManager.start();
            perfMeasure.start();
            
            const numTasks = 100;
            const tasks = [];
            
            // Schedule large number of tasks rapidly
            for (let i = 0; i < numTasks; i++) {
                const jobTypes = [MockCPUJob, MockGPUJob, MockWebNNJob];
                const JobClass = jobTypes[i % jobTypes.length];
                
                const job = new JobClass({ 
                    duration: 50 + Math.random() * 100,
                    complexity: 1
                });
                
                const taskId = taskManager.scheduleTask(job, Math.floor(Math.random() * 20));
                tasks.push(taskId);
            }
            
            let completedTasks = 0;
            let failedTasks = 0;
            
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    perfMeasure.recordTask(task.result);
                    completedTasks++;
                    if (completedTasks + failedTasks >= numTasks) resolve();
                });
                
                taskManager.on('taskFailed', (task) => {
                    failedTasks++;
                    if (completedTasks + failedTasks >= numTasks) resolve();
                });
            });
            
            const perfResults = perfMeasure.finish();
            const stats = taskManager.getStats();
            taskManager.stop();
            
            return {
                perfResults,
                stats,
                completedTasks,
                failedTasks,
                successRate: completedTasks / numTasks,
                scalabilityScore: perfResults.tasksPerSecond * stats.workers.cpu.total * stats.workers.gpu.total
            };
        });
        
        expect(result.successRate).toBeGreaterThan(0.95); // 95% success rate
        expect(result.perfResults.tasksPerSecond).toBeGreaterThan(15); // High throughput under load
        expect(result.scalabilityScore).toBeGreaterThan(100); // Good scalability metric
        expect(result.perfResults.peakMemoryUsageMB).toBeLessThan(800); // Memory stays under control
    });

    test('Worker efficiency and utilization', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const taskManager = new TaskManager({
                cpuWorkers: 3,
                gpuWorkers: 2,
                maxConcurrentTasks: 5
            });
            
            taskManager.start();
            
            const utilizationSamples = [];
            const sampleInterval = 100; // Sample every 100ms
            
            // Start utilization monitoring
            const monitorInterval = setInterval(() => {
                const stats = taskManager.getStats();
                utilizationSamples.push({
                    timestamp: Date.now(),
                    cpuUtilization: stats.workers.cpu.busy / stats.workers.cpu.total,
                    gpuUtilization: stats.workers.gpu.busy / stats.workers.gpu.total,
                    totalUtilization: (stats.workers.cpu.busy + stats.workers.gpu.busy) / 
                                     (stats.workers.cpu.total + stats.workers.gpu.total)
                });
            }, sampleInterval);
            
            // Schedule burst of tasks
            const tasks = [];
            for (let i = 0; i < 25; i++) {
                const job = i % 2 === 0 ? 
                    new MockCPUJob({ duration: 400, complexity: 1 }) :
                    new MockGPUJob({ duration: 300, complexity: 1 });
                
                const taskId = taskManager.scheduleTask(job, Math.floor(Math.random() * 10));
                tasks.push(taskId);
                
                // Add some delay to create a realistic workload pattern
                if (i % 5 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
            }
            
            let completedTasks = 0;
            await new Promise(resolve => {
                taskManager.on('taskCompleted', (task) => {
                    completedTasks++;
                    if (completedTasks === tasks.length) resolve();
                });
            });
            
            clearInterval(monitorInterval);
            const finalStats = taskManager.getStats();
            taskManager.stop();
            
            // Calculate efficiency metrics
            const avgCpuUtilization = utilizationSamples.reduce((sum, s) => sum + s.cpuUtilization, 0) / utilizationSamples.length;
            const avgGpuUtilization = utilizationSamples.reduce((sum, s) => sum + s.gpuUtilization, 0) / utilizationSamples.length;
            const avgTotalUtilization = utilizationSamples.reduce((sum, s) => sum + s.totalUtilization, 0) / utilizationSamples.length;
            
            const peakUtilization = Math.max(...utilizationSamples.map(s => s.totalUtilization));
            const utilizationVariance = utilizationSamples.reduce((sum, s) => 
                sum + Math.pow(s.totalUtilization - avgTotalUtilization, 2), 0) / utilizationSamples.length;
            
            return {
                finalStats,
                utilizationMetrics: {
                    avgCpuUtilization,
                    avgGpuUtilization,
                    avgTotalUtilization,
                    peakUtilization,
                    utilizationVariance,
                    samplesCollected: utilizationSamples.length
                },
                efficiencyScore: avgTotalUtilization * (1 - utilizationVariance), // Higher is better
                completedTasks
            };
        });
        
        expect(result.completedTasks).toBe(25);
        expect(result.utilizationMetrics.avgTotalUtilization).toBeGreaterThan(0.5); // Good utilization
        expect(result.utilizationMetrics.peakUtilization).toBeGreaterThan(0.8); // Can achieve high utilization
        expect(result.utilizationMetrics.utilizationVariance).toBeLessThan(0.25); // Stable utilization
        expect(result.efficiencyScore).toBeGreaterThan(0.4); // Good efficiency score
    });
});

test.describe('TaskManager Benchmarks', () => {
    test('FLOPS benchmark comparison', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const benchmarks = {};
            
            // Test different worker configurations
            const configs = [
                { name: 'single_cpu', cpuWorkers: 1, gpuWorkers: 0, webnnWorkers: 0 },
                { name: 'multi_cpu', cpuWorkers: 4, gpuWorkers: 0, webnnWorkers: 0 },
                { name: 'cpu_gpu_mix', cpuWorkers: 2, gpuWorkers: 2, webnnWorkers: 0 },
                { name: 'full_stack', cpuWorkers: 2, gpuWorkers: 1, webnnWorkers: 1 }
            ];
            
            for (const config of configs) {
                const perfMeasure = new PerfMeasure();
                const taskManager = new TaskManager({
                    cpuWorkers: config.cpuWorkers,
                    gpuWorkers: config.gpuWorkers,
                    webnnWorkers: config.webnnWorkers,
                    maxConcurrentTasks: config.cpuWorkers + config.gpuWorkers + config.webnnWorkers
                });
                
                taskManager.start();
                perfMeasure.start();
                
                // Schedule appropriate tasks for each config
                const tasks = [];
                const numTasks = 15;
                
                for (let i = 0; i < numTasks; i++) {
                    let job;
                    if (config.gpuWorkers > 0 && i % 3 === 0) {
                        job = new MockGPUJob({ duration: 200, complexity: 2 });
                    } else if (config.webnnWorkers > 0 && i % 4 === 0) {
                        job = new MockWebNNJob({ duration: 300, complexity: 1 });
                    } else {
                        job = new MockCPUJob({ duration: 250, complexity: 2 });
                    }
                    
                    const taskId = taskManager.scheduleTask(job, Math.floor(Math.random() * 5));
                    tasks.push(taskId);
                }
                
                let completedTasks = 0;
                await new Promise(resolve => {
                    taskManager.on('taskCompleted', (task) => {
                        perfMeasure.recordTask(task.result);
                        completedTasks++;
                        if (completedTasks === numTasks) resolve();
                    });
                });
                
                const perfResults = perfMeasure.finish();
                taskManager.stop();
                
                benchmarks[config.name] = {
                    flopsPerSecond: perfResults.avgFlopsPerSecond,
                    tasksPerSecond: perfResults.tasksPerSecond,
                    avgExecutionTime: perfResults.avgTaskExecutionTime,
                    memoryUsage: perfResults.peakMemoryUsageMB,
                    totalWorkers: config.cpuWorkers + config.gpuWorkers + config.webnnWorkers
                };
            }
            
            return benchmarks;
        });
        
        // Log results for analysis
        console.log('FLOPS Benchmark Results:', JSON.stringify(result, null, 2));
        
        // Performance expectations
        expect(result.multi_cpu.flopsPerSecond).toBeGreaterThan(result.single_cpu.flopsPerSecond);
        expect(result.cpu_gpu_mix.tasksPerSecond).toBeGreaterThan(result.multi_cpu.tasksPerSecond);
        expect(result.full_stack.flopsPerSecond).toBeGreaterThan(result.cpu_gpu_mix.flopsPerSecond);
        
        // Efficiency checks
        const singleCpuEfficiency = result.single_cpu.flopsPerSecond / result.single_cpu.totalWorkers;
        const multiCpuEfficiency = result.multi_cpu.flopsPerSecond / result.multi_cpu.totalWorkers;
        
        // Multi-CPU should have reasonable scaling (at least 2x with 4x workers)
        expect(multiCpuEfficiency).toBeGreaterThan(singleCpuEfficiency * 0.5);
    });
});
