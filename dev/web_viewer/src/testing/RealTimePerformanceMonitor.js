/**
 * Real-time Performance Monitor for Hardware Backends
 * Continuously monitors system performance and provides real-time feedback
 */

class RealTimePerformanceMonitor {
    constructor(taskManager, options = {}) {
        this.taskManager = taskManager;
        this.isMonitoring = false;
        this.monitoringInterval = options.interval || 1000; // 1 second
        this.performanceHistory = {
            cpu: [],
            gpu: [],
            webnn: [],
            wasm: []
        };
        this.maxHistoryLength = options.maxHistory || 60; // 60 seconds of history
        this.callbacks = {
            onUpdate: options.onUpdate || (() => {}),
            onAlert: options.onAlert || (() => {})
        };
        this.thresholds = {
            memoryWarning: 100, // MB/s
            computeWarning: 1,  // GFLOPS
            latencyWarning: 50  // ms
        };
        this.intervalId = null;
    }

    start() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.intervalId = setInterval(() => {
            this.performMicroBenchmark();
        }, this.monitoringInterval);
        
        console.log('Real-time performance monitoring started');
    }

    stop() {
        if (!this.isMonitoring) return;
        
        this.isMonitoring = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        
        console.log('Real-time performance monitoring stopped');
    }

    async performMicroBenchmark() {
        const timestamp = Date.now();
        const backends = ['cpu', 'gpu', 'webnn', 'wasm'];
        
        for (const backend of backends) {
            try {
                const metrics = await this.runMicroBenchmark(backend);
                this.updateHistory(backend, timestamp, metrics);
                this.checkThresholds(backend, metrics);
            } catch (error) {
                console.warn(`Micro-benchmark failed for ${backend}:`, error);
            }
        }
        
        this.callbacks.onUpdate(this.getCurrentMetrics());
    }

    async runMicroBenchmark(backend) {
        const startTime = performance.now();
        
        // Quick memory test
        const memoryMetrics = await this.quickMemoryTest();
        
        // Quick compute test
        const computeMetrics = await this.quickComputeTest(backend);
        
        // Task manager stats
        const taskStats = this.taskManager.getStats();
        const workerStats = taskStats.workers[backend] || {};
        
        const endTime = performance.now();
        
        return {
            timestamp: Date.now(),
            latency: endTime - startTime,
            memory: memoryMetrics,
            compute: computeMetrics,
            workers: workerStats,
            systemLoad: await this.estimateSystemLoad()
        };
    }

    async quickMemoryTest() {
        const size = 1024 * 1024; // 1MB
        const iterations = 10;
        
        const source = new Float32Array(size / 4);
        const dest = new Float32Array(size / 4);
        
        for (let i = 0; i < source.length; i++) {
            source[i] = Math.random();
        }
        
        const startTime = performance.now();
        
        for (let i = 0; i < iterations; i++) {
            dest.set(source);
        }
        
        const duration = performance.now() - startTime;
        const bytesTransferred = size * iterations * 2; // Read + Write
        const bandwidth = (bytesTransferred / 1024 / 1024) / (duration / 1000); // MB/s
        
        return {
            bandwidth: bandwidth,
            latency: duration / iterations,
            unit: 'MB/s'
        };
    }

    async quickComputeTest(backend) {
        const size = 10000;
        const a = new Float32Array(size);
        const b = new Float32Array(size);
        const c = new Float32Array(size);
        
        for (let i = 0; i < size; i++) {
            a[i] = Math.random();
            b[i] = Math.random();
        }
        
        const startTime = performance.now();
        
        // Fused multiply-add operations
        for (let i = 0; i < size; i++) {
            c[i] = a[i] * b[i] + c[i];
        }
        
        const duration = performance.now() - startTime;
        const operations = size * 2; // Multiply + Add
        const flops = operations / (duration / 1000);
        const gflops = flops / 1e9;
        
        return {
            gflops: gflops,
            latency: duration,
            operations: operations,
            unit: 'GFLOPS'
        };
    }

    async estimateSystemLoad() {
        // Use various browser APIs to estimate system load
        const load = {
            memory: this.getMemoryInfo(),
            cpu: await this.estimateCPULoad(),
            battery: await this.getBatteryInfo(),
            network: this.getNetworkInfo()
        };
        
        return load;
    }

    getMemoryInfo() {
        if (performance.memory) {
            return {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit,
                usage: performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize
            };
        }
        return { available: false };
    }

    async estimateCPULoad() {
        // Estimate CPU load by measuring time for a fixed computation
        const startTime = performance.now();
        const iterations = 100000;
        let result = 0;
        
        for (let i = 0; i < iterations; i++) {
            result += Math.sin(i) * Math.cos(i);
        }
        
        const duration = performance.now() - startTime;
        const expectedDuration = 10; // Expected duration on a baseline system
        const load = Math.min(1.0, duration / expectedDuration);
        
        return {
            estimatedLoad: load,
            computeTime: duration,
            baseline: expectedDuration
        };
    }

    async getBatteryInfo() {
        if ('getBattery' in navigator) {
            try {
                const battery = await navigator.getBattery();
                return {
                    level: battery.level,
                    charging: battery.charging,
                    chargingTime: battery.chargingTime,
                    dischargingTime: battery.dischargingTime
                };
            } catch (error) {
                return { available: false };
            }
        }
        return { available: false };
    }

    getNetworkInfo() {
        if ('connection' in navigator) {
            const connection = navigator.connection;
            return {
                effectiveType: connection.effectiveType,
                downlink: connection.downlink,
                rtt: connection.rtt,
                saveData: connection.saveData
            };
        }
        return { available: false };
    }

    updateHistory(backend, timestamp, metrics) {
        const history = this.performanceHistory[backend];
        history.push({ timestamp, ...metrics });
        
        // Keep only recent history
        while (history.length > this.maxHistoryLength) {
            history.shift();
        }
    }

    checkThresholds(backend, metrics) {
        const alerts = [];
        
        if (metrics.memory.bandwidth < this.thresholds.memoryWarning) {
            alerts.push({
                type: 'warning',
                backend: backend,
                metric: 'memory',
                message: `Low memory bandwidth: ${metrics.memory.bandwidth.toFixed(1)} MB/s`,
                value: metrics.memory.bandwidth,
                threshold: this.thresholds.memoryWarning
            });
        }
        
        if (metrics.compute.gflops < this.thresholds.computeWarning) {
            alerts.push({
                type: 'warning',
                backend: backend,
                metric: 'compute',
                message: `Low compute performance: ${metrics.compute.gflops.toFixed(2)} GFLOPS`,
                value: metrics.compute.gflops,
                threshold: this.thresholds.computeWarning
            });
        }
        
        if (metrics.latency > this.thresholds.latencyWarning) {
            alerts.push({
                type: 'warning',
                backend: backend,
                metric: 'latency',
                message: `High latency: ${metrics.latency.toFixed(1)} ms`,
                value: metrics.latency,
                threshold: this.thresholds.latencyWarning
            });
        }
        
        if (alerts.length > 0) {
            this.callbacks.onAlert(alerts);
        }
    }

    getCurrentMetrics() {
        const current = {};
        
        Object.entries(this.performanceHistory).forEach(([backend, history]) => {
            if (history.length > 0) {
                const latest = history[history.length - 1];
                const avg5s = this.getAverageMetrics(backend, 5000); // Last 5 seconds
                
                current[backend] = {
                    latest: latest,
                    average5s: avg5s,
                    trend: this.calculateTrend(backend),
                    health: this.assessHealth(backend)
                };
            }
        });
        
        return current;
    }

    getAverageMetrics(backend, timeWindow) {
        const history = this.performanceHistory[backend];
        const cutoff = Date.now() - timeWindow;
        const recent = history.filter(entry => entry.timestamp > cutoff);
        
        if (recent.length === 0) return null;
        
        const totals = recent.reduce((acc, entry) => {
            acc.memoryBandwidth += entry.memory.bandwidth;
            acc.computeGflops += entry.compute.gflops;
            acc.latency += entry.latency;
            return acc;
        }, { memoryBandwidth: 0, computeGflops: 0, latency: 0 });
        
        return {
            memoryBandwidth: totals.memoryBandwidth / recent.length,
            computeGflops: totals.computeGflops / recent.length,
            latency: totals.latency / recent.length,
            sampleCount: recent.length
        };
    }

    calculateTrend(backend) {
        const history = this.performanceHistory[backend];
        if (history.length < 10) return 'insufficient_data';
        
        const recent = history.slice(-10);
        const older = history.slice(-20, -10);
        
        if (older.length === 0) return 'insufficient_data';
        
        const recentAvg = recent.reduce((sum, entry) => sum + entry.compute.gflops, 0) / recent.length;
        const olderAvg = older.reduce((sum, entry) => sum + entry.compute.gflops, 0) / older.length;
        
        const change = (recentAvg - olderAvg) / olderAvg;
        
        if (change > 0.05) return 'improving';
        if (change < -0.05) return 'degrading';
        return 'stable';
    }

    assessHealth(backend) {
        const latest = this.performanceHistory[backend].slice(-1)[0];
        if (!latest) return 'unknown';
        
        let score = 0;
        let maxScore = 0;
        
        // Memory health
        if (latest.memory.bandwidth > this.thresholds.memoryWarning * 2) score += 3;
        else if (latest.memory.bandwidth > this.thresholds.memoryWarning) score += 2;
        else if (latest.memory.bandwidth > this.thresholds.memoryWarning * 0.5) score += 1;
        maxScore += 3;
        
        // Compute health
        if (latest.compute.gflops > this.thresholds.computeWarning * 2) score += 3;
        else if (latest.compute.gflops > this.thresholds.computeWarning) score += 2;
        else if (latest.compute.gflops > this.thresholds.computeWarning * 0.5) score += 1;
        maxScore += 3;
        
        // Latency health (lower is better)
        if (latest.latency < this.thresholds.latencyWarning * 0.5) score += 3;
        else if (latest.latency < this.thresholds.latencyWarning) score += 2;
        else if (latest.latency < this.thresholds.latencyWarning * 2) score += 1;
        maxScore += 3;
        
        // Worker health
        if (latest.workers.available > 0) score += 2;
        if (latest.workers.busy === 0) score += 1;
        maxScore += 3;
        
        const healthRatio = score / maxScore;
        
        if (healthRatio > 0.8) return 'excellent';
        if (healthRatio > 0.6) return 'good';
        if (healthRatio > 0.4) return 'fair';
        if (healthRatio > 0.2) return 'poor';
        return 'critical';
    }

    getPerformanceReport() {
        const report = {
            timestamp: Date.now(),
            monitoring: this.isMonitoring,
            backends: {},
            summary: {
                bestBackend: null,
                worstBackend: null,
                averageLatency: 0,
                totalGFLOPS: 0
            }
        };
        
        let bestScore = -1;
        let worstScore = Infinity;
        let totalLatency = 0;
        let totalGFLOPS = 0;
        let backendCount = 0;
        
        Object.entries(this.performanceHistory).forEach(([backend, history]) => {
            if (history.length === 0) return;
            
            const metrics = this.getCurrentMetrics()[backend];
            if (!metrics) return;
            
            const score = metrics.latest.compute.gflops + metrics.latest.memory.bandwidth / 100;
            
            if (score > bestScore) {
                bestScore = score;
                report.summary.bestBackend = backend;
            }
            
            if (score < worstScore) {
                worstScore = score;
                report.summary.worstBackend = backend;
            }
            
            totalLatency += metrics.latest.latency;
            totalGFLOPS += metrics.latest.compute.gflops;
            backendCount++;
            
            report.backends[backend] = {
                performance: metrics,
                score: score,
                history: history.slice(-10) // Last 10 entries
            };
        });
        
        if (backendCount > 0) {
            report.summary.averageLatency = totalLatency / backendCount;
            report.summary.totalGFLOPS = totalGFLOPS;
        }
        
        return report;
    }

    exportData() {
        return {
            history: this.performanceHistory,
            thresholds: this.thresholds,
            configuration: {
                interval: this.monitoringInterval,
                maxHistory: this.maxHistoryLength
            },
            report: this.getPerformanceReport()
        };
    }

    importData(data) {
        if (data.history) {
            this.performanceHistory = data.history;
        }
        if (data.thresholds) {
            this.thresholds = { ...this.thresholds, ...data.thresholds };
        }
        if (data.configuration) {
            this.monitoringInterval = data.configuration.interval || this.monitoringInterval;
            this.maxHistoryLength = data.configuration.maxHistory || this.maxHistoryLength;
        }
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RealTimePerformanceMonitor };
} else if (typeof window !== 'undefined') {
    window.RealTimePerformanceMonitor = RealTimePerformanceMonitor;
}
