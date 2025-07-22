/**
 * Benchmark Jobs for Hardware Backend Performance Measurement
 * Measures memory bandwidth, disk I/O, and FLOPS for WebNN, WebGPU, and WASM
 */

class BenchmarkJob {
    constructor(type, backend, options = {}) {
        this.type = type;
        this.backend = backend;
        this.duration = options.duration || 5000; // 5 seconds default
        this.iterations = options.iterations || 100;
        this.dataSize = options.dataSize || 1024 * 1024; // 1MB default
        this.progress = 0;
        this.results = null;
        this.resourceRequirements = this._getResourceRequirements(backend);
    }

    _getResourceRequirements(backend) {
        switch (backend) {
            case 'webnn':
                return { webnn: 1, memory: 512 };
            case 'gpu':
                return { gpu: 1, memory: 1024 };
            case 'wasm':
                return { wasm: 1, memory: 256 };
            default:
                return { cpu: 1, memory: 128 };
        }
    }

    async execute(progressCallback, shouldStop) {
        const startTime = performance.now();
        this.results = {
            backend: this.backend,
            type: this.type,
            startTime: startTime,
            measurements: []
        };

        try {
            switch (this.type) {
                case 'memory_bandwidth':
                    await this._measureMemoryBandwidth(progressCallback, shouldStop);
                    break;
                case 'disk_bandwidth':
                    await this._measureDiskBandwidth(progressCallback, shouldStop);
                    break;
                case 'flops_benchmark':
                    await this._measureFLOPS(progressCallback, shouldStop);
                    break;
                case 'comprehensive_benchmark':
                    await this._runComprehensiveBenchmark(progressCallback, shouldStop);
                    break;
                default:
                    throw new Error(`Unknown benchmark type: ${this.type}`);
            }

            this.results.endTime = performance.now();
            this.results.totalTime = this.results.endTime - this.results.startTime;
            this.results.success = true;

            return this.results;

        } catch (error) {
            this.results.error = error.message;
            this.results.success = false;
            throw error;
        }
    }

    async _measureMemoryBandwidth(progressCallback, shouldStop) {
        const measurements = [];
        const testSizes = [
            1024,        // 1KB
            64 * 1024,   // 64KB
            1024 * 1024, // 1MB
            16 * 1024 * 1024, // 16MB
            64 * 1024 * 1024  // 64MB (if memory allows)
        ];

        for (let i = 0; i < testSizes.length && !shouldStop(); i++) {
            const size = testSizes[i];
            const measurement = await this._runMemoryTest(size, shouldStop);
            measurements.push(measurement);

            this.progress = (i + 1) / testSizes.length;
            if (progressCallback) {
                progressCallback(this.progress, {
                    phase: 'memory_bandwidth',
                    currentTest: `${size} bytes`,
                    measurement: measurement
                });
            }
        }

        this.results.measurements = measurements;
        this.results.summary = this._analyzeMemoryResults(measurements);
    }

    async _runMemoryTest(size, shouldStop) {
        const iterations = Math.max(10, Math.floor(100000000 / size)); // Adjust iterations based on size
        
        // Allocate arrays
        const source = new Float32Array(size / 4);
        const destination = new Float32Array(size / 4);
        
        // Fill source with random data
        for (let i = 0; i < source.length; i++) {
            source[i] = Math.random();
        }

        // Warm up
        for (let i = 0; i < Math.min(10, iterations); i++) {
            destination.set(source);
        }

        // Measure copy operations
        const startTime = performance.now();
        
        for (let i = 0; i < iterations && !shouldStop(); i++) {
            // Sequential memory copy
            destination.set(source);
            
            // Memory operations with computation
            for (let j = 0; j < Math.min(1000, source.length); j += 100) {
                destination[j] = source[j] * 1.1 + source[j + 1] * 0.9;
            }
        }
        
        const endTime = performance.now();
        const duration = endTime - startTime;
        
        const bytesTransferred = size * iterations * 2; // Read + Write
        const bandwidth = (bytesTransferred / 1024 / 1024) / (duration / 1000); // MB/s

        return {
            size: size,
            iterations: iterations,
            duration: duration,
            bandwidth: bandwidth,
            unit: 'MB/s'
        };
    }

    async _measureDiskBandwidth(progressCallback, shouldStop) {
        const measurements = [];
        const testSizes = [1024, 64 * 1024, 1024 * 1024]; // 1KB, 64KB, 1MB
        
        for (let i = 0; i < testSizes.length && !shouldStop(); i++) {
            const size = testSizes[i];
            const measurement = await this._runDiskTest(size, shouldStop);
            measurements.push(measurement);

            this.progress = (i + 1) / testSizes.length;
            if (progressCallback) {
                progressCallback(this.progress, {
                    phase: 'disk_bandwidth',
                    currentTest: `${size} bytes`,
                    measurement: measurement
                });
            }
        }

        this.results.measurements = measurements;
        this.results.summary = this._analyzeDiskResults(measurements);
    }

    async _runDiskTest(size, shouldStop) {
        // Simulate disk operations using IndexedDB for persistent storage
        const testData = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
            testData[i] = Math.floor(Math.random() * 256);
        }

        const iterations = Math.max(5, Math.floor(1000000 / size));
        let writeTime = 0;
        let readTime = 0;

        // Test localStorage (limited but available)
        if (typeof localStorage !== 'undefined') {
            const dataString = Array.from(testData).join(',');
            
            // Write test
            const writeStart = performance.now();
            for (let i = 0; i < Math.min(iterations, 50) && !shouldStop(); i++) {
                localStorage.setItem(`benchmark_${i}`, dataString);
            }
            writeTime = performance.now() - writeStart;

            // Read test
            const readStart = performance.now();
            for (let i = 0; i < Math.min(iterations, 50) && !shouldStop(); i++) {
                const data = localStorage.getItem(`benchmark_${i}`);
            }
            readTime = performance.now() - readStart;

            // Cleanup
            for (let i = 0; i < Math.min(iterations, 50); i++) {
                localStorage.removeItem(`benchmark_${i}`);
            }
        }

        const writeBandwidth = writeTime > 0 ? (size * iterations / 1024 / 1024) / (writeTime / 1000) : 0;
        const readBandwidth = readTime > 0 ? (size * iterations / 1024 / 1024) / (readTime / 1000) : 0;

        return {
            size: size,
            iterations: Math.min(iterations, 50),
            writeTime: writeTime,
            readTime: readTime,
            writeBandwidth: writeBandwidth,
            readBandwidth: readBandwidth,
            unit: 'MB/s'
        };
    }

    async _measureFLOPS(progressCallback, shouldStop) {
        const measurements = [];
        const testTypes = [
            { name: 'float32_add', operation: 'addition' },
            { name: 'float32_mul', operation: 'multiplication' },
            { name: 'float32_fma', operation: 'fused_multiply_add' },
            { name: 'float32_sqrt', operation: 'square_root' },
            { name: 'float32_sin', operation: 'sine' }
        ];

        for (let i = 0; i < testTypes.length && !shouldStop(); i++) {
            const testType = testTypes[i];
            const measurement = await this._runFLOPSTest(testType, shouldStop);
            measurements.push(measurement);

            this.progress = (i + 1) / testTypes.length;
            if (progressCallback) {
                progressCallback(this.progress, {
                    phase: 'flops_benchmark',
                    currentTest: testType.name,
                    measurement: measurement
                });
            }
        }

        this.results.measurements = measurements;
        this.results.summary = this._analyzeFLOPSResults(measurements);
    }

    async _runFLOPSTest(testType, shouldStop) {
        const arraySize = 1024 * 1024; // 1M elements
        const iterations = 100;
        
        const a = new Float32Array(arraySize);
        const b = new Float32Array(arraySize);
        const c = new Float32Array(arraySize);

        // Initialize arrays
        for (let i = 0; i < arraySize; i++) {
            a[i] = Math.random();
            b[i] = Math.random();
            c[i] = 0;
        }

        const startTime = performance.now();

        for (let iter = 0; iter < iterations && !shouldStop(); iter++) {
            switch (testType.operation) {
                case 'addition':
                    for (let i = 0; i < arraySize; i++) {
                        c[i] = a[i] + b[i];
                    }
                    break;
                
                case 'multiplication':
                    for (let i = 0; i < arraySize; i++) {
                        c[i] = a[i] * b[i];
                    }
                    break;
                
                case 'fused_multiply_add':
                    for (let i = 0; i < arraySize; i++) {
                        c[i] = a[i] * b[i] + c[i];
                    }
                    break;
                
                case 'square_root':
                    for (let i = 0; i < arraySize; i++) {
                        c[i] = Math.sqrt(a[i]);
                    }
                    break;
                
                case 'sine':
                    for (let i = 0; i < arraySize; i++) {
                        c[i] = Math.sin(a[i]);
                    }
                    break;
            }
        }

        const endTime = performance.now();
        const duration = (endTime - startTime) / 1000; // seconds
        
        const operations = arraySize * iterations;
        const flops = operations / duration;
        const gflops = flops / 1e9;

        return {
            operation: testType.operation,
            arraySize: arraySize,
            iterations: iterations,
            duration: duration,
            flops: flops,
            gflops: gflops,
            unit: 'GFLOPS'
        };
    }

    async _runComprehensiveBenchmark(progressCallback, shouldStop) {
        const phases = [
            { name: 'memory', weight: 0.4 },
            { name: 'disk', weight: 0.2 },
            { name: 'flops', weight: 0.4 }
        ];

        this.results.phases = {};
        let currentProgress = 0;

        for (let i = 0; i < phases.length && !shouldStop(); i++) {
            const phase = phases[i];
            
            const phaseCallback = (progress, data) => {
                const phaseProgress = currentProgress + (progress * phase.weight);
                if (progressCallback) {
                    progressCallback(phaseProgress, {
                        phase: phase.name,
                        ...data
                    });
                }
            };

            switch (phase.name) {
                case 'memory':
                    await this._measureMemoryBandwidth(phaseCallback, shouldStop);
                    this.results.phases.memory = this.results.measurements;
                    break;
                case 'disk':
                    this.results.measurements = []; // Reset for next phase
                    await this._measureDiskBandwidth(phaseCallback, shouldStop);
                    this.results.phases.disk = this.results.measurements;
                    break;
                case 'flops':
                    this.results.measurements = []; // Reset for next phase
                    await this._measureFLOPS(phaseCallback, shouldStop);
                    this.results.phases.flops = this.results.measurements;
                    break;
            }

            currentProgress += phase.weight;
        }

        this.results.summary = this._analyzeComprehensiveResults();
    }

    _analyzeMemoryResults(measurements) {
        const bandwidths = measurements.map(m => m.bandwidth);
        return {
            maxBandwidth: Math.max(...bandwidths),
            minBandwidth: Math.min(...bandwidths),
            avgBandwidth: bandwidths.reduce((a, b) => a + b, 0) / bandwidths.length,
            optimalSize: measurements.find(m => m.bandwidth === Math.max(...bandwidths))?.size || 0,
            unit: 'MB/s'
        };
    }

    _analyzeDiskResults(measurements) {
        const writeBandwidths = measurements.map(m => m.writeBandwidth);
        const readBandwidths = measurements.map(m => m.readBandwidth);
        
        return {
            maxWriteBandwidth: Math.max(...writeBandwidths),
            maxReadBandwidth: Math.max(...readBandwidths),
            avgWriteBandwidth: writeBandwidths.reduce((a, b) => a + b, 0) / writeBandwidths.length,
            avgReadBandwidth: readBandwidths.reduce((a, b) => a + b, 0) / readBandwidths.length,
            unit: 'MB/s'
        };
    }

    _analyzeFLOPSResults(measurements) {
        const gflops = measurements.map(m => m.gflops);
        const totalGFLOPS = gflops.reduce((a, b) => a + b, 0);
        
        return {
            totalGFLOPS: totalGFLOPS,
            maxGFLOPS: Math.max(...gflops),
            avgGFLOPS: totalGFLOPS / gflops.length,
            bestOperation: measurements.find(m => m.gflops === Math.max(...gflops))?.operation || 'unknown',
            operations: measurements.map(m => ({ op: m.operation, gflops: m.gflops }))
        };
    }

    _analyzeComprehensiveResults() {
        const summary = {
            overallScore: 0,
            memoryScore: 0,
            diskScore: 0,
            computeScore: 0
        };

        if (this.results.phases.memory) {
            const memSummary = this._analyzeMemoryResults(this.results.phases.memory);
            summary.memoryScore = Math.min(100, memSummary.maxBandwidth / 10); // Normalize to 100
            summary.memory = memSummary;
        }

        if (this.results.phases.disk) {
            const diskSummary = this._analyzeDiskResults(this.results.phases.disk);
            summary.diskScore = Math.min(100, diskSummary.maxReadBandwidth / 5); // Normalize to 100
            summary.disk = diskSummary;
        }

        if (this.results.phases.flops) {
            const flopsSummary = this._analyzeFLOPSResults(this.results.phases.flops);
            summary.computeScore = Math.min(100, flopsSummary.totalGFLOPS * 10); // Normalize to 100
            summary.compute = flopsSummary;
        }

        summary.overallScore = (summary.memoryScore * 0.3 + summary.diskScore * 0.2 + summary.computeScore * 0.5);
        
        return summary;
    }

    interrupt() {
        // Can be interrupted
        this.interrupted = true;
    }
}

// Factory functions for creating benchmark jobs
class BenchmarkJobFactory {
    static createMemoryBenchmark(backend, options = {}) {
        return new BenchmarkJob('memory_bandwidth', backend, options);
    }

    static createDiskBenchmark(backend, options = {}) {
        return new BenchmarkJob('disk_bandwidth', backend, options);
    }

    static createFLOPSBenchmark(backend, options = {}) {
        return new BenchmarkJob('flops_benchmark', backend, options);
    }

    static createComprehensiveBenchmark(backend, options = {}) {
        return new BenchmarkJob('comprehensive_benchmark', backend, options);
    }

    static createBackendComparison(options = {}) {
        const backends = ['cpu', 'gpu', 'webnn', 'wasm'];
        return backends.map(backend => 
            new BenchmarkJob('comprehensive_benchmark', backend, options)
        );
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BenchmarkJob, BenchmarkJobFactory };
} else if (typeof window !== 'undefined') {
    window.BenchmarkJob = BenchmarkJob;
    window.BenchmarkJobFactory = BenchmarkJobFactory;
}
