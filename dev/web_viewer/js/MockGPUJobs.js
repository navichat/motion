/**
 * Mock GPU Jobs for testing task scheduling
 * Simulates WebGPU/WebNN computational workloads
 */

class MockGPUJob {
    constructor(type, duration, complexity = 1) {
        this.id = this._generateId();
        this.type = type;
        this.duration = duration; // milliseconds
        this.complexity = complexity; // 1-10 scale
        this.startTime = null;
        this.endTime = null;
        this.status = 'pending'; // pending, running, completed, failed, cancelled
        this.progress = 0; // 0-100
        this.result = null;
        this.error = null;
        this.interruptions = 0;
        this.resumptions = 0;
    }

    _generateId() {
        return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Simulate GPU computation with artificial delay and progress updates
     */
    async execute(progressCallback = null, shouldStop = null) {
        this.status = 'running';
        this.startTime = performance.now();
        this.progress = 0;

        try {
            const totalSteps = Math.max(10, this.complexity * 10);
            const stepDuration = this.duration / totalSteps;

            for (let step = 0; step < totalSteps; step++) {
                // Check if we should stop (for interruption)
                if (shouldStop && shouldStop()) {
                    this.status = 'cancelled';
                    return null;
                }

                // Simulate work with different patterns based on job type
                await this._simulateWork(stepDuration);
                
                this.progress = Math.round((step + 1) / totalSteps * 100);
                
                if (progressCallback) {
                    progressCallback(this.progress, this);
                }
            }

            this.endTime = performance.now();
            this.status = 'completed';
            this.result = this._generateResult();
            return this.result;

        } catch (error) {
            this.status = 'failed';
            this.error = error.message;
            this.endTime = performance.now();
            throw error;
        }
    }

    /**
     * Simulate different types of computational work
     */
    async _simulateWork(duration) {
        const startTime = performance.now();
        
        switch (this.type) {
            case 'JobA': // Matrix operations (CPU/GPU intensive)
                await this._simulateMatrixOperations(duration);
                break;
            case 'JobB': // Neural network inference (GPU/WebNN)
                await this._simulateNeuralNetwork(duration);
                break;
            case 'JobC': // Audio/Video processing (mixed workload)
                await this._simulateMediaProcessing(duration);
                break;
            default:
                await this._sleep(duration);
        }
    }

    async _simulateMatrixOperations(duration) {
        const startTime = performance.now();
        
        // Simulate matrix multiplication with actual computation
        const size = Math.min(100, this.complexity * 20);
        const matrixA = this._generateMatrix(size, size);
        const matrixB = this._generateMatrix(size, size);
        
        // Perform some actual computation to load CPU/GPU
        const result = this._multiplyMatrices(matrixA, matrixB);
        
        // Wait for remaining duration
        const elapsed = performance.now() - startTime;
        if (elapsed < duration) {
            await this._sleep(duration - elapsed);
        }
    }

    async _simulateNeuralNetwork(duration) {
        const startTime = performance.now();
        
        // Simulate neural network layers
        const layers = this.complexity;
        const layerTime = duration / layers;
        
        for (let i = 0; i < layers; i++) {
            // Simulate forward pass computation
            const weights = this._generateMatrix(64, 64);
            const input = this._generateMatrix(64, 1);
            const output = this._multiplyMatrices(weights, input);
            
            // Apply activation function (simulate)
            for (let j = 0; j < output.length; j++) {
                output[j] = Math.tanh(output[j][0]);
            }
            
            await this._sleep(layerTime * 0.1);
        }
        
        const elapsed = performance.now() - startTime;
        if (elapsed < duration) {
            await this._sleep(duration - elapsed);
        }
    }

    async _simulateMediaProcessing(duration) {
        const startTime = performance.now();
        
        // Simulate audio/video frame processing
        const frames = this.complexity * 10;
        const frameTime = duration / frames;
        
        for (let i = 0; i < frames; i++) {
            // Simulate frame processing
            const frameData = new Float32Array(1024);
            for (let j = 0; j < frameData.length; j++) {
                frameData[j] = Math.sin(i * j * 0.001) * Math.cos(j * 0.01);
            }
            
            // Simulate FFT or convolution
            const processed = this._applyFilter(frameData);
            
            await this._sleep(frameTime * 0.1);
        }
        
        const elapsed = performance.now() - startTime;
        if (elapsed < duration) {
            await this._sleep(duration - elapsed);
        }
    }

    _generateMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = Math.random() * 2 - 1; // -1 to 1
            }
        }
        return matrix;
    }

    _multiplyMatrices(a, b) {
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < b.length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    _applyFilter(data) {
        const filtered = new Float32Array(data.length);
        for (let i = 1; i < data.length - 1; i++) {
            filtered[i] = (data[i-1] + data[i] + data[i+1]) / 3;
        }
        return filtered;
    }

    async _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    _generateResult() {
        return {
            jobId: this.id,
            type: this.type,
            executionTime: this.endTime - this.startTime,
            complexity: this.complexity,
            dataPoints: Math.floor(Math.random() * 1000) + 100,
            accuracy: Math.random() * 0.1 + 0.9, // 90-100%
            memoryUsed: Math.floor(Math.random() * 500) + 100 // MB
        };
    }

    /**
     * Interrupt the job (for preemption)
     */
    interrupt() {
        if (this.status === 'running') {
            this.status = 'interrupted';
            this.interruptions++;
        }
    }

    /**
     * Resume the job after interruption
     */
    resume() {
        if (this.status === 'interrupted') {
            this.status = 'running';
            this.resumptions++;
        }
    }

    /**
     * Get job statistics
     */
    getStats() {
        return {
            id: this.id,
            type: this.type,
            status: this.status,
            progress: this.progress,
            duration: this.duration,
            complexity: this.complexity,
            interruptions: this.interruptions,
            resumptions: this.resumptions,
            executionTime: this.endTime ? this.endTime - this.startTime : null
        };
    }
}

/**
 * Factory for creating different types of mock GPU jobs
 */
class MockGPUJobFactory {
    static createJobA(complexity = 1) {
        const duration = 1000 + (complexity * 500); // 1-6 seconds
        return new MockGPUJob('JobA', duration, complexity);
    }

    static createJobB(complexity = 1) {
        const duration = 800 + (complexity * 300); // 0.8-3.8 seconds
        return new MockGPUJob('JobB', duration, complexity);
    }

    static createJobC(complexity = 1) {
        const duration = 1200 + (complexity * 400); // 1.2-5.2 seconds
        return new MockGPUJob('JobC', duration, complexity);
    }

    static createRandomJob() {
        const types = ['JobA', 'JobB', 'JobC'];
        const type = types[Math.floor(Math.random() * types.length)];
        const complexity = Math.floor(Math.random() * 5) + 1;
        
        switch (type) {
            case 'JobA': return this.createJobA(complexity);
            case 'JobB': return this.createJobB(complexity);
            case 'JobC': return this.createJobC(complexity);
        }
    }

    static createJobBatch(count = 10) {
        const jobs = [];
        for (let i = 0; i < count; i++) {
            jobs.push(this.createRandomJob());
        }
        return jobs;
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MockGPUJob, MockGPUJobFactory };
} else if (typeof window !== 'undefined') {
    window.MockGPUJob = MockGPUJob;
    window.MockGPUJobFactory = MockGPUJobFactory;
}
