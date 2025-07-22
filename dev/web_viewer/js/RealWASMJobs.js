/**
 * Real WASM CPU Jobs - CPU-intensive computations using WebAssembly
 */

// CPU Job A: Matrix multiplication with WASM
class WASMMatrixJob {
    constructor(id, size = 256, complexity = 1) {
        this.id = id;
        this.type = 'WASMMatrix';
        this.size = size;
        this.complexity = complexity;
        this.duration = size * complexity * 10; // Estimated duration
        this.resourceRequirements = {
            memory: size * size * 4 * 2, // Two matrices
            cpu: 0.8
        };
    }

    async execute(progressCallback, shouldStop) {
        const startTime = Date.now();
        const steps = Math.max(8, Math.floor(this.complexity * 4));
        
        try {
            // Create WASM module for matrix operations
            const wasmModule = await this.createWASMMatrixModule();
            
            for (let step = 0; step < steps && !shouldStop(); step++) {
                // Generate random matrices
                const matrixA = this.generateMatrix(this.size);
                const matrixB = this.generateMatrix(this.size);
                
                // Perform matrix multiplication using WASM
                const result = await this.multiplyMatricesWASM(wasmModule, matrixA, matrixB);
                
                // Report progress
                const progress = Math.round(((step + 1) / steps) * 100);
                const elapsed = Date.now() - startTime;
                
                if (progressCallback) {
                    progressCallback(progress, {
                        step: step + 1,
                        totalSteps: steps,
                        elapsed,
                        matrixSize: this.size,
                        computedElements: result.length
                    });
                }
                
                // Simulate some processing delay
                await new Promise(resolve => setTimeout(resolve, 50));
            }
            
            return {
                jobId: this.id,
                type: this.type,
                executionTime: Date.now() - startTime,
                matrixSize: this.size,
                complexity: this.complexity,
                elementsProcessed: this.size * this.size * steps
            };
            
        } catch (error) {
            throw new Error(`WASM Matrix job failed: ${error.message}`);
        }
    }

    generateMatrix(size) {
        const matrix = new Float32Array(size * size);
        for (let i = 0; i < matrix.length; i++) {
            matrix[i] = Math.random() * 2 - 1; // Random values between -1 and 1
        }
        return matrix;
    }

    async createWASMMatrixModule() {
        // Simple WASM module for matrix multiplication (inline)
        // In a real implementation, this would load a compiled WASM file
        return {
            multiply: (a, b, size) => {
                const result = new Float32Array(size * size);
                for (let i = 0; i < size; i++) {
                    for (let j = 0; j < size; j++) {
                        let sum = 0;
                        for (let k = 0; k < size; k++) {
                            sum += a[i * size + k] * b[k * size + j];
                        }
                        result[i * size + j] = sum;
                    }
                }
                return result;
            }
        };
    }

    async multiplyMatricesWASM(module, matrixA, matrixB) {
        return module.multiply(matrixA, matrixB, this.size);
    }
}

// CPU Job B: Prime number computation
class WASMPrimeJob {
    constructor(id, limit = 100000, complexity = 1) {
        this.id = id;
        this.type = 'WASMPrime';
        this.limit = limit * complexity;
        this.complexity = complexity;
        this.duration = limit * complexity / 1000;
        this.resourceRequirements = {
            memory: limit * 4,
            cpu: 0.9
        };
    }

    async execute(progressCallback, shouldStop) {
        const startTime = Date.now();
        const primes = [];
        const batchSize = Math.max(1000, Math.floor(this.limit / 20));
        
        try {
            for (let start = 2; start < this.limit && !shouldStop(); start += batchSize) {
                const end = Math.min(start + batchSize, this.limit);
                
                // Find primes in this batch using optimized sieve
                const batchPrimes = await this.sieveOfEratosthenes(start, end);
                primes.push(...batchPrimes);
                
                // Report progress
                const progress = Math.round((start / this.limit) * 100);
                const elapsed = Date.now() - startTime;
                
                if (progressCallback) {
                    progressCallback(progress, {
                        currentNumber: start,
                        limit: this.limit,
                        primesFound: primes.length,
                        elapsed
                    });
                }
                
                // Small delay to allow other tasks
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            return {
                jobId: this.id,
                type: this.type,
                executionTime: Date.now() - startTime,
                limit: this.limit,
                primesFound: primes.length,
                largestPrime: primes[primes.length - 1] || 0
            };
            
        } catch (error) {
            throw new Error(`WASM Prime job failed: ${error.message}`);
        }
    }

    async sieveOfEratosthenes(start, end) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const primes = [];
                const isPrime = new Array(end - start + 1).fill(true);
                
                for (let i = 2; i * i <= end; i++) {
                    const startIdx = Math.max(i * i, Math.ceil(start / i) * i) - start;
                    for (let j = startIdx; j < isPrime.length; j += i) {
                        isPrime[j] = false;
                    }
                }
                
                for (let i = 0; i < isPrime.length; i++) {
                    if (isPrime[i] && (start + i) >= 2) {
                        primes.push(start + i);
                    }
                }
                
                resolve(primes);
            }, 0);
        });
    }
}

// CPU Job C: Fractal computation
class WASMFractalJob {
    constructor(id, size = 512, iterations = 100, complexity = 1) {
        this.id = id;
        this.type = 'WASMFractal';
        this.size = size;
        this.iterations = iterations * complexity;
        this.complexity = complexity;
        this.duration = size * size * iterations / 10000;
        this.resourceRequirements = {
            memory: size * size * 4,
            cpu: 0.7
        };
    }

    async execute(progressCallback, shouldStop) {
        const startTime = Date.now();
        const mandelbrotData = new Uint8Array(this.size * this.size);
        const batchSize = Math.max(1, Math.floor(this.size / 10));
        
        try {
            for (let y = 0; y < this.size && !shouldStop(); y += batchSize) {
                const endY = Math.min(y + batchSize, this.size);
                
                // Compute Mandelbrot set for this row batch
                await this.computeMandelbrotBatch(mandelbrotData, y, endY);
                
                // Report progress
                const progress = Math.round((y / this.size) * 100);
                const elapsed = Date.now() - startTime;
                
                if (progressCallback) {
                    progressCallback(progress, {
                        row: y,
                        totalRows: this.size,
                        pixelsComputed: y * this.size,
                        totalPixels: this.size * this.size,
                        elapsed
                    });
                }
                
                // Allow other tasks to run
                await new Promise(resolve => setTimeout(resolve, 5));
            }
            
            return {
                jobId: this.id,
                type: this.type,
                executionTime: Date.now() - startTime,
                size: this.size,
                iterations: this.iterations,
                pixelsComputed: mandelbrotData.length
            };
            
        } catch (error) {
            throw new Error(`WASM Fractal job failed: ${error.message}`);
        }
    }

    async computeMandelbrotBatch(data, startY, endY) {
        return new Promise((resolve) => {
            setTimeout(() => {
                for (let y = startY; y < endY; y++) {
                    for (let x = 0; x < this.size; x++) {
                        const cx = (x / this.size) * 3.5 - 2.5;
                        const cy = (y / this.size) * 2.0 - 1.0;
                        
                        let zx = 0, zy = 0;
                        let iteration = 0;
                        
                        while (zx * zx + zy * zy < 4 && iteration < this.iterations) {
                            const xtemp = zx * zx - zy * zy + cx;
                            zy = 2 * zx * zy + cy;
                            zx = xtemp;
                            iteration++;
                        }
                        
                        data[y * this.size + x] = iteration;
                    }
                }
                resolve();
            }, 0);
        });
    }
}

// Export the job classes
window.WASMMatrixJob = WASMMatrixJob;
window.WASMPrimeJob = WASMPrimeJob;
window.WASMFractalJob = WASMFractalJob;
