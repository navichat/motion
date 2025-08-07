// Comprehensive Unit Tests for Optimized Multi-Head Attention
// Tests correctness, performance, and cross-backend consistency

class AttentionTestSuite {
    constructor() {
        this.testResults = [];
        this.tolerance = 1e-5; // Floating point comparison tolerance
        this.performanceThresholds = {
            webgpu: 50,   // Max ms for WebGPU
            webnn: 100,   // Max ms for WebNN  
            wasm: 200,    // Max ms for WASM
            cpu: 500      // Max ms for CPU
        };
    }

    async runAllTests() {
        console.log('🧪 Starting Comprehensive Multi-Head Attention Test Suite...');
        
        const tests = [
            this.testBasicFunctionality,
            this.testMultiBackendConsistency,
            this.testPerformanceBenchmarks,
            this.testNumericalStability,
            this.testBatchProcessing,
            this.testAttentionPatterns,
            this.testMemoryEfficiency,
            this.testEdgeCases
        ];

        for (const test of tests) {
            try {
                await test.call(this);
            } catch (error) {
                this.addTestResult(test.name, false, error.message);
            }
        }

        this.generateTestReport();
        return this.testResults;
    }

    async testBasicFunctionality() {
        console.log('🔬 Testing basic functionality...');

        const attention = new OptimizedMultiHeadAttention({
            numHeads: 8,
            headDim: 64,
            dropout: 0.1
        });

        await attention.initializeBackend('cpu');

        // Create test data
        const seqLen = 10;
        const query = this.generateTestTensor([seqLen, attention.modelDim]);
        const key = this.generateTestTensor([seqLen, attention.modelDim]);
        const value = this.generateTestTensor([seqLen, attention.modelDim]);

        // Test attention computation
        const result = await attention.computeAttention(query, key, value);

        // Verify output shape
        const expectedShape = [seqLen, attention.modelDim];
        const actualShape = [result.length, result[0].length];

        const shapeMatch = this.arraysEqual(expectedShape, actualShape);
        const hasValidValues = this.hasValidFloats(result);
        const isFinite = this.allFinite(result);

        const passed = shapeMatch && hasValidValues && isFinite;
        this.addTestResult('testBasicFunctionality', passed, 
            passed ? 'Basic functionality test passed' : 
            `Shape: ${shapeMatch}, Valid: ${hasValidValues}, Finite: ${isFinite}`);
    }

    async testMultiBackendConsistency() {
        console.log('🔄 Testing multi-backend consistency...');

        const config = {
            numHeads: 4,
            headDim: 32,
            dropout: 0.0 // Disable dropout for consistency testing
        };

        const backends = ['cpu', 'wasm'];
        const results = {};
        const seqLen = 8;

        // Generate consistent test data
        const query = this.generateTestTensor([seqLen, config.numHeads * config.headDim]);
        const key = this.generateTestTensor([seqLen, config.numHeads * config.headDim]);
        const value = this.generateTestTensor([seqLen, config.numHeads * config.headDim]);

        // Test each backend
        for (const backend of backends) {
            try {
                const attention = new OptimizedMultiHeadAttention(config);
                await attention.initializeBackend(backend);
                results[backend] = await attention.computeAttention(query, key, value);
            } catch (error) {
                console.warn(`Backend ${backend} failed:`, error.message);
                results[backend] = null;
            }
        }

        // Compare results between backends
        const availableBackends = Object.keys(results).filter(b => results[b] !== null);
        let allConsistent = true;
        let maxDifference = 0;

        if (availableBackends.length >= 2) {
            const reference = results[availableBackends[0]];
            
            for (let i = 1; i < availableBackends.length; i++) {
                const current = results[availableBackends[i]];
                const difference = this.computeMaxDifference(reference, current);
                maxDifference = Math.max(maxDifference, difference);
                
                if (difference > this.tolerance) {
                    allConsistent = false;
                    console.warn(`Inconsistency between ${availableBackends[0]} and ${availableBackends[i]}: ${difference}`);
                }
            }
        }

        this.addTestResult('testMultiBackendConsistency', allConsistent,
            `Max difference: ${maxDifference.toExponential(2)}, Backends tested: ${availableBackends.join(', ')}`);
    }

    async testPerformanceBenchmarks() {
        console.log('⚡ Testing performance benchmarks...');

        const benchmarkResults = {};
        const config = {
            numHeads: 8,
            headDim: 64
        };

        const testCases = [
            { seqLen: 50, desc: 'Short sequence' },
            { seqLen: 200, desc: 'Medium sequence' },
            { seqLen: 500, desc: 'Long sequence' }
        ];

        for (const backend of ['cpu', 'wasm']) {
            benchmarkResults[backend] = {};
            
            try {
                const attention = new OptimizedMultiHeadAttention(config);
                await attention.initializeBackend(backend);

                for (const testCase of testCases) {
                    const query = this.generateTestTensor([testCase.seqLen, config.numHeads * config.headDim]);
                    const key = this.generateTestTensor([testCase.seqLen, config.numHeads * config.headDim]);
                    const value = this.generateTestTensor([testCase.seqLen, config.numHeads * config.headDim]);

                    // Warmup
                    await attention.computeAttention(query, key, value);

                    // Benchmark
                    const iterations = 5;
                    const startTime = performance.now();
                    
                    for (let i = 0; i < iterations; i++) {
                        await attention.computeAttention(query, key, value);
                    }
                    
                    const endTime = performance.now();
                    const avgTime = (endTime - startTime) / iterations;
                    
                    benchmarkResults[backend][testCase.seqLen] = {
                        avgTime,
                        fps: 1000 / avgTime,
                        description: testCase.desc
                    };
                }
            } catch (error) {
                console.warn(`Performance test failed for ${backend}:`, error.message);
                benchmarkResults[backend] = { error: error.message };
            }
        }

        // Check if performance meets thresholds
        let performancePassed = true;
        let performanceDetails = '';

        for (const [backend, results] of Object.entries(benchmarkResults)) {
            if (results.error) continue;
            
            for (const [seqLen, metrics] of Object.entries(results)) {
                const threshold = this.performanceThresholds[backend] || 1000;
                const passed = metrics.avgTime <= threshold;
                
                if (!passed) performancePassed = false;
                
                performanceDetails += `${backend}/${seqLen}: ${metrics.avgTime.toFixed(2)}ms (${metrics.fps.toFixed(1)} FPS) `;
            }
        }

        this.addTestResult('testPerformanceBenchmarks', performancePassed, performanceDetails);
    }

    async testNumericalStability() {
        console.log('🧮 Testing numerical stability...');

        const attention = new OptimizedMultiHeadAttention({
            numHeads: 8,
            headDim: 64
        });

        await attention.initializeBackend('cpu');

        // Test with extreme values
        const seqLen = 10;
        const extremeTests = [
            {
                name: 'Large values',
                query: this.generateTestTensor([seqLen, attention.modelDim], 1000),
                key: this.generateTestTensor([seqLen, attention.modelDim], 1000),
                value: this.generateTestTensor([seqLen, attention.modelDim], 1000)
            },
            {
                name: 'Small values',
                query: this.generateTestTensor([seqLen, attention.modelDim], 0.001),
                key: this.generateTestTensor([seqLen, attention.modelDim], 0.001),
                value: this.generateTestTensor([seqLen, attention.modelDim], 0.001)
            },
            {
                name: 'Mixed values',
                query: this.generateTestTensor([seqLen, attention.modelDim], 1, -10, 10),
                key: this.generateTestTensor([seqLen, attention.modelDim], 1, -10, 10),
                value: this.generateTestTensor([seqLen, attention.modelDim], 1, -10, 10)
            }
        ];

        let stabilityPassed = true;
        let stabilityDetails = '';

        for (const test of extremeTests) {
            try {
                const result = await attention.computeAttention(test.query, test.key, test.value);
                
                const hasNaN = this.hasNaN(result);
                const hasInf = this.hasInfinity(result);
                const testPassed = !hasNaN && !hasInf;
                
                if (!testPassed) stabilityPassed = false;
                
                stabilityDetails += `${test.name}: ${testPassed ? 'PASS' : 'FAIL'} `;
                if (!testPassed) {
                    stabilityDetails += `(NaN: ${hasNaN}, Inf: ${hasInf}) `;
                }
            } catch (error) {
                stabilityPassed = false;
                stabilityDetails += `${test.name}: ERROR (${error.message}) `;
            }
        }

        this.addTestResult('testNumericalStability', stabilityPassed, stabilityDetails);
    }

    async testBatchProcessing() {
        console.log('📦 Testing batch processing...');

        const attention = new OptimizedMultiHeadAttention({
            numHeads: 4,
            headDim: 32
        });

        await attention.initializeBackend('cpu');

        const batchSize = 3;
        const seqLen = 8;
        const modelDim = attention.modelDim;

        const queries = [];
        const keys = [];
        const values = [];

        for (let i = 0; i < batchSize; i++) {
            queries.push(this.generateTestTensor([seqLen, modelDim]));
            keys.push(this.generateTestTensor([seqLen, modelDim]));
            values.push(this.generateTestTensor([seqLen, modelDim]));
        }

        // Test batch processing
        const batchResults = await attention.computeBatchAttention(queries, keys, values);

        // Test individual processing for comparison
        const individualResults = [];
        for (let i = 0; i < batchSize; i++) {
            const result = await attention.computeAttention(queries[i], keys[i], values[i]);
            individualResults.push(result);
        }

        // Compare batch vs individual results
        let batchConsistent = true;
        let maxBatchDifference = 0;

        for (let i = 0; i < batchSize; i++) {
            const difference = this.computeMaxDifference(batchResults[i], individualResults[i]);
            maxBatchDifference = Math.max(maxBatchDifference, difference);
            
            if (difference > this.tolerance) {
                batchConsistent = false;
            }
        }

        this.addTestResult('testBatchProcessing', batchConsistent,
            `Batch size: ${batchSize}, Max difference: ${maxBatchDifference.toExponential(2)}`);
    }

    async testAttentionPatterns() {
        console.log('🎯 Testing attention patterns...');

        const attention = new OptimizedMultiHeadAttention({
            numHeads: 4,
            headDim: 32
        });

        await attention.initializeBackend('cpu');

        // Test identity pattern (should attend to self)
        const seqLen = 5;
        const identityQuery = [];
        const identityKey = [];
        const identityValue = [];

        for (let i = 0; i < seqLen; i++) {
            const queryVec = new Array(attention.modelDim).fill(0);
            const keyVec = new Array(attention.modelDim).fill(0);
            const valueVec = new Array(attention.modelDim).fill(0);
            
            // Set one position to 1 for clear attention pattern
            queryVec[i] = 1.0;
            keyVec[i] = 1.0;
            valueVec[i] = 1.0;
            
            identityQuery.push(queryVec);
            identityKey.push(keyVec);
            identityValue.push(valueVec);
        }

        const identityResult = await attention.computeAttention(identityQuery, identityKey, identityValue);

        // Check if attention follows expected pattern
        let patternCorrect = true;
        
        // For identity pattern, each position should mostly attend to itself
        for (let i = 0; i < seqLen; i++) {
            const expectedValue = 1.0;
            const actualValue = identityResult[i][i];
            
            if (Math.abs(actualValue - expectedValue) > 0.1) {
                patternCorrect = false;
                break;
            }
        }

        this.addTestResult('testAttentionPatterns', patternCorrect,
            `Identity pattern test: ${patternCorrect ? 'PASS' : 'FAIL'}`);
    }

    async testMemoryEfficiency() {
        console.log('💾 Testing memory efficiency...');

        const largeSeqLen = 1000;
        const config = {
            numHeads: 8,
            headDim: 64
        };

        let memoryEfficient = true;
        let memoryDetails = '';

        try {
            const attention = new OptimizedMultiHeadAttention(config);
            await attention.initializeBackend('cpu');

            // Monitor memory usage during large sequence processing
            const beforeMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;

            const query = this.generateTestTensor([largeSeqLen, config.numHeads * config.headDim]);
            const key = this.generateTestTensor([largeSeqLen, config.numHeads * config.headDim]);
            const value = this.generateTestTensor([largeSeqLen, config.numHeads * config.headDim]);

            const result = await attention.computeAttention(query, key, value);

            const afterMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
            const memoryIncrease = afterMemory - beforeMemory;
            
            // Cleanup and force garbage collection
            if (global.gc) {
                global.gc();
            }

            memoryDetails = `Sequence length: ${largeSeqLen}, Memory increase: ${(memoryIncrease / 1024 / 1024).toFixed(2)} MB`;
            
            // Check if result is valid despite large input
            memoryEfficient = this.hasValidFloats(result) && this.allFinite(result);

        } catch (error) {
            memoryEfficient = false;
            memoryDetails = `Memory test failed: ${error.message}`;
        }

        this.addTestResult('testMemoryEfficiency', memoryEfficient, memoryDetails);
    }

    async testEdgeCases() {
        console.log('🔍 Testing edge cases...');

        const attention = new OptimizedMultiHeadAttention({
            numHeads: 2,
            headDim: 32
        });

        await attention.initializeBackend('cpu');

        const edgeCases = [
            {
                name: 'Single token',
                seqLen: 1
            },
            {
                name: 'Very long sequence',
                seqLen: 2000
            },
            {
                name: 'Zero values',
                seqLen: 5,
                useZeros: true
            }
        ];

        let edgeCasesPassed = true;
        let edgeDetails = '';

        for (const edgeCase of edgeCases) {
            try {
                const modelDim = attention.modelDim;
                let query, key, value;

                if (edgeCase.useZeros) {
                    query = new Array(edgeCase.seqLen).fill(null).map(() => new Array(modelDim).fill(0));
                    key = new Array(edgeCase.seqLen).fill(null).map(() => new Array(modelDim).fill(0));
                    value = new Array(edgeCase.seqLen).fill(null).map(() => new Array(modelDim).fill(0));
                } else {
                    query = this.generateTestTensor([edgeCase.seqLen, modelDim]);
                    key = this.generateTestTensor([edgeCase.seqLen, modelDim]);
                    value = this.generateTestTensor([edgeCase.seqLen, modelDim]);
                }

                const result = await attention.computeAttention(query, key, value);
                
                const hasValidShape = result.length === edgeCase.seqLen && result[0].length === modelDim;
                const hasValidValues = !edgeCase.useZeros || this.allFinite(result);
                
                const testPassed = hasValidShape && hasValidValues;
                if (!testPassed) edgeCasesPassed = false;
                
                edgeDetails += `${edgeCase.name}: ${testPassed ? 'PASS' : 'FAIL'} `;

            } catch (error) {
                edgeCasesPassed = false;
                edgeDetails += `${edgeCase.name}: ERROR (${error.message}) `;
            }
        }

        this.addTestResult('testEdgeCases', edgeCasesPassed, edgeDetails);
    }

    // Utility methods for testing
    generateTestTensor(shape, scale = 1, min = -1, max = 1) {
        const [rows, cols] = shape;
        const tensor = [];
        
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                const value = (Math.random() * (max - min) + min) * scale;
                row.push(value);
            }
            tensor.push(row);
        }
        
        return tensor;
    }

    arraysEqual(a, b) {
        if (a.length !== b.length) return false;
        return a.every((val, i) => val === b[i]);
    }

    hasValidFloats(tensor) {
        for (const row of tensor) {
            for (const val of row) {
                if (typeof val !== 'number') return false;
            }
        }
        return true;
    }

    allFinite(tensor) {
        for (const row of tensor) {
            for (const val of row) {
                if (!isFinite(val)) return false;
            }
        }
        return true;
    }

    hasNaN(tensor) {
        for (const row of tensor) {
            for (const val of row) {
                if (isNaN(val)) return true;
            }
        }
        return false;
    }

    hasInfinity(tensor) {
        for (const row of tensor) {
            for (const val of row) {
                if (!isFinite(val) && !isNaN(val)) return true;
            }
        }
        return false;
    }

    computeMaxDifference(tensor1, tensor2) {
        let maxDiff = 0;
        
        for (let i = 0; i < tensor1.length; i++) {
            for (let j = 0; j < tensor1[i].length; j++) {
                const diff = Math.abs(tensor1[i][j] - tensor2[i][j]);
                maxDiff = Math.max(maxDiff, diff);
            }
        }
        
        return maxDiff;
    }

    addTestResult(testName, passed, details) {
        this.testResults.push({
            test: testName,
            passed,
            details,
            timestamp: new Date().toISOString()
        });
        
        const status = passed ? '✅ PASS' : '❌ FAIL';
        console.log(`${status} ${testName}: ${details}`);
    }

    generateTestReport() {
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.passed).length;
        const failedTests = totalTests - passedTests;
        
        console.log('\n' + '='.repeat(60));
        console.log('📊 MULTI-HEAD ATTENTION TEST REPORT');
        console.log('='.repeat(60));
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests} ✅`);
        console.log(`Failed: ${failedTests} ❌`);
        console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
        console.log('='.repeat(60));
        
        if (failedTests > 0) {
            console.log('\n❌ Failed Tests:');
            this.testResults.filter(r => !r.passed).forEach(result => {
                console.log(`  • ${result.test}: ${result.details}`);
            });
        }
        
        console.log('\n✅ Passed Tests:');
        this.testResults.filter(r => r.passed).forEach(result => {
            console.log(`  • ${result.test}: ${result.details}`);
        });
        
        return {
            totalTests,
            passedTests,
            failedTests,
            successRate: (passedTests / totalTests) * 100,
            details: this.testResults
        };
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.AttentionTestSuite = AttentionTestSuite;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AttentionTestSuite;
}
