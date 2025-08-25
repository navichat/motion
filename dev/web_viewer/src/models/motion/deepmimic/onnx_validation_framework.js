/**
 * JavaScript ONNX Runtime Validation Framework
 * 
 * Tests JavaScript ONNX inference against Python reference data to ensure
 * consistency across different execution providers (WebGPU, WebNN, WebGL, WASM)
 */

class ONNXValidationFramework {
    constructor(options = {}) {
        this.tolerance = options.tolerance || {
            absolute: 1e-5,
            relative: 1e-4
        };
        
        this.testResults = {};
        this.executionProviders = ['webgl', 'wasm']; // Start with most compatible
        
        // Try to add advanced providers if available
        if (typeof navigator !== 'undefined') {
            if (navigator.gpu) {
                this.executionProviders.unshift('webgpu');
            }
            if ('ml' in navigator) {
                this.executionProviders.unshift('webnn');
            }
        }
        
        this.currentProvider = null;
        this.loadedModels = new Map();
        this.validationData = null;
    }
    
    /**
     * Initialize ONNX Runtime and load validation data
     */
    async initialize() {
        try {
            // Ensure ONNX Runtime is loaded
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded. Please include onnxruntime-web script.');
            }
            
            console.log('ONNX Validation Framework initialized');
            console.log('Available execution providers:', this.executionProviders);
            
            return true;
        } catch (error) {
            console.error('Failed to initialize validation framework:', error);
            throw error;
        }
    }
    
    /**
     * Load validation reference data from Python tests
     */
    async loadValidationData(validationDataPath) {
        try {
            const response = await fetch(validationDataPath);
            if (!response.ok) {
                throw new Error(`Failed to load validation data: ${response.status}`);
            }
            
            this.validationData = await response.json();
            console.log('Loaded validation data:', {
                model: this.validationData.model_name,
                testCases: Object.keys(this.validationData.javascript_test_data.test_cases).length,
                timestamp: this.validationData.timestamp
            });
            
            return this.validationData;
        } catch (error) {
            console.error('Error loading validation data:', error);
            throw error;
        }
    }
    
    /**
     * Load ONNX model for testing
     */
    async loadModel(modelPath, modelName) {
        try {
            const session = await ort.InferenceSession.create(modelPath, {
                executionProviders: this.getONNXProviders()
            });
            
            this.loadedModels.set(modelName, {
                session,
                path: modelPath,
                inputName: session.inputNames[0],
                outputName: session.outputNames[0],
                inputShape: session.inputMetadata[session.inputNames[0]].dims,
                outputShape: session.outputMetadata[session.outputNames[0]].dims
            });
            
            // Get actual execution provider used
            this.currentProvider = this.detectUsedProvider(session);
            
            console.log(`Loaded model: ${modelName}`);
            console.log(`Using execution provider: ${this.currentProvider}`);
            console.log(`Input shape: ${session.inputMetadata[session.inputNames[0]].dims}`);
            console.log(`Output shape: ${session.outputMetadata[session.outputNames[0]].dims}`);
            
            return this.loadedModels.get(modelName);
        } catch (error) {
            console.error(`Error loading model ${modelName}:`, error);
            throw error;
        }
    }
    
    /**
     * Get ONNX Runtime execution providers in priority order
     */
    getONNXProviders() {
        const providerMap = {
            'webgpu': 'webgpu',
            'webnn': 'webnn',
            'webgl': 'webgl',
            'wasm': 'wasm'
        };
        
        return this.executionProviders.map(p => providerMap[p]).filter(Boolean);
    }
    
    /**
     * Detect which execution provider is actually being used
     */
    detectUsedProvider(session) {
        // This is tricky to detect directly, so we make educated guesses
        // based on available APIs and performance characteristics
        
        if (typeof navigator !== 'undefined') {
            if (navigator.gpu && this.executionProviders.includes('webgpu')) {
                return 'webgpu';
            }
            if ('ml' in navigator && this.executionProviders.includes('webnn')) {
                return 'webnn';
            }
        }
        
        // Default assumption based on what we requested
        return this.executionProviders.includes('webgl') ? 'webgl' : 'wasm';
    }
    
    /**
     * Run inference on a single test case
     */
    async runInference(modelName, testInput) {
        const model = this.loadedModels.get(modelName);
        if (!model) {
            throw new Error(`Model ${modelName} not loaded`);
        }
        
        try {
            // Prepare input tensor
            const inputTensor = new ort.Tensor('float32', 
                new Float32Array(testInput), 
                [1, testInput.length]
            );
            
            // Run inference
            const startTime = performance.now();
            const results = await model.session.run({
                [model.inputName]: inputTensor
            });
            const inferenceTime = performance.now() - startTime;
            
            // Extract output
            const output = Array.from(results[model.outputName].data);
            
            return {
                output,
                inferenceTime,
                executionProvider: this.currentProvider,
                inputShape: [1, testInput.length],
                outputShape: results[model.outputName].dims
            };
            
        } catch (error) {
            console.error(`Inference error for ${modelName}:`, error);
            throw error;
        }
    }
    
    /**
     * Compare JavaScript output with reference data
     */
    compareOutputs(jsOutput, referenceOutput, testName) {
        if (jsOutput.length !== referenceOutput.length) {
            return {
                status: 'fail',
                error: `Shape mismatch: JS ${jsOutput.length} vs Reference ${referenceOutput.length}`
            };
        }
        
        const jsArray = Array.isArray(jsOutput) ? jsOutput : Array.from(jsOutput);
        const refArray = Array.isArray(referenceOutput) ? referenceOutput : Array.from(referenceOutput);
        
        // Calculate differences
        const absDiffs = jsArray.map((val, i) => Math.abs(val - refArray[i]));
        const relDiffs = jsArray.map((val, i) => {
            const ref = refArray[i];
            return Math.abs((val - ref) / (Math.abs(ref) + 1e-8));
        });
        
        const maxAbsDiff = Math.max(...absDiffs);
        const maxRelDiff = Math.max(...relDiffs);
        const meanAbsDiff = absDiffs.reduce((a, b) => a + b, 0) / absDiffs.length;
        const meanRelDiff = relDiffs.reduce((a, b) => a + b, 0) / relDiffs.length;
        
        // Check tolerances
        const absOk = maxAbsDiff < this.tolerance.absolute;
        const relOk = maxRelDiff < this.tolerance.relative;
        const withinTolerance = absOk && relOk;
        
        return {
            status: withinTolerance ? 'pass' : 'fail',
            maxAbsoluteDiff: maxAbsDiff,
            maxRelativeDiff: maxRelDiff,
            meanAbsoluteDiff: meanAbsDiff,
            meanRelativeDiff: meanRelDiff,
            withinTolerance,
            absoluteOk: absOk,
            relativeOk: relOk,
            tolerance: this.tolerance
        };
    }
    
    /**
     * Validate a single model against reference data
     */
    async validateModel(modelName, modelPath, validationDataPath) {
        console.log(`\n=== Validating ${modelName} ===`);
        
        try {
            // Load validation data and model
            await this.loadValidationData(validationDataPath);
            await this.loadModel(modelPath, modelName);
            
            const testCases = this.validationData.javascript_test_data.test_cases;
            const results = {};
            
            let totalTests = 0;
            let passedTests = 0;
            let totalInferenceTime = 0;
            
            // Run all test cases
            for (const [testName, testData] of Object.entries(testCases)) {
                console.log(`Running test: ${testName}`);
                
                try {
                    // Run JavaScript inference
                    const jsResult = await this.runInference(modelName, testData.input);
                    totalInferenceTime += jsResult.inferenceTime;
                    
                    // Compare with reference
                    const comparison = this.compareOutputs(
                        jsResult.output, 
                        testData.expected_output, 
                        testName
                    );
                    
                    results[testName] = {
                        description: testData.description,
                        status: comparison.status,
                        inferenceTime: jsResult.inferenceTime,
                        executionProvider: jsResult.executionProvider,
                        comparison,
                        inputShape: jsResult.inputShape,
                        outputShape: jsResult.outputShape
                    };
                    
                    totalTests++;
                    if (comparison.status === 'pass') {
                        passedTests++;
                        console.log(`  ✅ ${testName}: PASS (${jsResult.inferenceTime.toFixed(2)}ms)`);
                    } else {
                        console.log(`  ❌ ${testName}: FAIL (max_abs: ${comparison.maxAbsoluteDiff.toFixed(2e)}, max_rel: ${comparison.maxRelativeDiff.toFixed(2e)})`);
                    }
                    
                } catch (error) {
                    console.log(`  🚫 ${testName}: ERROR - ${error.message}`);
                    results[testName] = {
                        description: testData.description,
                        status: 'error',
                        error: error.message
                    };
                    totalTests++;
                }
            }
            
            // Create summary
            const summary = {
                modelName,
                executionProvider: this.currentProvider,
                totalTests,
                passedTests,
                failedTests: totalTests - passedTests,
                passRate: totalTests > 0 ? passedTests / totalTests : 0,
                averageInferenceTime: totalInferenceTime / totalTests,
                timestamp: new Date().toISOString(),
                results
            };
            
            this.testResults[modelName] = summary;
            
            // Print summary
            this.printValidationSummary(summary);
            
            return summary;
            
        } catch (error) {
            console.error(`Error validating model ${modelName}:`, error);
            throw error;
        }
    }
    
    /**
     * Validate multiple execution providers
     */
    async validateAllProviders(modelName, modelPath, validationDataPath) {
        const providerResults = {};
        
        console.log(`\n=== Testing all execution providers for ${modelName} ===`);
        
        for (const provider of this.executionProviders) {
            console.log(`\nTesting execution provider: ${provider}`);
            
            try {
                // Temporarily set single provider
                const originalProviders = this.executionProviders;
                this.executionProviders = [provider];
                
                // Clear loaded models to force reload with new provider
                this.loadedModels.clear();
                
                // Run validation
                const result = await this.validateModel(modelName, modelPath, validationDataPath);
                providerResults[provider] = result;
                
                // Restore original providers
                this.executionProviders = originalProviders;
                
            } catch (error) {
                console.error(`Error testing provider ${provider}:`, error);
                providerResults[provider] = {
                    status: 'error',
                    error: error.message
                };
            }
        }
        
        // Print provider comparison
        this.printProviderComparison(providerResults);
        
        return providerResults;
    }
    
    /**
     * Print validation summary for a single model
     */
    printValidationSummary(summary) {
        console.log(`\n--- Validation Summary for ${summary.modelName} ---`);
        console.log(`Execution Provider: ${summary.executionProvider}`);
        console.log(`Total tests: ${summary.totalTests}`);
        console.log(`Passed: ${summary.passedTests}`);
        console.log(`Failed: ${summary.failedTests}`);
        console.log(`Pass rate: ${(summary.passRate * 100).toFixed(1)}%`);
        console.log(`Average inference time: ${summary.averageInferenceTime.toFixed(2)}ms`);
    }
    
    /**
     * Print comparison across execution providers
     */
    printProviderComparison(providerResults) {
        console.log(`\n--- Execution Provider Comparison ---`);
        
        const headers = ['Provider', 'Status', 'Pass Rate', 'Avg Time (ms)', 'Tests'];
        const rows = [];
        
        for (const [provider, result] of Object.entries(providerResults)) {
            if (result.status === 'error') {
                rows.push([provider, 'ERROR', '-', '-', '-']);
            } else {
                rows.push([
                    provider,
                    result.passRate === 1 ? 'PASS' : 'FAIL',
                    `${(result.passRate * 100).toFixed(1)}%`,
                    result.averageInferenceTime.toFixed(2),
                    `${result.passedTests}/${result.totalTests}`
                ]);
            }
        }
        
        // Simple table formatting
        const colWidths = headers.map((header, i) => 
            Math.max(header.length, ...rows.map(row => row[i].length))
        );
        
        const formatRow = (row) => 
            row.map((cell, i) => cell.padEnd(colWidths[i])).join(' | ');
        
        console.log(formatRow(headers));
        console.log(colWidths.map(w => '-'.repeat(w)).join('-|-'));
        rows.forEach(row => console.log(formatRow(row)));
    }
    
    /**
     * Generate detailed validation report
     */
    generateReport() {
        const report = {
            timestamp: new Date().toISOString(),
            tolerance: this.tolerance,
            availableProviders: this.executionProviders,
            results: this.testResults,
            summary: {
                totalModels: Object.keys(this.testResults).length,
                allPassed: Object.values(this.testResults).every(r => r.passRate === 1),
                averagePassRate: Object.values(this.testResults).reduce((sum, r) => sum + r.passRate, 0) / Object.keys(this.testResults).length
            }
        };
        
        return report;
    }
    
    /**
     * Export results as JSON
     */
    exportResults() {
        const report = this.generateReport();
        const blob = new Blob([JSON.stringify(report, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `validation_report_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Get validation status for a model
     */
    getValidationStatus(modelName) {
        return this.testResults[modelName] || null;
    }
    
    /**
     * Check if all validations passed
     */
    allValidationsPassed() {
        return Object.values(this.testResults).every(result => result.passRate === 1);
    }
}

// Export for use in modules or global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXValidationFramework;
} else if (typeof window !== 'undefined') {
    window.ONNXValidationFramework = ONNXValidationFramework;
}
