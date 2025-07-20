/**
 * DeepMimic Cross-Platform Validation Tests
 * Compare outputs between Python TensorFlow and JavaScript ONNX Runtime
 */

class DeepMimicValidator {
    constructor() {
        this.inference = new DeepMimicInference();
        this.testResults = [];
        this.tolerance = 1e-4; // Default tolerance for comparison
    }

    /**
     * Initialize the validator
     */
    async initialize() {
        await this.inference.initialize();
        console.log('DeepMimic Validator initialized');
    }

    /**
     * Load test data from Python validation results
     * @param {string} testDataPath - Path to JSON file with Python reference data
     */
    async loadTestData(testDataPath) {
        try {
            const response = await fetch(testDataPath);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const testData = await response.json();
            return testData;
        } catch (error) {
            console.error('Failed to load test data:', error);
            throw error;
        }
    }

    /**
     * Validate a single model against Python reference data
     * @param {string} modelPath - Path to ONNX model
     * @param {Object} referenceData - Python reference test data
     * @param {string} executionProvider - Execution provider to test
     */
    async validateModel(modelPath, referenceData, executionProvider = 'wasm') {
        try {
            console.log(`\n=== Validating Model: ${modelPath} ===`);
            console.log(`Execution Provider: ${executionProvider}`);

            // Load the model
            await this.inference.loadModel(modelPath, executionProvider);

            const modelResults = {
                modelPath,
                executionProvider,
                testCases: [],
                summary: {
                    total: 0,
                    passed: 0,
                    failed: 0,
                    averageError: 0,
                    maxError: 0
                }
            };

            // Test each reference test case
            for (let i = 0; i < referenceData.test_cases.length; i++) {
                const testCase = referenceData.test_cases[i];
                const input = new Float32Array(testCase.input[0]); // Remove batch dimension
                const expectedOutput = new Float32Array(testCase.expected_output[0]);

                console.log(`Testing case ${i + 1}/${referenceData.test_cases.length}...`);

                try {
                    // Run JavaScript inference
                    const result = await this.inference.predict(input);
                    const actualOutput = result.actions;

                    // Compare outputs
                    const comparison = this.compareOutputs(actualOutput, expectedOutput);
                    
                    const testResult = {
                        caseIndex: i,
                        passed: comparison.withinTolerance,
                        maxDifference: comparison.maxDifference,
                        averageDifference: comparison.averageDifference,
                        inferenceTime: result.inferenceTime,
                        input: Array.from(input.slice(0, 5)), // Store first 5 elements for debugging
                        expectedOutput: Array.from(expectedOutput.slice(0, 5)),
                        actualOutput: Array.from(actualOutput.slice(0, 5))
                    };

                    modelResults.testCases.push(testResult);
                    modelResults.summary.total++;
                    
                    if (testResult.passed) {
                        modelResults.summary.passed++;
                        console.log(`  ✅ Case ${i + 1}: PASSED (max diff: ${comparison.maxDifference.toExponential(3)})`);
                    } else {
                        modelResults.summary.failed++;
                        console.log(`  ❌ Case ${i + 1}: FAILED (max diff: ${comparison.maxDifference.toExponential(3)})`);
                    }

                } catch (error) {
                    console.error(`  💥 Case ${i + 1}: ERROR - ${error.message}`);
                    modelResults.testCases.push({
                        caseIndex: i,
                        passed: false,
                        error: error.message
                    });
                    modelResults.summary.failed++;
                    modelResults.summary.total++;
                }
            }

            // Calculate summary statistics
            const validCases = modelResults.testCases.filter(tc => !tc.error);
            if (validCases.length > 0) {
                modelResults.summary.averageError = validCases.reduce((sum, tc) => sum + tc.averageDifference, 0) / validCases.length;
                modelResults.summary.maxError = Math.max(...validCases.map(tc => tc.maxDifference));
            }

            const passRate = (modelResults.summary.passed / modelResults.summary.total * 100).toFixed(1);
            console.log(`\n📊 Model Validation Summary:`);
            console.log(`   Total test cases: ${modelResults.summary.total}`);
            console.log(`   Passed: ${modelResults.summary.passed} (${passRate}%)`);
            console.log(`   Failed: ${modelResults.summary.failed}`);
            console.log(`   Average error: ${modelResults.summary.averageError.toExponential(3)}`);
            console.log(`   Max error: ${modelResults.summary.maxError.toExponential(3)}`);
            console.log(`   Tolerance: ${this.tolerance.toExponential(3)}`);

            this.testResults.push(modelResults);
            return modelResults;

        } catch (error) {
            console.error('Model validation failed:', error);
            throw error;
        }
    }

    /**
     * Compare two output arrays
     */
    compareOutputs(actual, expected) {
        if (actual.length !== expected.length) {
            throw new Error(`Output length mismatch: expected ${expected.length}, got ${actual.length}`);
        }

        let maxDiff = 0;
        let totalDiff = 0;
        let withinTolerance = true;

        for (let i = 0; i < actual.length; i++) {
            const diff = Math.abs(actual[i] - expected[i]);
            maxDiff = Math.max(maxDiff, diff);
            totalDiff += diff;
            
            if (diff > this.tolerance) {
                withinTolerance = false;
            }
        }

        return {
            maxDifference: maxDiff,
            averageDifference: totalDiff / actual.length,
            withinTolerance
        };
    }

    /**
     * Run comprehensive validation across all models and execution providers
     * @param {string} testDataDir - Directory containing test data JSON files
     * @param {Array<string>} modelNames - List of model names to test
     */
    async runComprehensiveValidation(testDataDir, modelNames) {
        console.log('🚀 Starting Comprehensive DeepMimic Validation');
        console.log(`Testing ${modelNames.length} models across multiple execution providers`);

        const comprehensiveResults = {
            startTime: new Date().toISOString(),
            tolerance: this.tolerance,
            models: {},
            executionProviders: {},
            summary: {
                totalModels: modelNames.length,
                successfulModels: 0,
                totalTestCases: 0,
                passedTestCases: 0,
                averagePassRate: 0
            }
        };

        // Test each model
        for (const modelName of modelNames) {
            try {
                console.log(`\n🎯 Testing model: ${modelName}`);

                // Load reference test data
                const testDataPath = `${testDataDir}/${modelName}_test_data.json`;
                const referenceData = await this.loadTestData(testDataPath);
                
                const modelPath = `${modelName}.onnx`;
                const modelResults = {};

                // Test each available execution provider
                for (const provider of this.inference.supportedProviders) {
                    try {
                        console.log(`\n   🔧 Testing with ${provider}...`);
                        const result = await this.validateModel(modelPath, referenceData, provider);
                        modelResults[provider] = result;

                        // Track execution provider performance
                        if (!comprehensiveResults.executionProviders[provider]) {
                            comprehensiveResults.executionProviders[provider] = {
                                modelsTestedSuccessfully: 0,
                                totalTestCases: 0,
                                passedTestCases: 0,
                                averageInferenceTime: 0,
                                times: []
                            };
                        }

                        const providerStats = comprehensiveResults.executionProviders[provider];
                        providerStats.modelsTestedSuccessfully++;
                        providerStats.totalTestCases += result.summary.total;
                        providerStats.passedTestCases += result.summary.passed;
                        
                        // Collect inference times
                        const validTimes = result.testCases.filter(tc => tc.inferenceTime).map(tc => tc.inferenceTime);
                        providerStats.times.push(...validTimes);

                    } catch (error) {
                        console.error(`   ❌ Failed with ${provider}: ${error.message}`);
                        modelResults[provider] = { error: error.message };
                    }
                }

                comprehensiveResults.models[modelName] = modelResults;

                // Check if at least one provider succeeded
                const hasSuccessfulProvider = Object.values(modelResults).some(r => !r.error);
                if (hasSuccessfulProvider) {
                    comprehensiveResults.summary.successfulModels++;
                }

            } catch (error) {
                console.error(`💥 Failed to test model ${modelName}: ${error.message}`);
                comprehensiveResults.models[modelName] = { error: error.message };
            }
        }

        // Calculate final statistics
        for (const [provider, stats] of Object.entries(comprehensiveResults.executionProviders)) {
            if (stats.times.length > 0) {
                stats.averageInferenceTime = stats.times.reduce((a, b) => a + b, 0) / stats.times.length;
            }
            stats.passRate = stats.totalTestCases > 0 ? (stats.passedTestCases / stats.totalTestCases * 100) : 0;
        }

        // Calculate overall summary
        const allResults = Object.values(comprehensiveResults.models).flatMap(model => 
            Object.values(model).filter(result => result.summary)
        );
        
        comprehensiveResults.summary.totalTestCases = allResults.reduce((sum, r) => sum + r.summary.total, 0);
        comprehensiveResults.summary.passedTestCases = allResults.reduce((sum, r) => sum + r.summary.passed, 0);
        comprehensiveResults.summary.averagePassRate = comprehensiveResults.summary.totalTestCases > 0 ? 
            (comprehensiveResults.summary.passedTestCases / comprehensiveResults.summary.totalTestCases * 100) : 0;

        comprehensiveResults.endTime = new Date().toISOString();

        this.printComprehensiveReport(comprehensiveResults);
        return comprehensiveResults;
    }

    /**
     * Print comprehensive validation report
     */
    printComprehensiveReport(results) {
        console.log('\n\n📋 COMPREHENSIVE VALIDATION REPORT');
        console.log('=' .repeat(60));
        console.log(`🕐 Start Time: ${results.startTime}`);
        console.log(`🕐 End Time: ${results.endTime}`);
        console.log(`🎯 Tolerance: ${results.tolerance.toExponential(3)}`);
        console.log(`📊 Models Tested: ${results.summary.totalModels}`);
        console.log(`✅ Successful Models: ${results.summary.successfulModels}`);
        console.log(`🧪 Total Test Cases: ${results.summary.totalTestCases}`);
        console.log(`✅ Passed Test Cases: ${results.summary.passedTestCases}`);
        console.log(`📈 Overall Pass Rate: ${results.summary.averagePassRate.toFixed(1)}%`);

        console.log('\n🔧 EXECUTION PROVIDER PERFORMANCE:');
        for (const [provider, stats] of Object.entries(results.executionProviders)) {
            console.log(`\n   ${provider.toUpperCase()}:`);
            console.log(`     Models: ${stats.modelsTestedSuccessfully}`);
            console.log(`     Test Cases: ${stats.passedTestCases}/${stats.totalTestCases} (${stats.passRate.toFixed(1)}%)`);
            console.log(`     Avg Inference Time: ${stats.averageInferenceTime.toFixed(2)}ms`);
        }

        console.log('\n📋 MODEL RESULTS:');
        for (const [modelName, modelResults] of Object.entries(results.models)) {
            console.log(`\n   ${modelName}:`);
            for (const [provider, result] of Object.entries(modelResults)) {
                if (result.error) {
                    console.log(`     ${provider}: ❌ ERROR - ${result.error}`);
                } else if (result.summary) {
                    const passRate = (result.summary.passed / result.summary.total * 100).toFixed(1);
                    console.log(`     ${provider}: ${result.summary.passed}/${result.summary.total} (${passRate}%)`);
                }
            }
        }

        console.log('\n' + '=' .repeat(60));
    }

    /**
     * Export test results to JSON
     */
    exportResults(filename = 'deepmimic_validation_results.json') {
        const resultsJson = JSON.stringify({
            testResults: this.testResults,
            timestamp: new Date().toISOString(),
            tolerance: this.tolerance
        }, null, 2);

        // Create download link
        const blob = new Blob([resultsJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);

        console.log(`Results exported to ${filename}`);
    }

    /**
     * Set comparison tolerance
     */
    setTolerance(tolerance) {
        this.tolerance = tolerance;
        console.log(`Tolerance set to ${tolerance.toExponential(3)}`);
    }

    /**
     * Get test results
     */
    getResults() {
        return this.testResults;
    }

    /**
     * Clear test results
     */
    clearResults() {
        this.testResults = [];
        console.log('Test results cleared');
    }
}

// Export for both module and global usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeepMimicValidator;
} else {
    window.DeepMimicValidator = DeepMimicValidator;
}
