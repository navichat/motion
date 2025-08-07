#!/usr/bin/env node

/**
 * Node.js Command Line DeepMimic Validation
 * Compare Python TensorFlow outputs with JavaScript ONNX Runtime outputs
 */

const fs = require('fs');
const path = require('path');

// Since we're in Node.js, we need to use onnxruntime-node instead of onnxruntime-web
let ort;
try {
    ort = require('onnxruntime-node');
} catch (e) {
    console.error('onnxruntime-node not installed. Install with: npm install onnxruntime-node');
    process.exit(1);
}

class NodeDeepMimicValidator {
    constructor() {
        this.tolerance = 1e-4;
        this.results = [];
    }

    /**
     * Load ONNX model
     */
    async loadModel(modelPath) {
        try {
            console.log(`Loading model: ${modelPath}`);
            const session = await ort.InferenceSession.create(modelPath);
            
            const modelInfo = {
                inputName: session.inputNames[0],
                outputName: session.outputNames[0],
                session: session
            };
            
            console.log(`✅ Model loaded: input=${modelInfo.inputName}, output=${modelInfo.outputName}`);
            return modelInfo;
            
        } catch (error) {
            console.error(`❌ Failed to load model: ${error.message}`);
            throw error;
        }
    }

    /**
     * Run inference
     */
    async predict(session, inputName, outputName, inputData) {
        try {
            const inputTensor = new ort.Tensor('float32', inputData, [1, inputData.length]);
            
            const startTime = process.hrtime.bigint();
            const outputs = await session.run({
                [inputName]: inputTensor
            });
            const endTime = process.hrtime.bigint();
            
            const inferenceTime = Number(endTime - startTime) / 1000000; // Convert to milliseconds
            const outputData = outputs[outputName].data;
            
            return {
                actions: new Float32Array(outputData),
                inferenceTime: inferenceTime
            };
            
        } catch (error) {
            console.error(`❌ Inference failed: ${error.message}`);
            throw error;
        }
    }

    /**
     * Load test data from JSON file
     */
    loadTestData(filePath) {
        try {
            const data = fs.readFileSync(filePath, 'utf8');
            return JSON.parse(data);
        } catch (error) {
            console.error(`❌ Failed to load test data: ${error.message}`);
            throw error;
        }
    }

    /**
     * Compare two arrays
     */
    compareArrays(actual, expected) {
        if (actual.length !== expected.length) {
            throw new Error(`Array length mismatch: expected ${expected.length}, got ${actual.length}`);
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
     * Validate a single model
     */
    async validateModel(modelPath, testDataPath) {
        try {
            console.log(`\n🧪 Validating: ${path.basename(modelPath)}`);
            
            // Load model
            const modelInfo = await this.loadModel(modelPath);
            
            // Load test data
            const testData = this.loadTestData(testDataPath);
            console.log(`📋 Loaded ${testData.test_cases.length} test cases`);
            
            const results = {
                modelPath: modelPath,
                testCases: [],
                summary: {
                    total: testData.test_cases.length,
                    passed: 0,
                    failed: 0,
                    averageError: 0,
                    maxError: 0,
                    averageInferenceTime: 0
                }
            };

            let totalInferenceTime = 0;
            const errors = [];

            // Run each test case
            for (let i = 0; i < testData.test_cases.length; i++) {
                const testCase = testData.test_cases[i];
                const input = new Float32Array(testCase.input[0]); // Remove batch dimension
                const expected = new Float32Array(testCase.expected_output[0]);

                process.stdout.write(`  Test ${i + 1}/${testData.test_cases.length}... `);

                try {
                    // Run inference
                    const result = await this.predict(
                        modelInfo.session, 
                        modelInfo.inputName, 
                        modelInfo.outputName, 
                        input
                    );

                    totalInferenceTime += result.inferenceTime;

                    // Compare outputs
                    const comparison = this.compareArrays(result.actions, expected);
                    errors.push(comparison.averageDifference);

                    const testResult = {
                        caseIndex: i,
                        passed: comparison.withinTolerance,
                        maxDifference: comparison.maxDifference,
                        averageDifference: comparison.averageDifference,
                        inferenceTime: result.inferenceTime
                    };

                    results.testCases.push(testResult);

                    if (testResult.passed) {
                        results.summary.passed++;
                        console.log(`✅ PASS (${comparison.maxDifference.toExponential(2)})`);
                    } else {
                        results.summary.failed++;
                        console.log(`❌ FAIL (${comparison.maxDifference.toExponential(2)})`);
                    }

                } catch (error) {
                    console.log(`💥 ERROR: ${error.message}`);
                    results.summary.failed++;
                    results.testCases.push({
                        caseIndex: i,
                        passed: false,
                        error: error.message
                    });
                }
            }

            // Calculate summary statistics
            if (results.summary.passed > 0) {
                results.summary.averageError = errors.reduce((a, b) => a + b, 0) / errors.length;
                results.summary.maxError = Math.max(...errors);
                results.summary.averageInferenceTime = totalInferenceTime / testData.test_cases.length;
            }

            const passRate = (results.summary.passed / results.summary.total * 100).toFixed(1);
            
            console.log(`\n📊 Results Summary:`);
            console.log(`   Pass Rate: ${passRate}% (${results.summary.passed}/${results.summary.total})`);
            console.log(`   Average Error: ${results.summary.averageError.toExponential(3)}`);
            console.log(`   Max Error: ${results.summary.maxError.toExponential(3)}`);
            console.log(`   Average Inference Time: ${results.summary.averageInferenceTime.toFixed(2)}ms`);
            console.log(`   Tolerance: ${this.tolerance.toExponential(3)}`);

            this.results.push(results);
            return results;

        } catch (error) {
            console.error(`💥 Model validation failed: ${error.message}`);
            throw error;
        }
    }

    /**
     * Validate all models in a directory
     */
    async validateAllModels(modelsDir, testDataDir) {
        console.log('🚀 Starting Node.js DeepMimic Validation');
        console.log(`Models directory: ${modelsDir}`);
        console.log(`Test data directory: ${testDataDir}`);
        console.log(`Tolerance: ${this.tolerance.toExponential(3)}`);

        // Find all compatible ONNX models with test data
        const modelFiles = fs.readdirSync(modelsDir)
            .filter(file => file.endsWith('.onnx') && file.startsWith('compatible_humanoid3d_'))
            .sort();

        console.log(`\n📁 Found ${modelFiles.length} models to validate:`);
        modelFiles.forEach((file, i) => console.log(`  ${i + 1}. ${file}`));

        const comprehensiveResults = {
            startTime: new Date().toISOString(),
            tolerance: this.tolerance,
            models: {},
            summary: {
                totalModels: modelFiles.length,
                successfulModels: 0,
                totalTestCases: 0,
                passedTestCases: 0,
                averagePassRate: 0,
                averageInferenceTime: 0
            }
        };

        // Validate each model
        for (const modelFile of modelFiles) {
            try {
                const modelPath = path.join(modelsDir, modelFile);
                const modelName = modelFile.replace('.onnx', '').replace('compatible_', '');
                const testDataPath = path.join(testDataDir, `${modelName}_test_data.json`);

                if (!fs.existsSync(testDataPath)) {
                    console.log(`⚠️ Skipping ${modelName}: test data not found`);
                    continue;
                }

                const result = await this.validateModel(modelPath, testDataPath);
                comprehensiveResults.models[modelName] = result;
                
                if (result.summary.passed > 0) {
                    comprehensiveResults.summary.successfulModels++;
                }
                
                comprehensiveResults.summary.totalTestCases += result.summary.total;
                comprehensiveResults.summary.passedTestCases += result.summary.passed;

            } catch (error) {
                console.error(`💥 Failed to validate ${modelFile}: ${error.message}`);
                comprehensiveResults.models[modelFile.replace('.onnx', '')] = { error: error.message };
            }
        }

        // Calculate final statistics
        if (comprehensiveResults.summary.totalTestCases > 0) {
            comprehensiveResults.summary.averagePassRate = 
                (comprehensiveResults.summary.passedTestCases / comprehensiveResults.summary.totalTestCases) * 100;
        }

        const allResults = Object.values(comprehensiveResults.models).filter(r => r.summary);
        if (allResults.length > 0) {
            comprehensiveResults.summary.averageInferenceTime = 
                allResults.reduce((sum, r) => sum + r.summary.averageInferenceTime, 0) / allResults.length;
        }

        comprehensiveResults.endTime = new Date().toISOString();

        this.printFinalReport(comprehensiveResults);
        
        // Save results to file
        const outputFile = 'node_validation_results.json';
        fs.writeFileSync(outputFile, JSON.stringify(comprehensiveResults, null, 2));
        console.log(`\n💾 Results saved to: ${outputFile}`);

        return comprehensiveResults;
    }

    /**
     * Print final validation report
     */
    printFinalReport(results) {
        console.log('\n\n' + '='.repeat(60));
        console.log('🏁 FINAL VALIDATION REPORT');
        console.log('='.repeat(60));
        console.log(`⏱️  Duration: ${results.startTime} → ${results.endTime}`);
        console.log(`🎯 Tolerance: ${results.tolerance.toExponential(3)}`);
        console.log(`📊 Models: ${results.summary.successfulModels}/${results.summary.totalModels} successful`);
        console.log(`🧪 Test Cases: ${results.summary.passedTestCases}/${results.summary.totalTestCases} passed`);
        console.log(`📈 Overall Pass Rate: ${results.summary.averagePassRate.toFixed(1)}%`);
        console.log(`⚡ Average Inference Time: ${results.summary.averageInferenceTime.toFixed(2)}ms`);
        
        console.log('\n📋 Individual Model Results:');
        for (const [modelName, result] of Object.entries(results.models)) {
            if (result.error) {
                console.log(`  ❌ ${modelName}: ERROR - ${result.error}`);
            } else if (result.summary) {
                const passRate = (result.summary.passed / result.summary.total * 100).toFixed(1);
                console.log(`  ${result.summary.passed === result.summary.total ? '✅' : '⚠️'} ${modelName}: ${passRate}% (${result.summary.passed}/${result.summary.total})`);
            }
        }
        
        console.log('='.repeat(60));
    }

    /**
     * Set tolerance
     */
    setTolerance(tolerance) {
        this.tolerance = tolerance;
        console.log(`Tolerance set to ${tolerance.toExponential(3)}`);
    }
}

// Command line interface
async function main() {
    const args = process.argv.slice(2);
    
    if (args.length < 2) {
        console.log('Usage: node validate-deepmimic.js <models_directory> <test_data_directory> [tolerance]');
        console.log('Example: node validate-deepmimic.js ./models ./validation_results 0.0001');
        process.exit(1);
    }

    const modelsDir = args[0];
    const testDataDir = args[1];
    const tolerance = args[2] ? parseFloat(args[2]) : 1e-4;

    if (!fs.existsSync(modelsDir)) {
        console.error(`❌ Models directory not found: ${modelsDir}`);
        process.exit(1);
    }

    if (!fs.existsSync(testDataDir)) {
        console.error(`❌ Test data directory not found: ${testDataDir}`);
        process.exit(1);
    }

    try {
        const validator = new NodeDeepMimicValidator();
        validator.setTolerance(tolerance);
        
        const results = await validator.validateAllModels(modelsDir, testDataDir);
        
        // Exit with appropriate code
        const success = results.summary.averagePassRate > 95;
        process.exit(success ? 0 : 1);
        
    } catch (error) {
        console.error('💥 Validation failed:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = NodeDeepMimicValidator;
