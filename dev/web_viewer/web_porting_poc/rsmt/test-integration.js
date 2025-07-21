#!/usr/bin/env node

/**
 * RSMT Integration Test Script
 * Tests the complete RSMT JavaScript implementation
 */

const fs = require('fs').promises;
const path = require('path');

class RSMTIntegrationTest {
    constructor() {
        this.testResults = {
            timestamp: new Date().toISOString(),
            tests: {},
            summary: {
                total: 0,
                passed: 0,
                failed: 0,
                errors: []
            }
        };
    }

    /**
     * Run all integration tests
     */
    async runTests() {
        console.log('🚀 Starting RSMT Integration Tests...\n');

        const tests = [
            { name: 'File Structure', fn: this.testFileStructure },
            { name: 'ONNX Models', fn: this.testONNXModels },
            { name: 'JavaScript Files', fn: this.testJavaScriptFiles },
            { name: 'Dependencies', fn: this.testDependencies },
            { name: 'HTML Demo', fn: this.testHTMLDemo },
            { name: 'Integration Points', fn: this.testIntegrationPoints }
        ];

        for (const test of tests) {
            console.log(`📋 Running test: ${test.name}`);
            try {
                const result = await test.fn.call(this);
                this.testResults.tests[test.name] = {
                    status: 'passed',
                    result: result,
                    timestamp: new Date().toISOString()
                };
                this.testResults.summary.passed++;
                console.log(`✅ ${test.name}: PASSED\n`);
            } catch (error) {
                this.testResults.tests[test.name] = {
                    status: 'failed',
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
                this.testResults.summary.failed++;
                this.testResults.summary.errors.push(`${test.name}: ${error.message}`);
                console.error(`❌ ${test.name}: FAILED - ${error.message}\n`);
            }
            this.testResults.summary.total++;
        }

        this.printSummary();
        await this.saveReport();
        
        return this.testResults;
    }

    /**
     * Test file structure
     */
    async testFileStructure() {
        const requiredFiles = [
            'rsmt-inference.js',
            'rsmt-bvh-integration.js',
            'rsmt-demo.html',
            'rsmt-validator.js',
            'deepphase.onnx',
            'stylevae.onnx',
            'transitionnet.onnx'
        ];

        const missingFiles = [];
        const existingFiles = [];

        for (const file of requiredFiles) {
            try {
                await fs.access(file);
                existingFiles.push(file);
            } catch (error) {
                missingFiles.push(file);
            }
        }

        if (missingFiles.length > 0) {
            throw new Error(`Missing required files: ${missingFiles.join(', ')}`);
        }

        return {
            requiredFiles: requiredFiles.length,
            existingFiles: existingFiles.length,
            files: existingFiles
        };
    }

    /**
     * Test ONNX models
     */
    async testONNXModels() {
        const models = ['deepphase.onnx', 'stylevae.onnx', 'transitionnet.onnx'];
        const modelInfo = {};

        for (const model of models) {
            try {
                const stats = await fs.stat(model);
                modelInfo[model] = {
                    size: stats.size,
                    modified: stats.mtime.toISOString(),
                    exists: true
                };

                // Check if file is not empty
                if (stats.size === 0) {
                    throw new Error(`${model} is empty`);
                }

                // Basic ONNX file format check (should start with specific bytes)
                const buffer = await fs.readFile(model);
                if (buffer.length < 4) {
                    throw new Error(`${model} is too small to be a valid ONNX file`);
                }

            } catch (error) {
                modelInfo[model] = {
                    exists: false,
                    error: error.message
                };
                throw new Error(`ONNX model ${model}: ${error.message}`);
            }
        }

        return modelInfo;
    }

    /**
     * Test JavaScript files
     */
    async testJavaScriptFiles() {
        const jsFiles = [
            'rsmt-inference.js',
            'rsmt-bvh-integration.js',
            'rsmt-validator.js'
        ];

        const fileInfo = {};

        for (const file of jsFiles) {
            try {
                const content = await fs.readFile(file, 'utf8');
                
                // Basic syntax checks
                const checks = {
                    hasClassDefinition: /class\s+\w+/.test(content),
                    hasConstructor: /constructor\s*\(/.test(content),
                    hasAsyncMethods: /async\s+\w+\s*\(/.test(content),
                    hasExports: /module\.exports|window\.\w+/.test(content),
                    lineCount: content.split('\n').length,
                    size: content.length
                };

                // Check for required classes
                if (file === 'rsmt-inference.js' && !content.includes('class RSMTInference')) {
                    throw new Error(`${file} missing RSMTInference class`);
                }
                if (file === 'rsmt-bvh-integration.js' && !content.includes('class RSMTBVHIntegration')) {
                    throw new Error(`${file} missing RSMTBVHIntegration class`);
                }
                if (file === 'rsmt-validator.js' && !content.includes('class RSMTValidator')) {
                    throw new Error(`${file} missing RSMTValidator class`);
                }

                fileInfo[file] = checks;

            } catch (error) {
                throw new Error(`JavaScript file ${file}: ${error.message}`);
            }
        }

        return fileInfo;
    }

    /**
     * Test dependencies
     */
    async testDependencies() {
        const dependencies = {
            'ONNX Runtime Web': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js',
            'BVH Timeline': '../bvh-timeline.js'
        };

        const depInfo = {};

        for (const [name, path] of Object.entries(dependencies)) {
            if (path.startsWith('http')) {
                // External dependency - check if referenced correctly
                const demoContent = await fs.readFile('rsmt-demo.html', 'utf8');
                if (demoContent.includes(path)) {
                    depInfo[name] = { status: 'referenced', url: path };
                } else {
                    depInfo[name] = { status: 'missing_reference', url: path };
                }
            } else {
                // Local dependency - check if file exists
                try {
                    await fs.access(path);
                    depInfo[name] = { status: 'exists', path: path };
                } catch (error) {
                    depInfo[name] = { status: 'missing', path: path, error: error.message };
                }
            }
        }

        // Check for any missing critical dependencies
        const missing = Object.entries(depInfo).filter(([_, info]) => 
            info.status === 'missing' || info.status === 'missing_reference'
        );

        if (missing.length > 0) {
            const missingNames = missing.map(([name]) => name);
            console.warn(`⚠️  Missing dependencies: ${missingNames.join(', ')}`);
        }

        return depInfo;
    }

    /**
     * Test HTML demo file
     */
    async testHTMLDemo() {
        const htmlFile = 'rsmt-demo.html';
        
        try {
            const content = await fs.readFile(htmlFile, 'utf8');
            
            const checks = {
                hasHTMLStructure: content.includes('<!DOCTYPE html>'),
                hasTitle: content.includes('<title>'),
                hasCanvas: content.includes('<canvas'),
                hasScriptTags: content.includes('<script'),
                referencesONNXRuntime: content.includes('onnxruntime-web'),
                referencesRSMTFiles: content.includes('rsmt-inference.js') && 
                                  content.includes('rsmt-bvh-integration.js'),
                hasEventListeners: content.includes('addEventListener'),
                hasInitialization: content.includes('initialize'),
                lineCount: content.split('\n').length,
                size: content.length
            };

            // Check for critical missing elements
            const criticalChecks = [
                'hasHTMLStructure', 'hasScriptTags', 'referencesONNXRuntime', 'referencesRSMTFiles'
            ];

            const missingCritical = criticalChecks.filter(check => !checks[check]);
            if (missingCritical.length > 0) {
                throw new Error(`HTML demo missing critical elements: ${missingCritical.join(', ')}`);
            }

            return checks;

        } catch (error) {
            throw new Error(`HTML demo file: ${error.message}`);
        }
    }

    /**
     * Test integration points
     */
    async testIntegrationPoints() {
        const checks = {
            rsmt_to_bvh: false,
            bvh_to_timeline: false,
            timeline_playback: false,
            style_selection: false,
            performance_tracking: false
        };

        try {
            // Check RSMT-BVH integration
            const integrationContent = await fs.readFile('rsmt-bvh-integration.js', 'utf8');
            
            checks.rsmt_to_bvh = integrationContent.includes('addStylizedTransition') &&
                                integrationContent.includes('generateRSMTTransition');
            
            checks.bvh_to_timeline = integrationContent.includes('timeline.addClip') &&
                                   integrationContent.includes('BVHClip');

            checks.performance_tracking = integrationContent.includes('performanceStats') &&
                                        integrationContent.includes('getPerformanceStats');

            // Check demo integration
            const demoContent = await fs.readFile('rsmt-demo.html', 'utf8');
            
            checks.timeline_playback = demoContent.includes('playTimeline') &&
                                     demoContent.includes('pauseTimeline');
            
            checks.style_selection = demoContent.includes('style-btn') &&
                                   demoContent.includes('selectedStyle');

            // Verify all integration points are working
            const missingIntegrations = Object.entries(checks)
                .filter(([_, working]) => !working)
                .map(([name]) => name);

            if (missingIntegrations.length > 0) {
                console.warn(`⚠️  Missing integration points: ${missingIntegrations.join(', ')}`);
            }

            return checks;

        } catch (error) {
            throw new Error(`Integration points: ${error.message}`);
        }
    }

    /**
     * Print test summary
     */
    printSummary() {
        console.log('\n' + '='.repeat(50));
        console.log('📊 RSMT INTEGRATION TEST SUMMARY');
        console.log('='.repeat(50));
        console.log(`Total Tests: ${this.testResults.summary.total}`);
        console.log(`✅ Passed: ${this.testResults.summary.passed}`);
        console.log(`❌ Failed: ${this.testResults.summary.failed}`);
        
        if (this.testResults.summary.failed > 0) {
            console.log('\n🔍 Failed Tests:');
            this.testResults.summary.errors.forEach(error => {
                console.log(`   • ${error}`);
            });
        }

        const successRate = (this.testResults.summary.passed / this.testResults.summary.total) * 100;
        console.log(`\n📈 Success Rate: ${successRate.toFixed(1)}%`);

        if (successRate === 100) {
            console.log('\n🎉 All tests passed! RSMT integration is ready.');
        } else if (successRate >= 80) {
            console.log('\n⚠️  Most tests passed. Minor issues detected.');
        } else {
            console.log('\n🚨 Multiple test failures. Review required before deployment.');
        }

        console.log('='.repeat(50) + '\n');
    }

    /**
     * Save test report
     */
    async saveReport() {
        const reportFile = 'rsmt-test-report.json';
        try {
            await fs.writeFile(reportFile, JSON.stringify(this.testResults, null, 2));
            console.log(`📄 Test report saved to: ${reportFile}`);
        } catch (error) {
            console.error(`Failed to save test report: ${error.message}`);
        }
    }
}

// Run tests if this script is executed directly
if (require.main === module) {
    const tester = new RSMTIntegrationTest();
    
    // Change to the RSMT directory
    process.chdir(path.join(__dirname));
    
    tester.runTests()
        .then(results => {
            const exitCode = results.summary.failed > 0 ? 1 : 0;
            process.exit(exitCode);
        })
        .catch(error => {
            console.error('❌ Test execution failed:', error.message);
            process.exit(1);
        });
}

module.exports = RSMTIntegrationTest;
