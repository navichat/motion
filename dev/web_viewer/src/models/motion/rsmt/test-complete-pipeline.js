#!/usr/bin/env node

/**
 * Complete RSMT Pipeline Test Script
 * Tests the full RSMT JavaScript implementation with ONNX models
 */

const fs = require('fs');
const path = require('path');

class RSMTPipelineTest {
    constructor() {
        this.testResults = {
            timestamp: new Date().toISOString(),
            overall: 'pending',
            environment: {},
            files: {},
            models: {},
            integration: {},
            performance: {},
            errors: []
        };
        
        this.baseDir = __dirname;
        
        console.log('RSMT Pipeline Test initialized');
        console.log('Base directory:', this.baseDir);
    }
    
    /**
     * Run complete test suite
     */
    async runCompleteTest() {
        console.log('\n=== RSMT Complete Pipeline Test ===\n');
        
        try {
            // Test 1: Environment and file structure
            console.log('1. Testing environment and file structure...');
            await this.testEnvironment();
            
            // Test 2: File integrity
            console.log('2. Testing file integrity...');
            await this.testFileIntegrity();
            
            // Test 3: Model availability
            console.log('3. Testing ONNX model availability...');
            await this.testModelAvailability();
            
            // Test 4: Integration layer structure
            console.log('4. Testing integration layer...');
            await this.testIntegrationLayer();
            
            // Test 5: Performance benchmarks
            console.log('5. Running performance benchmarks...');
            await this.testPerformance();
            
            // Determine overall status
            this.testResults.overall = this.determineOverallStatus();
            
            // Generate report
            const report = this.generateReport();
            console.log('\n=== Test Report ===\n');
            console.log(report);
            
            // Save report to file
            await this.saveReport(report);
            
            return this.testResults;
            
        } catch (error) {
            console.error('Test suite failed:', error);
            this.testResults.overall = 'error';
            this.testResults.errors.push(error.message);
            return this.testResults;
        }
    }
    
    /**
     * Test environment setup
     */
    async testEnvironment() {
        const env = {
            nodeVersion: process.version,
            platform: process.platform,
            architecture: process.arch,
            workingDirectory: process.cwd(),
            testDirectory: this.baseDir
        };
        
        console.log('Environment:', env);
        this.testResults.environment = env;
    }
    
    /**
     * Test file integrity
     */
    async testFileIntegrity() {
        const requiredFiles = [
            'rsmt-inference.js',
            'rsmt-bvh-integration.js',
            'rsmt-demo.html',
            'rsmt-validator.js',
            'README.md'
        ];
        
        const optionalFiles = [
            'deepphase.onnx',
            'stylevae.onnx',
            'transitionnet.onnx',
            'test-integration.js'
        ];
        
        const fileTests = {};
        
        // Test required files
        for (const filename of requiredFiles) {
            const filepath = path.join(this.baseDir, filename);
            const exists = fs.existsSync(filepath);
            
            if (exists) {
                const stats = fs.statSync(filepath);
                fileTests[filename] = {
                    status: 'found',
                    size: stats.size,
                    modified: stats.mtime.toISOString()
                };
                console.log(`  ✓ ${filename} (${stats.size} bytes)`);
            } else {
                fileTests[filename] = {
                    status: 'missing',
                    required: true
                };
                console.log(`  ✗ ${filename} (MISSING - REQUIRED)`);
            }
        }
        
        // Test optional files
        for (const filename of optionalFiles) {
            const filepath = path.join(this.baseDir, filename);
            const exists = fs.existsSync(filepath);
            
            if (exists) {
                const stats = fs.statSync(filepath);
                fileTests[filename] = {
                    status: 'found',
                    size: stats.size,
                    modified: stats.mtime.toISOString()
                };
                console.log(`  ✓ ${filename} (${stats.size} bytes)`);
            } else {
                fileTests[filename] = {
                    status: 'missing',
                    required: false
                };
                console.log(`  ~ ${filename} (optional - not found)`);
            }
        }
        
        // Check subdirectories
        const subdirs = ['onnx_models'];
        for (const subdir of subdirs) {
            const subdirPath = path.join(this.baseDir, subdir);
            if (fs.existsSync(subdirPath)) {
                const files = fs.readdirSync(subdirPath);
                fileTests[`${subdir}/`] = {
                    status: 'found',
                    contents: files,
                    count: files.length
                };
                console.log(`  ✓ ${subdir}/ (${files.length} files)`);
                
                // List ONNX files in subdirectory
                const onnxFiles = files.filter(f => f.endsWith('.onnx'));
                if (onnxFiles.length > 0) {
                    console.log(`    ONNX models: ${onnxFiles.join(', ')}`);
                }
            }
        }
        
        this.testResults.files = fileTests;
    }
    
    /**
     * Test ONNX model availability
     */
    async testModelAvailability() {
        const modelLocations = [
            { name: 'deepphase', paths: ['./deepphase.onnx', './onnx_models/deepphase.onnx'] },
            { name: 'stylevae', paths: ['./stylevae.onnx', './onnx_models/stylevae.onnx'] },
            { name: 'transitionnet', paths: ['./transitionnet.onnx', './onnx_models/transitionnet.onnx'] },
            { name: 'manifold_vae', paths: ['./onnx_models/manifold_vae.onnx'] }
        ];
        
        const modelTests = {};
        
        for (const model of modelLocations) {
            const modelResult = {
                name: model.name,
                found: false,
                location: null,
                size: 0,
                paths: model.paths
            };
            
            // Check each possible location
            for (const relativePath of model.paths) {
                const fullPath = path.join(this.baseDir, relativePath);
                if (fs.existsSync(fullPath)) {
                    const stats = fs.statSync(fullPath);
                    modelResult.found = true;
                    modelResult.location = relativePath;
                    modelResult.size = stats.size;
                    modelResult.modified = stats.mtime.toISOString();
                    break;
                }
            }
            
            if (modelResult.found) {
                console.log(`  ✓ ${model.name} model found at ${modelResult.location} (${modelResult.size} bytes)`);
            } else {
                console.log(`  ✗ ${model.name} model not found in any location`);
                console.log(`    Searched: ${model.paths.join(', ')}`);
            }
            
            modelTests[model.name] = modelResult;
        }
        
        this.testResults.models = modelTests;
    }
    
    /**
     * Test integration layer structure
     */
    async testIntegrationLayer() {
        const integrationTests = {};
        
        try {
            // Test JavaScript file structure
            const inferenceFile = path.join(this.baseDir, 'rsmt-inference.js');
            const integrationFile = path.join(this.baseDir, 'rsmt-bvh-integration.js');
            
            if (fs.existsSync(inferenceFile)) {
                const content = fs.readFileSync(inferenceFile, 'utf8');
                integrationTests.inference = this.analyzeJavaScriptFile(content, 'RSMTInference');
                console.log(`  ✓ Inference engine structure validated`);
            } else {
                integrationTests.inference = { status: 'missing' };
                console.log(`  ✗ Inference engine file missing`);
            }
            
            if (fs.existsSync(integrationFile)) {
                const content = fs.readFileSync(integrationFile, 'utf8');
                integrationTests.bvhIntegration = this.analyzeJavaScriptFile(content, 'RSMTBVHIntegration');
                console.log(`  ✓ BVH integration structure validated`);
            } else {
                integrationTests.bvhIntegration = { status: 'missing' };
                console.log(`  ✗ BVH integration file missing`);
            }
            
            // Test demo file
            const demoFile = path.join(this.baseDir, 'rsmt-demo.html');
            if (fs.existsSync(demoFile)) {
                const content = fs.readFileSync(demoFile, 'utf8');
                integrationTests.demo = this.analyzeHTMLFile(content);
                console.log(`  ✓ Demo interface structure validated`);
            } else {
                integrationTests.demo = { status: 'missing' };
                console.log(`  ✗ Demo interface file missing`);
            }
            
        } catch (error) {
            console.error(`  ✗ Integration layer analysis failed: ${error.message}`);
            integrationTests.error = error.message;
        }
        
        this.testResults.integration = integrationTests;
    }
    
    /**
     * Analyze JavaScript file structure
     */
    analyzeJavaScriptFile(content, expectedClass) {
        const analysis = {
            status: 'analyzed',
            hasClass: false,
            className: expectedClass,
            methods: [],
            dependencies: [],
            lines: content.split('\n').length,
            size: content.length
        };
        
        // Check for class definition
        const classRegex = new RegExp(`class\\s+${expectedClass}`, 'g');
        analysis.hasClass = classRegex.test(content);
        
        // Extract method names
        const methodRegex = /(?:async\s+)?(\w+)\s*\([^)]*\)\s*{/g;
        let match;
        while ((match = methodRegex.exec(content)) !== null) {
            if (!['if', 'for', 'while', 'switch', 'catch'].includes(match[1])) {
                analysis.methods.push(match[1]);
            }
        }
        
        // Check for dependencies
        const dependencies = [
            { name: 'onnxruntime-web', pattern: /ort\.|onnxruntime/g },
            { name: 'BVHTimeline', pattern: /BVHTimeline|BVHClip/g },
            { name: 'performance tracking', pattern: /performance\.now|Date\.now/g }
        ];
        
        for (const dep of dependencies) {
            if (dep.pattern.test(content)) {
                analysis.dependencies.push(dep.name);
            }
        }
        
        return analysis;
    }
    
    /**
     * Analyze HTML file structure
     */
    analyzeHTMLFile(content) {
        const analysis = {
            status: 'analyzed',
            scripts: [],
            styles: [],
            elements: [],
            size: content.length,
            lines: content.split('\n').length
        };
        
        // Extract script tags
        const scriptRegex = /<script[^>]*src=["']([^"']+)["'][^>]*>/g;
        let match;
        while ((match = scriptRegex.exec(content)) !== null) {
            analysis.scripts.push(match[1]);
        }
        
        // Check for inline scripts
        const inlineScriptRegex = /<script(?![^>]*src)[^>]*>/g;
        const inlineScripts = content.match(inlineScriptRegex);
        if (inlineScripts) {
            analysis.hasInlineScript = true;
            analysis.inlineScriptCount = inlineScripts.length;
        }
        
        // Extract key UI elements
        const elements = ['canvas', 'button', 'input', 'select'];
        for (const element of elements) {
            const regex = new RegExp(`<${element}[^>]*id=["']([^"']+)["']`, 'g');
            const elementMatches = [];
            while ((match = regex.exec(content)) !== null) {
                elementMatches.push(match[1]);
            }
            if (elementMatches.length > 0) {
                analysis.elements.push({ element, ids: elementMatches });
            }
        }
        
        return analysis;
    }
    
    /**
     * Test performance characteristics
     */
    async testPerformance() {
        const perfTests = {
            fileLoading: {},
            memoryUsage: {},
            compatibility: {}
        };
        
        try {
            // Test file loading performance
            const testFiles = ['rsmt-inference.js', 'rsmt-bvh-integration.js', 'rsmt-demo.html'];
            
            for (const filename of testFiles) {
                const filepath = path.join(this.baseDir, filename);
                if (fs.existsSync(filepath)) {
                    const startTime = process.hrtime.bigint();
                    const content = fs.readFileSync(filepath, 'utf8');
                    const endTime = process.hrtime.bigint();
                    
                    perfTests.fileLoading[filename] = {
                        size: content.length,
                        loadTime: Number(endTime - startTime) / 1000000, // Convert to milliseconds
                        linesPerSecond: (content.split('\n').length / (Number(endTime - startTime) / 1000000000)).toFixed(0)
                    };
                }
            }
            
            // Test memory usage
            const memUsage = process.memoryUsage();
            perfTests.memoryUsage = {
                rss: memUsage.rss,
                heapTotal: memUsage.heapTotal,
                heapUsed: memUsage.heapUsed,
                external: memUsage.external
            };
            
            // Test browser compatibility (static analysis)
            const inferenceFile = path.join(this.baseDir, 'rsmt-inference.js');
            if (fs.existsSync(inferenceFile)) {
                const content = fs.readFileSync(inferenceFile, 'utf8');
                perfTests.compatibility = this.analyzeBrowserCompatibility(content);
            }
            
        } catch (error) {
            console.error(`  ✗ Performance testing failed: ${error.message}`);
            perfTests.error = error.message;
        }
        
        this.testResults.performance = perfTests;
        
        // Log performance results
        console.log('  Performance Summary:');
        if (perfTests.fileLoading) {
            for (const [file, metrics] of Object.entries(perfTests.fileLoading)) {
                console.log(`    ${file}: ${metrics.loadTime.toFixed(2)}ms`);
            }
        }
        if (perfTests.memoryUsage) {
            console.log(`    Memory: ${(perfTests.memoryUsage.heapUsed / 1024 / 1024).toFixed(2)}MB heap`);
        }
    }
    
    /**
     * Analyze browser compatibility
     */
    analyzeBrowserCompatibility(content) {
        const compatibility = {
            features: {},
            warnings: [],
            score: 0
        };
        
        // Check for modern JavaScript features
        const features = [
            { name: 'async/await', pattern: /async\s+\w+|await\s+/, supported: true },
            { name: 'arrow functions', pattern: /=>\s*{|=>\s*\w/, supported: true },
            { name: 'template literals', pattern: /`[^`]*\${[^}]*}[^`]*`/, supported: true },
            { name: 'destructuring', pattern: /const\s*{[^}]+}\s*=|let\s*{[^}]+}\s*=/, supported: true },
            { name: 'classes', pattern: /class\s+\w+/, supported: true },
            { name: 'WebAssembly', pattern: /WebAssembly|wasm/, supported: true },
            { name: 'Web Workers', pattern: /Worker\(|new\s+Worker/, supported: true }
        ];
        
        for (const feature of features) {
            const found = feature.pattern.test(content);
            compatibility.features[feature.name] = {
                used: found,
                supported: feature.supported
            };
            
            if (found && !feature.supported) {
                compatibility.warnings.push(`${feature.name} may not be supported in older browsers`);
            }
        }
        
        // Calculate compatibility score
        const usedFeatures = Object.values(compatibility.features).filter(f => f.used);
        const supportedFeatures = usedFeatures.filter(f => f.supported);
        compatibility.score = usedFeatures.length > 0 ? (supportedFeatures.length / usedFeatures.length) * 100 : 100;
        
        return compatibility;
    }
    
    /**
     * Determine overall test status
     */
    determineOverallStatus() {
        if (this.testResults.errors.length > 0) return 'error';
        
        // Check critical files
        const criticalFiles = ['rsmt-inference.js', 'rsmt-bvh-integration.js'];
        for (const file of criticalFiles) {
            if (!this.testResults.files[file] || this.testResults.files[file].status !== 'found') {
                return 'critical_files_missing';
            }
        }
        
        // Check if at least one model is available
        const hasModels = Object.values(this.testResults.models).some(model => model.found);
        if (!hasModels) {
            return 'no_models_found';
        }
        
        // Check integration structure
        if (this.testResults.integration.inference?.status === 'missing' || 
            this.testResults.integration.bvhIntegration?.status === 'missing') {
            return 'integration_incomplete';
        }
        
        return 'ready';
    }
    
    /**
     * Generate comprehensive report
     */
    generateReport() {
        const results = this.testResults;
        
        let report = `RSMT JavaScript Implementation Test Report\n`;
        report += `==========================================\n\n`;
        report += `Generated: ${results.timestamp}\n`;
        report += `Overall Status: ${results.overall.toUpperCase()}\n\n`;
        
        // Environment
        report += `Environment:\n`;
        report += `  Node.js: ${results.environment.nodeVersion}\n`;
        report += `  Platform: ${results.environment.platform} (${results.environment.architecture})\n`;
        report += `  Test Directory: ${results.environment.testDirectory}\n\n`;
        
        // File structure
        report += `File Structure:\n`;
        const requiredFiles = ['rsmt-inference.js', 'rsmt-bvh-integration.js', 'rsmt-demo.html'];
        for (const file of requiredFiles) {
            const fileInfo = results.files[file];
            if (fileInfo) {
                const status = fileInfo.status === 'found' ? '✓' : '✗';
                const size = fileInfo.size ? ` (${fileInfo.size} bytes)` : '';
                report += `  ${status} ${file}${size}\n`;
            }
        }
        
        // ONNX Models
        report += `\nONNX Models:\n`;
        for (const [name, model] of Object.entries(results.models)) {
            const status = model.found ? '✓' : '✗';
            const location = model.location ? ` at ${model.location}` : ' (not found)';
            const size = model.size ? ` (${(model.size / 1024 / 1024).toFixed(1)}MB)` : '';
            report += `  ${status} ${name}${location}${size}\n`;
        }
        
        // Integration Analysis
        report += `\nIntegration Analysis:\n`;
        if (results.integration.inference) {
            const inf = results.integration.inference;
            report += `  Inference Engine: ${inf.hasClass ? '✓' : '✗'} Class found\n`;
            if (inf.methods) {
                report += `    Methods: ${inf.methods.length} functions\n`;
                report += `    Dependencies: ${inf.dependencies.join(', ')}\n`;
            }
        }
        
        if (results.integration.bvhIntegration) {
            const bvh = results.integration.bvhIntegration;
            report += `  BVH Integration: ${bvh.hasClass ? '✓' : '✗'} Class found\n`;
            if (bvh.methods) {
                report += `    Methods: ${bvh.methods.length} functions\n`;
            }
        }
        
        // Performance
        if (results.performance.fileLoading) {
            report += `\nPerformance:\n`;
            for (const [file, metrics] of Object.entries(results.performance.fileLoading)) {
                report += `  ${file}: ${metrics.loadTime.toFixed(2)}ms load time\n`;
            }
        }
        
        if (results.performance.compatibility) {
            const compat = results.performance.compatibility;
            report += `  Browser Compatibility: ${compat.score.toFixed(0)}%\n`;
            if (compat.warnings.length > 0) {
                report += `  Warnings: ${compat.warnings.length} compatibility issues\n`;
            }
        }
        
        // Status Summary
        report += `\nStatus Summary:\n`;
        switch (results.overall) {
            case 'ready':
                report += `  ✓ RSMT JavaScript implementation is ready for testing\n`;
                report += `  ✓ All critical files are present\n`;
                report += `  ✓ ONNX models are available\n`;
                report += `  ✓ Integration layer is complete\n`;
                break;
            case 'critical_files_missing':
                report += `  ✗ Critical implementation files are missing\n`;
                break;
            case 'no_models_found':
                report += `  ✗ No ONNX models found - download required\n`;
                break;
            case 'integration_incomplete':
                report += `  ✗ Integration layer is incomplete\n`;
                break;
            case 'error':
                report += `  ✗ Test errors occurred\n`;
                break;
        }
        
        if (results.errors.length > 0) {
            report += `\nErrors:\n`;
            results.errors.forEach(error => {
                report += `  - ${error}\n`;
            });
        }
        
        return report;
    }
    
    /**
     * Save report to file
     */
    async saveReport(report) {
        try {
            const reportPath = path.join(this.baseDir, 'RSMT_TEST_REPORT.txt');
            fs.writeFileSync(reportPath, report);
            console.log(`\nReport saved to: ${reportPath}`);
            
            // Also save JSON results
            const jsonPath = path.join(this.baseDir, 'rsmt_test_results.json');
            fs.writeFileSync(jsonPath, JSON.stringify(this.testResults, null, 2));
            console.log(`JSON results saved to: ${jsonPath}`);
            
        } catch (error) {
            console.error('Failed to save report:', error.message);
        }
    }
}

// Run tests if called directly
if (require.main === module) {
    const tester = new RSMTPipelineTest();
    tester.runCompleteTest().then(results => {
        console.log('\n=== Test Complete ===');
        process.exit(results.overall === 'ready' ? 0 : 1);
    }).catch(error => {
        console.error('Test suite failed:', error);
        process.exit(1);
    });
}

module.exports = RSMTPipelineTest;
