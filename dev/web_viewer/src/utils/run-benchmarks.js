#!/usr/bin/env node

/**
 * TaskManager Performance Benchmark Runner
 * Runs comprehensive performance tests and generates optimization reports
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class BenchmarkRunner {
    constructor() {
        this.resultsDir = './test-results';
        this.reportDir = './performance-reports';
        this.startTime = Date.now();
        
        // Ensure directories exist
        if (!fs.existsSync(this.resultsDir)) {
            fs.mkdirSync(this.resultsDir, { recursive: true });
        }
        if (!fs.existsSync(this.reportDir)) {
            fs.mkdirSync(this.reportDir, { recursive: true });
        }
    }

    log(message, level = 'INFO') {
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp}] ${level}: ${message}`);
    }

    async runBenchmarks() {
        this.log('Starting TaskManager Performance Benchmark Suite...');
        
        try {
            // Check if Playwright is installed
            this.log('Checking Playwright installation...');
            execSync('npx playwright --version', { stdio: 'pipe' });
            
            // Install browsers if needed
            this.log('Installing Playwright browsers...');
            execSync('npx playwright install', { stdio: 'inherit' });
            
            // Run the performance tests
            this.log('Executing performance tests...');
            const testResult = execSync('npx playwright test tests/taskmanager.test.js --reporter=json', {
                stdio: 'pipe',
                encoding: 'utf8'
            });
            
            // Parse results
            const results = JSON.parse(testResult);
            this.generateReport(results);
            
        } catch (error) {
            this.log(`Benchmark failed: ${error.message}`, 'ERROR');
            if (error.stdout) {
                this.log(`STDOUT: ${error.stdout}`, 'DEBUG');
            }
            if (error.stderr) {
                this.log(`STDERR: ${error.stderr}`, 'DEBUG');
            }
            process.exit(1);
        }
    }

    generateReport(results) {
        const reportTimestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const reportFile = path.join(this.reportDir, `performance-report-${reportTimestamp}.json`);
        const htmlReportFile = path.join(this.reportDir, `performance-report-${reportTimestamp}.html`);
        
        // Extract performance metrics from test results
        const performanceMetrics = this.extractPerformanceMetrics(results);
        
        // Generate JSON report
        const report = {
            timestamp: new Date().toISOString(),
            duration: Date.now() - this.startTime,
            summary: {
                totalTests: results.stats?.total || 0,
                passed: results.stats?.passed || 0,
                failed: results.stats?.failed || 0,
                flaky: results.stats?.flaky || 0
            },
            performanceMetrics,
            recommendations: this.generateRecommendations(performanceMetrics),
            testResults: results
        };
        
        fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
        this.log(`JSON report generated: ${reportFile}`);
        
        // Generate HTML report
        this.generateHTMLReport(report, htmlReportFile);
        this.log(`HTML report generated: ${htmlReportFile}`);
        
        // Display summary
        this.displaySummary(report);
    }

    extractPerformanceMetrics(results) {
        const metrics = {
            flops: { tests: [], avg: 0, max: 0, min: Infinity },
            throughput: { tests: [], avg: 0, max: 0, min: Infinity },
            memoryUsage: { tests: [], avg: 0, max: 0, min: Infinity },
            latency: { tests: [], avg: 0, max: 0, min: Infinity }
        };

        // This would be populated from actual test output parsing
        // For now, we'll include placeholder structure
        return metrics;
    }

    generateRecommendations(metrics) {
        const recommendations = [];
        
        if (metrics.flops.avg < 1000000) {
            recommendations.push({
                type: 'performance',
                severity: 'medium',
                message: 'FLOPS performance below optimal threshold. Consider optimizing computational kernels.',
                suggestion: 'Increase worker complexity or optimize mathematical operations in job execution.'
            });
        }
        
        if (metrics.throughput.avg < 10) {
            recommendations.push({
                type: 'scalability',
                severity: 'high',
                message: 'Task throughput below optimal. Worker pool may be underutilized.',
                suggestion: 'Increase worker pool size or reduce task execution overhead.'
            });
        }
        
        if (metrics.memoryUsage.max > 500) {
            recommendations.push({
                type: 'memory',
                severity: 'medium',
                message: 'Peak memory usage exceeds recommended limits.',
                suggestion: 'Implement memory pooling or reduce task memory footprint.'
            });
        }
        
        return recommendations;
    }

    generateHTMLReport(report, filePath) {
        const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TaskManager Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }
        .metric-title { font-weight: bold; color: #495057; margin-bottom: 5px; }
        .metric-value { font-size: 1.5em; color: #007bff; font-weight: bold; }
        .recommendations { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 6px; margin: 20px 0; }
        .recommendation { margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }
        .severity-high { border-left: 4px solid #dc3545; }
        .severity-medium { border-left: 4px solid #ffc107; }
        .severity-low { border-left: 4px solid #28a745; }
        .test-summary { background: #e9ecef; padding: 15px; border-radius: 6px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TaskManager Performance Report</h1>
            <p>Generated: ${report.timestamp}</p>
            <p>Duration: ${(report.duration / 1000).toFixed(2)} seconds</p>
        </div>
        
        <div class="test-summary">
            <h2>Test Summary</h2>
            <p><strong>Total Tests:</strong> ${report.summary.totalTests}</p>
            <p><strong>Passed:</strong> ${report.summary.passed}</p>
            <p><strong>Failed:</strong> ${report.summary.failed}</p>
            <p><strong>Flaky:</strong> ${report.summary.flaky}</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">Avg FLOPS</div>
                <div class="metric-value">${this.formatNumber(report.performanceMetrics.flops?.avg || 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Avg Throughput</div>
                <div class="metric-value">${(report.performanceMetrics.throughput?.avg || 0).toFixed(2)} tasks/s</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Peak Memory</div>
                <div class="metric-value">${(report.performanceMetrics.memoryUsage?.max || 0).toFixed(1)} MB</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Avg Latency</div>
                <div class="metric-value">${(report.performanceMetrics.latency?.avg || 0).toFixed(1)} ms</div>
            </div>
        </div>
        
        <div class="recommendations">
            <h2>Performance Recommendations</h2>
            ${report.recommendations.map(rec => `
                <div class="recommendation severity-${rec.severity}">
                    <h4>${rec.type.toUpperCase()}: ${rec.message}</h4>
                    <p><strong>Suggestion:</strong> ${rec.suggestion}</p>
                </div>
            `).join('')}
        </div>
    </div>
</body>
</html>`;
        
        fs.writeFileSync(filePath, html);
    }

    formatNumber(num) {
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'G';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toFixed(0);
    }

    displaySummary(report) {
        console.log('\n' + '='.repeat(60));
        console.log('  TASKMANAGER PERFORMANCE BENCHMARK SUMMARY');
        console.log('='.repeat(60));
        console.log(`Test Results: ${report.summary.passed}/${report.summary.totalTests} passed`);
        console.log(`Duration: ${(report.duration / 1000).toFixed(2)} seconds`);
        console.log(`Average FLOPS: ${this.formatNumber(report.performanceMetrics.flops?.avg || 0)}`);
        console.log(`Average Throughput: ${(report.performanceMetrics.throughput?.avg || 0).toFixed(2)} tasks/s`);
        console.log(`Peak Memory Usage: ${(report.performanceMetrics.memoryUsage?.max || 0).toFixed(1)} MB`);
        
        if (report.recommendations.length > 0) {
            console.log(`\nRecommendations: ${report.recommendations.length} optimization opportunities found`);
            report.recommendations.forEach(rec => {
                console.log(`  • ${rec.type.toUpperCase()}: ${rec.message}`);
            });
        } else {
            console.log('\n✅ No performance issues detected');
        }
        
        console.log('='.repeat(60));
    }
}

// CLI interface
if (require.main === module) {
    const runner = new BenchmarkRunner();
    
    process.on('SIGINT', () => {
        runner.log('Benchmark interrupted by user', 'WARN');
        process.exit(1);
    });
    
    runner.runBenchmarks().catch(error => {
        runner.log(`Unexpected error: ${error.message}`, 'ERROR');
        process.exit(1);
    });
}

module.exports = BenchmarkRunner;
