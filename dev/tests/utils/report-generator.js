/**
 * Test Report Generator and Artifact Manager
 * 
 * Generates comprehensive test reports and manages artifacts for CI/CD
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class TestReportGenerator {
  constructor(options = {}) {
    this.projectRoot = options.projectRoot || path.resolve(__dirname, '../../..');
    this.reportsDir = path.join(this.projectRoot, 'dev/tests/reports');
    this.outputFormats = options.outputFormats || ['html', 'json', 'junit'];
    this.includeArtifacts = options.includeArtifacts !== false;
  }

  /**
   * Generate comprehensive test report
   */
  async generateReport(testResults) {
    const timestamp = new Date().toISOString();
    const reportData = {
      timestamp,
      summary: this.generateSummary(testResults),
      results: testResults,
      artifacts: this.includeArtifacts ? await this.collectArtifacts() : {},
      environment: this.getEnvironmentInfo(),
      performance: await this.getPerformanceMetrics()
    };

    // Generate reports in different formats
    const reports = {};
    
    if (this.outputFormats.includes('html')) {
      reports.html = await this.generateHTMLReport(reportData);
    }
    
    if (this.outputFormats.includes('json')) {
      reports.json = await this.generateJSONReport(reportData);
    }
    
    if (this.outputFormats.includes('junit')) {
      reports.junit = await this.generateJUnitReport(reportData);
    }

    // Generate summary for CI
    await this.generateCISummary(reportData);

    return reports;
  }

  /**
   * Generate test summary
   */
  generateSummary(testResults) {
    const summary = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      errors: 0,
      duration: 0,
      suites: {}
    };

    for (const suite of testResults.suites || []) {
      summary.total += suite.total_tests;
      summary.passed += suite.passed;
      summary.failed += suite.failed;
      summary.skipped += suite.skipped;
      summary.errors += suite.errors;
      summary.duration += suite.duration;
      
      summary.suites[suite.suite_name] = {
        status: suite.status,
        passed: suite.passed,
        failed: suite.failed,
        duration: suite.duration
      };
    }

    summary.passRate = summary.total > 0 ? (summary.passed / summary.total) * 100 : 0;
    summary.status = summary.failed === 0 && summary.errors === 0 ? 'passed' : 'failed';

    return summary;
  }

  /**
   * Generate HTML report
   */
  async generateHTMLReport(reportData) {
    const reportPath = path.join(this.reportsDir, 'test-report.html');
    
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motion Viewer Test Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .header h1 { margin: 0 0 10px 0; color: #333; }
        .header .meta { color: #666; font-size: 14px; }
        .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
        .status-passed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }
        .summary-card h3 { margin: 0 0 10px 0; font-size: 32px; font-weight: bold; }
        .summary-card p { margin: 0; color: #666; font-size: 14px; }
        .summary-card.passed h3 { color: #28a745; }
        .summary-card.failed h3 { color: #dc3545; }
        .summary-card.skipped h3 { color: #ffc107; }
        .summary-card.errors h3 { color: #6f42c1; }
        .suite-section { background: white; border-radius: 12px; margin-bottom: 20px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .suite-header { padding: 20px; background: #f8f9fa; border-bottom: 1px solid #dee2e6; }
        .suite-header h3 { margin: 0; }
        .suite-stats { font-size: 14px; color: #666; margin-top: 5px; }
        .test-list { padding: 0; margin: 0; list-style: none; }
        .test-item { padding: 15px 20px; border-bottom: 1px solid #dee2e6; display: flex; justify-content: space-between; align-items: center; }
        .test-item:last-child { border-bottom: none; }
        .test-name { font-weight: 500; }
        .test-duration { font-size: 12px; color: #666; }
        .test-passed { border-left: 4px solid #28a745; }
        .test-failed { border-left: 4px solid #dc3545; }
        .test-skipped { border-left: 4px solid #ffc107; }
        .artifacts-section { background: white; padding: 30px; border-radius: 12px; margin-top: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .artifacts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .artifact-card { border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }
        .artifact-card h4 { margin: 0 0 10px 0; }
        .artifact-link { color: #007bff; text-decoration: none; }
        .artifact-link:hover { text-decoration: underline; }
        .performance-section { background: white; padding: 30px; border-radius: 12px; margin-top: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .perf-metric { text-align: center; }
        .perf-metric .value { font-size: 24px; font-weight: bold; color: #007bff; }
        .perf-metric .label { font-size: 14px; color: #666; margin-top: 5px; }
        .environment-info { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px; }
        .environment-info h4 { margin: 0 0 15px 0; }
        .env-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .env-item { font-size: 14px; }
        .env-label { font-weight: bold; margin-right: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Motion Viewer Test Report</h1>
            <div class="meta">
                Generated: ${reportData.timestamp} • 
                <span class="status-badge ${reportData.summary.status === 'passed' ? 'status-passed' : 'status-failed'}">
                    ${reportData.summary.status}
                </span>
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>${reportData.summary.total}</h3>
                <p>Total Tests</p>
            </div>
            <div class="summary-card passed">
                <h3>${reportData.summary.passed}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card failed">
                <h3>${reportData.summary.failed}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card skipped">
                <h3>${reportData.summary.skipped}</h3>
                <p>Skipped</p>
            </div>
            <div class="summary-card">
                <h3>${reportData.summary.passRate.toFixed(1)}%</h3>
                <p>Pass Rate</p>
            </div>
            <div class="summary-card">
                <h3>${(reportData.summary.duration / 1000).toFixed(1)}s</h3>
                <p>Duration</p>
            </div>
        </div>

        ${this.generateSuiteSections(reportData.results.suites || [])}

        ${reportData.performance ? this.generatePerformanceSection(reportData.performance) : ''}

        ${this.includeArtifacts ? this.generateArtifactsSection(reportData.artifacts) : ''}

        <div class="environment-info">
            <h4>Environment Information</h4>
            <div class="env-grid">
                <div class="env-item">
                    <span class="env-label">Node.js:</span>
                    ${reportData.environment.node}
                </div>
                <div class="env-item">
                    <span class="env-label">OS:</span>
                    ${reportData.environment.os}
                </div>
                <div class="env-item">
                    <span class="env-label">Browser:</span>
                    ${reportData.environment.browser || 'N/A'}
                </div>
                <div class="env-item">
                    <span class="env-label">CI:</span>
                    ${reportData.environment.ci ? 'Yes' : 'No'}
                </div>
            </div>
        </div>
    </div>
</body>
</html>`;

    await fs.writeFile(reportPath, html);
    return reportPath;
  }

  /**
   * Generate suite sections HTML
   */
  generateSuiteSections(suites) {
    return suites.map(suite => `
        <div class="suite-section">
            <div class="suite-header">
                <h3>${suite.suite_name}</h3>
                <div class="suite-stats">
                    ${suite.passed} passed, ${suite.failed} failed, ${suite.skipped} skipped • 
                    ${(suite.duration / 1000).toFixed(2)}s
                </div>
            </div>
            <ul class="test-list">
                ${(suite.tests || []).map(test => `
                    <li class="test-item test-${test.status}">
                        <div>
                            <div class="test-name">${test.name}</div>
                            ${test.message ? `<div style="font-size: 12px; color: #dc3545; margin-top: 4px;">${test.message}</div>` : ''}
                        </div>
                        <div class="test-duration">${(test.duration * 1000).toFixed(0)}ms</div>
                    </li>
                `).join('')}
            </ul>
        </div>
    `).join('');
  }

  /**
   * Generate performance section HTML
   */
  generatePerformanceSection(performance) {
    return `
        <div class="performance-section">
            <h4>Performance Metrics</h4>
            <div class="perf-grid">
                <div class="perf-metric">
                    <div class="value">${performance.fps?.average || 'N/A'}</div>
                    <div class="label">Average FPS</div>
                </div>
                <div class="perf-metric">
                    <div class="value">${performance.memory?.peak || 'N/A'}MB</div>
                    <div class="label">Peak Memory</div>
                </div>
                <div class="perf-metric">
                    <div class="value">${performance.loadTime || 'N/A'}ms</div>
                    <div class="label">Load Time</div>
                </div>
                <div class="perf-metric">
                    <div class="value">${performance.lighthouse?.performance || 'N/A'}</div>
                    <div class="label">Lighthouse Score</div>
                </div>
            </div>
        </div>
    `;
  }

  /**
   * Generate artifacts section HTML
   */
  generateArtifactsSection(artifacts) {
    return `
        <div class="artifacts-section">
            <h4>Test Artifacts</h4>
            <div class="artifacts-grid">
                ${Object.entries(artifacts).map(([type, files]) => `
                    <div class="artifact-card">
                        <h4>${type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                        ${files.map(file => `
                            <div><a href="${file.path}" class="artifact-link">${file.name}</a></div>
                        `).join('')}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
  }

  /**
   * Generate JSON report
   */
  async generateJSONReport(reportData) {
    const reportPath = path.join(this.reportsDir, 'test-report.json');
    await fs.writeFile(reportPath, JSON.stringify(reportData, null, 2));
    return reportPath;
  }

  /**
   * Generate JUnit XML report
   */
  async generateJUnitReport(reportData) {
    const reportPath = path.join(this.reportsDir, 'junit-report.xml');
    
    const xml = `<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Motion Viewer Tests" tests="${reportData.summary.total}" failures="${reportData.summary.failed}" errors="${reportData.summary.errors}" time="${reportData.summary.duration / 1000}">
${(reportData.results.suites || []).map(suite => `
    <testsuite name="${suite.suite_name}" tests="${suite.total_tests}" failures="${suite.failed}" errors="${suite.errors}" time="${suite.duration / 1000}">
        ${(suite.tests || []).map(test => `
        <testcase name="${test.name}" time="${test.duration}">
            ${test.status === 'failed' ? `<failure message="${test.message || 'Test failed'}">${test.message || ''}</failure>` : ''}
            ${test.status === 'skipped' ? '<skipped/>' : ''}
        </testcase>`).join('')}
    </testsuite>`).join('')}
</testsuites>`;

    await fs.writeFile(reportPath, xml);
    return reportPath;
  }

  /**
   * Generate CI summary
   */
  async generateCISummary(reportData) {
    const summary = {
      status: reportData.summary.status,
      total: reportData.summary.total,
      passed: reportData.summary.passed,
      failed: reportData.summary.failed,
      passRate: reportData.summary.passRate,
      duration: reportData.summary.duration,
      suites: Object.keys(reportData.summary.suites).length,
      timestamp: reportData.timestamp
    };

    const summaryPath = path.join(this.reportsDir, 'ci-summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2));

    // Generate GitHub Actions summary
    if (process.env.GITHUB_ACTIONS) {
      await this.generateGitHubSummary(reportData);
    }

    return summaryPath;
  }

  /**
   * Generate GitHub Actions summary
   */
  async generateGitHubSummary(reportData) {
    const summary = `# 🎭 Motion Viewer Test Results

## Summary
- **Status**: ${reportData.summary.status === 'passed' ? '✅ PASSED' : '❌ FAILED'}
- **Tests**: ${reportData.summary.passed}/${reportData.summary.total} passed (${reportData.summary.passRate.toFixed(1)}%)
- **Duration**: ${(reportData.summary.duration / 1000).toFixed(1)}s

## Test Suites
${Object.entries(reportData.summary.suites).map(([name, suite]) => 
  `- **${name}**: ${suite.status === 'passed' ? '✅' : '❌'} ${suite.passed}/${suite.passed + suite.failed} passed`
).join('\n')}

## Performance
${reportData.performance ? `
- **Average FPS**: ${reportData.performance.fps?.average || 'N/A'}
- **Peak Memory**: ${reportData.performance.memory?.peak || 'N/A'}MB
- **Load Time**: ${reportData.performance.loadTime || 'N/A'}ms
` : 'No performance data available'}

---
Generated: ${reportData.timestamp}`;

    if (process.env.GITHUB_STEP_SUMMARY) {
      await fs.appendFile(process.env.GITHUB_STEP_SUMMARY, summary);
    }
  }

  /**
   * Collect test artifacts
   */
  async collectArtifacts() {
    const artifacts = {};
    const artifactDirs = {
      screenshots: 'screenshots',
      videos: 'videos',
      traces: 'traces',
      logs: 'logs'
    };

    for (const [type, dir] of Object.entries(artifactDirs)) {
      const fullPath = path.join(this.reportsDir, '..', dir);
      try {
        const files = await fs.readdir(fullPath);
        artifacts[type] = files.map(file => ({
          name: file,
          path: path.join(dir, file),
          size: 0 // Could be populated with actual file size
        }));
      } catch {
        artifacts[type] = [];
      }
    }

    return artifacts;
  }

  /**
   * Get environment information
   */
  getEnvironmentInfo() {
    return {
      node: process.version,
      os: `${process.platform} ${process.arch}`,
      ci: process.env.CI === 'true',
      browser: process.env.BROWSER_NAME || null,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get performance metrics
   */
  async getPerformanceMetrics() {
    try {
      const perfFile = path.join(this.reportsDir, 'performance', 'metrics.json');
      const perfData = await fs.readFile(perfFile, 'utf8');
      return JSON.parse(perfData);
    } catch {
      return null;
    }
  }
}

export default TestReportGenerator;
