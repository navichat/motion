#!/usr/bin/env node

/**
 * Motion Viewer Comprehensive Test Runner
 * 
 * Orchestrates all testing phases with detailed reporting and CI/CD integration
 */

import { program } from 'commander';
import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import TestEnvironment from './setup/global-setup.js';
import TestReportGenerator from './utils/report-generator.js';
import VisualAnalyzer from './utils/visual-analyzer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MotionTestRunner {
  constructor(options = {}) {
    this.options = {
      verbose: false,
      parallel: true,
      generateReport: true,
      updateBaseline: false,
      skipSetup: false,
      outputFormats: ['html', 'json'],
      ...options
    };

    this.testEnvironment = new TestEnvironment();
    this.reportGenerator = new TestReportGenerator({
      outputFormats: this.options.outputFormats
    });
    this.visualAnalyzer = new VisualAnalyzer();
    
    this.results = {
      timestamp: new Date().toISOString(),
      suites: [],
      overall_status: 'pending',
      total_duration: 0
    };
  }

  /**
   * Run all tests
   */
  async runAll() {
    console.log('🎭 Motion Viewer Comprehensive Test Suite');
    console.log('==========================================');

    const startTime = Date.now();

    try {
      // Setup test environment
      if (!this.options.skipSetup) {
        await this.testEnvironment.globalSetup();
      }

      // Initialize visual analyzer
      await this.visualAnalyzer.initialize();

      // Run test suites
      const suites = await this.getTestSuites();
      
      if (this.options.parallel && suites.length > 1) {
        await this.runSuitesParallel(suites);
      } else {
        await this.runSuitesSequential(suites);
      }

      // Run visual analysis
      if (suites.some(s => s.name === 'visual')) {
        await this.runVisualAnalysis();
      }

      // Calculate overall results
      this.calculateOverallResults();

      // Generate comprehensive report
      if (this.options.generateReport) {
        await this.generateReports();
      }

      // Cleanup
      await this.testEnvironment.globalTeardown();

      // Exit with appropriate code
      const success = this.results.overall_status === 'passed';
      process.exit(success ? 0 : 1);

    } catch (error) {
      console.error('❌ Test runner failed:', error);
      await this.testEnvironment.globalTeardown();
      process.exit(1);
    } finally {
      this.results.total_duration = Date.now() - startTime;
    }
  }

  /**
   * Get available test suites
   */
  async getTestSuites() {
    const suites = [
      {
        name: 'unit',
        command: 'npm run test:unit',
        description: 'Unit tests for individual components',
        timeout: 60000
      },
      {
        name: 'integration',
        command: 'npm run test:integration',
        description: 'API and backend integration tests',
        timeout: 120000
      },
      {
        name: 'visual',
        command: 'npm run test:visual',
        description: 'Visual regression tests',
        timeout: 300000
      },
      {
        name: 'performance',
        command: 'npm run test:performance',
        description: 'Performance and FPS monitoring',
        timeout: 180000
      }
    ];

    // Filter suites based on options
    if (this.options.suites) {
      return suites.filter(suite => this.options.suites.includes(suite.name));
    }

    return suites;
  }

  /**
   * Run test suites in parallel
   */
  async runSuitesParallel(suites) {
    console.log(`🔄 Running ${suites.length} test suites in parallel...`);

    const promises = suites.map(suite => this.runSuite(suite));
    const results = await Promise.allSettled(promises);

    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const suite = suites[i];

      if (result.status === 'fulfilled') {
        this.results.suites.push(result.value);
      } else {
        console.error(`❌ Suite ${suite.name} failed:`, result.reason);
        this.results.suites.push({
          suite_name: suite.name,
          status: 'error',
          error: result.reason.message,
          total_tests: 0,
          passed: 0,
          failed: 1,
          skipped: 0,
          errors: 1,
          duration: 0
        });
      }
    }
  }

  /**
   * Run test suites sequentially
   */
  async runSuitesSequential(suites) {
    console.log(`🔄 Running ${suites.length} test suites sequentially...`);

    for (const suite of suites) {
      try {
        const result = await this.runSuite(suite);
        this.results.suites.push(result);
      } catch (error) {
        console.error(`❌ Suite ${suite.name} failed:`, error);
        this.results.suites.push({
          suite_name: suite.name,
          status: 'error',
          error: error.message,
          total_tests: 0,
          passed: 0,
          failed: 1,
          skipped: 0,
          errors: 1,
          duration: 0
        });
      }
    }
  }

  /**
   * Run individual test suite
   */
  async runSuite(suite) {
    console.log(`\n🧪 Running ${suite.name} tests: ${suite.description}`);
    
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
      const process = spawn('npm', ['run', `test:${suite.name}`], {
        stdio: this.options.verbose ? 'inherit' : 'pipe',
        shell: true,
        timeout: suite.timeout
      });

      let stdout = '';
      let stderr = '';

      if (!this.options.verbose) {
        process.stdout?.on('data', (data) => {
          stdout += data.toString();
        });

        process.stderr?.on('data', (data) => {
          stderr += data.toString();
        });
      }

      process.on('close', (code) => {
        const duration = Date.now() - startTime;
        
        if (code === 0) {
          console.log(`✅ ${suite.name} tests completed (${(duration / 1000).toFixed(1)}s)`);
          resolve(this.parseSuiteResults(suite.name, stdout, duration));
        } else {
          console.log(`❌ ${suite.name} tests failed (${(duration / 1000).toFixed(1)}s)`);
          reject(new Error(`Suite ${suite.name} exited with code ${code}\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Parse suite results from output
   */
  parseSuiteResults(suiteName, output, duration) {
    // This is a simplified parser - in reality you'd parse actual test output
    const defaultResult = {
      suite_name: suiteName,
      tool: this.getToolForSuite(suiteName),
      status: 'passed',
      total_tests: 1,
      passed: 1,
      failed: 0,
      skipped: 0,
      errors: 0,
      duration,
      tests: []
    };

    // Try to parse actual test results from output
    try {
      if (output.includes('failing') || output.includes('failed')) {
        defaultResult.status = 'failed';
        defaultResult.failed = 1;
        defaultResult.passed = 0;
      }

      // Parse test counts if available
      const testMatch = output.match(/(\d+) passing/);
      if (testMatch) {
        defaultResult.passed = parseInt(testMatch[1]);
        defaultResult.total_tests = defaultResult.passed + defaultResult.failed;
      }

      const failMatch = output.match(/(\d+) failing/);
      if (failMatch) {
        defaultResult.failed = parseInt(failMatch[1]);
        defaultResult.total_tests = defaultResult.passed + defaultResult.failed;
        defaultResult.status = 'failed';
      }

    } catch (error) {
      console.warn(`Warning: Could not parse results for ${suiteName}:`, error.message);
    }

    return defaultResult;
  }

  /**
   * Get testing tool for suite
   */
  getToolForSuite(suiteName) {
    const tools = {
      unit: 'vitest',
      integration: 'playwright',
      visual: 'playwright',
      performance: 'playwright'
    };
    return tools[suiteName] || 'unknown';
  }

  /**
   * Run visual analysis
   */
  async runVisualAnalysis() {
    console.log('\n📸 Analyzing visual regression tests...');
    
    try {
      const analysis = await this.visualAnalyzer.analyzeAll();
      
      console.log(`📊 Visual analysis complete:`);
      console.log(`   • Total screenshots: ${analysis.summary.total}`);
      console.log(`   • Passed: ${analysis.summary.passed}`);
      console.log(`   • Failed: ${analysis.summary.failed}`);
      console.log(`   • New: ${analysis.summary.new}`);
      console.log(`   • Removed: ${analysis.summary.removed}`);

      // Update baseline if requested
      if (this.options.updateBaseline) {
        await this.visualAnalyzer.updateBaseline();
        console.log('✅ Baseline screenshots updated');
      }

    } catch (error) {
      console.error('❌ Visual analysis failed:', error);
    }
  }

  /**
   * Calculate overall test results
   */
  calculateOverallResults() {
    const totals = this.results.suites.reduce((acc, suite) => ({
      total: acc.total + suite.total_tests,
      passed: acc.passed + suite.passed,
      failed: acc.failed + suite.failed,
      errors: acc.errors + suite.errors
    }), { total: 0, passed: 0, failed: 0, errors: 0 });

    this.results.overall_status = (totals.failed === 0 && totals.errors === 0) ? 'passed' : 'failed';

    console.log('\n📊 Overall Test Results:');
    console.log(`   • Status: ${this.results.overall_status.toUpperCase()}`);
    console.log(`   • Tests: ${totals.passed}/${totals.total} passed`);
    console.log(`   • Duration: ${(this.results.total_duration / 1000).toFixed(1)}s`);
  }

  /**
   * Generate comprehensive reports
   */
  async generateReports() {
    console.log('\n📝 Generating test reports...');

    try {
      const reports = await this.reportGenerator.generateReport(this.results);
      
      console.log('✅ Reports generated:');
      for (const [format, path] of Object.entries(reports)) {
        console.log(`   • ${format.toUpperCase()}: ${path}`);
      }

    } catch (error) {
      console.error('❌ Report generation failed:', error);
    }
  }
}

// CLI Configuration
program
  .name('motion-test')
  .description('Motion Viewer Comprehensive Test Runner')
  .version('1.0.0');

program
  .command('run')
  .description('Run all test suites')
  .option('-v, --verbose', 'Verbose output')
  .option('-p, --parallel', 'Run suites in parallel', true)
  .option('--no-parallel', 'Run suites sequentially')
  .option('-s, --suites <suites>', 'Comma-separated list of suites to run')
  .option('--skip-setup', 'Skip test environment setup')
  .option('--update-baseline', 'Update visual regression baselines')
  .option('--no-report', 'Skip report generation')
  .option('--format <formats>', 'Report formats (html,json,junit)', 'html,json')
  .action(async (options) => {
    const runner = new MotionTestRunner({
      verbose: options.verbose,
      parallel: options.parallel,
      skipSetup: options.skipSetup,
      updateBaseline: options.updateBaseline,
      generateReport: options.report,
      outputFormats: options.format.split(','),
      suites: options.suites?.split(',')
    });

    await runner.runAll();
  });

program
  .command('visual')
  .description('Run visual regression tests only')
  .option('--update-baseline', 'Update baseline screenshots')
  .action(async (options) => {
    const runner = new MotionTestRunner({
      suites: ['visual'],
      updateBaseline: options.updateBaseline
    });

    await runner.runAll();
  });

program
  .command('performance')
  .description('Run performance tests only')
  .action(async () => {
    const runner = new MotionTestRunner({
      suites: ['performance']
    });

    await runner.runAll();
  });

program
  .command('setup')
  .description('Setup test environment only')
  .action(async () => {
    const env = new TestEnvironment();
    await env.globalSetup();
    console.log('✅ Test environment setup complete');
  });

program
  .command('cleanup')
  .description('Cleanup test environment')
  .action(async () => {
    const env = new TestEnvironment();
    await env.globalTeardown();
    console.log('✅ Test environment cleanup complete');
  });

// Run CLI
if (import.meta.url === `file://${process.argv[1]}`) {
  program.parse();
}

export default MotionTestRunner;
