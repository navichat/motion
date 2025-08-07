# Development Workflow Documentation

## Overview

This document outlines the complete development workflow for the WebNN/WebGPU/WASM powered avatar system. It covers everything from component development to testing, integration, and deployment in the reorganized project structure.

## Development Environment Setup

### Prerequisites
```bash
# Ensure Node.js and npm are installed
node --version  # v18+ recommended
npm --version   # v9+ recommended

# Install Playwright for testing
npm install -g @playwright/test
npx playwright install

# Python for development server
python3 --version  # v3.8+ recommended
```

### Project Setup
```bash
# Navigate to project directory
cd /home/barberb/motion/dev/web_viewer

# Start development server
python3 -m http.server 8081

# Access development interface
# http://localhost:8081
```

### IDE Configuration
```javascript
// VS Code settings.json recommendations
{
  "files.associations": {
    "*.spec.js": "javascript"
  },
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "playwright.showTrace": true,
  "playwright.reuseBrowser": true
}
```

## Component Development Lifecycle

### 1. Planning Phase

#### Define Component Requirements
```markdown
## Component: NewAIModel
- **Purpose**: Implement new AI model for [specific use case]
- **Input**: [Input data format and structure]
- **Output**: [Expected output format]
- **Dependencies**: [Required modules and backends]
- **Performance**: [Target execution time and memory usage]
- **Testing**: [Unit test requirements and integration points]
```

#### Architecture Design
```javascript
// Component interface design
class NewAIModel {
  constructor(config) {
    this.config = config;
    this.initialized = false;
  }
  
  async initialize() {
    // Model initialization logic
  }
  
  async process(input) {
    // Main processing logic
  }
  
  cleanup() {
    // Resource cleanup
  }
}
```

### 2. Implementation Phase

#### Create Component Structure
```bash
# Create component files
mkdir -p src/ai/models/new-model/
touch src/ai/models/new-model/NewAIModel.js
touch src/ai/models/new-model/NewAIModelWorker.js
touch src/ai/jobs/NewAIModelJob.js

# Create test files
mkdir -p tests/unit/ai/new-model/
touch tests/unit/ai/new-model/new-ai-model.spec.js

# Create demo
mkdir -p demos/ai-models/new-model/
touch demos/ai-models/new-model/new-model-demo.html
```

#### Implement Core Component
```javascript
// src/ai/models/new-model/NewAIModel.js
export class NewAIModel {
  constructor(config = {}) {
    this.config = {
      modelPath: config.modelPath || '/assets/models/new-model.onnx',
      precision: config.precision || 'fp32',
      maxBatchSize: config.maxBatchSize || 1,
      ...config
    };
    
    this.model = null;
    this.backend = null;
    this.initialized = false;
  }
  
  async initialize(backend) {
    try {
      this.backend = backend;
      this.model = await this.backend.loadModel(this.config.modelPath);
      this.initialized = true;
      
      console.log(`NewAIModel initialized with ${backend.type} backend`);
    } catch (error) {
      console.error('NewAIModel initialization failed:', error);
      throw error;
    }
  }
  
  async process(input) {
    if (!this.initialized) {
      throw new Error('Model not initialized');
    }
    
    try {
      // Preprocess input
      const preprocessed = this.preprocessInput(input);
      
      // Run inference
      const output = await this.backend.inference(this.model, preprocessed);
      
      // Postprocess output
      const result = this.postprocessOutput(output);
      
      return result;
    } catch (error) {
      console.error('NewAIModel processing failed:', error);
      throw error;
    }
  }
  
  preprocessInput(input) {
    // Input preprocessing logic
    return input;
  }
  
  postprocessOutput(output) {
    // Output postprocessing logic
    return output;
  }
  
  cleanup() {
    if (this.model && this.backend) {
      this.backend.unloadModel(this.model);
    }
    this.initialized = false;
  }
}
```

#### Implement Worker
```javascript
// src/ai/models/new-model/NewAIModelWorker.js
import { NewAIModel } from './NewAIModel.js';

export class NewAIModelWorker {
  constructor() {
    this.model = null;
    this.type = 'new-ai-model';
  }
  
  async initialize(backend, config) {
    this.model = new NewAIModel(config);
    await this.model.initialize(backend);
  }
  
  async processJob(job) {
    if (!this.model) {
      throw new Error('Worker not initialized');
    }
    
    return await this.model.process(job.input);
  }
  
  cleanup() {
    if (this.model) {
      this.model.cleanup();
      this.model = null;
    }
  }
}
```

#### Implement Job Class
```javascript
// src/ai/jobs/NewAIModelJob.js
export class NewAIModelJob {
  constructor(id, input, config = {}) {
    this.id = id;
    this.type = 'new-ai-model';
    this.input = input;
    this.config = config;
    this.priority = config.priority || 1;
    this.timeout = config.timeout || 30000; // 30 seconds
  }
  
  async execute(worker) {
    const startTime = performance.now();
    
    try {
      const result = await worker.processJob(this);
      const endTime = performance.now();
      
      return {
        success: true,
        result: result,
        executionTime: endTime - startTime,
        jobId: this.id
      };
    } catch (error) {
      const endTime = performance.now();
      
      return {
        success: false,
        error: error.message,
        executionTime: endTime - startTime,
        jobId: this.id
      };
    }
  }
  
  validate() {
    if (!this.input) {
      throw new Error('Job input is required');
    }
    
    if (typeof this.input !== 'object') {
      throw new Error('Job input must be an object');
    }
    
    return true;
  }
}
```

### 3. Testing Phase

#### Unit Tests
```javascript
// tests/unit/ai/new-model/new-ai-model.spec.js
import { test, expect } from '@playwright/test';

test.describe('NewAIModel Component Tests', () => {
  test('should initialize model successfully', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/ai/new-model/new-model-test.html');
    
    const result = await page.evaluate(async () => {
      const { NewAIModel } = await import('/src/ai/models/new-model/NewAIModel.js');
      const { WASMBackend } = await import('/src/compute/backends/WASMBackend.js');
      
      try {
        const backend = new WASMBackend();
        await backend.initialize();
        
        const model = new NewAIModel({
          modelPath: '/assets/models/test-model.onnx'
        });
        
        await model.initialize(backend);
        
        return {
          initialized: model.initialized,
          hasModel: !!model.model,
          hasBackend: !!model.backend,
          backendType: model.backend.type
        };
      } catch (error) {
        return {
          initialized: false,
          error: error.message
        };
      }
    });
    
    expect(result.initialized).toBe(true);
    expect(result.hasModel).toBe(true);
    expect(result.hasBackend).toBe(true);
  });
  
  test('should process input correctly', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/ai/new-model/new-model-test.html');
    
    const result = await page.evaluate(async () => {
      const { NewAIModel } = await import('/src/ai/models/new-model/NewAIModel.js');
      const { WASMBackend } = await import('/src/compute/backends/WASMBackend.js');
      
      const backend = new WASMBackend();
      await backend.initialize();
      
      const model = new NewAIModel();
      await model.initialize(backend);
      
      const testInput = {
        data: [1, 2, 3, 4],
        shape: [1, 4]
      };
      
      try {
        const output = await model.process(testInput);
        
        return {
          processed: true,
          hasOutput: !!output,
          outputType: typeof output,
          outputKeys: Object.keys(output)
        };
      } catch (error) {
        return {
          processed: false,
          error: error.message
        };
      }
    });
    
    expect(result.processed).toBe(true);
    expect(result.hasOutput).toBe(true);
  });
  
  test('should handle errors gracefully', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/ai/new-model/new-model-test.html');
    
    const result = await page.evaluate(async () => {
      const { NewAIModel } = await import('/src/ai/models/new-model/NewAIModel.js');
      
      const model = new NewAIModel();
      
      try {
        // Try to process without initialization
        await model.process({ data: [1, 2, 3] });
        return { errorHandled: false };
      } catch (error) {
        return {
          errorHandled: true,
          errorMessage: error.message,
          errorType: error.constructor.name
        };
      }
    });
    
    expect(result.errorHandled).toBe(true);
    expect(result.errorMessage).toContain('not initialized');
  });
});
```

#### Integration Tests
```javascript
// tests/integration/new-model-integration.spec.js
import { test, expect } from '@playwright/test';

test.describe('NewAIModel Integration Tests', () => {
  test('should integrate with TaskManager', async ({ page }) => {
    await page.goto('http://localhost:8081/demos/ai-models/new-model/new-model-demo.html');
    
    const result = await page.evaluate(async () => {
      const taskManager = new window.TaskManager();
      await taskManager.initialize();
      
      // Create new model job
      const job = await taskManager.createTask('new-ai-model', {
        input: { data: [1, 2, 3, 4], shape: [1, 4] },
        config: { priority: 1 }
      });
      
      // Execute job
      const result = await taskManager.executeTask(job.id);
      
      return {
        jobCreated: !!job,
        jobId: job.id,
        executed: !!result,
        success: result.success,
        hasResult: !!result.result,
        executionTime: result.executionTime
      };
    });
    
    expect(result.jobCreated).toBe(true);
    expect(result.executed).toBe(true);
    expect(result.success).toBe(true);
    expect(result.executionTime).toBeLessThan(5000); // < 5 seconds
  });
});
```

### 4. Documentation Phase

#### Component Documentation
```javascript
/**
 * NewAIModel - Advanced AI Model Implementation
 * 
 * @description
 * Implements a new AI model for [specific purpose]. Supports multiple compute
 * backends (WebGPU, WebNN, WASM) for optimal performance across platforms.
 * 
 * @example
 * ```javascript
 * const model = new NewAIModel({
 *   modelPath: '/assets/models/new-model.onnx',
 *   precision: 'fp16'
 * });
 * 
 * const backend = new WebGPUBackend();
 * await backend.initialize();
 * await model.initialize(backend);
 * 
 * const result = await model.process({
 *   data: inputData,
 *   shape: [1, 256]
 * });
 * ```
 * 
 * @performance
 * - WebGPU: ~5ms for typical input
 * - WebNN: ~8ms for typical input  
 * - WASM: ~15ms for typical input
 * 
 * @memory
 * - Model size: ~10MB
 * - Runtime memory: ~50MB peak
 * - GPU memory: ~25MB (WebGPU only)
 */
export class NewAIModel {
  // Implementation...
}
```

#### Update Import Map
```javascript
// config/import-map.js
export const ImportMap = {
  ai: {
    // Existing entries...
    NewAIModel: '/src/ai/models/new-model/NewAIModel.js',
    NewAIModelWorker: '/src/ai/models/new-model/NewAIModelWorker.js',
    NewAIModelJob: '/src/ai/jobs/NewAIModelJob.js'
  }
  // Other categories...
};
```

## Quality Assurance Workflow

### Code Review Checklist

#### Component Implementation
- [ ] Follows established naming conventions
- [ ] Implements proper error handling
- [ ] Includes comprehensive documentation
- [ ] Supports multiple compute backends
- [ ] Implements proper resource cleanup
- [ ] Includes performance optimizations

#### Testing
- [ ] Unit tests cover all public methods
- [ ] Integration tests validate TaskManager compatibility
- [ ] Error cases are properly tested
- [ ] Performance benchmarks are included
- [ ] Memory leak tests pass
- [ ] Cross-browser compatibility verified

#### Documentation
- [ ] Component purpose clearly explained
- [ ] API documentation is complete
- [ ] Usage examples are provided
- [ ] Performance characteristics documented
- [ ] Integration points identified
- [ ] Troubleshooting guide included

### Automated Testing Pipeline

#### Pre-commit Hooks
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# Run unit tests for modified components
npm run test:unit:changed

# Run linting
npm run lint:fix

# Check import map consistency
npm run validate:imports

# Performance regression test
npm run test:performance:regression

echo "Pre-commit checks completed successfully!"
```

#### Continuous Integration
```yaml
# .github/workflows/ci.yml
name: Avatar System CI

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install
      - run: npx playwright install
      - run: npm run test:unit
      
  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npx playwright install
      - run: npm run test:integration
      
  performance-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npx playwright install
      - run: npm run test:performance
      
  deploy:
    needs: [unit-tests, integration-tests, performance-tests]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - run: npm run build:production
      - run: npm run deploy
```

## Performance Optimization Workflow

### Profiling and Benchmarking

#### Component Profiling
```javascript
// tools/performance/component-profiler.js
export class ComponentProfiler {
  static async profileComponent(ComponentClass, testCases) {
    const results = {};
    
    for (const [testName, testData] of Object.entries(testCases)) {
      const component = new ComponentClass(testData.config);
      
      // Warmup runs
      for (let i = 0; i < 5; i++) {
        await component.process(testData.input);
      }
      
      // Benchmark runs
      const times = [];
      const memoryBefore = performance.memory.usedJSHeapSize;
      
      for (let i = 0; i < 50; i++) {
        const start = performance.now();
        await component.process(testData.input);
        times.push(performance.now() - start);
      }
      
      const memoryAfter = performance.memory.usedJSHeapSize;
      
      results[testName] = {
        averageTime: times.reduce((a, b) => a + b) / times.length,
        minTime: Math.min(...times),
        maxTime: Math.max(...times),
        memoryDelta: memoryAfter - memoryBefore,
        throughput: 1000 / (times.reduce((a, b) => a + b) / times.length)
      };
      
      component.cleanup();
    }
    
    return results;
  }
}
```

#### Backend Comparison
```javascript
// tools/performance/backend-benchmark.js
export class BackendBenchmark {
  static async compareBackends(model, testInput) {
    const backends = ['webgpu', 'webnn', 'wasm'];
    const results = {};
    
    for (const backendType of backends) {
      try {
        const backend = await this.createBackend(backendType);
        await backend.initialize();
        
        const modelInstance = new model.constructor();
        await modelInstance.initialize(backend);
        
        // Benchmark
        const times = [];
        for (let i = 0; i < 20; i++) {
          const start = performance.now();
          await modelInstance.process(testInput);
          times.push(performance.now() - start);
        }
        
        results[backendType] = {
          supported: true,
          averageTime: times.reduce((a, b) => a + b) / times.length,
          minTime: Math.min(...times),
          maxTime: Math.max(...times),
          standardDeviation: this.calculateStdDev(times)
        };
        
        modelInstance.cleanup();
      } catch (error) {
        results[backendType] = {
          supported: false,
          error: error.message
        };
      }
    }
    
    return results;
  }
}
```

### Memory Management

#### Memory Leak Detection
```javascript
// tools/debugging/memory-leak-detector.js
export class MemoryLeakDetector {
  static async detectLeaks(operation, iterations = 100) {
    const measurements = [];
    
    // Force initial garbage collection
    if (window.gc) window.gc();
    
    const initialMemory = performance.memory.usedJSHeapSize;
    
    for (let i = 0; i < iterations; i++) {
      await operation();
      
      // Periodic garbage collection
      if (i % 10 === 0 && window.gc) {
        window.gc();
      }
      
      measurements.push(performance.memory.usedJSHeapSize);
    }
    
    // Final garbage collection
    if (window.gc) window.gc();
    const finalMemory = performance.memory.usedJSHeapSize;
    
    return {
      initialMemory,
      finalMemory,
      memoryDelta: finalMemory - initialMemory,
      measurements,
      leakDetected: finalMemory > initialMemory * 1.1, // 10% threshold
      trend: this.calculateMemoryTrend(measurements)
    };
  }
}
```

## Deployment Workflow

### Build Process

#### Production Build
```javascript
// tools/build/production-build.js
export class ProductionBuilder {
  static async build() {
    console.log('Starting production build...');
    
    // 1. Validate all components
    await this.validateComponents();
    
    // 2. Run comprehensive tests
    await this.runTests();
    
    // 3. Optimize assets
    await this.optimizeAssets();
    
    // 4. Bundle modules
    await this.bundleModules();
    
    // 5. Generate documentation
    await this.generateDocs();
    
    console.log('Production build completed successfully!');
  }
  
  static async validateComponents() {
    // Validate import map consistency
    // Check for missing dependencies
    // Verify API compatibility
  }
  
  static async optimizeAssets() {
    // Compress models
    // Optimize textures
    // Minify code
  }
}
```

#### Deployment Configuration
```javascript
// config/deployment.js
export const DeploymentConfig = {
  production: {
    baseUrl: 'https://avatar-system.example.com',
    modelCDN: 'https://models.example.com',
    enableAnalytics: true,
    enableDebugMode: false,
    cacheMaxAge: 86400, // 24 hours
    backends: ['webgpu', 'webnn', 'wasm']
  },
  
  staging: {
    baseUrl: 'https://staging.avatar-system.example.com',
    modelCDN: 'https://staging-models.example.com',
    enableAnalytics: false,
    enableDebugMode: true,
    cacheMaxAge: 3600, // 1 hour
    backends: ['webgpu', 'webnn', 'wasm']
  },
  
  development: {
    baseUrl: 'http://localhost:8081',
    modelCDN: 'http://localhost:8081/assets/models',
    enableAnalytics: false,
    enableDebugMode: true,
    cacheMaxAge: 0, // No caching
    backends: ['wasm'] // WASM only for reliability
  }
};
```

## Monitoring and Maintenance

### Performance Monitoring

#### Real-time Metrics
```javascript
// tools/monitoring/performance-monitor.js
export class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.alerts = [];
  }
  
  startMonitoring() {
    setInterval(() => {
      this.collectMetrics();
      this.checkAlerts();
    }, 5000); // Every 5 seconds
  }
  
  collectMetrics() {
    const memory = performance.memory;
    const now = Date.now();
    
    this.metrics.set(now, {
      memoryUsed: memory.usedJSHeapSize,
      memoryTotal: memory.totalJSHeapSize,
      activeComponents: this.getActiveComponentCount(),
      runningTasks: this.getRunningTaskCount(),
      frameRate: this.getCurrentFrameRate()
    });
    
    // Keep only last 100 measurements
    if (this.metrics.size > 100) {
      const oldestKey = Math.min(...this.metrics.keys());
      this.metrics.delete(oldestKey);
    }
  }
  
  checkAlerts() {
    const latest = this.getLatestMetrics();
    
    // Memory usage alert
    if (latest.memoryUsed > latest.memoryTotal * 0.9) {
      this.addAlert('HIGH_MEMORY_USAGE', {
        current: latest.memoryUsed,
        total: latest.memoryTotal,
        percentage: (latest.memoryUsed / latest.memoryTotal * 100).toFixed(1)
      });
    }
    
    // Frame rate alert
    if (latest.frameRate < 30) {
      this.addAlert('LOW_FRAME_RATE', {
        current: latest.frameRate,
        target: 60
      });
    }
  }
}
```

### Error Tracking

#### Error Reporter
```javascript
// tools/monitoring/error-reporter.js
export class ErrorReporter {
  static initialize() {
    window.addEventListener('error', this.handleError.bind(this));
    window.addEventListener('unhandledrejection', this.handleRejection.bind(this));
  }
  
  static handleError(event) {
    const errorInfo = {
      type: 'javascript-error',
      message: event.error.message,
      stack: event.error.stack,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    
    this.reportError(errorInfo);
  }
  
  static handleRejection(event) {
    const errorInfo = {
      type: 'unhandled-promise-rejection',
      message: event.reason.message || event.reason,
      stack: event.reason.stack,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    
    this.reportError(errorInfo);
  }
  
  static reportError(errorInfo) {
    // Send to monitoring service
    // Store locally for offline analysis
    // Trigger alerts for critical errors
    console.error('Error reported:', errorInfo);
  }
}
```

For more information, see the main [REORGANIZATION_STATUS.md](../REORGANIZATION_STATUS.md) documentation.
