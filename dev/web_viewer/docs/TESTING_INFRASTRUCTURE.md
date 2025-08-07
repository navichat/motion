# Testing Infrastructure Documentation

## Overview

The testing infrastructure provides comprehensive validation for the WebNN/WebGPU/WASM powered avatar system. This multi-layered testing approach ensures component reliability, system integration, and performance validation across different compute backends and use cases.

## Testing Architecture

### Test Hierarchy

```
tests/
├── unit/                     # Component isolation tests
│   ├── ai/                   # AI model component tests
│   ├── avatar/               # Avatar system component tests
│   ├── audio/                # Audio processing component tests
│   ├── compute/              # Compute backend component tests
│   ├── motion/               # Motion processing component tests
│   └── system/               # System integration component tests
├── integration/              # Cross-component integration tests
│   ├── e2e/                  # End-to-end workflow tests
│   └── master-test-suite.spec.js  # Comprehensive system validation
└── manual/                   # Interactive testing interfaces
    ├── test_conversation.html # Manual conversation testing
    ├── test_script.html      # Basic functionality testing
    └── three_test_simple.html # 3D rendering validation
```

### Testing Frameworks

#### Playwright (E2E and Integration)
- **Browser Testing**: Cross-browser compatibility validation
- **Real Environment**: Tests run in actual browser environments
- **Network Simulation**: Test network conditions and latency
- **Performance Monitoring**: Built-in performance metrics collection

#### Unit Testing Framework
- **Component Isolation**: Test individual components without dependencies
- **Mock Support**: Mock external dependencies for focused testing
- **Coverage Reporting**: Track test coverage across codebase
- **Async Testing**: Full support for Promise-based and async/await patterns

## Test Categories

### Unit Tests

#### AI Model Tests (`tests/unit/ai/ai-model-jobs.spec.js`)
```javascript
test('should create TinyLlama inference job', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const jobFactory = new window.RealJobFactory();
    const job = jobFactory.createJob('tinyllama', {
      prompt: 'Hello, how are you?',
      maxTokens: 50,
      temperature: 0.7
    });
    
    return {
      created: !!job,
      hasPrompt: !!job.prompt,
      hasConfig: !!job.config,
      jobType: job.type
    };
  });
  
  expect(result.created).toBe(true);
  expect(result.hasPrompt).toBe(true);
  expect(result.jobType).toBe('tinyllama');
});
```

**Coverage Areas:**
- Job creation and configuration
- Worker initialization and management
- Model loading and inference execution
- Error handling and recovery
- Memory management and cleanup

#### Avatar Animation Tests (`tests/unit/avatar/avatar-animation.spec.js`)
```javascript
test('should blend animations smoothly', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const blender = new window.AnimationBlender();
    const blendResult = blender.blend(
      idleAnimation,
      walkAnimation,
      0.5, // 50% blend weight
      { duration: 1.0, easing: 'linear' }
    );
    
    return {
      blended: !!blendResult,
      hasFrames: blendResult.frames && blendResult.frames.length > 0,
      duration: blendResult.duration,
      frameCount: blendResult.frames ? blendResult.frames.length : 0
    };
  });
  
  expect(result.blended).toBe(true);
  expect(result.hasFrames).toBe(true);
  expect(result.duration).toBeCloseTo(1.0, 1);
});
```

**Coverage Areas:**
- VRM character loading and validation
- BVH motion parsing and processing
- Animation blending and transitions
- Timeline management and synchronization
- Expression control and morphing

#### Audio Processing Tests (`tests/unit/audio/audio-processing.spec.js`)
```javascript
test('should process audio with Kokoro TTS', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const kokoroTTS = new window.KokoroTTS();
    const audioData = await kokoroTTS.synthesize(
      "Hello, this is a test message",
      { voice: 'neutral', speed: 1.0, pitch: 1.0 }
    );
    
    return {
      synthesized: true,
      hasAudioData: !!audioData,
      audioLength: audioData ? audioData.length : 0,
      isFloat32Array: audioData instanceof Float32Array
    };
  });
  
  expect(result.synthesized).toBe(true);
  expect(result.hasAudioData).toBe(true);
  expect(result.isFloat32Array).toBe(true);
});
```

**Coverage Areas:**
- Text-to-speech synthesis
- Speech-to-text transcription
- Voice activity detection
- Audio processing and filtering
- Real-time audio streaming

#### Compute Backend Tests (`tests/unit/compute/compute-backend.spec.js`)
```javascript
test('should execute model inference with different backends', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const backends = ['webgpu', 'webnn', 'wasm'];
    const results = {};
    
    for (const backendType of backends) {
      try {
        const backend = window.createBackend(backendType);
        await backend.initialize();
        
        const input = new Float32Array([1, 2, 3, 4]);
        const output = await backend.inference(mockModel, input);
        
        results[backendType] = {
          initialized: true,
          hasOutput: !!output,
          outputCorrect: validateOutput(output)
        };
      } catch (error) {
        results[backendType] = {
          initialized: false,
          error: error.message
        };
      }
    }
    
    return results;
  });
  
  // Verify at least one backend works
  const workingBackends = Object.keys(result).filter(b => result[b].initialized);
  expect(workingBackends.length).toBeGreaterThan(0);
});
```

**Coverage Areas:**
- Backend initialization and capability detection
- Model loading and inference execution
- Memory management and buffer operations
- Performance benchmarking and optimization
- Error handling and fallback mechanisms

### Integration Tests

#### End-to-End Workflow Tests (`tests/integration/e2e/`)
```javascript
test('should complete full avatar conversation pipeline', async ({ page }) => {
  await page.goto('http://localhost:8081/demos/ai-inference/task-manager-demo.html');
  
  // Start the workload
  await page.click('button:has-text("Real WASM/GPU/WebNN Workload")');
  
  // Wait for completion
  await page.waitForTimeout(180000);
  
  // Validate results
  const results = await page.evaluate(() => {
    return {
      completedTasks: window.taskManager.completedTasks.length,
      jobTypes: window.taskManager.getAllJobTypes(),
      successRate: window.taskManager.getSuccessRate()
    };
  });
  
  expect(results.completedTasks).toBeGreaterThan(40);
  expect(results.jobTypes.length).toBeGreaterThan(16);
  expect(results.successRate).toBeGreaterThan(0.95);
});
```

**Coverage Areas:**
- Complete AI inference pipeline
- Cross-component data flow
- Performance under realistic workloads
- Error recovery and graceful degradation
- Resource management and cleanup

#### Master Test Suite (`tests/integration/master-test-suite.spec.js`)
```javascript
test('should validate all test files exist and are accessible', async ({ page }) => {
  const testFiles = [
    '/tests/unit/ai/ai-model-jobs.spec.js',
    '/tests/unit/avatar/avatar-animation.spec.js',
    '/tests/unit/audio/audio-processing.spec.js',
    '/tests/unit/compute/compute-backend.spec.js',
    '/tests/unit/motion/motion-processing.spec.js',
    '/tests/unit/system/system-integration.spec.js'
  ];
  
  for (const testFile of testFiles) {
    const response = await page.goto(`http://localhost:8081${testFile}`);
    expect(response.status()).toBeLessThan(400);
  }
});
```

**Coverage Areas:**
- System architecture validation
- Import path verification
- Component accessibility testing
- Configuration validation
- Overall system health checks

### Manual Testing

#### Interactive Testing Interfaces (`tests/manual/`)

**Conversation Testing** (`test_conversation.html`)
- Real-time audio conversation testing
- AI model loading and initialization
- Voice activity detection validation
- TTS and STT integration testing
- Error handling and recovery testing

**Component Testing** (`test_script.html`)
- Basic functionality validation
- Library loading verification
- Browser compatibility testing
- Performance monitoring
- Debug information display

**3D Rendering Testing** (`three_test_simple.html`)
- THREE.js library validation
- WebGL capability testing
- Basic 3D scene rendering
- Performance benchmarking
- Graphics driver compatibility

## Test Execution

### Running Tests

#### Unit Tests
```bash
# Run all unit tests
npx playwright test tests/unit/

# Run specific component tests
npx playwright test tests/unit/ai/
npx playwright test tests/unit/avatar/
npx playwright test tests/unit/audio/
npx playwright test tests/unit/compute/
npx playwright test tests/unit/motion/
npx playwright test tests/unit/system/

# Run with verbose output
npx playwright test tests/unit/ --reporter=list
```

#### Integration Tests
```bash
# Run all integration tests
npx playwright test tests/integration/

# Run E2E tests
npx playwright test tests/integration/e2e/

# Run master test suite
npx playwright test tests/integration/master-test-suite.spec.js

# Run with debugging
npx playwright test tests/integration/ --debug
```

#### Manual Tests
```bash
# Start development server
cd /home/barberb/motion/dev/web_viewer
python3 -m http.server 8081

# Access manual tests
# http://localhost:8081/tests/manual/test_conversation.html
# http://localhost:8081/tests/manual/test_script.html
# http://localhost:8081/tests/manual/three_test_simple.html
```

### Continuous Integration

#### Test Pipeline
```yaml
# Example CI configuration
name: Avatar System Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npx playwright install
      - run: npx playwright test tests/unit/
      
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npx playwright install
      - run: npx playwright test tests/integration/
      
  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npx playwright install
      - run: npx playwright test tests/performance/
```

## Performance Testing

### Benchmarking Framework
```javascript
class PerformanceTester {
  async benchmarkComponent(component, operations) {
    const results = {};
    
    for (const [name, operation] of Object.entries(operations)) {
      const measurements = [];
      
      // Run multiple iterations
      for (let i = 0; i < 10; i++) {
        const start = performance.now();
        await operation();
        const end = performance.now();
        measurements.push(end - start);
      }
      
      results[name] = {
        mean: measurements.reduce((a, b) => a + b) / measurements.length,
        min: Math.min(...measurements),
        max: Math.max(...measurements),
        std: this.calculateStandardDeviation(measurements)
      };
    }
    
    return results;
  }
}
```

### Memory Testing
```javascript
test('should not leak memory during extended operation', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const initialMemory = performance.memory.usedJSHeapSize;
    
    // Run intensive operations
    for (let i = 0; i < 100; i++) {
      const taskManager = new TaskManager();
      await taskManager.initialize();
      await taskManager.createTask('ai-inference', { prompt: 'test' });
      taskManager.cleanup();
    }
    
    // Force garbage collection
    if (window.gc) window.gc();
    
    const finalMemory = performance.memory.usedJSHeapSize;
    const memoryDelta = finalMemory - initialMemory;
    
    return {
      initialMemory,
      finalMemory,
      memoryDelta,
      memoryLeakDetected: memoryDelta > initialMemory * 0.1 // 10% threshold
    };
  });
  
  expect(result.memoryLeakDetected).toBe(false);
});
```

## Test Configuration

### Playwright Configuration
```javascript
// playwright.config.js
export default {
  testDir: './tests',
  timeout: 300000, // 5 minutes for AI model tests
  expect: {
    timeout: 30000
  },
  use: {
    baseURL: 'http://localhost:8081',
    headless: false, // Set to true for CI
    viewport: { width: 1280, height: 720 },
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }
    }
  ],
  webServer: {
    command: 'python3 -m http.server 8081',
    port: 8081,
    cwd: '/home/barberb/motion/dev/web_viewer'
  }
};
```

### Test Environment Setup
```javascript
// Global test setup
export async function globalSetup() {
  // Start required services
  await startTestServer();
  await initializeTestDatabase();
  await warmupAIModels();
}

export async function globalTeardown() {
  // Cleanup after all tests
  await stopTestServer();
  await cleanupTestData();
  await releaseResources();
}
```

## Test Data Management

### Mock Data Generation
```javascript
class TestDataGenerator {
  generateVRMCharacter() {
    return {
      humanoid: { bones: this.generateBoneHierarchy() },
      materials: this.generateMaterials(),
      meshes: this.generateMeshes(),
      expressions: this.generateExpressions()
    };
  }
  
  generateBVHMotion(frameCount = 100) {
    return {
      hierarchy: this.generateBoneHierarchy(),
      motion: {
        frames: frameCount,
        frameTime: 1/30,
        data: this.generateMotionFrames(frameCount)
      }
    };
  }
  
  generateAudioData(duration = 1.0, sampleRate = 16000) {
    const sampleCount = duration * sampleRate;
    return new Float32Array(sampleCount).map(() => 
      Math.sin(2 * Math.PI * 440 * Math.random()) * 0.1
    );
  }
}
```

### Test Asset Management
```javascript
class TestAssetManager {
  constructor() {
    this.assetCache = new Map();
  }
  
  async loadTestAsset(assetPath) {
    if (this.assetCache.has(assetPath)) {
      return this.assetCache.get(assetPath);
    }
    
    const asset = await fetch(`/assets/test/${assetPath}`);
    const data = await asset.arrayBuffer();
    
    this.assetCache.set(assetPath, data);
    return data;
  }
  
  getTestVRM() {
    return this.loadTestAsset('characters/test-character.vrm');
  }
  
  getTestBVH() {
    return this.loadTestAsset('motions/test-motion.bvh');
  }
  
  getTestAudio() {
    return this.loadTestAsset('audio/test-speech.wav');
  }
}
```

## Debugging and Diagnostics

### Test Debugging Tools
```javascript
class TestDebugger {
  static async captureState(page) {
    return await page.evaluate(() => {
      return {
        taskManager: window.taskManager ? {
          totalTasks: window.taskManager.tasks.length,
          completedTasks: window.taskManager.completedTasks.length,
          runningTasks: window.taskManager.runningTasks.length,
          workers: window.taskManager.workers.map(w => w.type)
        } : null,
        memory: performance.memory ? {
          used: performance.memory.usedJSHeapSize,
          total: performance.memory.totalJSHeapSize,
          limit: performance.memory.jsHeapSizeLimit
        } : null,
        errors: window.testErrors || []
      };
    });
  }
  
  static async captureScreenshot(page, testName) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    await page.screenshot({ 
      path: `test-results/${testName}-${timestamp}.png`,
      fullPage: true
    });
  }
}
```

### Error Collection
```javascript
// Global error handler for tests
window.addEventListener('error', (event) => {
  if (!window.testErrors) {
    window.testErrors = [];
  }
  
  window.testErrors.push({
    message: event.error.message,
    stack: event.error.stack,
    timestamp: Date.now(),
    url: event.filename,
    line: event.lineno,
    column: event.colno
  });
});
```

## Test Reporting

### Coverage Reports
```bash
# Generate coverage report
npx playwright test --reporter=html

# View coverage
open playwright-report/index.html
```

### Performance Reports
```javascript
class PerformanceReporter {
  static generateReport(testResults) {
    const report = {
      summary: {
        totalTests: testResults.length,
        passedTests: testResults.filter(t => t.passed).length,
        failedTests: testResults.filter(t => !t.passed).length,
        averageExecutionTime: this.calculateAverage(testResults.map(t => t.duration))
      },
      performance: {
        slowestTests: testResults.sort((a, b) => b.duration - a.duration).slice(0, 10),
        memoryUsage: this.analyzeMemoryUsage(testResults),
        backendPerformance: this.analyzeBackendPerformance(testResults)
      },
      errors: this.categorizeErrors(testResults.filter(t => !t.passed))
    };
    
    return report;
  }
}
```

For more information, see the main [REORGANIZATION_STATUS.md](../REORGANIZATION_STATUS.md) documentation.
