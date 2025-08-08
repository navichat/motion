# Testing Guide

## Overview

Comprehensive testing framework for individual component validation and integration testing of the WebNN/WebGPU/WASM avatar system.

## Test Structure

```
tests/
├── unit/                     # Individual component tests
│   ├── ai-models.spec.js     # AI model testing
│   ├── audio-processing.spec.js # Audio pipeline testing
│   ├── avatar-motion.spec.js # Motion system testing
│   └── compute-backends.spec.js # Backend testing
├── integration/              # Integration tests
│   ├── avatar-animation.spec.js # Full pipeline testing
│   └── collect-all-outputs.spec.js # Output collection
├── legacy/                   # Legacy test files
└── legacy-root-tests/        # Previously scattered tests
```

## Individual Component Testing

### AI Models Testing

Test individual AI models and their backends:

```bash
# Test all AI models
npx playwright test tests/unit/ai-models.spec.js

# Test specific models
npx playwright test tests/unit/ai-models.spec.js --grep "TinyLlama"
npx playwright test tests/unit/ai-models.spec.js --grep "Whisper"
npx playwright test tests/unit/ai-models.spec.js --grep "Audio2Gesture"
```

**What it tests:**
- Model loading and initialization
- Inference functionality
- Backend switching (WebNN/WebGPU/WASM)
- Memory management
- Error handling

### Audio Processing Testing

Test audio pipeline components individually:

```bash
# Test audio processing
npx playwright test tests/unit/audio-processing.spec.js

# Test specific audio components
npx playwright test tests/unit/audio-processing.spec.js --grep "VAD"
npx playwright test tests/unit/audio-processing.spec.js --grep "TTS"
npx playwright test tests/unit/audio-processing.spec.js --grep "Conversation"
```

**What it tests:**
- Voice activity detection
- Speech synthesis
- Audio feature extraction
- Conversation workers
- Audio format handling

### Avatar Motion Testing

Test motion and animation systems:

```bash
# Test avatar motion
npx playwright test tests/unit/avatar-motion.spec.js

# Test specific motion components
npx playwright test tests/unit/avatar-motion.spec.js --grep "Gesture"
npx playwright test tests/unit/avatar-motion.spec.js --grep "Facial"
npx playwright test tests/unit/avatar-motion.spec.js --grep "RSMT"
```

**What it tests:**
- Gesture generation
- Facial expression control
- Motion blending
- Animation timing
- VRM compatibility

### Compute Backend Testing

Test individual compute backends:

```bash
# Test all backends
npx playwright test tests/unit/compute-backends.spec.js

# Test specific backends
npx playwright test tests/unit/compute-backends.spec.js --grep "WebNN"
npx playwright test tests/unit/compute-backends.spec.js --grep "WebGPU"
npx playwright test tests/unit/compute-backends.spec.js --grep "WASM"
```

**What it tests:**
- Backend initialization
- Model loading per backend
- Inference performance
- Memory usage
- Error handling

## Integration Testing

### Full Avatar Animation Pipeline

Test the complete avatar animation workflow:

```bash
npx playwright test tests/integration/avatar-animation.spec.js
```

**What it tests:**
- Audio input → gesture generation
- Speech synthesis → facial animation
- Motion blending and synchronization
- Real-time performance
- Cross-component communication

### Comprehensive Output Collection

Collect and analyze all AI model outputs:

```bash
npx playwright test tests/integration/collect-all-outputs.spec.js
```

**What it generates:**
- `collected-ai-model-outputs.json` - Raw data
- `ai-model-outputs-report.txt` - Detailed report
- `ai-model-outputs-quick-summary.json` - Summary

**Analysis includes:**
- Neural network output detection
- Model-specific output categorization
- Performance metrics
- Error tracking
- Console message analysis

## Testing Workflow

### 1. Individual Component Development

```bash
# 1. Test the specific component you're working on
npx playwright test tests/unit/ai-models.spec.js --grep "YourModel"

# 2. Run related component tests
npx playwright test tests/unit/audio-processing.spec.js

# 3. Test backend compatibility
npx playwright test tests/unit/compute-backends.spec.js --grep "WebNN"
```

### 2. Integration Validation

```bash
# 1. Test component integration
npx playwright test tests/integration/avatar-animation.spec.js

# 2. Collect comprehensive outputs
npx playwright test tests/integration/collect-all-outputs.spec.js

# 3. Analyze results in docs/reports/
```

### 3. Debug and Analysis

```bash
# 1. Run with debug output
npx playwright test --debug tests/unit/your-test.spec.js

# 2. Check generated reports
ls docs/reports/

# 3. Analyze specific issues
npx playwright test --headed tests/integration/collect-all-outputs.spec.js
```

## Test Configuration

### Playwright Configuration

The system uses a custom Playwright configuration optimized for avatar testing:

```javascript
// playwright.config.js highlights
{
  timeout: 300000, // 5 minutes for AI model loading
  retries: 2,
  workers: 1, // Prevent resource conflicts
  use: {
    headless: false, // For visual debugging
    viewport: { width: 1920, height: 1080 },
    ignoreHTTPSErrors: true
  }
}
```

### Test Data

Test data is organized in `data/test-data/`:
- Audio samples for processing tests
- Model configuration files
- Expected output samples
- Performance baselines

## Output Analysis

### Understanding Test Reports

#### Neural Network Detection
The output collection identifies neural network usage through keywords:
- `neural_network_used`, `model_output`, `inference_result`
- `webgpu`, `webnn`, `tensor`, `execution_provider`
- `transformer_blocks`, `attention_weights`

#### Model-Specific Outputs
Categorized by model patterns:
- **Language Models**: TinyLlama, DiabloGPT outputs
- **Audio Models**: Whisper, VAD, Kokoro, SpeechT5 outputs
- **Motion Models**: RSMT, DeepMimic, FaceFormer outputs
- **Compute Models**: WASM matrix operations

#### Performance Metrics
- Model loading times
- Inference latency
- Memory usage patterns
- Error rates and types

### Debug Information

#### Console Output Analysis
- Categorized by log level and timestamp
- Model-specific message filtering
- Error stack trace collection
- Performance timing data

#### Visual Debugging
- Screenshot capture on failures
- Element interaction recording
- Network request monitoring
- Resource loading tracking

## Best Practices

### 1. Test Organization
- Start with unit tests for individual components
- Progress to integration tests
- Use output collection for comprehensive analysis
- Run legacy tests for regression checking

### 2. Debugging Strategy
- Use `--headed` mode for visual debugging
- Enable `--debug` for step-by-step execution
- Check `docs/reports/` for detailed analysis
- Use grep filters for targeted testing

### 3. Performance Testing
- Monitor memory usage during tests
- Track model loading and inference times
- Compare backends for performance
- Use quantization for optimization testing

### 4. Continuous Integration
- Run unit tests frequently during development
- Run integration tests before commits
- Collect outputs for analysis in CI
- Monitor performance regression

## Troubleshooting

### Common Issues

#### Model Loading Failures
```bash
# Test specific backend
npx playwright test tests/unit/compute-backends.spec.js --grep "WebNN"

# Check model paths and availability
npx playwright test tests/unit/ai-models.spec.js --grep "ModelName"
```

#### Performance Issues
```bash
# Collect performance data
npx playwright test tests/integration/collect-all-outputs.spec.js

# Check memory usage
# Review reports in docs/reports/
```

#### Integration Failures
```bash
# Test components individually first
npx playwright test tests/unit/

# Then test integration
npx playwright test tests/integration/
```

### Debug Tools

- **Console Analysis**: Comprehensive console message collection
- **Error Tracking**: Detailed error reporting with stack traces
- **Performance Profiling**: Timing and memory usage analysis
- **Visual Debugging**: Screenshot and interaction recording

## Advanced Testing

### Custom Test Creation

Create new tests following the established patterns:

```javascript
import { test, expect } from '@playwright/test';

test.describe('Your Component', () => {
  test('should test specific functionality', async ({ page }) => {
    // Navigate to test page
    await page.goto('http://localhost:8080/dev/web_viewer/demos/html-tests/your-test.html');
    
    // Test component functionality
    // Collect outputs and analyze results
  });
});
```

### Backend-Specific Testing

Create backend-specific test variations:

```javascript
['webnn', 'webgpu', 'wasm'].forEach(backend => {
  test(`should work with ${backend} backend`, async ({ page }) => {
    // Test with specific backend
  });
});
```
