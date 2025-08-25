# Component Testing Guide

This guide explains how to test individual components of the WebNN/WebGPU/WASM avatar system using the newly organized structure.

## 🎯 Overview

With the complete reorganization, each component can now be tested individually, making it easier to:
- Debug specific functionality
- Optimize performance for WebNN/WebGPU
- Validate WASM integration
- Test individual motion models
- Verify VRM avatar components

## 🗂️ Component Categories

### 1. Motion Models (`src/models/motion/`)

#### Audio2Gesture Testing
```bash
# Location: src/models/motion/audio2gesture/
# Key Files:
- Audio2GestureBVHConverter.js     # Main converter
- audio2gesture_step_fixed.onnx    # Neural network model
- audio2gesture_bundle.js          # Web bundle
- test_*.js                        # Test files

# Test Individual Components:
open src/models/motion/audio2gesture/test_webgpu_webnn.js
open src/models/motion/audio2gesture/benchmark.js
```

#### RSMT Testing
```bash
# Location: src/models/motion/rsmt/
# Key Files:
- RSMTBVHConverter.js              # Main converter
- deepphase.onnx                   # Phase encoding model
- stylevae.onnx                    # Style encoding model
- transitionnet.onnx               # Transition generation

# Test Components:
open src/models/motion/rsmt/test-complete-pipeline.js
open src/models/motion/rsmt/rsmt-validator.js
```

#### DeepMimic Testing
```bash
# Location: src/models/motion/deepmimic/
# Key Files:
- DeepMimicBVHConverter.js         # Main converter
- compatible_humanoid3d_*.onnx     # Motion models
- deepmimic-validator.js           # Validation

# Test Humanoid Animations:
open src/models/motion/deepmimic/validation_demo.html
open src/models/motion/deepmimic/deepmimic_demo.html
```

#### FaceFormer Testing
```bash
# Location: src/models/motion/faceformer/
# Key Files:
- FaceFormerBVHConverter.js        # Main converter
- faceformer_core_step.onnx        # Core model
- faceformer_working_generator.js  # Generator

# Test Facial Animation:
open src/models/motion/faceformer/full_faceformer_demo.html
open src/models/motion/faceformer/test_comparison.html
```

### 2. Animation Components (`src/components/animation/`)

#### VRM Avatar System Testing
```bash
# Location: src/components/animation/vrm/
# 25 Files Including:
- VRMBVHAdapter.js                 # Core VRM integration
- conversation/                    # Conversation interfaces
- diagnostics/                     # Debug tools

# Test VRM Components:
cd src/components/animation/vrm/
# Each file can be tested individually
```

#### Timeline System Testing
```bash
# Location: src/components/animation/timeline/
# Key Files:
- BVHTimeline.js                   # Main timeline
- TimelineIntegration.js           # Integration layer

# Test Timeline:
cd src/components/animation/timeline/
```

### 3. Testing Infrastructure (`src/testing/`)

#### Organized Test Categories
```bash
# End-to-End Tests
cd src/testing/e2e/
npx playwright test

# Unit Tests
cd src/testing/unit/
npm test

# Integration Tests  
cd src/testing/integration/
npm run test:integration

# Performance Tests
cd src/testing/performance/
npm run benchmark

# Demo Applications
cd src/testing/demos/
python3 -m http.server 8000
```

### 4. Utilities (`src/utils/`)

#### Audio Processing (Kokoro.js)
```bash
# Location: src/utils/kokoro.js/
# Audio synthesis and processing utilities

cd src/utils/kokoro.js/
npm test
```

#### Web Workers
```bash
# Location: src/utils/workers/
# Background processing for neural inference

cd src/utils/workers/
# Test individual worker files
```

## 🧪 Testing Workflows

### Individual Component Testing
```bash
# 1. Test a single motion model
cd src/models/motion/audio2gesture/
python3 -m http.server 8000
open http://localhost:8000/test_webgpu_webnn.js

# 2. Test VRM integration
cd src/components/animation/vrm/
# Import and test individual components

# 3. Test utilities
cd src/utils/kokoro.js/
npm run test
```

### Integration Testing
```bash
# Test multiple components together
cd src/testing/integration/
npx playwright test complete-system-test.spec.js
```

### Performance Testing
```bash
# Benchmark individual components
cd src/testing/performance/
node benchmark-audio2gesture.js
node benchmark-rsmt.js
node benchmark-deepmimic.js
```

## 🚀 WebNN/WebGPU/WASM Testing

### WebNN Testing
```javascript
// Test neural network acceleration
// Location: src/models/motion/*/test_webnn.js
const session = await ort.InferenceSession.create(modelPath, {
  executionProviders: ['webnn']
});
```

### WebGPU Testing
```javascript
// Test GPU acceleration
// Location: src/models/motion/*/test_webgpu.js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
```

### WASM Testing
```javascript
// Test WebAssembly modules
// Location: src/utils/*/test_wasm.js
const wasmModule = await WebAssembly.instantiateStreaming(
  fetch('module.wasm')
);
```

## 📊 Testing Results

### Expected Outputs
- **Motion Models**: BVH animation files
- **VRM Components**: 3D avatar animations
- **Performance Tests**: Benchmark results in JSON
- **Integration Tests**: Multi-component validation reports

### Validation Criteria
- ✅ Individual components load without errors
- ✅ Neural networks inference correctly
- ✅ WebNN/WebGPU acceleration works
- ✅ WASM modules execute properly
- ✅ Components integrate successfully

## 🛠️ Debugging Tips

### Component-Specific Debugging
1. **Check Console**: Each component has detailed logging
2. **Validate Models**: Use validator files for each motion model
3. **Test Incrementally**: Start with individual components, then integrate
4. **Performance Monitor**: Use browser dev tools for WebNN/WebGPU profiling

### Common Issues
- **Import Paths**: Verify ES6 module paths after reorganization
- **Model Loading**: Check ONNX model file paths
- **Worker Contexts**: Validate web worker script paths
- **CORS Issues**: Use proper HTTP server for local testing

## 📚 Additional Resources

- **[Main README](README.md)**: Complete system overview
- **[Component Documentation](../src/)**: Individual component docs
- **[Performance Guide](PERFORMANCE.md)**: Optimization techniques
- **[API Reference](API.md)**: Component interfaces

---

This organized structure enables systematic testing of each avatar system component individually or in combination, supporting comprehensive WebNN/WebGPU/WASM validation.
