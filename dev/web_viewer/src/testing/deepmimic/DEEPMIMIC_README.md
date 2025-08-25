# DeepMimic JavaScript Porting and Validation

This directory contains the JavaScript port of DeepMimic inference capabilities with comprehensive validation testing to ensure outputs match the original Python TensorFlow implementation.

## 🎯 Overview

This project ports DeepMimic pretrained models from Python/TensorFlow to JavaScript/ONNX Runtime Web, enabling:

- **Cross-platform inference**: Run DeepMimic models in browsers and Node.js
- **Multiple execution providers**: WebGPU, WebNN, and WebAssembly backends
- **Validation framework**: Comprehensive testing to ensure output consistency
- **Performance benchmarking**: Compare inference speeds across platforms

## 📁 Project Structure

```
deepmimic/
├── index.html                    # Web-based validation interface
├── deepmimic-inference.js        # Core inference engine
├── deepmimic-validator.js        # Validation framework
├── validate-deepmimic.js         # Node.js command-line validator
├── package.json                  # Node.js dependencies
├── *.onnx                        # Converted ONNX models
├── validation_results/           # Python reference test data
│   ├── *_test_data.json         # Per-model test cases
│   └── all_models_test_data.json # Combined test data
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (>= 14.0.0) for command-line validation
- **Modern browser** with WebGPU/WebNN support (optional)
- **Python environment** with ONNX models (already converted)

### Installation

1. Install Node.js dependencies:
```bash
cd /home/barberb/motion/dev/web_viewer/web_porting_poc/deepmimic
npm install
```

2. Start a local web server for browser testing:
```bash
npm run serve
# Or manually: python3 -m http.server 8080
```

3. Open the web interface:
```
http://localhost:8080
```

## 🧪 Validation Testing

### Web Interface Testing

1. Open `http://localhost:8080` in your browser
2. Click "Initialize System" to set up the validation framework
3. Click "Detect Providers" to find available execution providers
4. Select a model and execution provider
5. Choose from these testing options:
   - **Single Test**: Validate one model with current settings
   - **Comprehensive Validation**: Test all models across all providers
   - **Benchmark**: Measure inference performance
   - **Compare Providers**: Test output consistency across backends

### Command Line Testing

```bash
# Basic validation with default tolerance (1e-4)
node validate-deepmimic.js . validation_results

# Strict validation with tighter tolerance
npm run validate-strict

# Relaxed validation for quick testing
npm run validate-relaxed

# Custom tolerance
node validate-deepmimic.js . validation_results 1e-5
```

## 🎛️ Execution Providers

### WebAssembly (WASM)
- ✅ **Always available** in all browsers
- 🔧 **CPU-based** execution
- 📊 **Baseline performance** for comparison

### WebGPU
- 🌟 **GPU acceleration** for faster inference
- 🔧 **Chrome/Edge 113+** with experimental flags
- 📊 **Best performance** on supported hardware

### WebNN (Web Neural Network)
- 🚧 **Experimental** browser API
- 🔧 **Hardware-optimized** neural network execution
- 📊 **Future-proof** for next-generation browsers

## 📊 Model Coverage

### Successfully Converted Models (14 total)

All models use **197-dimensional input** (humanoid state) and **36-dimensional output** (action vector):

- `humanoid3d_humanoid3d_backflip` - Backflip motion
- `humanoid3d_humanoid3d_cartwheel` - Cartwheel motion  
- `humanoid3d_humanoid3d_crawl` - Crawling motion
- `humanoid3d_humanoid3d_dance_a` - Dance sequence A
- `humanoid3d_humanoid3d_dance_b` - Dance sequence B
- `humanoid3d_humanoid3d_getup_facedown` - Get up from face down
- `humanoid3d_humanoid3d_getup_faceup` - Get up from face up
- `humanoid3d_humanoid3d_jump` - Jumping motion
- `humanoid3d_humanoid3d_kick` - Kicking motion
- `humanoid3d_humanoid3d_punch` - Punching motion
- `humanoid3d_humanoid3d_roll` - Rolling motion
- `humanoid3d_humanoid3d_run` - Running motion
- `humanoid3d_humanoid3d_spinkick` - Spinning kick motion
- `humanoid3d_humanoid3d_walk` - Walking motion

## 🔬 Validation Methodology

### Test Data Generation

1. **Python Reference**: Load TensorFlow checkpoints and generate test cases
2. **ONNX Conversion**: Convert checkpoints to ONNX format with corrected dimensions
3. **JavaScript Testing**: Run identical inputs through ONNX Runtime Web
4. **Comparison**: Compare outputs with configurable numerical tolerance

### Validation Metrics

- **Pass Rate**: Percentage of test cases within tolerance
- **Maximum Error**: Largest absolute difference between outputs
- **Average Error**: Mean absolute difference across all outputs
- **Inference Time**: Performance measurement per execution provider

### Tolerance Settings

- **Strict** (`1e-6`): Research-grade precision
- **Standard** (`1e-4`): Production-ready precision  
- **Relaxed** (`1e-3`): Quick validation testing

## 📈 Expected Results

### Typical Validation Results

- **Pass Rate**: >99% for most models with standard tolerance
- **WebAssembly**: Consistent baseline performance (~5-10ms per inference)
- **WebGPU**: 2-5x faster than WebAssembly on compatible hardware
- **Cross-Provider Consistency**: <1e-5 difference between execution providers

### Known Issues

1. **Browser Compatibility**: WebGPU requires Chrome 113+ with flags enabled
2. **Model Precision**: Some numerical differences due to float32 precision
3. **File Serving**: ONNX models must be served via HTTP (not file://)

## 🔧 API Reference

### DeepMimicInference Class

```javascript
const inference = new DeepMimicInference();

// Initialize and load model
await inference.initialize();
await inference.loadModel('humanoid3d_humanoid3d_walk.onnx', 'webgpu');

// Run inference
const input = new Float32Array(197); // State vector
const result = await inference.predict(input);
console.log(result.actions); // 36-dimensional action vector

// Benchmark performance
const benchmarkResults = await inference.benchmark(100);
console.log(`Average: ${benchmarkResults.averageTime}ms`);
```

### DeepMimicValidator Class

```javascript
const validator = new DeepMimicValidator();

// Run comprehensive validation
await validator.initialize();
const results = await validator.runComprehensiveValidation(
    'validation_results', 
    ['humanoid3d_humanoid3d_walk', 'humanoid3d_humanoid3d_run']
);

// Export results
validator.exportResults('validation_results.json');
```

## 🚀 Performance Optimization

### Tips for Best Performance

1. **Use WebGPU** when available for GPU acceleration
2. **Batch Processing**: Process multiple inputs together when possible
3. **Model Caching**: Load models once and reuse sessions
4. **Warm-up Runs**: Execute a few predictions before benchmarking

### Expected Inference Times

| Execution Provider | Typical Range | Best Case |
|-------------------|---------------|-----------|
| WebAssembly       | 5-15ms        | 3ms       |
| WebGPU            | 1-5ms         | 0.5ms     |
| WebNN             | 2-8ms         | 1ms       |

## 🐛 Troubleshooting

### Common Issues

**Models not loading**:
- Ensure files are served via HTTP server (not file://)
- Check browser console for CORS errors
- Verify ONNX model files are accessible

**WebGPU not available**:
- Enable experimental WebGPU flag in Chrome: `chrome://flags/#enable-unsafe-webgpu`
- Use Chrome/Edge 113+ or Firefox Nightly
- Fallback to WebAssembly automatically

**Validation failures**:
- Check tolerance settings (try relaxed mode first)
- Verify test data files are accessible
- Check model input/output dimensions match expectations

**Performance issues**:
- Try different execution providers
- Check for browser extensions blocking WebGPU
- Monitor browser DevTools performance tab

## 📚 References

- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/get-started/with-javascript.html)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebNN API Draft](https://www.w3.org/TR/webnn/)
- [DeepMimic Original Paper](https://arxiv.org/abs/1804.02717)

## 🤝 Contributing

1. Test new models by adding them to the model list
2. Improve execution provider detection and fallback logic
3. Add new validation metrics and benchmark tests
4. Optimize performance for specific hardware configurations

## 📄 License

This project is part of the larger motion repository and follows the same license terms.
