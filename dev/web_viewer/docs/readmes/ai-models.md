# AI Models Documentation

## Overview

The AI models system provides a comprehensive framework for managing various AI models including language models, audio processing models, and motion generation models with support for multiple compute backends.

## Components

### Core AI Components

#### AIModelJobs.js
Main AI job management system that handles:
- Model loading and initialization
- Job queuing and execution
- Backend switching (WebNN/WebGPU/WASM)
- Performance monitoring

```javascript
import { AIModelJobs } from '../ai/AIModelJobs.js';

const aiJobs = new AIModelJobs({
  backend: 'webnn', // or 'webgpu', 'wasm'
  models: ['tinyLlama', 'whisper', 'audio2gesture']
});
```

#### KNNJobs.js
K-nearest neighbor job system for:
- Similarity search
- Embedding-based retrieval
- Motion pattern matching
- Audio feature matching

```javascript
import { KNNJobs } from '../ai/KNNJobs.js';

const knnJobs = new KNNJobs({
  dimensions: 512,
  algorithm: 'bruteforce'
});
```

### Model Categories

#### Language Models
- **TinyLlama**: Lightweight language model for conversation
- **DiabloGPT**: Dialog-focused language model

#### Audio Processing Models
- **Whisper**: Speech recognition and transcription
- **VAD**: Voice activity detection
- **Kokoro**: Text-to-speech synthesis
- **SpeechT5**: Advanced speech synthesis

#### Motion Models
- **Audio2Gesture**: Audio-driven gesture generation
- **FaceFormer**: Facial animation from audio
- **RSMT**: Real-time stylized motion transition
- **DeepMimic**: Deep reinforcement learning for motion

### Backend Support

#### WebNN Backend
- Optimized for neural network inference
- Hardware acceleration support
- Quantization support

#### WebGPU Backend
- GPU-accelerated computation
- Memory management
- Parallel processing

#### WASM Backend
- Cross-platform compatibility
- CPU-optimized execution
- Deterministic results

## Usage Examples

### Basic Model Loading

```javascript
// Load a language model
const model = await aiJobs.loadModel('tinyLlama', {
  backend: 'webnn',
  quantization: 'int8'
});

// Run inference
const result = await model.generate({
  prompt: "Hello, how are you?",
  maxTokens: 50
});
```

### Audio Processing

```javascript
// Load audio processing models
const whisper = await aiJobs.loadModel('whisper');
const vad = await aiJobs.loadModel('vad');

// Process audio
const transcript = await whisper.transcribe(audioBuffer);
const voiceActivity = await vad.detect(audioBuffer);
```

### Motion Generation

```javascript
// Load motion models
const audio2gesture = await aiJobs.loadModel('audio2gesture');
const faceFormer = await aiJobs.loadModel('faceFormer');

// Generate motion from audio
const gestures = await audio2gesture.generate(audioFeatures);
const facialAnimation = await faceFormer.animate(audioFeatures);
```

## Testing

### Unit Tests

```bash
# Test AI model loading and inference
npx playwright test tests/unit/ai-models.spec.js

# Test specific models
npx playwright test tests/unit/ai-models.spec.js --grep "TinyLlama"
npx playwright test tests/unit/ai-models.spec.js --grep "Whisper"
```

### Output Collection

```bash
# Collect comprehensive AI model outputs
npx playwright test tests/integration/collect-all-outputs.spec.js
```

This generates detailed reports about:
- Model loading times
- Inference performance
- Memory usage
- Error rates
- Output quality metrics

## Performance Optimization

### Backend Selection
Choose the optimal backend based on your requirements:
- **WebNN**: Best for neural network inference, hardware acceleration
- **WebGPU**: Best for parallel processing, large models
- **WASM**: Best for compatibility, deterministic results

### Model Quantization
Reduce model size and improve performance:
```javascript
const model = await aiJobs.loadModel('tinyLlama', {
  quantization: 'int8', // or 'int4', 'fp16'
  backend: 'webnn'
});
```

### Memory Management
Monitor and optimize memory usage:
```javascript
// Check memory usage
const memoryUsage = aiJobs.getMemoryUsage();

// Cleanup unused models
await aiJobs.unloadModel('modelName');
```

## Debugging

### Output Analysis
Use the output collection system to analyze model behavior:
1. Run `collect-all-outputs.spec.js`
2. Check generated reports in `docs/reports/`
3. Analyze model-specific outputs and performance

### Error Handling
Common issues and solutions:
- **Model loading failures**: Check model paths and backend compatibility
- **Inference errors**: Verify input format and model requirements
- **Performance issues**: Consider quantization and backend optimization

### Debug Tools
- Console output collection
- Performance profiling
- Memory usage tracking
- Error reporting and stack traces

## Advanced Features

### Custom Model Integration
Add new models to the system:
1. Create model configuration in `model-configs/`
2. Implement model worker in `model-workers/`
3. Add model patterns to output collection
4. Create unit tests for the new model

### Backend Extension
Implement new compute backends:
1. Create backend implementation in `src/compute/[backend]/`
2. Add backend support to AIModelJobs
3. Create backend-specific tests
4. Update documentation

## Best Practices

1. **Model Loading**: Load models asynchronously and handle errors gracefully
2. **Memory Management**: Unload unused models to free memory
3. **Performance**: Use appropriate quantization and backend selection
4. **Testing**: Test each model individually before integration
5. **Debugging**: Use output collection for comprehensive analysis
