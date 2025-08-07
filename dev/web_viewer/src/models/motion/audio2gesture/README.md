# Audio2Gesture Web Porting

This folder contains files related to porting the Audio2Gesture model (Audio-to-Body Motion) to web deployment using ONNX Runtime Web, WebNN, and WebGPU.

## 🎯 Status: 🔧 IN DEVELOPMENT

Building on the successful FaceFormer porting, this will implement full-body gesture generation from audio.

## 📁 Current File Structure

### Existing Files (from previous work)
- `export_model.py` - Export script for MotionGenerator_RNN model
- `motion_generator.onnx` - Exported ONNX model (~50MB)
- `sample_data.json` - Sample input/output data for testing
- `test_model.test.js` - Jest test suite for ONNX model validation
- `test_model_simple.js` - Simple JavaScript test for model loading

## 🏗️ Audio2Gesture System Architecture

Based on the codebase analysis, the Audio2Gesture system has multiple components:

### 1. **Audio Processing Pipeline**
- **Audio2Pace**: Converts audio to "pace" representations (rhythm/timing features)
- **Input**: Raw audio (16kHz sampling rate)
- **Output**: Pace codes for gesture timing

### 2. **Gesture Generation Models**
- **Streaming Engine**: Real-time gesture generation for live applications
- **Offline Engine**: High-quality batch processing with global optimization
- **Model Types**: TensorRT models (.trt files) for inference

### 3. **Animation Pipeline**
- **Skeleton System**: 92-joint full-body skeleton
- **Style Codes**: Different gesture styles (Neutral, etc.)
- **Output**: 3D translations and quaternion rotations per joint

### 4. **Motion Matching & Stitching**
- **MatchingGenerator**: Finds best matching animation segments
- **Stitching**: Smooth transitions between animation pieces
- **Global Optimization**: Multi-option generation with quality ranking

## 🎯 Web Porting Strategy

### Phase 1: Model Analysis & Export ✅
- [x] Analyze existing PyTorch/TensorRT models
- [x] Export MotionGenerator_RNN to ONNX
- [x] Basic JavaScript loading test

### Phase 2: Core Components (Current Focus)
- [ ] **Audio2Pace Web Implementation**
  - Port audio feature extraction to Web Audio API
  - Export pace generation model to ONNX
  - Implement real-time audio processing

- [ ] **Gesture Generation Core**
  - Export main gesture generation model
  - Handle autoregressive generation (similar to FaceFormer)
  - Implement fixed-size buffer approach for web compatibility

- [ ] **Motion Data Management**
  - Port skeleton data and style mappings
  - Implement motion database for web
  - Optimize data loading for browser constraints

### Phase 3: Advanced Features
- [ ] **Real-time Streaming**
  - Implement streaming gesture generation
  - Add frame-by-frame animation output
  - Optimize for low-latency performance

- [ ] **Style Control**
  - Multiple gesture styles (Neutral, Expressive, etc.)
  - Dynamic style switching during generation
  - User-configurable style parameters

- [ ] **3D Integration**
  - Three.js integration for real-time rendering
  - Bone animation and skinning
  - Camera controls and environment setup

## 🔧 Technical Challenges & Solutions

### Challenge 1: Large Model Size
**Problem**: TensorRT models are optimized for server deployment
**Solution**: 
- Convert to ONNX with quantization (FP16/INT8)
- Split large models into smaller chunks
- Progressive loading for web deployment

### Challenge 2: Real-time Performance
**Problem**: 20+ FPS generation with 92 joints
**Solution**:
- WebNN/WebGPU acceleration
- Batched processing
- Predictive pre-generation

### Challenge 3: Motion Database Size
**Problem**: Large animation datasets (Corvallis dataset)
**Solution**:
- Compressed motion representations
- Streaming motion data
- On-demand loading of motion segments

### Challenge 4: Autoregressive Generation
**Problem**: Similar to FaceFormer - dynamic sequence lengths
**Solution**: Apply the same fixed-buffer approach proven successful with FaceFormer

## 📊 Model Specifications

### Current MotionGenerator_RNN
- **Input**: Audio features, Motion features, Landmark features
- **Output**: Body joint positions and rotations
- **Architecture**: LSTM-based recurrent neural network
- **Sequence Length**: Variable (100+ frames typical)

### Target Web Architecture
```javascript
// Proposed web implementation structure
class Audio2GestureWebGenerator {
    async initialize() {
        this.audioPaceModel = await loadONNX('audio2pace.onnx');
        this.gestureModel = await loadONNX('gesture_generator.onnx');
        this.motionData = await loadMotionDatabase();
    }
    
    async generateGestures(audioBuffer, style = 'neutral') {
        const paceFeatures = await this.audioPaceModel.run({ audio: audioBuffer });
        const gestures = await this.gestureModel.run({ 
            pace: paceFeatures, 
            style: style 
        });
        return this.postProcessGestures(gestures);
    }
}
```

## 🎮 Integration Examples

### Web Application Usage
```javascript
// Initialize the generator
const gestureGen = new Audio2GestureWebGenerator();
await gestureGen.initialize();

// Process audio file
const audioFile = await loadAudioFile('speech.wav');
const gestures = await gestureGen.generateGestures(audioFile, 'expressive');

// Render with Three.js
const scene = new THREE.Scene();
const avatar = new SkeletalAvatar(gestures.skeleton);
gestures.frames.forEach((frame, i) => {
    setTimeout(() => avatar.updatePose(frame), i * 50); // 20 FPS
});
```

### Real-time Streaming
```javascript
// Live audio processing
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        const gestureStream = new Audio2GestureStream();
        gestureStream.onGesture = (frame) => {
            avatar.updatePose(frame);
        };
        gestureStream.connectAudio(stream);
    });
```

## 📝 Development Roadmap

### Week 1-2: Foundation
- [ ] Analyze and export Audio2Pace model
- [ ] Create Web Audio API preprocessing pipeline
- [ ] Port skeleton data and basic animation structures

### Week 3-4: Core Generation
- [ ] Export main gesture generation model with fixed-size approach
- [ ] Implement JavaScript autoregressive generation
- [ ] Add motion matching and stitching algorithms

### Week 5-6: Integration & Optimization
- [ ] Three.js skeletal animation integration
- [ ] WebNN/WebGPU optimization
- [ ] Performance testing and tuning

### Week 7-8: Advanced Features
- [ ] Multiple style support
- [ ] Real-time streaming implementation
- [ ] User interface and controls

## 🧪 Testing Strategy

### Unit Tests
- Audio processing pipeline validation
- Model output comparison (Python vs JavaScript)
- Gesture quality metrics

### Integration Tests  
- End-to-end audio-to-animation pipeline
- Real-time performance benchmarks
- Cross-browser compatibility

### Quality Assurance
- Motion smoothness validation
- Audio-gesture synchronization accuracy
- Style consistency across different inputs

## 🚀 Success Metrics

- **Latency**: <100ms audio-to-gesture generation
- **Quality**: Smooth, natural-looking gestures
- **Performance**: 20+ FPS real-time generation
- **Compatibility**: Works across modern browsers
- **File Size**: <200MB total including motion data

---

This Audio2Gesture web porting builds directly on the successful FaceFormer implementation, providing a complete audio-to-animation pipeline for web applications! 🎭
