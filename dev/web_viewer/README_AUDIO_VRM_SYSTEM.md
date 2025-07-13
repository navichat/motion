# VRM Audio-Driven Animation System

This system integrates Audio2Gesture and FaceFormer models to create audio-driven animations for VRM characters in a 3D classroom environment.

## Files Created

### Main Application
- `vrm_test_mesh_audio_animation_classroom.html` - Main HTML file with audio-driven VRM animation system

### Supporting Modules
- `js/AudioProcessor.js` - Audio feature extraction and processing
- `js/AnimationSync.js` - Synchronization between audio and animation

### Updated Web Generators
- `audio2gesture/audio2gesture_web_generator.js` - Browser-compatible Audio2Gesture generator
- `faceformer/faceformer_web_generator.js` - Browser-compatible FaceFormer generator

## Features

### 🎵 Audio Input
- **File Upload**: Load WAV, MP3, or OGG audio files
- **Live Recording**: Record audio directly in the browser using microphone
- **Audio Visualization**: Real-time waveform display
- **Playback Controls**: Play, pause, and stop audio

### 🎭 Gesture Generation (Audio2Gesture)
- **Neural Network**: Uses ONNX model for gesture generation
- **Style Control**: Choose between neutral, expressive, and subtle gesture styles
- **Real-time Processing**: Generate gestures synchronized to audio
- **Performance Metrics**: Track generation speed and frame rates

### 😊 Facial Animation (FaceFormer)
- **Facial Expressions**: Generate facial vertex animations from audio
- **Intensity Control**: Adjust facial expression intensity (0.1x to 2.0x)
- **FLAME Topology**: Compatible with FLAME facial model
- **Synchronized Playback**: Facial animations synchronized with gestures

### 🎬 Animation Playback
- **Synchronized Playback**: Audio and animation perfectly synchronized
- **Real-time Preview**: See animations as they're generated
- **Export Functionality**: Export generated animations as BVH files
- **Loop Playback**: Continuous animation loops

### 🏫 3D Environment
- **VRM Characters**: Full VRM character support with expressions
- **Classroom Scene**: GLB classroom environment with collision detection
- **Optimized Lighting**: VRM-specific lighting for realistic rendering
- **Camera Controls**: Orbit controls for viewing from different angles

## Technical Architecture

### Audio Processing Pipeline
1. **Audio Input** → Browser Web Audio API
2. **Feature Extraction** → MFCC, rhythm, energy analysis
3. **Neural Processing** → ONNX models via onnxruntime-web
4. **Animation Output** → BVH-compatible gesture data

### Synchronization System
- **Frame-accurate Timing**: 30 FPS animation sync
- **Beat Detection**: Automatic beat detection for gesture emphasis
- **Drift Compensation**: Automatic timing adjustments
- **Quality Metrics**: Sync quality monitoring

### Integration Points
- **VRM Adapter**: Converts neural network outputs to VRM bone rotations
- **Expression Manager**: Maps facial outputs to VRM blend shapes
- **Lighting Manager**: Optimized lighting for VRM characters
- **Collision System**: Character movement within classroom bounds

## Usage Instructions

### Getting Started
1. Open `vrm_test_mesh_audio_animation_classroom.html` in a modern browser
2. Wait for the system to initialize (models will load automatically)
3. A default VRM character and classroom will load if available

### Loading Audio
1. **File Upload**: Click "Audio File" and select an audio file
2. **Recording**: Click "🎤 Record" to record audio from microphone
3. **Playback**: Use "▶️ Play" to test audio playback

### Generating Animations
1. **Gestures**: 
   - Select gesture style (neutral/expressive/subtle)
   - Click "Generate Gestures" to create body animations
2. **Facial**:
   - Adjust intensity slider (0.1 to 2.0)
   - Click "Generate Facial" to create facial animations

### Playback and Export
1. **Play Generated**: View the generated animation
2. **Sync Audio**: Play audio and animation synchronized
3. **Export BVH**: Download the animation as a BVH file

## Model Requirements

### Audio2Gesture Model
- File: `audio2gesture/audio2gesture_step_fixed.onnx`
- Input: Audio features [1, 80, 30], Previous motion [1, 48], Lexeme [1, 96], Hidden state [4, 1, 1024]
- Output: New motion [1, 48], Updated hidden state [4, 1, 1024]

### FaceFormer Model
- File: `faceformer/faceformer_core_step.onnx`
- Input: Audio features [1, seq_len, 768], Template [1, 1, vertex_dim], One-hot [1, num_subjects]
- Output: Facial vertices [1, 1, vertex_dim], Updated embeddings

## Browser Compatibility

### Required Features
- **Web Audio API**: For audio processing and playback
- **WebAssembly**: For ONNX model execution
- **WebGL**: For 3D rendering with THREE.js
- **File API**: For file upload and download
- **MediaDevices API**: For microphone recording

### Supported Browsers
- Chrome 90+ (recommended)
- Firefox 85+
- Safari 14+
- Edge 90+

## Performance Considerations

### Audio Processing
- Real-time feature extraction may impact performance on slower devices
- Audio files longer than 30 seconds may take time to process
- Consider chunking long audio files for better responsiveness

### Neural Network Inference
- ONNX models run on WebAssembly (CPU) for broad compatibility
- WebGPU support may be added in future for GPU acceleration
- Memory usage scales with audio length and animation complexity

### 3D Rendering
- VRM character complexity affects rendering performance
- Classroom GLB models should be optimized for web use
- Disable shadows for better performance on mobile devices

## Troubleshooting

### Common Issues

**Models Not Loading**
- Ensure ONNX model files are present in correct directories
- Check browser console for specific error messages
- Verify onnxruntime-web is loading from CDN

**Audio Not Processing**
- Check microphone permissions for recording
- Ensure audio file format is supported (WAV, MP3, OGG)
- Verify Web Audio API support in browser

**Animation Not Synchronized**
- Check that both audio and animation data are loaded
- Verify frame rate settings (default: 30 FPS)
- Look for timing drift in debug panel

**Performance Issues**
- Reduce audio length for faster processing
- Lower animation quality settings if available
- Close other browser tabs to free memory

## Development Notes

### Extending the System
- Add new gesture styles by modifying lexeme encoding
- Implement custom audio feature extractors
- Add new VRM expression mappings
- Create custom lighting presets

### Integration with Other Systems
- The modular design allows easy integration with other applications
- Audio processing can be used independently
- VRM animation system is reusable
- Synchronization system works with any frame-based animation

## Future Enhancements

### Planned Features
- **Real-time Generation**: Live audio-to-animation conversion
- **Custom Training**: User-specific gesture and facial models
- **Multi-character**: Support for multiple VRM characters
- **Advanced Sync**: Music beat tracking and rhythm-aware gestures
- **Cloud Processing**: Server-side model inference for complex animations
- **VR Integration**: VR headset support for immersive experiences

### Technical Improvements
- **WebGPU Support**: GPU acceleration for faster inference
- **Worker Threads**: Background processing to maintain 60 FPS rendering
- **Progressive Loading**: Stream large models for faster startup
- **Caching System**: Cache processed audio features and animations
- **Compression**: Compress animation data for storage and transmission

## Credits

### Technologies Used
- **THREE.js**: 3D rendering and VRM support
- **@pixiv/three-vrm**: VRM character handling
- **ONNX Runtime Web**: Neural network inference
- **Web Audio API**: Audio processing and playback

### Models
- **Audio2Gesture**: Rhythmic Gesticulator implementation
- **FaceFormer**: Speech-driven 3D facial animation
- **VRM Characters**: Virtual avatar standard

### Development
- **Integration**: Custom audio-VRM integration system
- **Synchronization**: Frame-accurate audio-animation sync
- **Web Adaptation**: Browser-compatible neural network deployment
