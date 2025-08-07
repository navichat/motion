# FaceFormer BVH Timeline Integration Guide

## Overview

The FaceFormer BVH Converter bridges the gap between FaceFormer neural network (audio-to-facial animation) and the BVH Timeline system, enabling real-time audio-driven facial animation in your avatar system.

## Components Created

### 1. FaceformerBVHConverter.js
- **Location**: `/dev/web_viewer/js/FaceformerBVHConverter.js`
- **Purpose**: Converts FaceFormer neural network outputs to BVH bone transformations
- **Key Features**:
  - ONNX Runtime integration for FaceFormer model
  - Audio preprocessing (16kHz conversion, feature extraction)
  - Vertex-to-bone mapping for facial animation
  - Timeline integration methods

### 2. Enhanced Demo Interface
- **Location**: `/dev/web_viewer/faceformer_timeline_demo_enhanced.html`
- **Purpose**: Comprehensive demo showing audio-to-animation pipeline
- **Features**:
  - Audio file loading and processing
  - Real-time waveform visualization
  - Timeline track visualization
  - Performance statistics
  - Configuration controls

## Quick Start

### 1. Basic Integration

```javascript
// Initialize the converter
const faceformerConverter = new FaceformerBVHConverter({
    faceformerPath: '../faceformer/faceformer_simple_step.onnx',
    framerate: 30,
    maxSeqLen: 20
});

await faceformerConverter.initialize();

// Initialize timeline
const timeline = new BVHTimeline({
    framerate: 30,
    enableRealTime: true
});
```

### 2. Process Audio to Animation

```javascript
// Load audio file
const audioBuffer = await loadAudioFile('speech.wav');

// Generate BVH animation from audio
const animationData = await faceformerConverter.generateBVHFromAudio(
    audioBuffer, 
    { 
        subjectId: 0,
        intensity: 1.0,
        maxFrames: Math.ceil(audioBuffer.duration * 30)
    }
);

// Create timeline clip
const facialClip = faceformerConverter.createTimelineClip(
    audioBuffer,
    0, // Start time
    { weight: 1.0, intensity: 1.0 }
);

// Add to timeline
const trackId = timeline.addTrack('facial', { priority: 10 });
timeline.addClip(trackId, facialClip);
```

### 3. Real-time Playback

```javascript
// Start timeline playback
timeline.play();

// Animation loop
function animate() {
    const currentTime = timeline.getCurrentTime();
    const frame = timeline.getFrameAtTime(currentTime);
    
    if (frame && frame.motionData) {
        // Apply BVH data to your avatar system
        applyBVHToAvatar(frame.motionData);
    }
    
    requestAnimationFrame(animate);
}

animate();
```

## Architecture Overview

```
Audio Input
    ↓
Audio Processing (16kHz, features)
    ↓
FaceFormer Neural Network (ONNX)
    ↓
Facial Mesh Vertices (15069-dim)
    ↓
Vertex-to-Bone Mapping
    ↓
BVH Bone Transformations
    ↓
Timeline Integration
    ↓
VRM Avatar Animation
```

## Key Features

### Audio Processing Pipeline
- **Input**: Audio files (MP3, WAV, etc.)
- **Preprocessing**: Resampling to 16kHz, feature extraction
- **Output**: Audio features compatible with FaceFormer

### Neural Network Integration
- **Model**: FaceFormer ONNX model (`faceformer_simple_step.onnx`)
- **Input**: Audio features + sequence buffer + template mesh
- **Output**: Facial mesh vertex positions (autoregressive generation)

### BVH Conversion
- **Landmark Extraction**: Key facial points from mesh vertices
- **Bone Mapping**: Convert landmarks to bone rotations
- **BVH Format**: Standard motion data format for timeline

### Timeline Integration
- **Multi-track**: Supports multiple animation sources
- **Blending**: Weight-based mixing with other animations
- **Real-time**: Frame-by-frame generation for live performance

## Configuration Options

### FaceformerBVHConverter Options
```javascript
{
    faceformerPath: 'path/to/model.onnx',  // Model file location
    framerate: 30,                         // Output framerate
    maxSeqLen: 20,                        // Sequence buffer length
    vertexCount: 5023,                    // Expected vertex count
    sampleRate: 16000                     // Audio sample rate
}
```

### Timeline Clip Options
```javascript
{
    weight: 1.0,           // Animation intensity
    blendMode: 'additive', // How to blend with other tracks
    intensity: 1.0,        // Scale factor for animations
    subjectId: 0          // FaceFormer subject variant
}
```

## Performance Characteristics

### Typical Performance (on modern hardware):
- **Audio Processing**: ~50ms per second of audio
- **Neural Inference**: ~200ms per frame (autoregressive)
- **BVH Conversion**: ~10ms per frame
- **Memory Usage**: ~100MB for model + buffers

### Optimization Tips:
1. Use WebGL execution provider for ONNX Runtime
2. Process audio in chunks for long sequences
3. Cache generated frames for repeated playback
4. Use lower framerates (15-20 FPS) for real-time performance

## Integration with Existing Systems

### With VRM Avatar System
```javascript
// Connect timeline to VRM adapter
const vrmIntegration = new BVHTimelineVRMIntegration(timeline, vrmAdapter);

// Enable real-time mode for live performance
vrmIntegration.enableRealTimeMode({
    targetFPS: 30,
    smoothing: true,
    smoothingFactor: 0.8
});
```

### With Multiple Animation Sources
```javascript
// Add multiple tracks for different animation types
const bodyTrack = timeline.addTrack('body', { priority: 5 });
const facialTrack = timeline.addTrack('facial', { priority: 10 }); // Higher priority
const gestureTrack = timeline.addTrack('gestures', { priority: 7 });

// FaceFormer facial animation will override lower priority tracks
```

## Troubleshooting

### Common Issues:

1. **Model Loading Fails**
   - Check ONNX model path
   - Verify ONNX Runtime is loaded
   - Ensure model is compatible version

2. **Audio Processing Errors**
   - Verify audio file format support
   - Check Web Audio API compatibility
   - Ensure proper sample rate conversion

3. **Performance Issues**
   - Try lower framerates
   - Use batch processing instead of real-time
   - Check WebGL/GPU acceleration

4. **Animation Quality**
   - Adjust intensity and weight parameters
   - Try different subject IDs
   - Verify bone mapping configuration

## File Structure

```
/dev/web_viewer/
├── js/
│   └── FaceformerBVHConverter.js     # Main converter class
├── BVHTimeline.js                    # Timeline system (existing)
├── BVHTimelineVRMIntegration.js     # VRM integration (existing)
├── faceformer_timeline_demo_enhanced.html  # Enhanced demo
└── faceformer/                       # FaceFormer model files
    ├── faceformer_simple_step.onnx   # ONNX model
    ├── faceformer_simple_generator.js # Generator class
    └── full_faceformer_web.js        # Full implementation
```

## Next Steps

1. **Test with Real Audio**: Try the demo with actual speech audio files
2. **Performance Tuning**: Optimize for your target hardware
3. **Custom Bone Mapping**: Adjust vertex-to-bone mapping for your avatar
4. **Integration**: Connect to your existing VRM/avatar system
5. **Real-time Streaming**: Add microphone input for live performance

## API Reference

### Main Classes

#### FaceformerBVHConverter
- `initialize()` - Load model and setup processing
- `generateBVHFromAudio(audioBuffer, options)` - Convert audio to BVH
- `createTimelineClip(audioBuffer, startTime, options)` - Create timeline-compatible clip
- `getStats()` - Get performance statistics

#### Timeline Integration
- Works with existing `BVHTimeline` class
- Compatible with `BVHTimelineVRMIntegration`
- Supports all timeline blending modes

This integration provides a complete pipeline from audio input to avatar facial animation, seamlessly connecting FaceFormer's neural network capabilities with your existing BVH Timeline system.
