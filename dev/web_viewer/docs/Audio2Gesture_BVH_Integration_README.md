# Audio2Gesture BVH Integration Documentation

## Overview

The Audio2Gesture BVH Integration system provides a complete pipeline for converting audio input into realistic body gesture animations using the BVH (Biovision Hierarchy) format. This system integrates Audio2Gesture neural networks with the BVH Timeline Compositor to create synchronized, multi-layered body animation from speech or music.

## Features

### Core Capabilities
- **Audio Feature Extraction**: Comprehensive audio analysis including amplitude, spectral features, rhythm, pitch, and prosody
- **Neural Network Integration**: Support for Audio2Gesture models with fallback mock implementation
- **Full Body Animation**: Complete skeleton support with 34+ bones including spine, arms, hands, fingers, and legs
- **Real-time Processing**: Live audio processing with low-latency gesture generation
- **Emotional Context**: Gesture adaptation based on detected emotional states
- **Timeline Integration**: Seamless integration with BVH Timeline Compositor for multi-track animation

### Gesture Features
- **Gesture Patterns**: Pre-defined gesture styles for different emotional contexts (neutral, excited, calm, emphatic, questioning)
- **Intensity Control**: Adjustable gesture magnitude based on audio characteristics
- **Smoothing System**: Advanced interpolation and constraint systems for natural movement
- **Body Part Separation**: Independent control for upper body, lower body, and hand gestures
- **Multi-format Support**: SMPL, pose matrices, joint rotations, and coordinate-based input formats

## Architecture

### Core Components

#### 1. Audio2GestureBVHConverter
Converts Audio2Gesture neural network output to BVH animation data.

```javascript
const converter = new Audio2GestureBVHConverter({
    scaleFactor: 1.0,
    smoothing: true,
    smoothingFactor: 0.7,
    gestureIntensity: 1.0,
    emotionalModulation: true,
    enableLowerBody: true,
    enableFingers: true
});
```

#### 2. Audio2GestureTimelineIntegration
Integrates Audio2Gesture with the BVH Timeline system.

```javascript
const integration = new Audio2GestureTimelineIntegration(timeline, {
    realtime: true,
    trackName: 'audio2gesture_body',
    priority: 80,
    channels: ['body', 'arms', 'hands']
});
```

#### 3. Audio2GestureAudioProcessor
Handles comprehensive audio feature extraction.

```javascript
const processor = new Audio2GestureAudioProcessor({
    sampleRate: 16000,
    frameSize: 1024,
    hopLength: 512,
    melBins: 80
});
```

## Quick Start

### Basic Setup

```javascript
// 1. Create timeline
const timeline = new BVHTimelineCompositor({
    frameRate: 30,
    maxTracks: 15
});

// 2. Create Audio2Gesture integration
const audio2gesture = new Audio2GestureTimelineIntegration(timeline, {
    realtime: false,
    trackName: 'body_gestures',
    priority: 80,
    channels: ['body', 'arms', 'hands']
});

// 3. Initialize model
await audio2gesture.initializeAudio2Gesture('/path/to/model');

// 4. Set up synchronization
audio2gesture.synchronizeWithTimeline();

// 5. Process audio
const audioData = new Float32Array(48000); // 3 seconds at 16kHz
const result = await audio2gesture.processAudio(audioData, {
    startTime: 0,
    emotionalContext: 'neutral'
});

// 6. Play timeline
timeline.play();
```

### HTML Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>Audio2Gesture Demo</title>
</head>
<body>
    <script src="BVHTimelineCompositor.js"></script>
    <script src="Audio2GestureBVHConverter.js"></script>
    <script src="Audio2GestureTimelineIntegration.js"></script>
    
    <script>
        // Your Audio2Gesture implementation
    </script>
</body>
</html>
```

## Audio Processing

### Supported Audio Features

The system extracts comprehensive audio features for gesture generation:

#### Time-Domain Features
- **Amplitude**: Peak audio level for gesture intensity
- **RMS Energy**: Root mean square energy for movement dynamics
- **Zero Crossing Rate**: Speech/music classification
- **Short-Time Energy**: Temporal energy variations

#### Spectral Features
- **Spectral Centroid**: Brightness/timbre for gesture style
- **Spectral Rolloff**: Frequency distribution for movement range
- **Spectral Flux**: Spectral change rate for gesture timing
- **Mel-scale Spectrogram**: Perceptual frequency analysis

#### Rhythm Features
- **Tempo Estimation**: Beat detection for gesture synchronization
- **Beat Strength**: Rhythmic regularity measurement
- **Rhythm Regularity**: Consistency of beat patterns

#### Pitch Features
- **Fundamental Frequency**: Base pitch for gesture coordination
- **Pitch Estimation**: MIDI note conversion
- **Pitch Variability**: Melodic range for expression

#### Prosodic Features
- **Loudness**: Perceptual volume for gesture magnitude
- **Prosody**: Speech rhythm and stress patterns

### Audio Input Formats

```javascript
// Float32Array (preferred)
const audioData = new Float32Array(sampleCount);

// AudioBuffer from Web Audio API
const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
const audioData = audioBuffer.getChannelData(0);

// With metadata
const audioInput = {
    audio: audioData,
    sampleRate: 16000,
    duration: 3.0,
    emotionalContext: 'excited'
};
```

## Gesture Generation

### Emotional Contexts

The system supports different emotional contexts that affect gesture style:

#### Neutral
- Balanced, natural movements
- Moderate gesture intensity
- Regular timing patterns

```javascript
await audio2gesture.processAudio(audioData, {
    emotionalContext: 'neutral'
});
```

#### Excited
- Large, dynamic gestures
- Increased movement speed
- Higher gesture intensity

```javascript
await audio2gesture.processAudio(audioData, {
    emotionalContext: 'excited'
});
```

#### Calm
- Smooth, flowing movements
- Reduced gesture intensity
- Slower timing

```javascript
await audio2gesture.processAudio(audioData, {
    emotionalContext: 'calm'
});
```

#### Emphatic
- Sharp, pronounced gestures
- Accent-synchronized movements
- High contrast timing

```javascript
await audio2gesture.processAudio(audioData, {
    emotionalContext: 'emphatic'
});
```

#### Questioning
- Hesitant, uncertain movements
- Upward gesture tendencies
- Variable timing patterns

```javascript
await audio2gesture.processAudio(audioData, {
    emotionalContext: 'questioning'
});
```

### Gesture Intensity Control

Adjust the overall magnitude of generated gestures:

```javascript
// Set gesture intensity (0.0 to 2.0)
audio2gesture.setGestureIntensity(1.5);

// Or per-processing
await audio2gesture.processAudio(audioData, {
    gestureIntensity: 0.8
});
```

### Body Part Control

Enable or disable specific body parts:

```javascript
const integration = new Audio2GestureTimelineIntegration(timeline, {
    converter: {
        enableLowerBody: true,   // Include legs and feet
        enableFingers: true,     // Include finger gestures
        enableFacial: false     // Exclude facial animation
    }
});
```

## Timeline Integration

### Track Configuration

Audio2Gesture creates specialized tracks for different body parts:

```javascript
// Main body track
trackName: 'audio2gesture_body'
priority: 80
channels: ['body', 'arms', 'hands']
blending: 'additive'

// Upper body track
trackName: 'audio2gesture_upper'
priority: 85
channels: ['arms', 'hands', 'spine', 'head']
blending: 'override'

// Hand gesture tracks
trackName: 'audio2gesture_left_hand'
trackName: 'audio2gesture_right_hand'
priority: 90
channels: ['left_hand', 'right_hand', 'fingers']
blending: 'override'
```

### Timeline Synchronization

```javascript
// Set up automatic synchronization
audio2gesture.synchronizeWithTimeline();

// Handle custom frame processing
audio2gesture.onAudio2GestureFrame = (frame, time) => {
    // Custom processing for each frame
    console.log('Audio2Gesture frame:', frame);
};
```

### Multi-Track Processing

```javascript
// Process with track separation
await audio2gesture.processAudio(audioData, {
    separateUpperLower: true,  // Create separate upper/lower body tracks
    separateHands: true,       // Create separate hand tracks
    startTime: 5.0            // Start at 5 seconds in timeline
});
```

## Real-Time Processing

### Live Audio Processing

```javascript
// Enable real-time mode
audio2gesture.setRealtime(true);

// Process live audio chunks
audio2gesture.processLiveAudio(audioChunk, timestamp);
```

### Performance Optimization

```javascript
// Adjust processing parameters for real-time
const integration = new Audio2GestureTimelineIntegration(timeline, {
    realtime: true,
    converter: {
        smoothing: true,        // Enable for quality
        smoothingFactor: 0.5,   // Reduce for speed
        cacheFrames: true       // Enable caching
    }
});
```

## Batch Processing

### Multiple Audio Segments

```javascript
const audioSegments = [
    {
        audio: audioData1,
        duration: 3.0,
        emotionalContext: 'neutral'
    },
    {
        audio: audioData2,
        duration: 2.5,
        emotionalContext: 'excited'
    }
];

const results = await audio2gesture.processBatchAudio(audioSegments, {
    startTime: 0,
    separateUpperLower: false
});
```

## BVH Output Format

### Bone Structure

The system generates BVH data for a complete body skeleton:

```
Root (Hips)
├── Spine
│   ├── Spine1
│   │   ├── Spine2
│   │   │   ├── Neck
│   │   │   │   └── Head
│   │   │   ├── LeftShoulder
│   │   │   │   ├── LeftArm
│   │   │   │   │   ├── LeftForeArm
│   │   │   │   │   │   └── LeftHand
│   │   │   │   │   │       ├── LeftHandThumb1-3
│   │   │   │   │   │       ├── LeftHandIndex1-3
│   │   │   │   │   │       ├── LeftHandMiddle1-3
│   │   │   │   │   │       ├── LeftHandRing1-3
│   │   │   │   │   │       └── LeftHandPinky1-3
│   │   │   │   │   └── (similar for right arm)
│   │   │   └── RightShoulder...
├── LeftUpLeg
│   ├── LeftLeg
│   │   ├── LeftFoot
│   │   │   └── LeftToeBase
└── RightUpLeg...
```

### Frame Data Structure

Each BVH frame contains rotation data for all bones:

```javascript
{
    frameNumber: 42,
    timestamp: 1400, // milliseconds
    bones: {
        'Hips': {
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0.1, y: -0.05, z: 0.02 }
        },
        'Spine': {
            rotation: { x: 0.15, y: 0.08, z: -0.03 }
        },
        // ... other bones
    },
    metadata: {
        source: 'audio2gesture',
        emotionalContext: 'neutral',
        audioFeatures: { ... }
    }
}
```

## Advanced Features

### Custom Gesture Patterns

Define custom gesture patterns for specific contexts:

```javascript
const customPattern = {
    name: 'presentation',
    baseIntensity: 1.2,
    preferredBones: ['LeftArm', 'RightArm', 'Spine'],
    gestureStyle: {
        armMovement: 'wide',
        handPositioning: 'open',
        spineMovement: 'upright'
    },
    timing: {
        beatSync: true,
        accentResponse: 'strong'
    }
};

audio2gesture.converter.addGesturePattern(customPattern);
```

### Audio Feature Customization

```javascript
const processor = new Audio2GestureAudioProcessor({
    sampleRate: 16000,
    frameSize: 2048,        // Larger for more frequency detail
    hopLength: 256,         // Smaller for more temporal detail
    melBins: 128,          // More frequency bands
    enablePitchTracking: true,
    enableBeatTracking: true
});
```

### Smoothing Configuration

```javascript
const converter = new Audio2GestureBVHConverter({
    smoothing: true,
    smoothingFactor: 0.8,          // Higher = smoother
    maxRotationChange: 0.5,        // Limit per-frame rotation change
    constraintSystem: 'anatomical', // Enforce realistic joint limits
    adaptiveSmoothing: true        // Adjust smoothing based on gesture type
});
```

## Performance Monitoring

### Statistics Collection

```javascript
const stats = audio2gesture.getStats();
console.log('Performance Stats:', {
    framesProcessed: stats.framesProcessed,
    averageLatency: stats.averageLatency,
    modelInferenceTime: stats.modelInferenceTime,
    conversionTime: stats.conversionTime,
    audioProcessingTime: stats.audioProcessingTime,
    isModelLoaded: stats.isModelLoaded,
    isMockModel: stats.isMockModel
});
```

### Performance Optimization

```javascript
// Optimize for real-time performance
audio2gesture.setRealtime(true);

// Adjust cache settings
timeline.setOptions({
    cacheFrames: true,
    maxCacheSize: 1000
});

// Monitor queue size
const queueSize = audio2gesture.getStats().queueSize;
if (queueSize > 10) {
    console.warn('Processing queue backing up');
}
```

## Error Handling

### Graceful Degradation

```javascript
try {
    await audio2gesture.processAudio(audioData);
} catch (error) {
    if (error.message.includes('model not loaded')) {
        console.warn('Using fallback gesture generation');
        // Fallback to procedural gestures
    } else {
        console.error('Audio processing failed:', error);
    }
}
```

### Model Loading Fallbacks

```javascript
// The system automatically falls back to mock model if real model fails
const modelLoaded = await audio2gesture.initializeAudio2Gesture(modelPath);
if (!modelLoaded) {
    console.warn('Using mock Audio2Gesture model for development');
}
```

## Integration Examples

### With Three.js VRM Avatars

```javascript
// VRM Avatar integration
const vrm = await VRMLoader.loadVRM('/models/avatar.vrm');

// Connect BVH timeline to VRM bones
timeline.onFrameUpdate = (frame) => {
    if (frame.bones) {
        Object.entries(frame.bones).forEach(([boneName, boneData]) => {
            const vrmBone = vrm.humanoid.getBoneNode(boneName);
            if (vrmBone && boneData.rotation) {
                vrmBone.rotation.setFromEuler(
                    boneData.rotation.x,
                    boneData.rotation.y,
                    boneData.rotation.z
                );
            }
        });
    }
};
```

### With Audio Analysis Libraries

```javascript
// Integration with Web Audio API
const analyser = audioContext.createAnalyser();
analyser.fftSize = 2048;

// Real-time frequency analysis
const processAudioFrame = () => {
    const frequencyData = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(frequencyData);
    
    // Convert to Audio2Gesture format
    const audioFeatures = processor.convertFrequencyData(frequencyData);
    audio2gesture.processLiveAudio(audioFeatures, audioContext.currentTime);
};
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check model path and format
   - Verify CORS settings for remote models
   - Falls back to mock model automatically

2. **Audio Processing Errors**
   - Ensure audio sample rate matches processor settings
   - Check audio format compatibility
   - Verify audio data is not empty

3. **Timeline Synchronization Issues**
   - Confirm timeline is initialized before integration
   - Check track naming conflicts
   - Verify frame rate compatibility

4. **Performance Issues**
   - Enable caching for repeated playback
   - Reduce audio processing resolution
   - Adjust smoothing parameters

### Debug Mode

```javascript
// Enable detailed logging
const integration = new Audio2GestureTimelineIntegration(timeline, {
    debug: true,
    logLevel: 'verbose'
});

// Monitor processing pipeline
integration.onProcessingStep = (step, data) => {
    console.log(`Processing step: ${step}`, data);
};
```

## Browser Compatibility

### Requirements
- Modern browsers with Web Audio API support
- ES6 module support or transpilation
- Float32Array support for audio data
- RequestAnimationFrame for timeline playback

### Tested Browsers
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## License and Credits

This Audio2Gesture BVH Integration system is designed to work with various Audio2Gesture neural network implementations. Please ensure you comply with the licensing terms of any specific Audio2Gesture models you use.

The system integrates with the BVH Timeline Compositor and supports standard BVH format specifications for maximum compatibility with existing 3D animation workflows.
