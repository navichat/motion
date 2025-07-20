# FaceFormer to BVH Timeline Integration

This documentation covers the complete system for integrating FaceFormer neural network facial animation with the BVH Timeline compositor.

## Overview

The FaceFormer BVH integration consists of three main components:

1. **FaceFormerBVHConverter.js** - Converts FaceFormer output to BVH format
2. **FaceFormerTimelineIntegration.js** - Integrates FaceFormer with the timeline system
3. **Demo and examples** - Shows how to use the complete system

## Quick Start

```javascript
// 1. Create timeline instance
const timeline = new BVHTimelineCompositor({
    frameRate: 30,
    maxTracks: 10
});

// 2. Create FaceFormer integration
const faceFormerIntegration = new FaceFormerTimelineIntegration(timeline, {
    realtime: true,
    trackName: 'facial_animation',
    priority: 100,
    channels: ['face', 'head']
});

// 3. Initialize FaceFormer model
await faceFormerIntegration.initializeFaceFormer('/path/to/faceformer/model');

// 4. Process audio for facial animation
const audioData = new Float32Array(16000); // 1 second of audio
const result = await faceFormerIntegration.processAudio(audioData, {
    startTime: 0
});

// 5. Play the timeline
timeline.play();
```

## Architecture

### FaceFormerBVHConverter

Converts various FaceFormer output formats to BVH-compatible facial bone rotations.

#### Supported Input Formats

1. **Facial Landmarks** (68-point standard)
2. **Blend Shapes** (ARKit-style)
3. **3D Mesh Vertices**
4. **Expression Coefficients**
5. **Raw Data** (fallback)

#### Output Format

BVH frames with facial bone rotations:
- Head, neck
- Left/right eyes
- Left/right eyebrows
- Jaw, left/right mouth corners
- Left/right cheeks
- Nose

### FaceFormerTimelineIntegration

Manages the complete integration between FaceFormer and the timeline system.

#### Key Features

- **Real-time processing** - Low-latency facial animation
- **Batch processing** - Process pre-recorded audio
- **Model management** - Load and manage FaceFormer models
- **Performance monitoring** - Track inference and conversion times
- **Timeline synchronization** - Seamless integration with compositor

## API Reference

### FaceFormerBVHConverter

#### Constructor Options

```javascript
const converter = new FaceFormerBVHConverter({
    // Conversion settings
    scaleFactor: 1.0,           // Scale rotation values
    smoothing: true,            // Enable frame smoothing
    smoothingFactor: 0.8,       // Smoothing strength (0-1)
    
    // Feature toggles
    enableEyeMovements: true,
    enableJawMovement: true,
    enableEyebrowMovement: true,
    enableCheekMovement: true,
    
    // Performance
    targetFrameRate: 30,
    maxLookAhead: 10,          // Frames for predictive smoothing
    
    // Debug
    verbose: false,
    logConversions: false
});
```

#### Main Methods

```javascript
// Convert single frame
const bvhFrame = await converter.convertToBVH(faceFormerOutput, timestamp);

// Create timeline clip from frame sequence
const clip = converter.createTimelineClip(faceFormerFrames, startTime, duration);

// Real-time frame processing
const bvhFrame = await converter.processLiveFrame(faceFormerOutput, timestamp);

// Get conversion statistics
const stats = converter.getStats();
```

### FaceFormerTimelineIntegration

#### Constructor Options

```javascript
const integration = new FaceFormerTimelineIntegration(timeline, {
    // Integration settings
    realtime: true,             // Real-time vs batch processing
    trackName: 'faceformer_facial',
    priority: 100,              // Track priority (higher = more important)
    channels: ['face', 'head'], // Target bone channels
    
    // Converter options
    converter: {
        scaleFactor: 1.0,
        smoothing: true,
        smoothingFactor: 0.8
    }
});
```

#### Main Methods

```javascript
// Initialize FaceFormer model
await integration.initializeFaceFormer(modelPath, options);

// Process audio for facial animation
const result = await integration.processAudio(audioData, {
    startTime: 0,
    realtime: false
});

// Real-time audio processing
const result = await integration.processLiveAudio(audioChunk, timestamp);

// Batch processing
const results = await integration.processBatchAudio(audioSegments, options);

// Configuration
integration.setRealtime(enabled);
integration.setTrackPriority(priority);
integration.setChannels(channels);

// Statistics
const stats = integration.getStats();
```

## Input Data Formats

### 1. Facial Landmarks (68-point)

Standard 68-point facial landmark format:

```javascript
const landmarkInput = {
    landmarks: [
        { x: 100, y: 200 }, // Point 0: Jaw left
        { x: 101, y: 201 }, // Point 1: Jaw
        // ... 66 more points
        { x: 150, y: 180 }  // Point 67: Jaw right
    ],
    confidence: 0.95,
    timestamp: Date.now()
};
```

### 2. Blend Shapes (ARKit-style)

ARKit-compatible blend shape weights:

```javascript
const blendShapeInput = {
    blendshapes: {
        // Eye movements
        eyeBlinkLeft: 0.2,
        eyeBlinkRight: 0.1,
        eyeLookUpLeft: 0.3,
        eyeLookDownLeft: 0.0,
        eyeLookInLeft: 0.1,
        eyeLookOutLeft: 0.0,
        
        // Jaw movement
        jawOpen: 0.4,
        jawLeft: 0.1,
        jawRight: 0.0,
        
        // Mouth expressions
        mouthSmileLeft: 0.6,
        mouthSmileRight: 0.6,
        mouthFrownLeft: 0.0,
        mouthFrownRight: 0.0,
        
        // Eyebrows
        browInnerUp: 0.2,
        browOuterUpLeft: 0.1,
        browOuterUpRight: 0.1,
        
        // Cheeks
        cheekPuffLeft: 0.0,
        cheekPuffRight: 0.0
    },
    confidence: 0.9,
    timestamp: Date.now()
};
```

### 3. Expression Coefficients

Numerical expression coefficients:

```javascript
const coefficientInput = {
    coefficients: [
        0.2,  // Jaw open
        0.1,  // Jaw side
        0.3,  // Left eye up/down
        0.1,  // Left eye left/right
        0.3,  // Right eye up/down
        -0.1, // Right eye left/right
        0.2,  // Left eyebrow
        0.2   // Right eyebrow
        // ... more coefficients
    ],
    confidence: 0.85,
    timestamp: Date.now()
};
```

## Timeline Integration

### Track Configuration

The system automatically creates and manages timeline tracks:

```javascript
// Facial animation track
{
    type: 'faceformer',
    priority: 100,
    channels: ['face', 'head'],
    blending: 'override'
}

// Eye tracking track (optional)
{
    type: 'faceformer_eyes',
    priority: 110,
    channels: ['eyes'],
    blending: 'override'
}

// Mouth animation track (optional)
{
    type: 'faceformer_mouth',
    priority: 105,
    channels: ['mouth', 'jaw'],
    blending: 'override'
}
```

### Clip Structure

Generated timeline clips follow this structure:

```javascript
{
    type: 'faceformer',
    startTime: 0.0,
    duration: 2.5,
    frames: [
        {
            frameNumber: 0,
            timestamp: 0.0,
            motionData: [/* BVH motion data array */],
            metadata: {
                type: 'facial',
                source: 'faceformer',
                format: 'blendshapes',
                confidence: 0.9,
                boneCount: 12,
                channels: 72,
                converter: 'FaceFormerBVHConverter'
            }
        },
        // ... more frames
    ],
    metadata: {
        sourceFrameCount: 75,
        targetFrameRate: 30,
        converter: 'FaceFormerBVHConverter'
    }
}
```

## Performance Optimization

### Real-time Processing

For real-time applications:

```javascript
const integration = new FaceFormerTimelineIntegration(timeline, {
    realtime: true,
    converter: {
        smoothing: true,
        smoothingFactor: 0.9  // Higher for smoother but more delayed
    }
});

// Process audio chunks as they arrive
audioStream.on('data', async (chunk) => {
    await integration.processLiveAudio(chunk, Date.now());
});
```

### Batch Processing

For pre-recorded content:

```javascript
const integration = new FaceFormerTimelineIntegration(timeline, {
    realtime: false,
    converter: {
        smoothing: true,
        smoothingFactor: 0.8
    }
});

// Process entire audio file
const audioSegments = splitAudioIntoSegments(audioFile, 2.0); // 2-second segments
const results = await integration.processBatchAudio(audioSegments);
```

### Performance Tips

1. **Adjust frame rate** - Lower frame rates (15-24 FPS) for better performance
2. **Enable smoothing** - Reduces jitter but adds slight delay
3. **Limit channels** - Only target necessary bone channels
4. **Batch processing** - More efficient for non-real-time use cases
5. **Model optimization** - Use quantized or optimized FaceFormer models

## Error Handling

### Common Error Scenarios

```javascript
try {
    const result = await integration.processAudio(audioData);
} catch (error) {
    if (error.message.includes('FaceFormer model not loaded')) {
        // Model initialization required
        await integration.initializeFaceFormer(modelPath);
        
    } else if (error.message.includes('Invalid audio format')) {
        // Audio preprocessing required
        audioData = preprocessAudio(audioData);
        
    } else if (error.message.includes('Timeline not ready')) {
        // Timeline initialization required
        timeline.initialize();
        
    } else {
        // Generic error handling
        console.error('FaceFormer processing failed:', error);
        
        // Fallback to neutral animation
        const neutralClip = createNeutralFacialClip(duration);
        timeline.addClip('facial_fallback', neutralClip);
    }
}
```

### Fallback Strategies

1. **Neutral frames** - Generate neutral facial pose when processing fails
2. **Previous frame** - Use last successful frame with decay
3. **Simplified animation** - Basic jaw movement from audio amplitude
4. **Mock model** - Built-in mock FaceFormer for development/testing

## Integration Examples

### With Audio2Gesture System

```javascript
// Create combined audio-to-animation pipeline
const timeline = new BVHTimelineCompositor({ frameRate: 30 });

// Body animation from Audio2Gesture
const bodyIntegration = new Audio2GestureIntegration(timeline, {
    trackName: 'body_animation',
    channels: ['body', 'arms', 'hands']
});

// Facial animation from FaceFormer
const faceIntegration = new FaceFormerTimelineIntegration(timeline, {
    trackName: 'facial_animation',
    channels: ['face', 'head']
});

// Process audio for both systems
const audioData = getAudioInput();
await Promise.all([
    bodyIntegration.processAudio(audioData),
    faceIntegration.processAudio(audioData)
]);

timeline.play();
```

### With VRM Avatar System

```javascript
// Connect to VRM avatar
const vrmIntegration = new BVHTimelineVRMIntegration(vrmBVHAdapter);
vrmIntegration.connectTimeline(timeline);

// Set up FaceFormer for facial animation
const faceFormer = new FaceFormerTimelineIntegration(timeline, {
    channels: ['face', 'head'],
    priority: 100
});

// Real-time facial animation
faceFormer.onFaceFormerFrame = (frame, time) => {
    // Custom VRM-specific processing
    vrmIntegration.applyFacialFrame(frame, time);
};
```

## Troubleshooting

### Common Issues

1. **No facial animation visible**
   - Check channel mapping between FaceFormer output and VRM bones
   - Verify track priority settings
   - Ensure timeline is playing

2. **Jittery animation**
   - Enable smoothing with higher smoothing factor
   - Check frame rate consistency
   - Verify input data quality

3. **Performance issues**
   - Reduce frame rate
   - Disable unnecessary features (eyebrows, cheeks)
   - Use batch processing instead of real-time

4. **Audio-visual sync issues**
   - Check audio processing latency
   - Adjust timeline start times
   - Verify frame timing accuracy

### Debug Mode

Enable verbose logging for troubleshooting:

```javascript
const converter = new FaceFormerBVHConverter({
    verbose: true,
    logConversions: true
});

const integration = new FaceFormerTimelineIntegration(timeline, {
    converter: converter
});

// Monitor statistics
setInterval(() => {
    console.log('Stats:', integration.getStats());
}, 1000);
```

## Demo and Testing

Use the provided demo page (`faceformer_timeline_demo.html`) to:

1. Test different FaceFormer input formats
2. Adjust conversion parameters
3. Monitor performance metrics
4. Visualize timeline tracks and clips
5. Export debug logs

The demo includes:
- Mock FaceFormer model for testing
- Real-time audio recording and processing
- Timeline visualization and controls
- Performance monitoring
- Configuration interface

## Conclusion

The FaceFormer BVH Timeline integration provides a complete solution for adding neural network-driven facial animation to your 3D avatar system. The modular design allows for easy customization and extension, while the timeline compositor ensures smooth integration with other animation sources.

For questions or issues, check the debug logs and refer to the troubleshooting section above.
