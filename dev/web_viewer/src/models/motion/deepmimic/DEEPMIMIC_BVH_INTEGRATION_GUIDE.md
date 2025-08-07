# DeepMimic BVH Timeline Integration Guide

## Overview

This enhanced DeepMimic system provides seamless integration with the BVH Timeline framework, enabling you to generate continuous motion sequences for VRM avatars using AI-powered motion synthesis. The system combines the power of DeepMimic neural networks with sophisticated timeline management and frame buffering.

## Architecture

### Core Components

1. **DeepMimicBVHGenerator** - Main motion synthesis engine
2. **BVHTimelineDeepMimicIntegration** - Timeline integration layer
3. **BVHTimeline** - Frame buffering and composition system
4. **DeepMimicInference** - ONNX model inference engine

### Key Features

- **Real-time Motion Generation**: Generate motion frames on-demand
- **Timeline Integration**: Seamless integration with BVH Timeline system
- **Motion Templates**: Pre-configured motion patterns for common actions
- **Smooth Transitions**: Temporal smoothing and motion blending
- **VRM Compatibility**: Direct output in VRM-compatible bone format
- **Multi-Model Support**: Switch between different motion models
- **Performance Optimization**: Frame buffering and lookahead generation

## Quick Start

### Basic Setup

```javascript
// Initialize the system
const timeline = new BVHTimeline({
    framerate: 30,
    maxBufferSize: 300,
    lookaheadFrames: 60
});

const generator = new DeepMimicBVHGenerator({
    frameRate: 30,
    smoothingFactor: 0.15
});

await generator.initialize();

const integration = new BVHTimelineDeepMimicIntegration(
    timeline, 
    generator,
    {
        defaultTrack: 'deepmimic',
        bufferAhead: 2.0,
        maxClips: 15
    }
);
```

### Adding Motion Clips

```javascript
// Add a walking motion for 5 seconds
const clipId = await integration.addMotionFromTemplate('walk', 0);

// Add custom motion with specific parameters
const customClipId = await integration.addMotionClip({
    model: 'dance_a',
    duration: 8.0,
    startTime: 5.0,
    weight: 1.0,
    blendMode: 'replace'
});
```

### Real-time Motion Control

```javascript
// Start real-time motion generation
const realtimeId = await integration.addRealTimeMotion({
    model: 'walk',
    targetMotion: {
        speed: 1.5,
        direction: [0, 0, 1] // Forward direction
    }
});

// Update motion parameters in real-time
integration.updateTargetMotion(realtimeId, {
    speed: 2.0,
    direction: [1, 0, 0] // Right direction
});
```

## Available Motion Templates

### Standard Templates

| Template | Model | Duration | Loop | Description |
|----------|-------|----------|------|-------------|
| `idle` | walk | 10.0s | Yes | Stationary idle pose with subtle movement |
| `walk` | walk | 4.0s | Yes | Natural walking gait |
| `run` | run | 3.0s | Yes | Running motion |
| `jump` | jump | 2.0s | No | Single jump action |
| `dance` | dance_a | 8.0s | Yes | Rhythmic dance movement |
| `combat` | punch | 1.5s | No | Combat action (punch) |
| `acrobatic` | backflip | 2.5s | No | Acrobatic movement (backflip) |

### Using Templates

```javascript
// Add a dance sequence starting at current time
await integration.addMotionFromTemplate('dance', timeline.currentTime);

// Create a motion sequence
const sequence = [
    { ...integration.motionTemplates.walk, duration: 3.0 },
    { ...integration.motionTemplates.jump, duration: 2.0 },
    { ...integration.motionTemplates.run, duration: 4.0 }
];

await integration.addMotionSequence(sequence, 0);
```

## Motion Parameters

### Target Motion Structure

```javascript
const targetMotion = {
    speed: 1.0,                    // Movement speed (0-3)
    direction: [x, y, z],          // 3D direction vector
    turnRate: 0.0,                 // Angular velocity (rad/s)
    rhythmic: false,               // Rhythmic movement flag
    aggressive: false              // Aggressive movement flag
};
```

### Motion Control

```javascript
// Get motion controller for real-time adjustment
const controller = integration.getMotionController(clipId);

controller.updateSpeed(2.0);
controller.updateDirection([1, 0, 0.5]);
controller.updateWeight(0.8);

// Check status
const status = controller.getStatus();
console.log('Motion active:', status.isActive);
```

## Advanced Usage

### Custom Motion Generation

```javascript
// Generate motion with specific parameters
const motionClip = await generator.generateMotionClip({
    duration: 10.0,
    model: 'walk',
    targetMotion: {
        speed: 1.5,
        direction: [0.7071, 0, 0.7071] // 45-degree angle
    }
});

// Add to timeline with custom settings
const clipId = timeline.addClip('custom_track', new BVHClip({
    id: motionClip.id,
    type: 'deepmimic_generated',
    startTime: 0,
    duration: 10.0,
    generator: integration.createFrameGenerator(motionClip)
}));
```

### Smooth Transitions

```javascript
// Transition from walking to running
await integration.transitionToMotion(
    { model: 'run', duration: 5.0 },
    { 
        transitionDuration: 1.0,
        blendDuringTransition: true
    }
);
```

### Motion Sequences from Text

```javascript
// Parse natural language description
const clipIds = await integration.createMotionFromDescription(
    "walk forward then jump and dance",
    { duration: 15.0 }
);
```

## Frame Format

### BVH Frame Structure

```javascript
const bvhFrame = {
    time: 1.5,                     // Timestamp in seconds
    motionData: {                  // VRM bone data
        'hips': [x, y, z, rx, ry, rz],        // Position + rotation
        'spine': [rx, ry, rz],                // Rotation only
        'chest': [rx, ry, rz],
        'neck': [rx, ry, rz],
        'head': [rx, ry, rz],
        'leftUpperLeg': [rx, ry, rz],
        'leftLowerLeg': [rx, ry, rz],
        'leftFoot': [rx, ry, rz],
        // ... more bones
    },
    metadata: {
        type: 'deepmimic',
        model: 'walk',
        phase: 1.23,
        timestamp: 1653123456789
    }
};
```

### VRM Bone Mapping

The system outputs motion data for all standard VRM bones:

- **Core**: hips, spine, chest, upperChest, neck, head
- **Left Leg**: leftUpperLeg, leftLowerLeg, leftFoot, leftToes
- **Right Leg**: rightUpperLeg, rightLowerLeg, rightFoot, rightToes
- **Left Arm**: leftShoulder, leftUpperArm, leftLowerArm, leftHand
- **Right Arm**: rightShoulder, rightUpperArm, rightLowerArm, rightHand

## Performance Optimization

### Frame Buffering

```javascript
// Configure buffer settings
const timeline = new BVHTimeline({
    framerate: 30,
    maxBufferSize: 300,           // Max frames to cache
    lookaheadFrames: 60,          // Frames to generate ahead
    bufferUpdateInterval: 100     // Buffer update frequency (ms)
});

// Monitor buffer performance
const bufferStats = timeline.getBufferStats();
console.log('Buffer hit rate:', bufferStats.hitRate);
console.log('Memory usage:', bufferStats.memoryUsage);
```

### Performance Monitoring

```javascript
// Get performance statistics
const stats = generator.getPerformanceStats();
console.log('Average frame time:', stats.averageTime, 'ms');
console.log('FPS:', stats.fps);
console.log('Frames generated:', stats.framesGenerated);

// Monitor motion statistics
const motionStats = integration.getMotionStats();
console.log('Active clips:', motionStats.activeClips);
console.log('Real-time clips:', motionStats.realtimeClips);
console.log('Models used:', motionStats.models);
```

## Error Handling

### Graceful Fallbacks

```javascript
try {
    const clipId = await integration.addMotionClip({
        model: 'walk',
        duration: 5.0
    });
} catch (error) {
    console.error('Motion generation failed:', error);
    
    // Fallback to default motion
    const fallbackId = await integration.addMotionFromTemplate('idle', 0);
}
```

### Model Loading

```javascript
// Check available models
const availableModels = generator.getAvailableModels();
console.log('Available models:', availableModels);

// Load additional models
try {
    await generator.loadModel('custom_model', './path/to/model.onnx');
} catch (error) {
    console.warn('Failed to load custom model:', error.message);
}
```

## Integration with VRM Avatar

### Applying Motion to VRM

```javascript
// Get current frame for VRM application
timeline.onFrameUpdate = async (frame) => {
    const currentFrame = await timeline.getCurrentFrame();
    
    if (currentFrame && currentFrame.motionData) {
        // Apply to VRM avatar
        applyBVHToVRM(vrmAvatar, currentFrame.motionData);
    }
};

function applyBVHToVRM(vrm, motionData) {
    for (const [boneName, boneData] of Object.entries(motionData)) {
        const vrmBone = vrm.humanoid.getBoneNode(boneName);
        
        if (vrmBone) {
            if (boneName === 'hips') {
                // Apply position + rotation
                vrmBone.position.set(boneData[0], boneData[1], boneData[2]);
                vrmBone.rotation.set(
                    boneData[3] * Math.PI / 180,
                    boneData[4] * Math.PI / 180,
                    boneData[5] * Math.PI / 180
                );
            } else {
                // Apply rotation only
                vrmBone.rotation.set(
                    boneData[0] * Math.PI / 180,
                    boneData[1] * Math.PI / 180,
                    boneData[2] * Math.PI / 180
                );
            }
        }
    }
}
```

## Cleanup and Resource Management

```javascript
// Proper cleanup
function cleanup() {
    // Stop real-time motions
    if (realtimeClipId) {
        integration.removeClip(realtimeClipId);
    }
    
    // Clear timeline
    integration.clearAllMotions();
    
    // Dispose components
    integration.dispose();
    generator.dispose();
    timeline.dispose();
}

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);
```

## Troubleshooting

### Common Issues

1. **Models not loading**: Check ONNX file paths and compatibility
2. **Low performance**: Reduce buffer sizes or lower frame rate
3. **Memory issues**: Enable auto-cleanup and reduce max clips
4. **Jerky motion**: Increase smoothing factor or reduce phase increment
5. **Timeline sync issues**: Check frame rate consistency

### Debug Information

```javascript
// Enable detailed logging
console.log('Timeline stats:', timeline.getStats());
console.log('Generator performance:', generator.getPerformanceStats());
console.log('Motion statistics:', integration.getMotionStats());

// Export motion data for analysis
const motionData = integration.exportMotionData(clipId);
console.log('Motion data:', motionData);
```

## API Reference

### DeepMimicBVHGenerator

- `initialize(modelPaths?)` - Initialize with optional model paths
- `generateFrame(time, options?)` - Generate single frame
- `generateMotionClip(options)` - Generate motion sequence
- `switchModel(modelName)` - Switch active model
- `getAvailableModels()` - Get available model names
- `dispose()` - Clean up resources

### BVHTimelineDeepMimicIntegration

- `addMotionFromTemplate(templateName, startTime, options?)` - Add template motion
- `addMotionClip(options)` - Add custom motion clip
- `addRealTimeMotion(options)` - Start real-time motion
- `transitionToMotion(newMotionOptions, options?)` - Smooth transition
- `removeClip(clipId)` - Remove motion clip
- `clearAllMotions()` - Clear all motions
- `getMotionStats()` - Get motion statistics

This system provides a complete solution for generating and managing AI-powered motion sequences for VRM avatars, with professional-grade performance and flexibility.
