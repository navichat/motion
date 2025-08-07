# BVH Timeline Compositor

A comprehensive JavaScript library for compositing multiple BVH animation sources into a single timeline for driving 3D avatars.

## Overview

The BVH Timeline Compositor allows you to:

- **Manage multiple animation tracks** with different priorities and blend modes
- **Composite BVH animations** from various sources (static files, neural networks, audio-driven systems)
- **Real-time blending** of animations with smooth transitions
- **Timeline-based playback** with frame-accurate timing
- **Integration with VRM avatars** and Three.js

## Key Features

### Multi-Track Timeline System
- **Track Types**: Static BVH, RSMT transitions, Neural networks, Audio2Gesture, Faceformer
- **Priority System**: Higher priority tracks override lower priority ones
- **Channel Targeting**: Tracks can target specific body parts (body, face, hands, arms, legs)
- **Blend Modes**: Replace, additive, multiply blending between tracks

### Animation Sources
- **Static BVH Files**: Pre-recorded animation clips
- **RSMT Integration**: Real-time motion transitions between clips
- **Neural Networks**: Deep learning-based gesture generation
- **Audio2Gesture**: Body gesture generation from audio input
- **Faceformer**: Facial animation synchronized to speech
- **Procedural Animation**: Custom generator functions

### Real-Time Compositing
- **Frame-by-frame compositing** at any timestamp
- **Intelligent caching** for performance optimization
- **Smooth blending** between different animation sources
- **Fade in/out support** for seamless transitions

## Usage

### Basic Setup

```javascript
// Create the compositor
const compositor = new BVHTimelineCompositor({
    frameRate: 30,
    blendMode: 'hierarchical',
    cacheSize: 1000
});

// Add tracks for different animation types
compositor.addTrack('base_animations', {
    type: 'static',
    priority: 100,
    weight: 1.0,
    channels: 'body',
    blendMode: 'replace'
});

compositor.addTrack('facial_animation', {
    type: 'faceformer',
    priority: 85,
    weight: 1.0,
    channels: 'face',
    blendMode: 'replace'
});

compositor.addTrack('body_gestures', {
    type: 'audio2gesture',
    priority: 80,
    weight: 0.7,
    channels: 'arms',
    blendMode: 'additive'
});
```

### Adding Animation Clips

```javascript
// Add a walking animation
compositor.addClip('base_animations', {
    id: 'walk_cycle',
    startTime: 0,
    duration: 5000,
    bvhFile: 'animations/walk_cycle.bvh',
    loop: true,
    weight: 1.0
});

// Add speech-driven facial animation
compositor.addClip('facial_animation', {
    id: 'speech_lips',
    startTime: 6000,
    duration: 8000,
    audioFile: 'audio/speech.wav',
    parameters: {
        visemeIntensity: 1.0,
        emotionalExpression: 'happy'
    }
});

// Add gesture animation from neural network
compositor.addClip('body_gestures', {
    id: 'expressive_gestures',
    startTime: 6500,
    duration: 7000,
    prompt: 'enthusiastic presentation gestures',
    parameters: {
        style: 'professional',
        intensity: 0.8
    }
});
```

### Timeline Playback

```javascript
// Set up event listeners
compositor.on('frameUpdate', (data) => {
    // Send frame to VRM avatar
    applyFrameToAvatar(data.frame);
});

// Play the timeline
compositor.play();

// Seek to specific time
await compositor.setTime(5000); // 5 seconds

// Get frame at specific timestamp
const frame = await compositor.getFrameAtTime(3000);
```

### Integration with VRM/Three.js

```javascript
function applyFrameToAvatar(frame) {
    if (!vrm || !vrm.humanoid) return;
    
    for (const [jointIndex, jointData] of Object.entries(frame)) {
        const vrmBoneName = mapBVHJointToVRM(parseInt(jointIndex));
        if (vrmBoneName && vrm.humanoid.getBoneNode(vrmBoneName)) {
            const bone = vrm.humanoid.getBoneNode(vrmBoneName);
            
            // Apply rotation
            if (jointData.rotation) {
                bone.rotation.set(
                    jointData.rotation.x * Math.PI / 180,
                    jointData.rotation.y * Math.PI / 180,
                    jointData.rotation.z * Math.PI / 180
                );
            }
            
            // Apply position (usually only for root)
            if (jointData.position && jointIndex === '0') {
                bone.position.set(
                    jointData.position.x,
                    jointData.position.y,
                    jointData.position.z
                );
            }
        }
    }
}
```

## Track Types

### 1. Static BVH (`type: 'static'`)
Pre-recorded BVH animation files.

```javascript
compositor.addClip('base_animations', {
    bvhFile: 'animations/walk.bvh',
    // or
    bvhData: parsedBVHData,
    loop: true,
    speed: 1.2
});
```

### 2. RSMT Transitions (`type: 'rsmt'`)
Real-time motion transitions between animations.

```javascript
compositor.addClip('transitions', {
    sourceClip: 'walk_cycle',
    targetClip: 'idle_pose',
    transitionDuration: 500,
    parameters: {
        smoothness: 0.8,
        preserveRootMotion: true
    }
});
```

### 3. Audio2Gesture (`type: 'audio2gesture'`)
Generate body gestures from audio input.

```javascript
compositor.addClip('body_gestures', {
    audioFile: 'speech.wav',
    parameters: {
        gestureIntensity: 0.7,
        gestureType: 'explanatory'
    }
});
```

### 4. Faceformer (`type: 'faceformer'`)
Generate facial animation from audio.

```javascript
compositor.addClip('facial_animation', {
    audioFile: 'speech.wav',
    parameters: {
        visemeIntensity: 1.0,
        emotionalExpression: 'neutral'
    }
});
```

### 5. Neural Networks (`type: 'neural'`)
Generate animations using neural networks.

```javascript
compositor.addClip('neural_gestures', {
    prompt: 'confident presentation gestures',
    parameters: {
        style: 'business',
        intensity: 0.6
    }
});
```

### 6. Custom Generators
Create custom animation generators.

```javascript
compositor.addTrack('custom_track', {
    type: 'custom',
    generator: (clip, localTime, weight) => {
        // Return BVH frame data
        const frame = {};
        // ... generate animation
        return frame;
    }
});
```

## Channel System

The compositor supports targeting specific body parts:

- **`'all'`**: All joints (default)
- **`'body'`**: Spine, arms, legs
- **`'face'`**: Facial joints
- **`'hands'`**: Hand and finger joints
- **`'arms'`**: Arm and shoulder joints
- **`'legs'`**: Leg and hip joints
- **`'spine'`**: Spine joints only

## Blend Modes

### Track Blend Modes
- **`'replace'`**: Replace base animation
- **`'additive'`**: Add to base animation
- **`'multiply'`**: Multiply with base animation

### Timeline Blend Mode
- **`'hierarchical'`**: Higher priority tracks override lower ones
- **`'additive'`**: All tracks are blended additively
- **`'override'`**: Latest track overrides all others

## Events

```javascript
compositor.on('frameUpdate', (data) => {
    console.log('New frame:', data.timestamp, data.frame);
});

compositor.on('trackAdded', (data) => {
    console.log('Track added:', data.trackId);
});

compositor.on('clipAdded', (data) => {
    console.log('Clip added:', data.clip.id);
});

compositor.on('playStarted', () => {
    console.log('Playback started');
});

compositor.on('playPaused', () => {
    console.log('Playback paused');
});
```

## Advanced Features

### Timeline Export/Import
```javascript
// Export timeline to JSON
const timelineData = compositor.exportTimeline();

// Import timeline from JSON
compositor.importTimeline(timelineData);
```

### Dynamic Clip Management
```javascript
// Add clips during playback
await addSpeechClip('audio/response.wav', 10000, 3000);

// Remove clips
compositor.removeClip('base_animations', 'walk_cycle');

// Modify track properties
compositor.tracks.get('body_gestures').weight = 0.5;
```

### Performance Optimization
```javascript
// Configure caching
const compositor = new BVHTimelineCompositor({
    cacheSize: 2000, // Number of frames to cache
    frameRate: 60    // Higher frame rate for smoother animation
});

// Clear cache when needed
compositor.clearCache();
```

## Integration Examples

### With Audio2Gesture System
```javascript
// Integrate with your Audio2Gesture system
window.Audio2GestureGenerator = {
    async generateFrame(audioData, localTime, parameters) {
        // Your Audio2Gesture implementation
        return bvhFrame;
    }
};
```

### With RSMT System
```javascript
// Integrate with your RSMT system
window.RSMTGenerator = {
    async generateTransitionFrame(sourceClip, targetClip, progress, duration, params) {
        // Your RSMT implementation
        return interpolatedFrame;
    }
};
```

### With Neural Networks
```javascript
// Integrate with your neural network systems
window.NeuralGenerator = {
    async generateFrame(prompt, localTime, parameters) {
        // Your neural network implementation
        return generatedFrame;
    }
};
```

## File Structure

```
/js/
├── BVHTimelineCompositor.js    # Main compositor class
├── BVHTimelineExample.js       # Usage examples
└── bvh_timeline_demo_test.html # Interactive demo
```

## Demo

Open `bvh_timeline_demo_test.html` in a web browser to see an interactive demonstration of the timeline compositor. The demo includes:

- Real-time timeline playback
- Track weight controls
- Dynamic clip addition
- Frame data visualization
- Timeline status monitoring

## API Reference

### Constructor
```javascript
new BVHTimelineCompositor(options)
```

### Methods
- `addTrack(trackId, config)` - Add a new track
- `removeTrack(trackId)` - Remove a track
- `addClip(trackId, clip)` - Add a clip to a track
- `removeClip(trackId, clipId)` - Remove a clip
- `getFrameAtTime(timestamp)` - Get composited frame at timestamp
- `setTime(timestamp)` - Set current playback time
- `play(startTime?)` - Start playback
- `pause()` - Pause playback
- `stop()` - Stop playback
- `exportTimeline()` - Export to JSON
- `importTimeline(data)` - Import from JSON

### Events
- `frameUpdate` - New frame available
- `trackAdded` - Track was added
- `clipAdded` - Clip was added
- `playStarted` - Playback started
- `playPaused` - Playback paused
- `playStopped` - Playback stopped

This system provides a powerful foundation for creating complex, multi-layered BVH animations that can drive realistic 3D avatars with natural-looking motion combining multiple AI and traditional animation sources.
