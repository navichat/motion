# RSMT BVH Integration

A comprehensive system for Realtime Stylized Motion Transition (RSMT) that enables seamless transitions between BVH animations using pose vector matching and DeepPhase neural network generation.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Installation & Setup](#installation--setup)
4. [Quick Start Guide](#quick-start-guide)
5. [API Reference](#api-reference)
6. [Pose Vector System](#pose-vector-system)
7. [Transition Generation](#transition-generation)
8. [Timeline Integration](#timeline-integration)
9. [Animation Library Management](#animation-library-management)
10. [Advanced Features](#advanced-features)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)

## Overview

The RSMT (Realtime Stylized Motion Transition) system provides intelligent transition generation between different BVH animations by:

1. **Pose Vector Analysis**: Converting BVH poses into high-dimensional vectors for similarity comparison
2. **Smart Matching**: Finding optimal transition points between animations using cosine similarity
3. **DeepPhase Generation**: Using neural networks to generate natural transition sequences
4. **Timeline Integration**: Seamlessly integrating transitions into the BVH Timeline system

### Key Features

- **Intelligent Pose Matching**: Automatic detection of similar poses across different animations
- **Neural Transition Generation**: DeepPhase-powered smooth transition creation
- **Real-time Processing**: Live transition generation during timeline playback
- **Animation Library Management**: Automatic loading and processing of animation assets
- **Transition Queuing**: Support for multiple concurrent transitions
- **Quality Assessment**: Automatic evaluation of transition smoothness and naturalness
- **Caching System**: Performance optimization through intelligent caching

## Core Components

### 1. RSMTBVHConverter

The core converter class that handles pose analysis and transition generation.

```javascript
const rsmt = new RSMTBVHConverter({
    frameRate: 30,
    vectorDimensions: 128,
    transitionDuration: 1.0,
    similarityThreshold: 0.8,
    includePosition: true,
    includeRotation: true,
    normalizeVectors: true
});
```

**Key Responsibilities:**
- Pose vector extraction from BVH frames
- Similarity computation between poses
- DeepPhase integration for transition generation
- Animation data processing and caching

### 2. RSMTTimelineIntegration

Integration layer connecting RSMT with the BVH Timeline system.

```javascript
const integration = new RSMTTimelineIntegration(timeline, {
    animationsPath: '/assets/animations/',
    autoLoadAnimations: true,
    transitionTrackName: 'rsmt_transitions',
    transitionPriority: 75,
    maxConcurrentTransitions: 3
});
```

**Key Responsibilities:**
- Timeline synchronization
- Animation library management
- Transition queue processing
- Real-time integration with timeline playback

## Installation & Setup

### Prerequisites

```bash
# Ensure you have the BVH Timeline Compositor
# Animation assets in JSON or BVH format
# Web server for loading animation files
```

### HTML Integration

```html
<!-- Include required scripts -->
<script src="BVHTimelineCompositor.js"></script>
<script src="RSMTBVHConverter.js"></script>
<script src="RSMTTimelineIntegration.js"></script>
```

### Basic Initialization

```javascript
// Initialize timeline
const timeline = new BVHTimelineCompositor({
    frameRate: 30,
    enableBlending: true
});

// Initialize RSMT integration
const rsmt = new RSMTTimelineIntegration(timeline, {
    animationsPath: '/assets/animations/',
    autoLoadAnimations: true
});

// Initialize the system
await rsmt.initialize();
```

## Quick Start Guide

### Loading Animations

```javascript
// Auto-load from assets folder
await rsmt.loadAnimationsFromAssets();

// Or load specific animation
await rsmt.loadAnimation('walk', '/assets/animations/walk.json');

// Check loaded animations
const loadedAnimations = rsmt.getLoadedAnimations();
console.log('Loaded animations:', loadedAnimations);
```

### Basic Transition

```javascript
// Request transition to specific animation
const transition = await rsmt.requestTransition('dance', {
    duration: 1.5,
    style: 'smooth'
});

console.log('Transition ID:', transition.id);
```

### Smart Transition

```javascript
// Automatically find best transition point
const smartTransition = await rsmt.smartTransition('jump', {
    duration: 1.0,
    quality: 'high'
});
```

### Chained Transitions

```javascript
// Create sequence of transitions
const sequence = ['walk', 'run', 'jump', 'land'];
const results = await rsmt.chainTransitions(sequence, {
    duration: 1.0
});
```

## API Reference

### RSMTBVHConverter

#### Constructor Options

```javascript
{
    frameRate: 30,                    // Animation frame rate
    vectorDimensions: 128,            // Pose vector size
    transitionDuration: 1.0,          // Default transition duration (seconds)
    similarityThreshold: 0.8,         // Minimum similarity for matches
    
    // Pose encoding configuration
    includePosition: true,            // Include bone positions in vectors
    includeRotation: true,            // Include bone rotations in vectors
    includeVelocity: false,           // Include velocity information
    positionWeight: 0.3,             // Weight for position components
    rotationWeight: 0.7,             // Weight for rotation components
    normalizeVectors: true           // Normalize pose vectors
}
```

#### Key Methods

##### `loadAnimation(name, data, options)`

Load and process an animation for RSMT use.

**Parameters:**
- `name` (String): Animation identifier
- `data` (String|Object): File path or animation data
- `options` (Object): Processing options

**Returns:** Promise resolving to processed animation data

##### `generateTransition(fromPose, targetAnimation, options)`

Generate transition between current pose and target animation.

**Parameters:**
- `fromPose` (Object): Starting BVH pose
- `targetAnimation` (String): Target animation name
- `options` (Object): Transition options

**Options:**
```javascript
{
    duration: 1.0,                   // Transition duration
    style: 'natural',                // Transition style
    quality: 'normal',               // Quality level
    constraints: {}                  // Motion constraints
}
```

**Returns:** Promise resolving to transition data

##### `findBestMatch(currentPose, targetAnimation)`

Find the best matching frame in target animation.

**Parameters:**
- `currentPose` (Object): Current BVH pose
- `targetAnimation` (String): Target animation name

**Returns:** Match object with similarity score and frame index

##### `extractPoseVector(frame)`

Extract pose vector from BVH frame.

**Parameters:**
- `frame` (Object): BVH frame data

**Returns:** Normalized pose vector array

### RSMTTimelineIntegration

#### Constructor Options

```javascript
{
    animationsPath: '/assets/animations/',    // Path to animation files
    autoLoadAnimations: true,                 // Auto-load on initialization
    transitionTrackName: 'rsmt_transitions', // Timeline track name
    transitionPriority: 75,                   // Track priority
    maxConcurrentTransitions: 3,              // Max simultaneous transitions
    preloadTransitions: false                 // Preload common transitions
}
```

#### Key Methods

##### `requestTransition(targetAnimation, options)`

Request transition to target animation.

**Parameters:**
- `targetAnimation` (String): Target animation name
- `options` (Object): Transition options

**Returns:** Promise resolving to transition request object

##### `smartTransition(targetAnimation, options)`

Request intelligent transition with automatic best match finding.

**Parameters:**
- `targetAnimation` (String): Target animation name
- `options` (Object): Transition options

**Returns:** Promise resolving to transition request object

##### `chainTransitions(animationSequence, options)`

Create sequence of connected transitions.

**Parameters:**
- `animationSequence` (Array): Array of animation names
- `options` (Object): Options applied to all transitions

**Returns:** Promise resolving to array of transition results

##### `loadAnimationsFromAssets()`

Automatically discover and load animations from assets folder.

**Returns:** Promise resolving to load results summary

##### `getLoadedAnimations()`

Get list of currently loaded animations.

**Returns:** Array of animation names

##### `getActiveTransitions()`

Get information about currently active transitions.

**Returns:** Array of active transition objects

##### `getStats()`

Get comprehensive system statistics.

**Returns:** Statistics object with performance metrics

## Pose Vector System

### Vector Composition

The RSMT system converts BVH poses into high-dimensional vectors for similarity analysis:

#### Bone Components

Each bone contributes to the pose vector based on:

```javascript
// Position components (if enabled)
vector.push(bone.position.x * weight * positionWeight);
vector.push(bone.position.y * weight * positionWeight);
vector.push(bone.position.z * weight * positionWeight);

// Rotation components (if enabled)
vector.push(bone.rotation.x * weight * rotationWeight);
vector.push(bone.rotation.y * weight * rotationWeight);
vector.push(bone.rotation.z * weight * rotationWeight);
vector.push(bone.rotation.w * weight * rotationWeight);
```

#### Bone Weighting

Different bones have different importance weights:

```javascript
const boneWeights = {
    'hips': 1.0,           // Root motion - highest weight
    'spine': 0.8,          // Core posture
    'chest': 0.7,          // Upper body orientation
    'leftUpperArm': 0.7,   // Arm gestures
    'rightUpperArm': 0.7,
    'leftUpperLeg': 0.8,   // Leg positioning
    'rightUpperLeg': 0.8,
    'head': 0.4,           // Head orientation
    'leftHand': 0.5,       // Hand positions
    'rightHand': 0.5,
    'leftFoot': 0.6,       // Foot placement
    'rightFoot': 0.6
};
```

### Similarity Computation

Similarity between pose vectors uses cosine similarity:

```javascript
function computeSimilarity(vector1, vector2) {
    const dotProduct = vector1.reduce((sum, val, i) => sum + val * vector2[i], 0);
    const norm1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
    const norm2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
    
    return dotProduct / (norm1 * norm2);
}
```

### Vector Optimization

- **Normalization**: Vectors are normalized to unit length for consistent comparison
- **Dimensionality**: Configurable vector dimensions (default 128) for optimal performance
- **Caching**: Computed vectors are cached to avoid recomputation

## Transition Generation

### DeepPhase Integration

The system integrates with DeepPhase neural networks for natural transition generation:

#### Mock Model (Development)

For development and testing, a mock DeepPhase model provides interpolated transitions:

```javascript
class MockDeepPhaseModel {
    async generateTransition(params) {
        // Simulate neural network processing
        const frames = [];
        
        for (let i = 0; i < frameCount; i++) {
            const progress = i / (frameCount - 1);
            const frame = interpolatePoses(
                params.fromPose, 
                params.toPose, 
                progress
            );
            frames.push(frame);
        }
        
        return { frames, quality: 0.85 };
    }
}
```

#### Real Model Integration

For production use, integrate with actual DeepPhase models:

```javascript
// Load real DeepPhase model
await rsmt.rsmt.initializeDeepPhase('/models/deepphase/model.onnx');

// Model will be used automatically for transition generation
```

### Transition Quality Assessment

The system evaluates transition quality through multiple metrics:

```javascript
const qualityMetrics = {
    smoothness: 0.92,        // Frame-to-frame variation
    naturalness: 0.88,       // Motion believability
    goalAchievement: 0.95,   // Target pose reaching
    physicsConsistency: 0.90 // Physics law adherence
};
```

### Fallback Mechanisms

If DeepPhase is unavailable, the system falls back to:

1. **Linear Interpolation**: Simple pose blending
2. **Spline Interpolation**: Smooth curve-based transitions
3. **Key Frame Matching**: Direct frame substitution

## Timeline Integration

### Track Management

RSMT creates a dedicated timeline track for transitions:

```javascript
// Track configuration
const transitionTrack = {
    name: 'rsmt_transitions',
    type: 'rsmt',
    priority: 75,
    weight: 1.0,
    channels: 'all',
    blendMode: 'replace'
};
```

### Clip Generation

Each transition generates a timeline clip:

```javascript
const clip = {
    id: 'rsmt_transition_12345',
    startTime: 5000,           // ms
    duration: 1500,            // ms
    type: 'rsmt_transition',
    frames: transitionFrames,
    metadata: {
        targetAnimation: 'dance',
        quality: 0.88,
        method: 'deepphase'
    }
};
```

### Synchronization

RSMT synchronizes with timeline events:

```javascript
// Timeline event handlers
timeline.on('play', () => rsmt.resumeProcessing());
timeline.on('pause', () => rsmt.pauseTransitions());
timeline.on('timeUpdate', (time) => rsmt.updateTime(time));
```

## Animation Library Management

### Auto-Discovery

The system automatically discovers animation files:

```javascript
// Common animation file patterns
const patterns = [
    'idle.json', 'walk.json', 'run.json',
    'dance.json', 'jump.json', 'wave.json',
    'sit.json', 'stand.json', 'turn.json'
];

// Discovery process
for (const pattern of patterns) {
    try {
        const response = await fetch(`${animationsPath}${pattern}`);
        if (response.ok) {
            await loadAnimation(pattern.replace('.json', ''), response);
        }
    } catch (error) {
        // File not found, continue
    }
}
```

### Animation Processing

Each loaded animation is processed for RSMT use:

```javascript
const processedAnimation = {
    name: 'walk',
    fps: 30,
    frames: convertedFrames,      // Normalized BVH frames
    duration: 3.5,               // seconds
    poseVectors: generatedVectors, // Pose vectors for all frames
    metadata: {
        totalFrames: 105,
        originalFormat: 'json_tracks',
        processedAt: timestamp
    }
};
```

### Format Support

RSMT supports multiple animation formats:

#### JSON Track Format
```javascript
{
    "body": {
        "fps": 30,
        "frames": 185,
        "tracks": [
            { "key": "hips.loc", "type": "vec3" },
            { "key": "hips.rot", "type": "quat" }
        ]
    }
}
```

#### BVH Array Format
```javascript
{
    "motionData": [
        [x, y, z, rx, ry, rz, ...], // Frame 0
        [x, y, z, rx, ry, rz, ...], // Frame 1
        // ...
    ]
}
```

## Advanced Features

### Transition Queuing

Multiple transitions can be queued and processed sequentially:

```javascript
// Queue multiple transitions
await rsmt.requestTransition('walk', { delay: 0 });
await rsmt.requestTransition('run', { delay: 2000 });
await rsmt.requestTransition('jump', { delay: 4000 });

// Queue is processed automatically
```

### Smart Batching

Related transitions are batched for efficiency:

```javascript
// Batch process related transitions
const batch = [
    { from: 'idle', to: 'walk' },
    { from: 'walk', to: 'run' },
    { from: 'run', to: 'jump' }
];

await rsmt.batchProcessTransitions(batch);
```

### Transition Styles

Different transition styles affect generation:

```javascript
const styles = {
    natural: {
        smoothness: 0.8,
        speed: 1.0,
        exaggeration: 0.2
    },
    smooth: {
        smoothness: 0.95,
        speed: 0.8,
        exaggeration: 0.1
    },
    dynamic: {
        smoothness: 0.6,
        speed: 1.3,
        exaggeration: 0.4
    },
    precise: {
        smoothness: 0.9,
        speed: 0.9,
        exaggeration: 0.0
    }
};
```

### Quality Control

Transitions can be filtered by quality thresholds:

```javascript
// Set quality requirements
rsmt.setQualityThreshold({
    minimum: 0.7,        // Reject transitions below this quality
    target: 0.85,        // Preferred quality level
    retries: 3           // Max attempts to achieve target quality
});
```

## Performance Optimization

### Caching Strategy

RSMT implements multi-level caching:

#### Pose Vector Caching
```javascript
// Cache pose vectors by animation and frame
const vectorCache = new Map();
const cacheKey = `${animationName}_${frameIndex}`;
vectorCache.set(cacheKey, poseVector);
```

#### Transition Caching
```javascript
// Cache generated transitions
const transitionCache = new Map();
const cacheKey = generateTransitionCacheKey(fromPose, toPose, options);
transitionCache.set(cacheKey, generatedTransition);
```

#### Similarity Caching
```javascript
// Cache similarity computations
const similarityCache = new Map();
const cacheKey = `${pose1Hash}_${pose2Hash}`;
similarityCache.set(cacheKey, similarity);
```

### Memory Management

Automatic cleanup prevents memory leaks:

```javascript
// Cache size limits
const cacheConfig = {
    maxPoseVectors: 1000,
    maxTransitions: 100,
    maxSimilarities: 5000,
    cleanupInterval: 30000,  // 30 seconds
    staleThreshold: 300000   // 5 minutes
};

// Automatic cleanup
setInterval(() => {
    rsmt.cleanupStaleCache();
}, cacheConfig.cleanupInterval);
```

### Parallel Processing

Where possible, operations are parallelized:

```javascript
// Parallel pose vector generation
const vectorPromises = frames.map(frame => 
    generatePoseVectorAsync(frame)
);
const vectors = await Promise.all(vectorPromises);

// Parallel similarity computation
const similarities = await Promise.all(
    targetVectors.map(vector => 
        computeSimilarityAsync(currentVector, vector)
    )
);
```

### Performance Monitoring

Built-in performance tracking:

```javascript
const perfStats = rsmt.getPerformanceStats();
console.log('Performance metrics:', {
    averageTransitionTime: perfStats.avgTransitionTime,
    cacheHitRate: perfStats.cacheHitRate,
    memoryUsage: perfStats.memoryUsage,
    vectorComputations: perfStats.vectorComputations,
    transitionsGenerated: perfStats.transitionsGenerated
});
```

## Troubleshooting

### Common Issues

#### 1. Animation Loading Failures

**Problem**: Animations fail to load from assets folder

**Solutions:**
```javascript
// Check file paths
console.log('Animations path:', rsmt.config.animationsPath);

// Verify file accessibility
try {
    const response = await fetch('/assets/animations/walk.json');
    console.log('File accessible:', response.ok);
} catch (error) {
    console.error('File access error:', error);
}

// Use absolute paths
const rsmt = new RSMTTimelineIntegration(timeline, {
    animationsPath: 'http://localhost:3000/assets/animations/'
});
```

#### 2. Poor Transition Quality

**Problem**: Generated transitions look unnatural

**Solutions:**
```javascript
// Increase similarity threshold
rsmt.setSimilarityThreshold(0.9);

// Adjust pose vector weights
const rsmt = new RSMTBVHConverter({
    positionWeight: 0.4,
    rotationWeight: 0.6,
    includeVelocity: true
});

// Use longer transition duration
await rsmt.requestTransition('target', { duration: 2.0 });
```

#### 3. Performance Issues

**Problem**: Slow transition generation

**Solutions:**
```javascript
// Enable caching
rsmt.enableCaching(true);

// Reduce vector dimensions
const rsmt = new RSMTBVHConverter({
    vectorDimensions: 64  // Reduced from 128
});

// Limit concurrent transitions
rsmt.setMaxConcurrentTransitions(2);

// Preload common transitions
await rsmt.preloadTransitions([
    { from: 'idle', to: 'walk' },
    { from: 'walk', to: 'run' }
]);
```

#### 4. Memory Leaks

**Problem**: Memory usage increases over time

**Solutions:**
```javascript
// Enable automatic cleanup
rsmt.enableAutoCleanup({
    interval: 30000,     // 30 seconds
    maxCacheSize: 100    // Maximum cached items
});

// Manual cleanup
rsmt.clearCache();

// Dispose when done
rsmt.dispose();
```

### Debug Tools

#### Enable Debug Logging

```javascript
// Enable detailed logging
rsmt.setDebugMode(true);

// Set log level
rsmt.setLogLevel('verbose');

// Custom log handler
rsmt.onLog = (level, message, data) => {
    console.log(`[RSMT ${level}] ${message}`, data);
};
```

#### Performance Profiling

```javascript
// Enable profiling
rsmt.enableProfiling(true);

// Get profiling data
const profile = rsmt.getProfilingData();
console.log('Performance profile:', profile);
```

#### Transition Visualization

```javascript
// Enable transition debug visualization
rsmt.enableDebugVisualization({
    showPoseVectors: true,
    showSimilarityScores: true,
    showTransitionPaths: true
});
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| RSMT001 | Animation not found | Check animation name and library |
| RSMT002 | Pose vector extraction failed | Verify BVH frame format |
| RSMT003 | No suitable match found | Lower similarity threshold |
| RSMT004 | Transition generation failed | Check DeepPhase model status |
| RSMT005 | Timeline integration error | Verify timeline compatibility |

### Best Practices

1. **Load animations early** in the application lifecycle
2. **Use appropriate similarity thresholds** (0.7-0.9 typical range)
3. **Monitor memory usage** and enable cleanup for long-running applications
4. **Cache frequently used transitions** to improve performance
5. **Test with various animation types** to ensure robust matching
6. **Use quality assessment** to filter poor transitions
7. **Implement fallback mechanisms** for critical transition points

---

For a complete interactive demonstration, see the `rsmt_timeline_demo.html` file which provides hands-on experience with all RSMT features.
