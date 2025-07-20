# DeepMimic BVH Integration

A comprehensive system for integrating DeepMimic physics-based character animation with the BVH Timeline Compositor. This implementation enables goal-driven, physics-accurate character movement with real-time simulation and timeline integration.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Installation & Setup](#installation--setup)
4. [Quick Start Guide](#quick-start-guide)
5. [API Reference](#api-reference)
6. [Physics System](#physics-system)
7. [Motion Styles](#motion-styles)
8. [Goal Processing](#goal-processing)
9. [Integration Examples](#integration-examples)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)

## Overview

The DeepMimic BVH Integration provides a complete pipeline for physics-based character animation using deep reinforcement learning principles. The system converts high-level motion goals into detailed BVH animations through physics simulation, constraint solving, and motion style processing.

### Key Features

- **Physics-Based Animation**: Full physics simulation with constraints and contact detection
- **Goal-Driven Motion**: Natural language goal processing for character actions
- **Motion Styles**: Athletic, dramatic, precise, and natural movement styles
- **Real-time Simulation**: Live physics processing with timeline synchronization
- **Contact Constraints**: Ground contact and collision handling
- **Root Motion**: Proper character locomotion and positioning
- **Quality Metrics**: Performance analysis and motion quality assessment

## Core Components

### 1. DeepMimicBVHConverter

The core converter class that transforms DeepMimic neural network outputs into BVH format.

```javascript
const converter = new DeepMimicBVHConverter({
    scaleFactor: 1.0,
    frameRate: 30,
    motionStyle: 'natural',
    energyLevel: 1.0,
    physicsIntegration: true,
    rootMotionEnabled: true,
    contactConstraints: true
});
```

**Key Methods:**
- `processPhysicsStep()`: Execute single physics simulation step
- `convertToBVH()`: Convert physics state to BVH frame
- `handleContacts()`: Process contact constraints
- `validatePhysics()`: Check physics constraints

### 2. DeepMimicTimelineIntegration

Integration layer connecting DeepMimic with the BVH Timeline system.

```javascript
const integration = new DeepMimicTimelineIntegration(timeline, {
    realtime: false,
    trackName: 'deepmimic_physics',
    priority: 100,
    channels: ['root', 'body', 'arms', 'legs', 'spine', 'head']
});
```

**Key Methods:**
- `processGoal()`: Process motion goals into animations
- `queueGoals()`: Process multiple sequential goals
- `synchronizeWithTimeline()`: Sync with timeline playback
- `setMotionStyle()`: Change movement characteristics

## Installation & Setup

### Prerequisites

```bash
# Ensure you have the BVH Timeline Compositor
# Include the required script files in your HTML
```

### HTML Integration

```html
<!-- Include required scripts -->
<script src="BVHTimelineCompositor.js"></script>
<script src="DeepMimicBVHConverter.js"></script>
<script src="DeepMimicTimelineIntegration.js"></script>
```

### Basic Initialization

```javascript
// Initialize timeline
const timeline = new BVHTimelineCompositor({
    frameRate: 30,
    maxTracks: 20,
    enableBlending: true
});

// Initialize DeepMimic integration
const deepMimicIntegration = new DeepMimicTimelineIntegration(timeline, {
    realtime: false,
    trackName: 'deepmimic_physics',
    priority: 100
});

// Initialize the model (will use mock for demo)
await deepMimicIntegration.initializeDeepMimic('./models/deepmimic');
```

## Quick Start Guide

### Basic Motion Generation

```javascript
// Process a simple walking goal
const result = await deepMimicIntegration.processGoal('walk forward naturally', {
    startTime: 0,
    duration: 3.0,
    motionStyle: 'natural',
    energyLevel: 1.0
});

console.log(`Generated ${result.clips.length} animation clips`);
console.log(`Quality score: ${result.quality}`);
```

### Multiple Sequential Goals

```javascript
// Process a sequence of connected movements
const goals = [
    { description: 'walk forward naturally', duration: 3.0 },
    { description: 'turn around smoothly', duration: 2.0 },
    { description: 'run back quickly', duration: 2.5 },
    { description: 'stop and wave', duration: 2.0 }
];

const results = await deepMimicIntegration.queueGoals(goals);
console.log(`Total duration: ${results.totalDuration}s`);
```

### Timeline Playback

```javascript
// Play the generated animation
timeline.play();

// Get current frame for visualization
const currentFrame = timeline.getCurrentFrame();
if (currentFrame && currentFrame.physics) {
    // Render physics-based character pose
    renderCharacterPose(currentFrame);
}
```

## API Reference

### DeepMimicBVHConverter

#### Constructor Options

```javascript
{
    scaleFactor: 1.0,           // Scale factor for motion
    frameRate: 30,              // Animation frame rate
    motionStyle: 'natural',     // Motion style preset
    energyLevel: 1.0,           // Energy/intensity level
    physicsIntegration: true,   // Enable physics simulation
    rootMotionEnabled: true,    // Enable root motion
    contactConstraints: true,   // Enable contact constraints
    stabilityFactor: 1.0,       // Movement stability factor
    smoothingFactor: 0.1        // Motion smoothing amount
}
```

#### Key Methods

##### `processPhysicsStep(action, currentState, deltaTime)`

Execute a single physics simulation step.

**Parameters:**
- `action` (Object): Neural network action output
- `currentState` (Object): Current physics state
- `deltaTime` (Number): Time step in seconds

**Returns:** Updated physics state

##### `convertToBVH(state, frameNumber, time)`

Convert physics state to BVH frame format.

**Parameters:**
- `state` (Object): Physics simulation state
- `frameNumber` (Number): Frame index
- `time` (Number): Animation time

**Returns:** BVH frame object

##### `setMotionStyle(style)`

Change the motion style characteristics.

**Parameters:**
- `style` (String): Style name ('natural', 'athletic', 'dramatic', 'precise')

##### `setEnergyLevel(level)`

Adjust movement energy and intensity.

**Parameters:**
- `level` (Number): Energy level (0.1 - 2.0)

### DeepMimicTimelineIntegration

#### Constructor Options

```javascript
{
    realtime: false,                    // Real-time processing mode
    trackName: 'deepmimic_physics',     // Timeline track name
    priority: 100,                      // Track priority (higher = more important)
    channels: ['root', 'body', 'arms', 'legs', 'spine', 'head'],
    converter: {                        // Converter options
        scaleFactor: 1.0,
        frameRate: 30,
        motionStyle: 'natural',
        energyLevel: 1.0
    }
}
```

#### Key Methods

##### `processGoal(goalDescription, options)`

Process a motion goal into animation clips.

**Parameters:**
- `goalDescription` (String): Natural language goal description
- `options` (Object): Processing options

**Options:**
```javascript
{
    startTime: 0,                   // Start time in timeline
    duration: 3.0,                  // Goal duration in seconds
    motionStyle: 'natural',         // Motion style
    energyLevel: 1.0,              // Energy level
    separateBodyParts: true,       // Create separate clips for body parts
    enablePhysics: true,           // Enable physics simulation
    contactConstraints: true       // Enable contact constraints
}
```

**Returns:** Promise resolving to result object with clips, quality score, and timing info

##### `queueGoals(goals, options)`

Process multiple sequential goals.

**Parameters:**
- `goals` (Array): Array of goal objects
- `options` (Object): Processing options

**Goal Object:**
```javascript
{
    description: 'walk forward',    // Goal description
    duration: 3.0,                  // Duration in seconds
    motionStyle: 'natural',         // Motion style
    energyLevel: 1.0               // Energy level
}
```

**Returns:** Promise resolving to results summary

##### `synchronizeWithTimeline()`

Synchronize DeepMimic processing with timeline playback.

##### `setPhysicsEnabled(enabled)`

Enable or disable physics simulation.

**Parameters:**
- `enabled` (Boolean): Physics simulation state

##### `getStats()`

Get integration statistics and performance metrics.

**Returns:** Statistics object

## Physics System

### Physics Simulation

The DeepMimic system implements a comprehensive physics engine for character animation:

#### Core Physics Components

1. **Rigid Body Dynamics**: Full 3D physics simulation for character bones
2. **Constraint Solving**: Joint limits and contact constraints
3. **Contact Detection**: Ground and object collision handling
4. **Force Integration**: Gravity, friction, and applied forces

#### Physics State Structure

```javascript
{
    position: { x: 0, y: 0, z: 0 },        // World position
    rotation: { x: 0, y: 0, z: 0, w: 1 },  // Quaternion rotation
    velocity: { x: 0, y: 0, z: 0 },        // Linear velocity
    angularVelocity: { x: 0, y: 0, z: 0 }, // Angular velocity
    contacts: [],                           // Active contact points
    constraints: [],                        // Active constraints
    energy: 0,                             // Kinetic energy
    stability: 1.0                         // Stability measure
}
```

#### Contact Processing

```javascript
// Contact constraint example
{
    bone: 'leftFoot',
    contactPoint: { x: 0, y: 0, z: 0 },
    contactNormal: { x: 0, y: 1, z: 0 },
    contactForce: 500,
    startTime: 0.0,
    endTime: 0.5
}
```

### Physics Quality Metrics

The system provides comprehensive quality assessment:

```javascript
const qualityMetrics = {
    stability: 0.95,           // Movement stability (0-1)
    energyEfficiency: 0.85,    // Energy usage efficiency (0-1)
    goalAchievement: 0.90,     // Goal completion score (0-1)
    physicsViolations: 0,      // Number of physics violations
    contactQuality: 0.88,      // Contact constraint quality (0-1)
    naturalness: 0.92          // Motion naturalness score (0-1)
};
```

## Motion Styles

### Available Styles

#### 1. Natural Style
- **Characteristics**: Relaxed, everyday movement
- **Use Cases**: Casual walking, normal gestures
- **Energy Range**: 0.5 - 1.5

```javascript
deepMimicIntegration.setMotionStyle('natural');
```

#### 2. Athletic Style
- **Characteristics**: Dynamic, energetic movement
- **Use Cases**: Sports, running, jumping
- **Energy Range**: 1.0 - 2.0

```javascript
deepMimicIntegration.setMotionStyle('athletic');
```

#### 3. Dramatic Style
- **Characteristics**: Exaggerated, expressive movement
- **Use Cases**: Acting, dancing, emotional expression
- **Energy Range**: 0.8 - 2.0

```javascript
deepMimicIntegration.setMotionStyle('dramatic');
```

#### 4. Precise Style
- **Characteristics**: Controlled, accurate movement
- **Use Cases**: Technical tasks, careful actions
- **Energy Range**: 0.3 - 1.2

```javascript
deepMimicIntegration.setMotionStyle('precise');
```

### Style Customization

```javascript
// Custom style parameters
const customStyle = {
    name: 'custom',
    damping: 0.8,
    stiffness: 0.9,
    responsiveness: 0.7,
    smoothness: 0.8,
    energyMultiplier: 1.2
};

deepMimicIntegration.setCustomMotionStyle(customStyle);
```

## Goal Processing

### Goal Types

The system supports various types of motion goals:

#### 1. Locomotion Goals
```javascript
'walk forward naturally'
'run to the right quickly'
'jog in place steadily'
'march with high knees'
```

#### 2. Action Goals
```javascript
'jump high in place'
'reach for object above'
'kick ball with right foot'
'throw ball forward'
```

#### 3. Posture Goals
```javascript
'sit down gracefully'
'stand up straight'
'maintain balance on one foot'
'crouch down low'
```

#### 4. Expression Goals
```javascript
'wave hello enthusiastically'
'gesture while speaking'
'dance to music'
'stretch arms and back'
```

### Goal Processing Pipeline

1. **Goal Parsing**: Natural language processing of goal description
2. **Motion Planning**: Generate motion trajectory and keyframes
3. **Physics Simulation**: Run physics-based character simulation
4. **Quality Assessment**: Evaluate motion quality and goal achievement
5. **BVH Conversion**: Convert physics state to BVH format
6. **Timeline Integration**: Add clips to timeline with proper timing

### Advanced Goal Options

```javascript
const advancedGoal = {
    description: 'walk forward and stop at marker',
    duration: 5.0,
    motionStyle: 'natural',
    energyLevel: 1.0,
    constraints: [
        {
            type: 'position',
            target: { x: 5, y: 0, z: 0 },
            time: 4.0,
            priority: 'high'
        }
    ],
    preferences: {
        preferredFoot: 'left',
        movementSpeed: 'moderate',
        stopType: 'gradual'
    }
};

const result = await deepMimicIntegration.processAdvancedGoal(advancedGoal);
```

## Integration Examples

### Example 1: Basic Character Walk Cycle

```javascript
// Initialize system
const timeline = new BVHTimelineCompositor();
const deepMimic = new DeepMimicTimelineIntegration(timeline);
await deepMimic.initializeDeepMimic();

// Generate walk cycle
const walkResult = await deepMimic.processGoal('walk forward naturally', {
    duration: 4.0,
    motionStyle: 'natural',
    energyLevel: 1.0
});

// Play animation
timeline.play();
```

### Example 2: Complex Action Sequence

```javascript
// Queue multiple connected actions
const actionSequence = [
    { description: 'walk to position', duration: 3.0, motionStyle: 'natural' },
    { description: 'reach up high', duration: 2.0, motionStyle: 'dramatic' },
    { description: 'grab object', duration: 1.0, motionStyle: 'precise' },
    { description: 'walk back', duration: 3.0, motionStyle: 'natural' }
];

const results = await deepMimic.queueGoals(actionSequence);
console.log(`Sequence duration: ${results.totalDuration}s`);
```

### Example 3: Real-time Interactive Control

```javascript
// Set up real-time mode
const realtimeDeepMimic = new DeepMimicTimelineIntegration(timeline, {
    realtime: true,
    trackName: 'realtime_physics'
});

// Process goals as they come in
async function handleUserInput(goalText) {
    const result = await realtimeDeepMimic.processGoal(goalText, {
        startTime: timeline.getCurrentTime(),
        duration: 2.0
    });
    
    // Animation starts immediately
    return result;
}

// Example usage
await handleUserInput('wave hello');
await handleUserInput('turn around');
```

### Example 4: Custom Physics Parameters

```javascript
// Configure custom physics settings
const customConverter = new DeepMimicBVHConverter({
    scaleFactor: 1.2,
    frameRate: 60,
    physicsIntegration: true,
    contactConstraints: true,
    stabilityFactor: 0.8,
    customPhysics: {
        gravity: { x: 0, y: -12, z: 0 },
        friction: 0.8,
        restitution: 0.2,
        damping: 0.95
    }
});

const customIntegration = new DeepMimicTimelineIntegration(timeline, {
    converter: customConverter
});
```

### Example 5: Multi-Character Coordination

```javascript
// Set up multiple characters
const character1 = new DeepMimicTimelineIntegration(timeline, {
    trackName: 'character1_physics',
    priority: 100
});

const character2 = new DeepMimicTimelineIntegration(timeline, {
    trackName: 'character2_physics',
    priority: 99
});

// Coordinate movements
await character1.processGoal('walk forward', { startTime: 0, duration: 3 });
await character2.processGoal('walk beside', { startTime: 0.5, duration: 3 });
```

## Performance Optimization

### Optimization Strategies

#### 1. Frame Rate Management

```javascript
// Optimize for target frame rate
const optimizedConverter = new DeepMimicBVHConverter({
    frameRate: 30,  // Lower frame rate for performance
    adaptiveQuality: true,
    qualityThreshold: 0.8
});
```

#### 2. Physics LOD (Level of Detail)

```javascript
// Adjust physics detail based on distance/importance
deepMimicIntegration.setPhysicsLOD({
    distance: 10,        // Distance from camera
    minQuality: 0.5,     // Minimum quality level
    maxSimSteps: 100     // Maximum simulation steps
});
```

#### 3. Caching and Precomputation

```javascript
// Enable result caching
deepMimicIntegration.enableCaching({
    maxCacheSize: 100,   // Maximum cached results
    cacheByGoal: true,   // Cache by goal description
    cacheByStyle: true   // Cache by motion style
});
```

#### 4. Batch Processing

```javascript
// Process multiple goals in batch
const batchResults = await deepMimicIntegration.batchProcessGoals([
    { description: 'walk', duration: 3 },
    { description: 'run', duration: 2 },
    { description: 'jump', duration: 1 }
], {
    parallel: true,      // Process in parallel
    maxConcurrency: 4    // Limit concurrent processes
});
```

### Performance Monitoring

```javascript
// Get performance metrics
const perfMetrics = deepMimicIntegration.getPerformanceMetrics();
console.log(`Average simulation time: ${perfMetrics.avgSimTime}ms`);
console.log(`Cache hit rate: ${perfMetrics.cacheHitRate}%`);
console.log(`Memory usage: ${perfMetrics.memoryUsage}MB`);
```

## Troubleshooting

### Common Issues

#### 1. Physics Instability

**Problem**: Character jitters or falls through ground
**Solution**: 
```javascript
// Increase stability factor
deepMimicIntegration.converter.setStabilityFactor(1.5);

// Enable contact constraints
deepMimicIntegration.converter.contactConstraints = true;

// Reduce energy level
deepMimicIntegration.setEnergyLevel(0.8);
```

#### 2. Poor Goal Achievement

**Problem**: Character doesn't follow goals accurately
**Solution**:
```javascript
// Increase goal processing iterations
const result = await deepMimicIntegration.processGoal(goal, {
    maxIterations: 200,
    convergenceThreshold: 0.95,
    goalWeight: 2.0
});
```

#### 3. Timeline Synchronization Issues

**Problem**: Animation doesn't sync with timeline
**Solution**:
```javascript
// Force synchronization
deepMimicIntegration.synchronizeWithTimeline();

// Check timeline state
if (!timeline.isPlaying()) {
    timeline.play();
}

// Verify frame rates match
console.log(`Timeline FPS: ${timeline.frameRate}`);
console.log(`DeepMimic FPS: ${deepMimicIntegration.converter.frameRate}`);
```

#### 4. Memory Leaks

**Problem**: Memory usage increases over time
**Solution**:
```javascript
// Enable automatic cleanup
deepMimicIntegration.enableAutoCleanup({
    maxClips: 50,        // Maximum clips to keep
    cleanupInterval: 5000, // Cleanup every 5 seconds
    preserveRecent: 10   // Keep recent clips
});

// Manual cleanup
deepMimicIntegration.cleanup();
```

#### 5. Quality Issues

**Problem**: Generated animation looks unnatural
**Solution**:
```javascript
// Adjust quality settings
deepMimicIntegration.setQualitySettings({
    minQualityThreshold: 0.8,
    maxRetries: 5,
    qualityWeights: {
        stability: 0.3,
        naturalness: 0.4,
        goalAchievement: 0.3
    }
});
```

### Debug Tools

#### 1. Enable Debug Logging

```javascript
// Enable detailed logging
deepMimicIntegration.setDebugMode(true);

// Set log level
deepMimicIntegration.setLogLevel('verbose');
```

#### 2. Physics Visualization

```javascript
// Enable physics debug rendering
deepMimicIntegration.enablePhysicsDebug({
    showConstraints: true,
    showContacts: true,
    showForces: true,
    showTrajectory: true
});
```

#### 3. Performance Profiling

```javascript
// Enable performance profiling
deepMimicIntegration.enableProfiling(true);

// Get detailed timing breakdown
const profile = deepMimicIntegration.getProfilingData();
console.log('Profiling results:', profile);
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| DM001 | Model not loaded | Call `initializeDeepMimic()` |
| DM002 | Invalid goal format | Check goal description format |
| DM003 | Physics simulation failed | Reduce energy level or increase stability |
| DM004 | Timeline sync error | Call `synchronizeWithTimeline()` |
| DM005 | Memory limit exceeded | Enable automatic cleanup |

### Best Practices

1. **Always initialize the model** before processing goals
2. **Use appropriate motion styles** for different types of movement
3. **Monitor quality metrics** to ensure good animation results
4. **Enable caching** for repeated similar goals
5. **Set reasonable energy levels** to avoid physics instability
6. **Use batch processing** for multiple goals when possible
7. **Clean up regularly** to prevent memory leaks
8. **Test with mock model first** before using real DeepMimic model

---

For more examples and advanced usage, see the `deepmimic_timeline_demo.html` file which provides a complete interactive demonstration of all features.
