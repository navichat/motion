# Pathfinding BVH Integration Documentation

## Overview

The Pathfinding BVH system provides intelligent path planning and animation generation for 3D avatars using multiple neural network backends. It creates keyframe-based movement sequences that integrate seamlessly with the BVH Timeline system.

## Core Components

### 1. PathfindingBVHPlanner (`PathfindingBVHPlanner.js`)

The main pathfinding engine that handles:
- **A* Pathfinding**: Optimal path calculation with obstacle avoidance
- **Animation Backend Integration**: Coordinates with Audio2Gesture, DeepMimic, RSMT, and FaceFormer
- **Keyframe Generation**: Creates timed movement sequences with animation metadata
- **Locomotion Library**: Manages walking, running, turning, and transition animations

#### Key Features:
```javascript
// Initialize pathfinder
const pathfinder = new PathfindingBVHPlanner({
    gridResolution: 0.5,        // meters per grid cell
    maxPlanningDistance: 50,    // maximum planning distance
    characterRadius: 0.3,       // character collision radius
    movementConfig: {
        maxSpeed: 2.0,          // m/s
        preferredSpeed: 1.2,    // m/s
        maxAcceleration: 3.0,   // m/s²
        maxTurnRate: Math.PI    // rad/s
    }
});

// Load environment and backends
await pathfinder.initialize(backends, environment);

// Plan path to destination
const plan = await pathfinder.planPathToDestination(destination, {
    startTime: 0,
    frameRate: 30,
    enablePhysics: true,
    animationStyle: 'natural'
});
```

### 2. PathfindingTimelineIntegration (`PathfindingTimelineIntegration.js`)

Integration layer that connects pathfinding with the BVH timeline system:
- **Real-time Path Monitoring**: Tracks character movement and triggers replanning
- **Animation Backend Coordination**: Orchestrates multiple animation systems
- **Timeline Layer Management**: Creates and manages pathfinding animation layers
- **Interactive Destination Setting**: Handles user input for path planning

#### Integration Example:
```javascript
// Initialize integration
const integration = new PathfindingTimelineIntegration(timeline, {
    pathfinding: {
        gridResolution: 0.5,
        autoReplan: true
    },
    layerName: 'pathfinding_movement',
    priority: 5
});

// Initialize with backends
await integration.initialize({
    audio2gesture: audio2gestureConverter,
    deepmimic: deepmimicConverter,
    rsmt: rsmtConverter,
    faceformer: faceformerConverter
}, environment);

// Plan path and add to timeline
const plan = await integration.planPathTo(destination, {
    enableGestures: true,
    enableFacial: true,
    enablePhysics: true,
    coordination: {
        enableTransitions: true,
        transitionDuration: 0.3
    }
});
```

### 3. Interactive Demo (`pathfinding_timeline_demo.html`)

Comprehensive demo showcasing all pathfinding features:
- **Visual Environment**: Interactive 2D navigation grid with obstacles and waypoints
- **Real-time Planning**: Click-to-set destinations with immediate path calculation
- **Backend Integration**: Mock implementations of all animation backends
- **Performance Monitoring**: Real-time statistics and system status
- **Timeline Playback**: Integration with BVH timeline system

## Technical Architecture

### Path Planning Pipeline

1. **Environment Analysis**
   ```javascript
   // Navigation grid initialization
   this.navigationGrid = {
       bounds: { minX: -25, maxX: 25, minZ: -25, maxZ: 25 },
       width: 100,
       height: 100,
       resolution: 0.5,
       cells: Array(10000).fill(0),  // 0 = free, 1 = blocked
       costs: Array(10000).fill(1.0) // movement cost multipliers
   };
   ```

2. **A* Pathfinding**
   ```javascript
   // Core A* algorithm with priority queue
   const openSet = new PriorityQueue();
   const gScore = new Map();
   const fScore = new Map();
   
   // Heuristic function (Manhattan + diagonal)
   heuristic(pos1, pos2) {
       const dx = Math.abs(pos1.x - pos2.x);
       const dz = Math.abs(pos1.z - pos2.z);
       return Math.max(dx, dz) + (Math.sqrt(2) - 1) * Math.min(dx, dz);
   }
   ```

3. **Path Smoothing**
   ```javascript
   // Multi-pass smoothing with collision checking
   smoothPath(path, options = {}) {
       const smoothingPasses = options.smoothingPasses || 2;
       for (let pass = 0; pass < smoothingPasses; pass++) {
           // Apply weighted averaging filter
           const smoothed = {
               x: prev.x * 0.25 + current.x * 0.5 + next.x * 0.25,
               y: prev.y * 0.25 + current.y * 0.5 + next.y * 0.25,
               z: prev.z * 0.25 + current.z * 0.5 + next.z * 0.25
           };
       }
   }
   ```

4. **Keyframe Generation**
   ```javascript
   // Generate timed keyframes with movement constraints
   async generateTimedKeyframes(path, options = {}) {
       const keyframes = [];
       let currentTime = options.startTime || 0;
       let currentVelocity = { ...this.planningState.currentVelocity };
       
       for (let i = 0; i < path.length; i++) {
           const keyframe = {
               time: currentTime,
               position: { ...path[i] },
               velocity: { ...currentVelocity },
               orientation: this.calculateOrientation(path[i], path[i + 1]),
               movementType: this.classifyMovement(segmentData),
               constraints: { maxSpeed, maxAcceleration }
           };
           keyframes.push(keyframe);
       }
   }
   ```

### Animation Backend Coordination

The system coordinates with multiple animation backends to create comprehensive character animation:

#### 1. Audio2Gesture Integration
```javascript
// Generate arm movements for locomotion
async coordinateWithAudio2Gesture(plan, options = {}) {
    for (const segment of plan.animationSequence) {
        if (segment.animationType === 'walk' || segment.animationType === 'run') {
            const gestureData = await this.generateMovementGestures(segment, options);
            await this.backends.audio2gesture.addGestureSequence({
                startTime: segment.startTime,
                duration: segment.duration,
                gestureType: 'locomotion',
                data: gestureData
            });
        }
    }
}
```

#### 2. FaceFormer Integration
```javascript
// Generate facial expressions based on movement
generateMovementExpressions(plan, options) {
    const expressions = [];
    for (const segment of plan.animationSequence) {
        let expression = 'neutral';
        if (segment.animationType === 'run') expression = 'focused';
        else if (segment.animationType === 'walk') expression = 'calm';
        
        expressions.push({
            startTime: segment.startTime,
            duration: segment.duration,
            expression: expression,
            intensity: 0.3
        });
    }
    return expressions;
}
```

#### 3. DeepMimic Integration
```javascript
// Physics-based refinement
async coordinateWithDeepMimic(plan, options = {}) {
    if (this.backends.deepmimic.refinePhysics) {
        const refinedPlan = await this.backends.deepmimic.refinePhysics({
            keyframes: plan.keyframes,
            constraints: options.constraints || {},
            quality: options.quality || 'high'
        });
        await this.applyPhysicsRefinement(refinedPlan);
    }
}
```

#### 4. RSMT Integration
```javascript
// Smooth transitions between animation types
async coordinateWithRSMT(plan, options = {}) {
    for (let i = 0; i < plan.animationSequence.length - 1; i++) {
        const currentSegment = plan.animationSequence[i];
        const nextSegment = plan.animationSequence[i + 1];
        
        if (currentSegment.animationType !== nextSegment.animationType) {
            const transition = await this.backends.rsmt.generateTransition(
                currentSegment.toKeyframe,
                nextSegment.animationType,
                { duration: 0.3, quality: 'high' }
            );
            await this.addTransitionToTimeline(transition, currentSegment.endTime);
        }
    }
}
```

### Timeline Integration

The pathfinding system creates timeline layers that integrate with the BVH compositor:

```javascript
// Create timeline layer
createTimelineLayer(bvhKeyframes, options = {}) {
    const layer = {
        id: options.layerId || `pathfinding_${Date.now()}`,
        name: 'Pathfinding Movement',
        type: 'pathfinding',
        priority: options.priority || 5,
        clips: [],
        metadata: {
            generatedBy: 'pathfinding',
            totalKeyframes: bvhKeyframes.length,
            startTime: bvhKeyframes[0]?.time || 0,
            endTime: bvhKeyframes[bvhKeyframes.length - 1]?.time || 0
        }
    };
    
    // Group keyframes into clips by animation segment
    let currentClip = null;
    for (const keyframe of bvhKeyframes) {
        if (!currentClip || keyframe.segment !== currentClip.segmentId) {
            if (currentClip) layer.clips.push(currentClip);
            currentClip = {
                id: `clip_${keyframe.segment}`,
                segmentId: keyframe.segment,
                startTime: keyframe.time,
                frames: [],
                animationType: keyframe.metadata.animationType
            };
        }
        currentClip.frames.push(keyframe.frame);
        currentClip.endTime = keyframe.time;
    }
    if (currentClip) layer.clips.push(currentClip);
    
    return layer;
}
```

## Usage Examples

### Basic Path Planning

```javascript
// Initialize system
const timeline = new BVHTimelineCompositor({ frameRate: 30 });
const pathfinding = new PathfindingTimelineIntegration(timeline);

// Set up environment
const environment = {
    bounds: { minX: -10, maxX: 10, minZ: -10, maxZ: 10 },
    obstacles: [
        { id: 'wall1', minX: 2, maxX: 3, minZ: -2, maxZ: 2 },
        { id: 'wall2', minX: -3, maxX: -2, minZ: 1, maxZ: 5 }
    ]
};

// Initialize with backends
await pathfinding.initialize(backends, environment);

// Plan path to destination
const destination = { x: 5, y: 0, z: 5 };
const plan = await pathfinding.planPathTo(destination, {
    enableGestures: true,
    enableFacial: false,
    animationStyle: 'energetic'
});

// Add to timeline and play
await pathfinding.addToTimeline(plan.timelineLayer, { autoPlay: true });
```

### Dynamic Obstacle Avoidance

```javascript
// Add obstacles dynamically
pathfinding.addObstacle('moving_obstacle', {
    minX: 1, maxX: 2, minZ: 1, maxZ: 2,
    cost: 5.0 // Higher cost = prefer to avoid
});

// Enable automatic replanning
pathfinding.config.autoReplan = true;
pathfinding.config.replanThreshold = 1.0; // meters

// Monitor for replanning events
pathfinding.on('pathReplanned', (data) => {
    console.log(`Path replanned (attempt #${data.replanCount})`);
});
```

### Multi-Modal Animation

```javascript
// Coordinate all animation backends
const plan = await pathfinding.planPathTo(destination, {
    coordination: {
        enableGestures: true,     // Audio2Gesture arm movements
        enableFacial: true,       // FaceFormer expressions
        enablePhysics: true,      // DeepMimic physics
        enableTransitions: true,  // RSMT smooth transitions
        gestures: {
            intensity: 0.8,
            style: 'confident'
        },
        facial: {
            expressiveness: 0.6,
            emotions: ['focused', 'determined']
        },
        physics: {
            quality: 'high',
            constraints: { maxForce: 1000 }
        },
        transitions: {
            duration: 0.3,
            quality: 'high'
        }
    }
});
```

### Interactive Destination Setting

```javascript
// Set up click-to-move
canvas.addEventListener('click', async (e) => {
    const worldPos = canvasToWorld(e.offsetX, e.offsetY);
    
    // Validate destination
    if (pathfinding.isValidDestination(worldPos)) {
        try {
            await pathfinding.setDestination(worldPos, {
                clearExisting: true,
                enableGestures: true
            });
        } catch (error) {
            console.error('Path planning failed:', error);
        }
    } else {
        console.warn('Invalid destination: blocked or out of bounds');
    }
});

// Preview path before committing
const preview = await pathfinding.previewPath(worldPos);
if (preview && preview.isValid) {
    // Show preview visualization
    drawPathPreview(preview.plan.path);
}
```

## Performance Optimization

### Caching System

```javascript
// Path caching for repeated destinations
const cacheKey = this.generatePathCacheKey(start, destination, options);
if (this.pathCache.has(cacheKey)) {
    return this.adaptCachedPath(this.pathCache.get(cacheKey), options);
}
```

### Grid Resolution Tuning

```javascript
// Balance between accuracy and performance
const pathfinder = new PathfindingBVHPlanner({
    gridResolution: 0.5,        // Fine grid for accuracy
    maxPlanningDistance: 25,    // Limit search space
    maxCacheSize: 100          // Limit memory usage
});
```

### Asynchronous Processing

```javascript
// Non-blocking path planning
async planPathToDestination(destination, options = {}) {
    // Use requestIdleCallback for heavy computations
    return new Promise((resolve) => {
        const planChunk = () => {
            // Process path planning in chunks
            if (this.shouldYield()) {
                requestIdleCallback(planChunk);
                return;
            }
            // Continue planning...
        };
        requestIdleCallback(planChunk);
    });
}
```

## API Reference

### PathfindingBVHPlanner

#### Constructor Options
- `gridResolution` (number): Grid cell size in meters (default: 0.5)
- `maxPlanningDistance` (number): Maximum planning distance (default: 50)
- `characterRadius` (number): Character collision radius (default: 0.3)
- `movementConfig` (object): Movement constraints and preferences

#### Methods
- `initialize(backends, environment)`: Initialize with animation backends and environment
- `planPathToDestination(destination, options)`: Plan path and generate animations
- `addObstacle(id, obstacle)`: Add obstacle to navigation grid
- `removeObstacle(id)`: Remove obstacle from navigation grid
- `updateCharacterState(position, orientation, velocity)`: Update character state

### PathfindingTimelineIntegration

#### Constructor Options
- `layerName` (string): Timeline layer name (default: 'pathfinding')
- `priority` (number): Animation priority (default: 5)
- `autoReplan` (boolean): Enable automatic replanning (default: true)
- `replanThreshold` (number): Distance threshold for replanning (default: 2.0)

#### Methods
- `initialize(backends, environment)`: Initialize integration system
- `planPathTo(destination, options)`: Plan path and integrate with timeline
- `setDestination(destination, options)`: Interactive destination setting
- `previewPath(destination, options)`: Preview path without execution
- `addObstacle(id, obstacle)`: Add dynamic obstacle
- `cancelCurrentPlan()`: Cancel active path planning

#### Events
- `pathPlanned`: Emitted when path planning completes
- `pathCompleted`: Emitted when character reaches destination
- `pathReplanned`: Emitted when automatic replanning occurs
- `planningError`: Emitted when path planning fails

## Troubleshooting

### Common Issues

1. **No Path Found**
   ```javascript
   // Check destination validity
   if (!pathfinder.isValidDestination(destination)) {
       console.error('Destination is blocked or out of bounds');
   }
   
   // Increase planning distance
   pathfinder.maxPlanningDistance = 100;
   
   // Reduce grid resolution for broader search
   pathfinder.gridResolution = 1.0;
   ```

2. **Performance Issues**
   ```javascript
   // Optimize grid size
   pathfinder.gridResolution = 1.0; // Larger cells = faster planning
   
   // Limit search area
   pathfinder.maxPlanningDistance = 25;
   
   // Use caching
   pathfinder.maxCacheSize = 50;
   ```

3. **Animation Coordination Issues**
   ```javascript
   // Check backend availability
   if (!backends.audio2gesture) {
       console.warn('Audio2Gesture backend not available');
   }
   
   // Disable problematic backends
   const plan = await pathfinding.planPathTo(destination, {
       coordination: {
           enableGestures: false, // Disable if causing issues
           enablePhysics: false
       }
   });
   ```

### Debug Mode

```javascript
// Enable debug logging
const integration = new PathfindingTimelineIntegration(timeline, {
    debugMode: true
});

// Monitor performance
integration.on('pathPlanned', (data) => {
    console.log(`Planning time: ${data.planningTime}ms`);
    console.log(`Keyframes generated: ${data.plan.keyframes.length}`);
});
```

## Integration with Existing Systems

### BVH Timeline Integration

The pathfinding system creates timeline layers that work seamlessly with the existing BVH timeline:

```javascript
// Add pathfinding layer to existing timeline
const timelineLayer = plan.timelineLayer;
await timeline.addLayer(timelineLayer);

// Blend with existing animations
timelineLayer.blendMode = 'additive';
timelineLayer.priority = 5; // Medium priority
```

### Character Controller Integration

```javascript
// Update pathfinding from character controller
characterController.on('positionChanged', (position, orientation, velocity) => {
    pathfinding.updateCharacterState(position, orientation, velocity);
});

// Use pathfinding to drive character controller
pathfinding.on('pathPlanned', (data) => {
    characterController.setTargetPath(data.plan.keyframes);
});
```

This pathfinding system provides a complete solution for intelligent character movement planning that integrates with all your existing animation backends and the BVH timeline system. The combination of A* pathfinding, multi-modal animation coordination, and real-time replanning creates a robust foundation for dynamic character navigation in your 3D avatar system.
