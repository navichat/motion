# Avatar Animation System Documentation

## Overview

The Avatar Animation System provides comprehensive support for VRM characters, BVH motion processing, and real-time animation blending in the WebNN/WebGPU/WASM powered avatar platform. This system enables realistic character animation through motion capture integration, AI-driven gesture generation, and smooth motion transitions.

## Architecture

### Core Components

#### VRM Character System (`src/avatar/vrm/`)
- **VRMLoader.js**: Load and parse VRM character files
- **VRMValidator.js**: Validate VRM file structure and compatibility
- **VRMOptimizer.js**: Optimize VRM assets for web performance
- **VRMAnimator.js**: Handle VRM-specific animation features

#### Animation Engine (`src/avatar/animation/`)
- **AnimationBlender.js**: Blend multiple animations smoothly
- **AnimationController.js**: High-level animation state management
- **TimelineManager.js**: Manage animation timing and synchronization
- **ExpressionController.js**: Handle facial expressions and morph targets

#### Motion Processing (`src/avatar/motion/`)
- **BVHProcessor.js**: Parse and process BVH motion capture data
- **MotionAnalyzer.js**: Analyze motion patterns and features
- **MotionRetargeter.js**: Adapt motions between different character rigs
- **RSMTProcessor.js**: Real-time Stylized Motion Transition processing

### Data Flow

```
Audio Input → Gesture Generation → Motion Processing → Animation Blending → VRM Rendering
     ↓              ↓                    ↓                    ↓              ↓
   VAD/STT → Audio2Gesture → BVH Processing → Timeline Mgmt → VRM Display
```

## Features

### VRM Character Support

#### File Format Compatibility
- **VRM 0.x**: Full support for VRM specification 0.x
- **VRM 1.0**: Enhanced support for latest VRM 1.0 features
- **GLB/GLTF**: Base 3D model format support
- **Extensions**: Support for VRM-specific extensions and metadata

#### Character Features
- **Bone Structure**: Humanoid bone mapping and hierarchy
- **Morph Targets**: Facial expressions and shape keys
- **Materials**: PBR materials with VRM-specific properties
- **Physics**: Spring bone physics for hair and clothing
- **Look-at**: Eye tracking and gaze control

#### Performance Optimization
- **LOD (Level of Detail)**: Multiple quality levels for performance scaling
- **Culling**: Frustum and occlusion culling for efficiency
- **Batching**: Draw call optimization for multiple characters
- **Compression**: Texture and geometry compression

### Motion Processing

#### BVH Motion Capture
```javascript
// Example BVH processing
const bvhProcessor = new BVHProcessor();
const motionData = await bvhProcessor.parse(bvhFileContent);

// Extract key motion features
const analysis = await bvhProcessor.analyze(motionData);
console.log('Motion duration:', analysis.duration);
console.log('Frame rate:', analysis.frameRate);
console.log('Joint count:', analysis.jointCount);
```

#### Motion Analysis
- **Velocity Analysis**: Calculate movement speed and acceleration
- **Rhythm Detection**: Identify periodic patterns in motion
- **Gesture Recognition**: Classify motion types (walking, waving, etc.)
- **Contact Detection**: Identify ground contact points for feet

#### Motion Retargeting
```javascript
// Retarget motion between different characters
const retargeter = new MotionRetargeter();
const retargetedMotion = await retargeter.retarget(
  sourceMotion,
  sourceCharacter.skeleton,
  targetCharacter.skeleton
);
```

### Animation Blending

#### Blend Types
- **Linear Blending**: Simple weighted average between animations
- **Spherical Blending**: Smooth rotation interpolation (SLERP)
- **Additive Blending**: Layer animations on top of base motion
- **Masking**: Blend different animations for different body parts

#### Timeline Management
```javascript
// Create animation timeline
const timeline = new TimelineManager();
timeline.addClip('idle', idleAnimation, { loop: true });
timeline.addClip('wave', waveAnimation, { duration: 2.0 });
timeline.addClip('walk', walkAnimation, { loop: true, weight: 0.5 });

// Control playback
timeline.play('wave');
timeline.crossfade('idle', 'walk', 1.0); // 1 second crossfade
```

#### Real-time Blending
- **Frame-based Blending**: Per-frame animation mixing
- **Bone-level Control**: Individual bone animation weights
- **Expression Layers**: Separate facial expression blending
- **Physics Integration**: Blend with physics-based animation

### RSMT Integration

#### Real-time Stylized Motion Transition
- **Motion Graphs**: Define transition paths between motion clips
- **Style Transfer**: Apply style variations to base motions
- **Context Awareness**: Adapt transitions based on environment
- **Quality Preservation**: Maintain motion quality during transitions

#### Transition Types
```javascript
// Define motion transitions
const rsmt = new RSMTProcessor();

// Create transition graph
rsmt.addMotion('idle', idleClip);
rsmt.addMotion('walk', walkClip);
rsmt.addMotion('run', runClip);

// Define transition rules
rsmt.addTransition('idle', 'walk', { duration: 0.5, conditions: ['speed > 0.1'] });
rsmt.addTransition('walk', 'run', { duration: 0.3, conditions: ['speed > 2.0'] });
rsmt.addTransition('*', 'idle', { duration: 1.0, conditions: ['speed < 0.05'] });
```

## Testing

### Unit Tests (`tests/unit/avatar/`)

#### VRM Loading Tests
```javascript
test('should load VRM character', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const loader = new VRMLoader();
    const vrm = await loader.load('/assets/models/test-character.vrm');
    return {
      loaded: !!vrm,
      hasBones: vrm.humanoid && vrm.humanoid.bones.length > 0,
      hasMaterials: vrm.materials && vrm.materials.length > 0
    };
  });
  
  expect(result.loaded).toBe(true);
  expect(result.hasBones).toBe(true);
});
```

#### Animation Blending Tests
```javascript
test('should blend animations smoothly', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const blender = new AnimationBlender();
    const result = blender.blend(
      idleAnimation, 
      walkAnimation, 
      0.5 // 50% blend weight
    );
    return {
      blended: !!result,
      frameCount: result.frames.length,
      hasTransitions: result.transitions.length > 0
    };
  });
  
  expect(result.blended).toBe(true);
  expect(result.frameCount).toBeGreaterThan(0);
});
```

#### Motion Processing Tests
```javascript
test('should process BVH motion data', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const processor = new BVHProcessor();
    const parsed = await processor.parse(sampleBVHData);
    return {
      parsed: !!parsed,
      hasHierarchy: !!parsed.hierarchy,
      frameCount: parsed.motion.frames,
      jointCount: parsed.hierarchy.joints.length
    };
  });
  
  expect(result.parsed).toBe(true);
  expect(result.frameCount).toBeGreaterThan(0);
});
```

### Integration Tests

#### End-to-end Animation Pipeline
```javascript
test('should complete full animation pipeline', async ({ page }) => {
  // Test: Audio → Gesture → Motion → Animation → VRM
  const result = await page.evaluate(async () => {
    // 1. Generate gesture from audio
    const gestureGen = new Audio2GestureProcessor();
    const gesture = await gestureGen.process(audioData);
    
    // 2. Convert to BVH motion
    const motionGen = new MotionGenerator();
    const motion = await motionGen.fromGesture(gesture);
    
    // 3. Apply to VRM character
    const animator = new VRMAnimator();
    const animation = await animator.applyMotion(vrmCharacter, motion);
    
    return {
      gestureGenerated: !!gesture,
      motionCreated: !!motion,
      animationApplied: !!animation
    };
  });
  
  expect(result.gestureGenerated).toBe(true);
  expect(result.motionCreated).toBe(true);
  expect(result.animationApplied).toBe(true);
});
```

## Performance

### Optimization Strategies

#### Character Optimization
- **Bone Reduction**: Reduce bone count for performance-critical scenarios
- **Texture Atlasing**: Combine textures to reduce draw calls
- **Mesh Simplification**: Reduce polygon count while preserving quality
- **Animation Compression**: Compress keyframe data for storage efficiency

#### Animation Optimization
- **Keyframe Reduction**: Remove redundant keyframes
- **Quaternion Compression**: Compress rotation data
- **Delta Compression**: Store only changes between frames
- **LOD Animation**: Use simplified animations for distant characters

#### Memory Management
```javascript
// Example memory management
class AnimationManager {
  constructor() {
    this.animationCache = new Map();
    this.maxCacheSize = 100; // MB
  }
  
  loadAnimation(url) {
    if (this.animationCache.has(url)) {
      return this.animationCache.get(url);
    }
    
    // Load new animation
    const animation = this.loadFromURL(url);
    
    // Check cache size
    if (this.getCacheSize() > this.maxCacheSize) {
      this.evictOldAnimations();
    }
    
    this.animationCache.set(url, animation);
    return animation;
  }
}
```

### Performance Targets
- **Frame Rate**: 60 FPS for single character, 30 FPS for multiple characters
- **Loading Time**: < 2 seconds for VRM character loading
- **Memory Usage**: < 100MB per character including animations
- **Animation Latency**: < 16ms for real-time gesture application

## Integration with AI Systems

### Audio-driven Animation
```javascript
// Real-time audio-to-animation pipeline
class AudioAnimationPipeline {
  async processAudio(audioChunk) {
    // 1. Speech recognition
    const text = await this.whisper.transcribe(audioChunk);
    
    // 2. Generate response
    const response = await this.llm.generate(text);
    
    // 3. Generate gestures
    const gestures = await this.audio2gesture.process(audioChunk);
    
    // 4. Apply to character
    this.character.speak(response, gestures);
  }
}
```

### Motion Style Transfer
```javascript
// Apply style to base motions
const styleTransfer = new MotionStyleTransfer();
const styledMotion = await styleTransfer.apply(baseMotion, {
  style: 'energetic',
  intensity: 0.8,
  personalityTraits: ['confident', 'friendly']
});
```

## Configuration

### Animation Settings
```javascript
// Global animation configuration
const animationConfig = {
  frameRate: 30,
  blendMode: 'linear',
  compressionLevel: 0.8,
  enablePhysics: true,
  lodLevels: 3,
  cacheSize: '100MB'
};
```

### Character Settings
```javascript
// Per-character configuration
const characterConfig = {
  enableMorphTargets: true,
  enableSpringBones: true,
  lookAtController: true,
  expressionBlending: true,
  maxAnimationLayers: 4
};
```

## Troubleshooting

### Common Issues

#### VRM Loading Failures
```javascript
// Debug VRM loading
try {
  const vrm = await loader.load(vrmUrl);
} catch (error) {
  if (error.message.includes('CORS')) {
    console.error('CORS issue - check server configuration');
  } else if (error.message.includes('format')) {
    console.error('Invalid VRM file format');
  }
}
```

#### Animation Synchronization
```javascript
// Debug animation timing
const timeline = new TimelineManager();
timeline.onUpdate = (time, deltaTime) => {
  console.log(`Animation time: ${time}ms, delta: ${deltaTime}ms`);
};
```

#### Performance Issues
```javascript
// Monitor performance
const performanceMonitor = new AnimationPerformanceMonitor();
performanceMonitor.track('animation_update', () => {
  character.updateAnimation(deltaTime);
});
```

For more information, see the main [REORGANIZATION_STATUS.md](../REORGANIZATION_STATUS.md) documentation.
