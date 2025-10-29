# Integration Testing Strategy for 3D Ichika VRM System

## Overview

This document outlines the comprehensive testing strategy for validating the integrated 3D animated Ichika VRM system. All tests are designed to work with Playwright and include proper shell timeout configurations as per repository standards.

## Test Categories

### 1. Component Validation Tests

#### VRM Loading and Rendering Tests
```javascript
// dev/web_viewer/tests/integration/vrm-loading-validation.spec.js
const { test, expect } = require('@playwright/test');

test('VRM avatar loads correctly in classroom scene', async ({ page }) => {
  test.setTimeout(120000); // Shell timeout compliance
  
  await page.goto('/demos/ichika_full_classroom_experience.html');
  
  // Wait for VRM loading
  await page.waitForSelector('[data-testid="vrm-loaded"]', { timeout: 60000 });
  
  // Validate VRM properties
  const vrmStats = await page.evaluate(() => {
    return window.ichikaVRM ? {
      loaded: !!window.ichikaVRM.scene,
      boneCount: window.ichikaVRM.humanoid?.bones.length || 0,
      meshCount: window.ichikaVRM.scene?.children.length || 0
    } : null;
  });
  
  expect(vrmStats).toBeTruthy();
  expect(vrmStats.loaded).toBe(true);
  expect(vrmStats.boneCount).toBeGreaterThan(20); // VRM humanoid bones
  expect(vrmStats.meshCount).toBeGreaterThan(0);
});

test('Classroom scene renders with proper lighting', async ({ page }) => {
  test.setTimeout(90000);
  
  await page.goto('/demos/ichika_full_classroom_experience.html');
  await page.waitForSelector('[data-testid="scene-ready"]');
  
  const sceneInfo = await page.evaluate(() => ({
    lights: window.classroomScene?.lights?.length || 0,
    meshes: window.classroomScene?.meshes?.length || 0,
    hasClassroom: !!window.classroomModel
  }));
  
  expect(sceneInfo.hasClassroom).toBe(true);
  expect(sceneInfo.lights).toBeGreaterThan(2); // Ambient + directional + point lights
});
```

#### Animation System Validation Tests
```javascript  
// dev/web_viewer/tests/integration/animation-system-validation.spec.js
test('BVH timeline system processes frames correctly', async ({ page }) => {
  test.setTimeout(150000);
  
  await page.goto('/demos/ichika_animation_showcase.html');
  
  // Start idle animation
  await page.click('[data-testid="start-idle"]');
  await page.waitForTimeout(2000);
  
  const animationStats = await page.evaluate(() => ({
    isPlaying: window.bvhTimeline?.isPlaying || false,
    currentFrame: window.bvhTimeline?.getCurrentFrame() || 0,
    frameRate: window.bvhTimeline?.frameRate || 0
  }));
  
  expect(animationStats.isPlaying).toBe(true);
  expect(animationStats.frameRate).toBe(30);
  expect(animationStats.currentFrame).toBeGreaterThan(0);
});

test('Animation preemption works correctly', async ({ page }) => {
  test.setTimeout(120000);
  
  await page.goto('/demos/ichika_animation_showcase.html');
  
  // Start idle, then trigger gesture
  await page.click('[data-testid="start-idle"]');
  await page.waitForTimeout(1000);
  await page.click('[data-testid="wave-gesture"]');
  
  // Verify preemption occurred
  const preemptionResult = await page.evaluate(() => ({
    activeAnimation: window.animationOrchestrator?.getCurrentAnimation(),
    transitionSmooth: window.animationOrchestrator?.lastTransitionSmooth || false
  }));
  
  expect(preemptionResult.activeAnimation).toBe('wave');
  expect(preemptionResult.transitionSmooth).toBe(true);
});
```

### 2. Audio-Visual Synchronization Tests

```javascript
// dev/web_viewer/tests/integration/audio-visual-sync.spec.js  
test('TTS audio drives mouth movements accurately', async ({ page }) => {
  test.setTimeout(180000); // Extended for TTS processing
  
  await page.goto('/demos/ichika_voice_sync_demo.html');
  
  // Trigger TTS with known text
  await page.fill('[data-testid="tts-input"]', 'Hello, welcome to our class');
  await page.click('[data-testid="speak-button"]');
  
  // Wait for TTS to start
  await page.waitForSelector('[data-testid="tts-playing"]');
  
  // Monitor viseme synchronization
  const syncResults = await page.evaluate(() => {
    return new Promise((resolve) => {
      const results = {
        audioStarted: false,
        visemesTriggered: false,
        syncAccuracy: 0,
        mouthMovementFrames: 0
      };
      
      let frameCount = 0;
      const checkSync = () => {
        frameCount++;
        if (window.ttsAudio?.currentTime > 0) results.audioStarted = true;
        if (window.visemeDriver?.activeViseme) {
          results.visemesTriggered = true;
          results.mouthMovementFrames++;
        }
        
        if (frameCount < 180) { // 6 seconds at 30fps
          requestAnimationFrame(checkSync);
        } else {
          results.syncAccuracy = results.mouthMovementFrames / frameCount;
          resolve(results);
        }
      };
      checkSync();
    });
  });
  
  expect(syncResults.audioStarted).toBe(true);
  expect(syncResults.visemesTriggered).toBe(true);
  expect(syncResults.syncAccuracy).toBeGreaterThan(0.3); // At least 30% of frames have mouth movement
});

test('Audio2Gesture generates appropriate gestures', async ({ page }) => {
  test.setTimeout(150000);
  
  await page.goto('/demos/ichika_voice_sync_demo.html');
  
  // Enable gesture generation  
  await page.check('[data-testid="enable-gestures"]');
  await page.fill('[data-testid="tts-input"]', 'Let me show you something important');
  await page.click('[data-testid="speak-button"]');
  
  await page.waitForSelector('[data-testid="gestures-active"]', { timeout: 30000 });
  
  const gestureResults = await page.evaluate(() => ({
    gesturesGenerated: window.gestureTimeline?.frames?.length || 0,
    armMovementDetected: window.gestureTracker?.armMovementCount || 0,
    gestureIntensity: window.gestureTracker?.averageIntensity || 0
  }));
  
  expect(gestureResults.gesturesGenerated).toBeGreaterThan(0);
  expect(gestureResults.armMovementDetected).toBeGreaterThan(0);
});
```

### 3. Interactive Behavior Tests

```javascript
// dev/web_viewer/tests/integration/interactive-behavior.spec.js
test('Voice input triggers appropriate responses', async ({ page }) => {
  test.setTimeout(200000); // Extended for voice processing
  
  await page.goto('/demos/ichika_full_classroom_experience.html');
  
  // Simulate voice input
  await page.evaluate(() => {
    window.simulateVoiceInput('Can you point to the blackboard?');
  });
  
  // Wait for processing and response
  await page.waitForSelector('[data-testid="response-generated"]', { timeout: 60000 });
  
  const responseResult = await page.evaluate(() => ({
    responseGenerated: !!window.lastResponse,
    animationTriggered: window.animationOrchestrator?.getCurrentAnimation(),
    audioPlaying: window.ttsAudio?.duration > 0
  }));
  
  expect(responseResult.responseGenerated).toBe(true);
  expect(responseResult.animationTriggered).toBeTruthy();
  expect(responseResult.audioPlaying).toBe(true);
});

test('Classroom actions execute naturally', async ({ page }) => {
  test.setTimeout(120000);
  
  await page.goto('/demos/ichika_full_classroom_experience.html');
  
  // Trigger classroom-specific action
  await page.click('[data-testid="point-at-board"]');
  await page.waitForTimeout(3000);
  
  const actionResult = await page.evaluate(() => ({
    actionCompleted: window.actionTracker?.lastAction === 'point_at_board',
    ichikaPosition: window.ichikaVRM?.scene?.position,
    lookDirection: window.ichikaVRM?.lookAt?.target
  }));
  
  expect(actionResult.actionCompleted).toBe(true);
  expect(actionResult.ichikaPosition).toBeTruthy();
});
```

### 4. Performance Validation Tests

```javascript
// dev/web_viewer/tests/integration/performance-validation.spec.js
test('System maintains target framerate during animation', async ({ page }) => {
  test.setTimeout(180000);
  
  await page.goto('/demos/performance_benchmark_suite.html');
  
  // Start performance monitoring
  await page.click('[data-testid="start-performance-test"]');
  await page.waitForTimeout(10000); // Run for 10 seconds
  
  const perfResults = await page.evaluate(() => window.performanceMonitor?.getResults());
  
  expect(perfResults.averageFPS).toBeGreaterThan(25); // Minimum acceptable
  expect(perfResults.frameTimeVariance).toBeLessThan(20); // Stable timing
  expect(perfResults.memoryUsage).toBeLessThan(512 * 1024 * 1024); // 512MB limit
});

test('WebGPU acceleration improves performance', async ({ page }) => {
  test.setTimeout(150000);
  
  await page.goto('/demos/performance_benchmark_suite.html');
  
  // Test with WebGPU disabled
  await page.evaluate(() => window.disableWebGPU());
  await page.click('[data-testid="run-cpu-benchmark"]');
  const cpuResults = await page.evaluate(() => window.getBenchmarkResults());
  
  // Test with WebGPU enabled  
  await page.evaluate(() => window.enableWebGPU());
  await page.click('[data-testid="run-gpu-benchmark"]');
  const gpuResults = await page.evaluate(() => window.getBenchmarkResults());
  
  // WebGPU should provide significant performance improvement
  expect(gpuResults.averageFPS).toBeGreaterThan(cpuResults.averageFPS * 1.2);
});
```

### 5. Visual Regression Tests

```javascript
// dev/web_viewer/tests/integration/visual-regression.spec.js
test('Animation quality remains consistent', async ({ page }) => {
  test.setTimeout(120000);
  
  await page.goto('/demos/ichika_animation_showcase.html');
  
  // Capture baseline animation frames
  await page.click('[data-testid="start-idle"]');
  await page.waitForTimeout(2000);
  
  const screenshot1 = await page.screenshot({ 
    clip: { x: 0, y: 100, width: 800, height: 600 }
  });
  
  await page.waitForTimeout(3000);
  const screenshot2 = await page.screenshot({
    clip: { x: 0, y: 100, width: 800, height: 600 }
  });
  
  // Verify animation is progressing (screenshots should be different)
  expect(screenshot1).not.toEqual(screenshot2);
});
```

## Test Execution Strategy

### Continuous Integration Setup
```bash
# Shell timeout compliant test execution
#!/bin/bash
set -e

# Component validation (fastest)
timeout 300s npm run test:integration:components

# Animation system tests  
timeout 600s npm run test:integration:animation

# Audio-visual sync (most time-intensive)
timeout 900s npm run test:integration:audio-sync

# Interactive behavior
timeout 600s npm run test:integration:behavior

# Performance validation
timeout 400s npm run test:integration:performance

# Visual regression
timeout 300s npm run test:integration:visual
```

### Package.json Integration
```json
{
  "scripts": {
    "test:integration": "timeout 1200s playwright test dev/web_viewer/tests/integration --reporter=line",
    "test:integration:components": "timeout 300s playwright test dev/web_viewer/tests/integration/vrm-loading-validation.spec.js dev/web_viewer/tests/integration/animation-system-validation.spec.js --reporter=line",
    "test:integration:audio-sync": "timeout 900s playwright test dev/web_viewer/tests/integration/audio-visual-sync.spec.js --reporter=line",
    "test:integration:behavior": "timeout 600s playwright test dev/web_viewer/tests/integration/interactive-behavior.spec.js --reporter=line",
    "test:integration:performance": "timeout 400s playwright test dev/web_viewer/tests/integration/performance-validation.spec.js --reporter=line",
    "test:integration:visual": "timeout 300s playwright test dev/web_viewer/tests/integration/visual-regression.spec.js --reporter=line"
  }
}
```

## Test Data and Fixtures

### Mock Voice Input
```javascript
// dev/web_viewer/tests/fixtures/voice-simulation.js
class VoiceSimulator {
  static simulateVoiceInput(text, options = {}) {
    const { duration = text.length * 100, energy = 0.8 } = options;
    
    // Simulate ASR processing
    window.speechRecognitionResult = {
      transcript: text,
      confidence: 0.95,
      timestamp: Date.now()
    };
    
    // Trigger speech recognition event
    window.dispatchEvent(new CustomEvent('speechresult', {
      detail: { results: [{ transcript: text }] }
    }));
  }
}
```

### Animation Test Data
```javascript
// dev/web_viewer/tests/fixtures/animation-data.js
const testAnimations = {
  idle: {
    duration: 5000,
    expectedBones: ['Hips', 'Spine', 'Head', 'LeftArm', 'RightArm'],
    frameCount: 150
  },
  wave: {
    duration: 3000,
    expectedBones: ['RightShoulder', 'RightArm', 'RightWrist'],
    frameCount: 90
  },
  pointAtBoard: {
    duration: 4000,
    expectedBones: ['RightArm', 'Head', 'Spine'],
    targetPosition: { x: 2, y: 1.5, z: -3 }
  }
};
```

## Validation Checkpoints

### Pre-Commit Tests
- Component loading validation
- Basic animation playback
- Audio system initialization

### Integration Tests (Full Suite)
- Audio-visual synchronization
- Interactive behavior responses  
- Performance benchmarks
- Visual regression detection

### Release Validation
- Full user journey simulation
- Cross-browser compatibility
- Performance under load
- Memory leak detection

## Test Environment Setup

### Browser Configuration
```javascript  
// playwright.config.js addition for integration tests
{
  name: 'integration-tests',
  testDir: './dev/web_viewer/tests/integration',
  use: {
    browserName: 'chromium',
    actionTimeout: 60_000,
    navigationTimeout: 60_000,
    expect: { timeout: 30_000 },
    launchOptions: {
      args: [
        '--enable-precise-memory-info',
        '--enable-features=WebGPU,SharedArrayBuffer',
        '--enable-webgl',
        '--disable-web-security',
        '--use-fake-device-for-media-stream',
        '--use-fake-ui-for-media-stream',
        '--enable-unsafe-webgpu' // For WebGPU testing
      ]
    }
  }
}
```

This testing strategy ensures comprehensive validation of all system components while maintaining compliance with the repository's shell timeout requirements.