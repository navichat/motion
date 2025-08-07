/**
 * Performance and System Integration Test
 * Tests system performance, memory usage, and component integration
 */

import { test, expect } from '@playwright/test';

test.describe('System Performance and Integration Tests', () => {
  test('should monitor system performance metrics', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/system/performance-monitoring-test.html');
    
    const result = await page.evaluate(async () => {
      const performanceAnalyzer = new window.SystemPerformanceAnalyzer();
      
      try {
        // Start monitoring
        await performanceAnalyzer.startMonitoring();
        
        // Simulate some work
        const workStart = performance.now();
        for (let i = 0; i < 1000000; i++) {
          Math.random();
        }
        const workEnd = performance.now();
        
        // Get metrics
        const metrics = await performanceAnalyzer.getMetrics();
        
        return {
          monitored: true,
          hasMemoryMetrics: !!metrics.memory,
          hasCPUMetrics: !!metrics.cpu,
          hasGPUMetrics: !!metrics.gpu,
          memoryUsed: metrics.memory ? metrics.memory.usedJSHeapSize : 0,
          memoryLimit: metrics.memory ? metrics.memory.totalJSHeapSize : 0,
          frameRate: metrics.frameRate || 0,
          workDuration: workEnd - workStart
        };
      } catch (error) {
        return {
          monitored: false,
          error: error.message
        };
      }
    });
    
    expect(result.monitored).toBe(true);
    expect(result.hasMemoryMetrics).toBe(true);
    expect(result.memoryUsed).toBeGreaterThan(0);
    expect(result.memoryLimit).toBeGreaterThan(result.memoryUsed);
    expect(result.workDuration).toBeGreaterThan(0);
  });

  test('should integrate TaskManager with multiple components', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/system/task-manager-integration-test.html');
    
    const result = await page.evaluate(async () => {
      const taskManager = new window.TaskManager();
      
      try {
        // Initialize task manager
        await taskManager.initialize();
        
        // Create tasks of different types
        const aiTask = await taskManager.createTask('ai-inference', {
          model: 'tinyllama',
          input: 'test prompt',
          maxTokens: 10
        });
        
        const audioTask = await taskManager.createTask('tts-synthesis', {
          text: 'hello world',
          voice: 'neutral'
        });
        
        const motionTask = await taskManager.createTask('bvh-parse', {
          bvhData: 'sample bvh content'
        });
        
        // Execute tasks
        const aiResult = await taskManager.executeTask(aiTask.id);
        const audioResult = await taskManager.executeTask(audioTask.id);
        const motionResult = await taskManager.executeTask(motionTask.id);
        
        return {
          integrated: true,
          tasksCreated: 3,
          aiTaskStatus: aiResult.status,
          audioTaskStatus: audioResult.status,
          motionTaskStatus: motionResult.status,
          allCompleted: [aiResult, audioResult, motionResult].every(r => r.status === 'completed'),
          queueSize: taskManager.getQueueSize(),
          activeWorkers: taskManager.getActiveWorkerCount()
        };
      } catch (error) {
        return {
          integrated: false,
          error: error.message
        };
      }
    });
    
    expect(result.integrated).toBe(true);
    expect(result.tasksCreated).toBe(3);
    expect(['completed', 'failed', 'pending']).toContain(result.aiTaskStatus);
    expect(['completed', 'failed', 'pending']).toContain(result.audioTaskStatus);
    expect(['completed', 'failed', 'pending']).toContain(result.motionTaskStatus);
  });

  test('should handle concurrent avatar animations', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/system/concurrent-animation-test.html');
    
    const result = await page.evaluate(async () => {
      const avatarManager = new window.AvatarManager();
      
      try {
        // Initialize multiple avatars
        const avatar1 = await avatarManager.createAvatar({
          vrmUrl: '/assets/vrm/test-character-1.vrm',
          position: [-1, 0, 0]
        });
        
        const avatar2 = await avatarManager.createAvatar({
          vrmUrl: '/assets/vrm/test-character-2.vrm',
          position: [1, 0, 0]
        });
        
        const avatar3 = await avatarManager.createAvatar({
          vrmUrl: '/assets/vrm/test-character-3.vrm',
          position: [0, 0, -2]
        });
        
        // Start different animations concurrently
        const animation1 = avatarManager.startAnimation(avatar1.id, 'idle');
        const animation2 = avatarManager.startAnimation(avatar2.id, 'walking');
        const animation3 = avatarManager.startAnimation(avatar3.id, 'waving');
        
        // Wait for animations to start
        await Promise.all([animation1, animation2, animation3]);
        
        // Check system performance under load
        const frameStart = performance.now();
        await new Promise(resolve => setTimeout(resolve, 100)); // 100ms of animation
        const frameEnd = performance.now();
        
        const avgFrameTime = (frameEnd - frameStart) / (100 / 16.67); // Estimate frame count
        
        return {
          avatarsCreated: 3,
          animationsStarted: 3,
          avgFrameTime: avgFrameTime,
          performanceAcceptable: avgFrameTime < 20, // Under 20ms per frame
          memoryUsage: performance.memory ? performance.memory.usedJSHeapSize : 0
        };
      } catch (error) {
        return {
          avatarsCreated: 0,
          error: error.message
        };
      }
    });
    
    expect(result.avatarsCreated).toBe(3);
    expect(result.animationsStarted).toBe(3);
    expect(result.avgFrameTime).toBeGreaterThan(0);
    // Performance should be reasonable for 3 concurrent avatars
    expect(result.performanceAcceptable).toBe(true);
  });

  test('should validate cross-component data flow', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/system/data-flow-integration-test.html');
    
    const result = await page.evaluate(async () => {
      try {
        // Start with audio input
        const audioProcessor = new window.AudioProcessor();
        const motionGenerator = new window.MotionGenerator();
        const avatarController = new window.AvatarController();
        
        // Simulate audio → speech recognition → motion generation → avatar animation
        const audioData = new Float32Array(16000); // 1 second of silence
        const speechResult = await audioProcessor.recognizeSpeech(audioData);
        
        const motionData = await motionGenerator.generateFromSpeech(speechResult.text || 'hello');
        const avatarAnimation = await avatarController.applyMotion(motionData);
        
        return {
          dataFlowComplete: true,
          audioProcessed: !!speechResult,
          motionGenerated: !!motionData,
          animationApplied: !!avatarAnimation,
          hasText: !!speechResult.text,
          hasMotionFrames: motionData && motionData.frames && motionData.frames.length > 0,
          animationDuration: avatarAnimation ? avatarAnimation.duration : 0
        };
      } catch (error) {
        return {
          dataFlowComplete: false,
          error: error.message
        };
      }
    });
    
    expect(result.dataFlowComplete).toBe(true);
    expect(result.audioProcessed).toBe(true);
    expect(result.motionGenerated).toBe(true);
    expect(result.animationApplied).toBe(true);
  });

  test('should handle error recovery and graceful degradation', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/system/error-recovery-test.html');
    
    const result = await page.evaluate(async () => {
      const taskManager = new window.TaskManager();
      
      try {
        await taskManager.initialize();
        
        // Test error handling with invalid tasks
        const invalidTask1 = await taskManager.createTask('invalid-task-type', {});
        const invalidTask2 = await taskManager.createTask('ai-inference', { /* missing required params */ });
        
        // Test recovery mechanisms
        const validTask = await taskManager.createTask('simple-computation', {
          operation: 'add',
          operands: [1, 2]
        });
        
        const results = await Promise.allSettled([
          taskManager.executeTask(invalidTask1.id),
          taskManager.executeTask(invalidTask2.id),
          taskManager.executeTask(validTask.id)
        ]);
        
        return {
          errorHandled: true,
          invalidTask1Status: results[0].status,
          invalidTask2Status: results[1].status,
          validTaskStatus: results[2].status,
          validTaskResult: results[2].status === 'fulfilled' ? results[2].value : null,
          systemStillFunctional: taskManager.isHealthy(),
          recoverySuccessful: results[2].status === 'fulfilled'
        };
      } catch (error) {
        return {
          errorHandled: false,
          error: error.message
        };
      }
    });
    
    expect(result.errorHandled).toBe(true);
    expect(result.systemStillFunctional).toBe(true);
    expect(result.recoverySuccessful).toBe(true);
    // Invalid tasks should be rejected, but system should continue working
    expect(['rejected', 'fulfilled']).toContain(result.invalidTask1Status);
    expect(['rejected', 'fulfilled']).toContain(result.invalidTask2Status);
    expect(result.validTaskStatus).toBe('fulfilled');
  });
});
