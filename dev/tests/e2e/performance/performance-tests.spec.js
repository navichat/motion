import { test, expect } from '@playwright/test';

/**
 * Performance Tests for Motion Viewer 3D Environment
 * 
 * These tests monitor rendering performance, memory usage, and frame rates
 * to ensure the 3D environment maintains good performance.
 */

test.describe('3D Environment Performance Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Enable performance monitoring
    await page.addInitScript(() => {
      // Performance monitoring setup
      window.performanceMetrics = {
        frameCount: 0,
        lastFrameTime: performance.now(),
        frameTimes: [],
        memoryUsage: [],
        renderErrors: []
      };
      
      // Monitor frame rate
      const originalRAF = window.requestAnimationFrame;
      window.requestAnimationFrame = function(callback) {
        return originalRAF(function(timestamp) {
          const metrics = window.performanceMetrics;
          metrics.frameCount++;
          
          if (metrics.lastFrameTime) {
            const frameTime = timestamp - metrics.lastFrameTime;
            metrics.frameTimes.push(frameTime);
            
            // Keep only last 60 frames for rolling average
            if (metrics.frameTimes.length > 60) {
              metrics.frameTimes.shift();
            }
          }
          
          metrics.lastFrameTime = timestamp;
          
          // Monitor memory if available
          if (performance.memory) {
            metrics.memoryUsage.push({
              timestamp: timestamp,
              used: performance.memory.usedJSHeapSize,
              total: performance.memory.totalJSHeapSize,
              limit: performance.memory.jsHeapSizeLimit
            });
            
            // Keep only last 100 memory samples
            if (metrics.memoryUsage.length > 100) {
              metrics.memoryUsage.shift();
            }
          }
          
          callback(timestamp);
        });
      };
      
      // Monitor WebGL errors
      const originalGetError = WebGLRenderingContext.prototype.getError;
      WebGLRenderingContext.prototype.getError = function() {
        const error = originalGetError.call(this);
        if (error !== this.NO_ERROR) {
          window.performanceMetrics.renderErrors.push({
            timestamp: performance.now(),
            error: error
          });
        }
        return error;
      };
    });
    
    await page.goto('/');
    await page.waitForSelector('.motion-viewer', { state: 'visible' });
    await page.waitForSelector('.loading-overlay', { state: 'hidden', timeout: 10000 });
  });

  test('performance: classroom environment maintains 30+ FPS', async ({ page }) => {
    // Switch to classroom environment
    await page.selectOption('.environment-selector', 'classroom');
    await page.waitForTimeout(2000);
    
    // Let it render for 5 seconds to collect performance data
    await page.waitForTimeout(5000);
    
    // Get performance metrics
    const metrics = await page.evaluate(() => {
      const perf = window.performanceMetrics;
      const avgFrameTime = perf.frameTimes.reduce((a, b) => a + b, 0) / perf.frameTimes.length;
      const fps = 1000 / avgFrameTime;
      
      return {
        frameCount: perf.frameCount,
        averageFPS: fps,
        minFrameTime: Math.min(...perf.frameTimes),
        maxFrameTime: Math.max(...perf.frameTimes),
        renderErrors: perf.renderErrors.length
      };
    });
    
    console.log('Classroom Performance Metrics:', metrics);
    
    // Assert performance requirements
    expect(metrics.averageFPS).toBeGreaterThan(30);
    expect(metrics.renderErrors).toBe(0);
    expect(metrics.frameCount).toBeGreaterThan(150); // ~30 FPS * 5 seconds
  });

  test('performance: memory usage remains stable', async ({ page }) => {
    // Only run if memory monitoring is available
    const hasMemoryAPI = await page.evaluate(() => !!performance.memory);
    test.skip(!hasMemoryAPI, 'Memory API not available');
    
    // Get initial memory usage
    const initialMemory = await page.evaluate(() => performance.memory.usedJSHeapSize);
    
    // Stress test: switch between environments multiple times
    const environments = ['classroom', 'stage', 'studio', 'outdoor'];
    
    for (let i = 0; i < 3; i++) {
      for (const env of environments) {
        await page.selectOption('.environment-selector', env);
        await page.waitForTimeout(1000);
      }
    }
    
    // Force garbage collection if available
    await page.evaluate(() => {
      if (window.gc) {
        window.gc();
      }
    });
    
    await page.waitForTimeout(2000);
    
    // Get final memory usage
    const finalMemory = await page.evaluate(() => performance.memory.usedJSHeapSize);
    const memoryIncrease = finalMemory - initialMemory;
    const memoryIncreaseMB = memoryIncrease / 1024 / 1024;
    
    console.log(`Memory increase: ${memoryIncreaseMB.toFixed(2)} MB`);
    
    // Memory increase should be reasonable (less than 50MB)
    expect(memoryIncreaseMB).toBeLessThan(50);
  });

  test('performance: environment switching is smooth', async ({ page }) => {
    const environments = ['classroom', 'stage', 'studio', 'outdoor'];
    const switchTimes = [];
    
    for (const env of environments) {
      const startTime = await page.evaluate(() => performance.now());
      
      await page.selectOption('.environment-selector', env);
      
      // Wait for environment to fully load (no loading indicators visible)
      await page.waitForFunction(() => {
        return !document.querySelector('.loading-overlay:not([style*="display: none"])');
      }, { timeout: 5000 });
      
      const endTime = await page.evaluate(() => performance.now());
      const switchTime = endTime - startTime;
      switchTimes.push(switchTime);
      
      console.log(`${env} environment switch time: ${switchTime.toFixed(2)}ms`);
    }
    
    // All environment switches should complete within 3 seconds
    switchTimes.forEach(time => {
      expect(time).toBeLessThan(3000);
    });
    
    // Average switch time should be reasonable
    const avgSwitchTime = switchTimes.reduce((a, b) => a + b, 0) / switchTimes.length;
    expect(avgSwitchTime).toBeLessThan(2000);
  });

  test('performance: canvas resizing maintains performance', async ({ page }) => {
    // Get initial FPS
    await page.waitForTimeout(2000);
    
    const initialMetrics = await page.evaluate(() => {
      const perf = window.performanceMetrics;
      const avgFrameTime = perf.frameTimes.reduce((a, b) => a + b, 0) / perf.frameTimes.length;
      return { fps: 1000 / avgFrameTime, frameCount: perf.frameCount };
    });
    
    // Resize viewport multiple times
    await page.setViewportSize({ width: 800, height: 600 });
    await page.waitForTimeout(1000);
    
    await page.setViewportSize({ width: 1200, height: 800 });
    await page.waitForTimeout(1000);
    
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForTimeout(1000);
    
    await page.setViewportSize({ width: 800, height: 600 });
    await page.waitForTimeout(2000);
    
    // Get final FPS after resizing
    const finalMetrics = await page.evaluate(() => {
      const perf = window.performanceMetrics;
      // Clear old frame times and get fresh measurement
      perf.frameTimes = [];
      return new Promise(resolve => {
        setTimeout(() => {
          const avgFrameTime = perf.frameTimes.reduce((a, b) => a + b, 0) / perf.frameTimes.length;
          resolve({ fps: 1000 / avgFrameTime, frameCount: perf.frameCount });
        }, 1000);
      });
    });
    
    console.log('Before resize FPS:', initialMetrics.fps);
    console.log('After resize FPS:', finalMetrics.fps);
    
    // FPS should not degrade significantly after resizing
    const fpsDrop = initialMetrics.fps - finalMetrics.fps;
    expect(fpsDrop).toBeLessThan(10); // Allow max 10 FPS drop
    expect(finalMetrics.fps).toBeGreaterThan(25); // Minimum acceptable FPS
  });

  test('performance: WebGL context remains stable', async ({ page }) => {
    // Monitor WebGL context for stability
    const contextEvents = await page.evaluate(() => {
      return new Promise(resolve => {
        const events = [];
        const canvas = document.querySelector('canvas');
        
        if (canvas) {
          canvas.addEventListener('webglcontextlost', (event) => {
            events.push({ type: 'lost', timestamp: performance.now() });
          });
          
          canvas.addEventListener('webglcontextrestored', (event) => {
            events.push({ type: 'restored', timestamp: performance.now() });
          });
        }
        
        // Test for 10 seconds
        setTimeout(() => resolve(events), 10000);
      });
    });
    
    // No context loss events should occur during normal operation
    expect(contextEvents.filter(e => e.type === 'lost')).toHaveLength(0);
    
    // Verify WebGL context is still active
    const contextActive = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return false;
      
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return gl && !gl.isContextLost();
    });
    
    expect(contextActive).toBe(true);
  });

  test('performance: animation playback maintains smooth framerate', async ({ page }) => {
    // This test will be expanded when animation system is fully implemented
    // For now, test basic animation interface performance
    
    // Click animation selector
    await page.click('.selector-button:has-text("Select Animation")');
    await page.waitForTimeout(500);
    
    // Select first animation (if available)
    const firstAnimation = page.locator('.dropdown-item').first();
    if (await firstAnimation.count() > 0) {
      await firstAnimation.click();
      await page.waitForTimeout(500);
      
      // Click play button
      await page.click('.play-button');
      
      // Monitor performance during animation
      await page.waitForTimeout(3000);
      
      const animationMetrics = await page.evaluate(() => {
        const perf = window.performanceMetrics;
        const avgFrameTime = perf.frameTimes.reduce((a, b) => a + b, 0) / perf.frameTimes.length;
        return {
          fps: 1000 / avgFrameTime,
          renderErrors: perf.renderErrors.length
        };
      });
      
      console.log('Animation Performance:', animationMetrics);
      
      // Animation should maintain good performance
      expect(animationMetrics.fps).toBeGreaterThan(25);
      expect(animationMetrics.renderErrors).toBe(0);
    } else {
      console.log('No animations available for testing');
    }
  });
});
