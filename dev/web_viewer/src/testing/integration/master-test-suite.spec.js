/**
 * Master Test Suite Configuration
 * Coordinates all unit and integration tests for the avatar system
 */

import { test, expect } from '@playwright/test';

test.describe('Avatar System Test Suite', () => {
  // Test configuration
  const testConfig = {
    baseUrl: 'http://localhost:8081',
    timeout: 30000,
    retries: 2
  };

  test.beforeEach(async ({ page }) => {
    // Set up common test environment
    await page.goto(testConfig.baseUrl);
    await page.waitForLoadState('networkidle');
  });

  test('should run comprehensive system validation', async ({ page }) => {
    // Load import map configuration
    await page.addScriptTag({ 
      type: 'module',
      content: `
        import { ImportMap } from '/config/import-map.js';
        window.testImportMap = ImportMap;
      `
    });

    const result = await page.evaluate(async () => {
      const results = {
        importMapLoaded: !!window.testImportMap,
        coreModulesAccessible: false,
        aiModulesAccessible: false,
        avatarModulesAccessible: false,
        audioModulesAccessible: false,
        computeModulesAccessible: false,
        errors: []
      };

      try {
        // Test core modules
        const coreModules = [
          'TaskManager',
          'FibonacciHeap', 
          'SystemPerformanceAnalyzer'
        ];
        
        let coreModulesFound = 0;
        for (const moduleName of coreModules) {
          const modulePath = window.testImportMap?.getPath('core', moduleName);
          if (modulePath) {
            coreModulesFound++;
          }
        }
        results.coreModulesAccessible = coreModulesFound === coreModules.length;

        // Test AI modules
        const aiModules = [
          'AIModelJobs',
          'TinyLlamaWorker',
          'WhisperWorker',
          'RealJobFactory'
        ];
        
        let aiModulesFound = 0;
        for (const moduleName of aiModules) {
          const modulePath = window.testImportMap?.getPath('ai', moduleName);
          if (modulePath) {
            aiModulesFound++;
          }
        }
        results.aiModulesAccessible = aiModulesFound > 0;

        // Test avatar modules
        const avatarModules = [
          'VRMLoader',
          'BVHProcessor',
          'AnimationBlender'
        ];
        
        let avatarModulesFound = 0;
        for (const moduleName of avatarModules) {
          const modulePath = window.testImportMap?.getPath('avatar', moduleName);
          if (modulePath) {
            avatarModulesFound++;
          }
        }
        results.avatarModulesAccessible = avatarModulesFound > 0;

        // Test compute modules
        const computeModules = [
          'WebGPUBackend',
          'WebNNBackend', 
          'WASMBackend'
        ];
        
        let computeModulesFound = 0;
        for (const moduleName of computeModules) {
          const modulePath = window.testImportMap?.getPath('compute', moduleName);
          if (modulePath) {
            computeModulesFound++;
          }
        }
        results.computeModulesAccessible = computeModulesFound > 0;

      } catch (error) {
        results.errors.push(error.message);
      }

      return results;
    });

    expect(result.importMapLoaded).toBe(true);
    expect(result.coreModulesAccessible).toBe(true);
    expect(result.errors.length).toBe(0);
  });

  test('should validate all test files exist and are accessible', async ({ page }) => {
    const testFiles = [
      '/tests/unit/ai/ai-model-jobs.spec.js',
      '/tests/unit/avatar/avatar-animation.spec.js',
      '/tests/unit/audio/audio-processing.spec.js',
      '/tests/unit/compute/compute-backend.spec.js',
      '/tests/unit/motion/motion-processing.spec.js',
      '/tests/unit/system/system-integration.spec.js',
      '/tests/integration/e2e/capture-ai-results.spec.js'
    ];

    for (const testFile of testFiles) {
      const response = await page.goto(testConfig.baseUrl + testFile);
      expect(response.status()).toBeLessThan(400);
    }
  });

  test('should validate demo files are accessible with correct imports', async ({ page }) => {
    const demoFiles = [
      '/demos/ai-inference/task-manager-demo.html',
      '/demos/avatar-animation/vrm_test_animation_conversation.html',
      '/demos/audio-processing/automated_kokoro_test.html'
    ];

    for (const demoFile of demoFiles) {
      await page.goto(testConfig.baseUrl + demoFile);
      
      // Check for JavaScript errors
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      
      // Wait for potential dynamic imports
      await page.waitForTimeout(2000);
      
      // Verify no critical import errors
      const hasImportErrors = errors.some(error => 
        error.includes('import') || error.includes('module') || error.includes('404')
      );
      
      expect(hasImportErrors).toBe(false);
    }
  });

  test('should verify folder structure integrity', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test that expected folders are accessible
      const expectedPaths = [
        '/src/core/',
        '/src/ai/jobs/',
        '/src/ai/workers/',
        '/src/avatar/vrm/',
        '/src/avatar/animation/',
        '/src/audio/tts/',
        '/src/audio/stt/',
        '/src/compute/backends/',
        '/tests/unit/',
        '/tests/integration/',
        '/demos/',
        '/config/',
        '/tools/',
        '/assets/'
      ];

      const results = {};
      
      for (const path of expectedPaths) {
        try {
          const response = await fetch(path);
          results[path] = {
            accessible: response.status < 400,
            status: response.status
          };
        } catch (error) {
          results[path] = {
            accessible: false,
            error: error.message
          };
        }
      }

      return results;
    });

    // Verify that most paths are accessible (some might be 403 forbidden but still exist)
    const accessiblePaths = Object.values(result).filter(r => r.accessible || r.status === 403);
    const totalPaths = Object.keys(result).length;
    
    expect(accessiblePaths.length / totalPaths).toBeGreaterThan(0.7); // At least 70% accessible
  });
});
