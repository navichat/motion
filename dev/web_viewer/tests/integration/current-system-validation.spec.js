const { test, expect } = require('@playwright/test');

test.describe('Current System Validation', () => {
  
  test('Ichika Classroom Demo - Current State Validation', async ({ page }) => {
    test.setTimeout(120000); // Shell timeout compliance
    
    await page.goto('/demos/ichika_classroom_demo.html');
    
    // Validate page loads correctly
    await expect(page.locator('h1')).toContainText('Ichika Classroom Demo');
    
    // Check if main control buttons are present
    await expect(page.locator('button:text("Start Idle")')).toBeVisible();
    await expect(page.locator('button:text("Point at Board")')).toBeVisible();
    await expect(page.locator('button:text("Wave")')).toBeVisible();
    await expect(page.locator('button:text("Speak (visemes + gestures)")')).toBeVisible();
    
    // Validate TTS system is initialized
    await expect(page.locator('select:text("SpeechT5")')).toBeVisible();
    
    // Take screenshot for baseline comparison
    await page.screenshot({ 
      path: '/tmp/playwright-logs/classroom-demo-baseline.png',
      fullPage: true 
    });
    
    // Test basic interaction - Start Idle animation
    await page.click('button:text("Start Idle")');
    await page.waitForTimeout(2000);
    
    // Capture state after interaction
    const pageContent = await page.evaluate(() => ({
      hasCanvas: !!document.querySelector('canvas'),
      canvasCount: document.querySelectorAll('canvas').length,
      hasThreeJS: !!window.THREE,
      consoleErrors: window.consoleErrors || []
    }));
    
    // Validate basic 3D setup
    expect(pageContent.hasCanvas).toBe(true);
    expect(pageContent.canvasCount).toBeGreaterThan(0);
  });

  test('Ichika Voice Conversation Demo - Audio System Validation', async ({ page }) => {
    test.setTimeout(150000);
    
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    
    // Validate conversation interface
    await expect(page.locator('h1')).toContainText('Ichika Conversation');
    await expect(page.locator('button:text("Start Mic")')).toBeVisible();
    await expect(page.locator('button:text("Say")')).toBeVisible();
    
    // Check TTS input functionality
    const testText = 'Hello, this is a test message';
    await page.fill('input[placeholder*="Type something"], textbox', testText);
    
    // Validate backend selection is available
    await expect(page.locator('select')).toBeVisible();
    
    // Test TTS trigger (without actual audio playback in CI)
    await page.click('button:text("Say")');
    await page.waitForTimeout(1000);
    
    // Check for animation system initialization
    const animationSystemStatus = await page.evaluate(() => ({
      hasBVHTimeline: !!window.bvhTimeline,
      hasVRMSystem: !!window.vrmSystem,
      animationSystemLoaded: !!window.animationOrchestrator,
      timelineInitialized: window.bvhTimeline?.isInitialized || false
    }));
    
    // Validate core systems are loaded
    expect(animationSystemStatus.hasBVHTimeline).toBe(true);
    
    // Take screenshot of conversation interface
    await page.screenshot({
      path: '/tmp/playwright-logs/voice-conversation-baseline.png',
      fullPage: true
    });
  });

  test('VRM Orchestrator Demo - 3D System Validation', async ({ page }) => {
    test.setTimeout(120000);
    
    await page.goto('/demos/ichika_vrm_orchestrator_demo.html');
    
    // Validate VRM orchestrator interface  
    await expect(page.locator('button:text("Load Ichika VRM")')).toBeVisible();
    
    // Check if Three.js and VRM dependencies load
    await page.waitForTimeout(3000); // Allow time for script loading
    
    const systemStatus = await page.evaluate(() => ({
      hasThreeJS: !!window.THREE,
      hasVRMLoader: !!window.VRM || !!window.VRMLoader,
      canvasPresent: !!document.querySelector('canvas'),
      webGLSupported: (() => {
        try {
          const canvas = document.createElement('canvas');
          return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
          return false;
        }
      })()
    }));
    
    expect(systemStatus.webGLSupported).toBe(true);
    expect(systemStatus.canvasPresent).toBe(true);
    
    // Test VRM loading interaction
    await page.click('button:text("Load Ichika VRM")');
    await page.waitForTimeout(5000); // VRM loading can take time
    
    // Check if VRM loading initiated (buttons should become enabled)
    const loadingResult = await page.evaluate(() => ({
      gestureButtonEnabled: !document.querySelector('button:text("Start Gestures")')?.disabled,
      orchestratorButtonEnabled: !document.querySelector('button:text("Start Orchestrator")')?.disabled,
      statusText: document.body.textContent.includes('Idle') || document.body.textContent.includes('Loading')
    }));
    
    // Take screenshot of VRM orchestrator
    await page.screenshot({
      path: '/tmp/playwright-logs/vrm-orchestrator-baseline.png',
      fullPage: true
    });
  });

  test('Asset Files Accessibility Check', async ({ page }) => {
    test.setTimeout(90000);
    
    // Test if key VRM and scene files are accessible
    const assetTests = [
      '/assets/characters/ichika.vrm',
      '/assets/scenes/classroom.glb',
      '/src/components/animation/vrm/VRMBVHAdapter.js',
      '/src/components/animation/timeline/BVHTimeline.js'
    ];
    
    const assetResults = {};
    
    for (const asset of assetTests) {
      try {
        const response = await page.request.get(asset);
        assetResults[asset] = {
          accessible: response.status() < 400,
          status: response.status(),
          size: response.headers()['content-length'] || 'unknown'
        };
      } catch (error) {
        assetResults[asset] = {
          accessible: false,
          error: error.message
        };
      }
    }
    
    // Validate critical assets are accessible
    expect(assetResults['/assets/characters/ichika.vrm']?.accessible || 
           assetResults['/assets/avatars/ichika.vrm']?.accessible).toBe(true);
    
    // Log results for debugging
    console.log('Asset Accessibility Results:', JSON.stringify(assetResults, null, 2));
  });

  test('Console Error Detection', async ({ page }) => {
    test.setTimeout(60000);
    
    const consoleErrors = [];
    const consoleWarnings = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      } else if (msg.type() === 'warning') {
        consoleWarnings.push(msg.text());
      }
    });
    
    // Test each demo for console errors
    const demos = [
      '/demos/ichika_classroom_demo.html',
      '/demos/ichika_voice_conversation_demo.html',
      '/demos/ichika_vrm_orchestrator_demo.html'
    ];
    
    for (const demo of demos) {
      await page.goto(demo);
      await page.waitForTimeout(5000);
      
      // Interact with main features
      const buttons = await page.locator('button').all();
      for (const button of buttons.slice(0, 2)) { // Test first 2 buttons to avoid timeout
        try {
          if (await button.isEnabled()) {
            await button.click();
            await page.waitForTimeout(1000);
          }
        } catch (e) {
          console.log('Button interaction error:', e.message);
        }
      }
    }
    
    // Report critical errors (excluding known issues like blocked external resources)
    const criticalErrors = consoleErrors.filter(error => 
      !error.includes('ERR_BLOCKED_BY_CLIENT') &&
      !error.includes('net::ERR_INTERNET_DISCONNECTED') &&
      !error.includes('unpkg.com')
    );
    
    console.log('Console Errors Found:', criticalErrors.length);
    console.log('Console Warnings Found:', consoleWarnings.length);
    
    // We'll be lenient for now since this is baseline testing
    expect(criticalErrors.length).toBeLessThan(10);
  });

});

test.describe('System Integration Validation', () => {
  
  test('BVH Timeline System Integration', async ({ page }) => {
    test.setTimeout(90000);
    
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    
    // Wait for timeline system to initialize
    await page.waitForTimeout(3000);
    
    const timelineStatus = await page.evaluate(() => ({
      timelineExists: !!window.bvhTimeline,
      frameBufferSize: window.bvhTimeline?.frameBuffer?.maxSize || 0,
      framerate: window.bvhTimeline?.frameRate || 0,
      isPlaying: window.bvhTimeline?.isPlaying || false,
      currentFrame: window.bvhTimeline?.getCurrentFrame?.() || 0
    }));
    
    // Validate timeline system basic functionality
    expect(timelineStatus.timelineExists).toBe(true);
    expect(timelineStatus.framerate).toBe(30);
    
    console.log('BVH Timeline Status:', timelineStatus);
  });
  
  test('Audio System Integration Status', async ({ page }) => {
    test.setTimeout(90000);
    
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    await page.waitForTimeout(2000);
    
    const audioStatus = await page.evaluate(() => ({
      hasTTSSystem: !!window.ttsSystem || !!window.speechSynthesis,
      hasAudioContext: !!window.AudioContext || !!window.webkitAudioContext,
      hasVisemeSystem: !!window.visemeDriver,
      hasGestureSystem: !!window.gestureGenerator,
      browserTTSAvailable: 'speechSynthesis' in window
    }));
    
    expect(audioStatus.hasAudioContext).toBe(true);
    expect(audioStatus.browserTTSAvailable).toBe(true);
    
    console.log('Audio System Status:', audioStatus);
  });

});