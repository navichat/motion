const { test, expect } = require('@playwright/test');

test.describe('3D Animated Ichika VRM System - Progress Validation', () => {
  
  test('Classroom Demo - Animation System Validation', async ({ page }) => {
    test.setTimeout(120000); // Shell timeout compliance
    
    await page.goto('/demos/ichika_classroom_demo.html');
    
    // Wait for page to load and scripts to initialize
    await page.waitForTimeout(3000);
    
    // Validate page structure
    await expect(page.locator('h1')).toContainText('Ichika Classroom Demo');
    
    // Test animation control buttons
    const startButton = page.locator('button:text("Start Idle")');
    const pointButton = page.locator('button:text("Point at Board")');
    const waveButton = page.locator('button:text("Wave")');
    const speechButton = page.locator('button:text("Speak (visemes + gestures)")');
    
    await expect(startButton).toBeVisible();
    await expect(pointButton).toBeVisible();
    await expect(waveButton).toBeVisible();
    await expect(speechButton).toBeVisible();
    
    // Test animation triggering
    console.log('Testing Start Idle animation...');
    await startButton.click();
    await page.waitForTimeout(2000);
    
    // Check if orchestrator system loaded
    const orchLoaded = await page.evaluate(() => {
      return {
        hasIchikaDemo: !!window.__ichikaDemo,
        hasOrchestrator: !!window.__ichikaDemo?.orch,
        hasStage: !!window.__ichikaDemo?.stage,
        hasRegistry: !!window.__ichikaDemo?.reg,
        registryEntries: window.__ichikaDemo?.reg?.list?.()?.length || 0
      };
    });
    
    console.log('Orchestrator System Status:', JSON.stringify(orchLoaded, null, 2));
    
    // Test other animations
    console.log('Testing Point at Board animation...');
    await pointButton.click();
    await page.waitForTimeout(1000);
    
    console.log('Testing Wave animation...');
    await waveButton.click();
    await page.waitForTimeout(1000);
    
    // Take screenshot of working demo
    await page.screenshot({ 
      path: '/tmp/playwright-logs/classroom-demo-working.png',
      fullPage: true 
    });
  });

  test('Voice Conversation Demo - Integration Test', async ({ page }) => {
    test.setTimeout(150000);
    
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    await page.waitForTimeout(3000);
    
    // Validate interface
    await expect(page.locator('h1')).toContainText('Ichika Conversation');
    
    // Test TTS functionality
    const testText = 'Hello everyone, welcome to our enhanced classroom experience!';
    await page.fill('input[placeholder*="Type something"], textbox', testText);
    
    // Check backend options
    const backendSelect = page.locator('select').first();
    await expect(backendSelect).toBeVisible();
    
    // Test TTS trigger
    const sayButton = page.locator('button:text("Say")');
    await expect(sayButton).toBeVisible();
    await sayButton.click();
    
    // Wait for processing
    await page.waitForTimeout(3000);
    
    // Check animation system status
    const systemStatus = await page.evaluate(() => {
      const status = {};
      
      // Check for BVH Timeline
      if (window.bvhTimeline) {
        status.bvhTimeline = {
          exists: true,
          isInitialized: window.bvhTimeline.isInitialized || false,
          frameRate: window.bvhTimeline.frameRate || 0,
          isPlaying: window.bvhTimeline.isPlaying || false
        };
      }
      
      // Check for VRM system
      if (window.vrmSystem || window.vrm || window.ichikaVRM) {
        status.vrmSystem = {
          exists: true,
          loaded: !!(window.vrmSystem || window.vrm || window.ichikaVRM)
        };
      }
      
      // Check for audio context
      status.audioContext = {
        supported: !!(window.AudioContext || window.webkitAudioContext),
        speechSynthesis: !!window.speechSynthesis
      };
      
      // Check canvas presence
      status.rendering = {
        hasCanvas: document.querySelectorAll('canvas').length,
        canvasElements: Array.from(document.querySelectorAll('canvas')).map(c => ({
          width: c.width,
          height: c.height,
          visible: c.style.display !== 'none'
        }))
      };
      
      return status;
    });
    
    console.log('Voice Demo System Status:', JSON.stringify(systemStatus, null, 2));
    
    // Take screenshot
    await page.screenshot({
      path: '/tmp/playwright-logs/voice-conversation-working.png',
      fullPage: true
    });
  });

  test('VRM Orchestrator Demo - 3D System Test', async ({ page }) => {
    test.setTimeout(120000);
    
    await page.goto('/demos/ichika_vrm_orchestrator_demo.html');
    await page.waitForTimeout(5000); // Allow time for ES modules to load
    
    // Check if interface loaded
    const loadButton = page.locator('button:text("Load Ichika VRM")');
    await expect(loadButton).toBeVisible();
    
    // Test VRM loading (may fail due to file access)
    console.log('Attempting VRM load...');
    await loadButton.click();
    await page.waitForTimeout(8000); // VRM loading can take time
    
    // Check if any progress was made
    const loadingResult = await page.evaluate(() => {
      return {
        hasCanvas: !!document.querySelector('canvas'),
        canvasContext: (() => {
          const canvas = document.querySelector('canvas');
          if (canvas) {
            try {
              return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
            } catch (e) {
              return false;
            }
          }
          return false;
        })(),
        buttonsEnabled: {
          gestures: !document.querySelector('button:text("Start Gestures")')?.disabled,
          orchestrator: !document.querySelector('button:text("Start Orchestrator")')?.disabled
        },
        logContent: document.querySelector('#log')?.textContent || '',
        statusText: document.querySelector('#status')?.textContent || ''
      };
    });
    
    console.log('VRM Orchestrator Status:', JSON.stringify(loadingResult, null, 2));
    
    // Take screenshot
    await page.screenshot({
      path: '/tmp/playwright-logs/vrm-orchestrator-test.png',
      fullPage: true
    });
  });

  test('Component Integration Analysis', async ({ page }) => {
    test.setTimeout(90000);
    
    // Test each demo and analyze available components
    const demos = [
      '/demos/ichika_classroom_demo.html',
      '/demos/ichika_voice_conversation_demo.html'
    ];
    
    const integrationData = {};
    
    for (const demo of demos) {
      await page.goto(demo);
      await page.waitForTimeout(3000);
      
      // Trigger some interactions to activate systems
      try {
        const buttons = await page.locator('button').all();
        for (const button of buttons.slice(0, 2)) {
          if (await button.isEnabled()) {
            await button.click();
            await page.waitForTimeout(1000);
          }
        }
      } catch (e) {
        console.log('Interaction error:', e.message);
      }
      
      // Analyze available components
      const analysis = await page.evaluate(() => {
        const components = {};
        
        // Check for global objects that indicate component presence
        const globalChecks = [
          'THREE', 'VRM', 'bvhTimeline', 'vrmSystem', 'ichikaVRM',
          'speechSynthesis', 'AudioContext', 'webkitAudioContext',
          'ClipRegistry', 'StageController', 'IchikaOrchestrator',
          '__ichikaDemo'
        ];
        
        globalChecks.forEach(check => {
          components[check] = !!window[check];
        });
        
        // Check DOM elements
        components.domElements = {
          canvas: document.querySelectorAll('canvas').length,
          buttons: document.querySelectorAll('button').length,
          inputs: document.querySelectorAll('input, textarea, select').length
        };
        
        // Check for specific functionality
        components.features = {
          webGL: (() => {
            try {
              const canvas = document.createElement('canvas');
              return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
            } catch (e) {
              return false;
            }
          })(),
          webGPU: !!navigator.gpu,
          speechAPI: 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window,
          mediaDevices: 'mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices
        };
        
        return components;
      });
      
      integrationData[demo] = analysis;
    }
    
    console.log('=== COMPONENT INTEGRATION ANALYSIS ===');
    console.log(JSON.stringify(integrationData, null, 2));
    
    // Generate integration recommendations
    const recommendations = [];
    
    if (integrationData['/demos/ichika_classroom_demo.html']?.__ichikaDemo) {
      recommendations.push('✅ Orchestrator system available in classroom demo');
    }
    
    if (integrationData['/demos/ichika_voice_conversation_demo.html']?.bvhTimeline) {
      recommendations.push('✅ BVH Timeline system active in voice demo');
    }
    
    const webGLSupport = Object.values(integrationData).some(demo => demo.features?.webGL);
    if (webGLSupport) {
      recommendations.push('✅ WebGL rendering support confirmed');
    }
    
    const audioSupport = Object.values(integrationData).some(demo => demo.features?.speechAPI);
    if (audioSupport) {
      recommendations.push('✅ Speech synthesis API available');
    }
    
    console.log('=== INTEGRATION RECOMMENDATIONS ===');
    recommendations.forEach(rec => console.log(rec));
  });

  test('Performance and Resource Validation', async ({ page }) => {
    test.setTimeout(90000);
    
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    await page.waitForTimeout(5000);
    
    // Measure initial performance
    const performanceData = await page.evaluate(() => {
      const data = {
        memory: {},
        timing: {},
        resources: {}
      };
      
      // Memory information (if available)
      if (performance.memory) {
        data.memory = {
          used: Math.round(performance.memory.usedJSHeapSize / 1048576), // MB
          total: Math.round(performance.memory.totalJSHeapSize / 1048576), // MB
          limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) // MB
        };
      }
      
      // Performance timing
      if (performance.timing) {
        const timing = performance.timing;
        data.timing = {
          domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
          pageLoad: timing.loadEventEnd - timing.navigationStart,
          domReady: timing.domInteractive - timing.navigationStart
        };
      }
      
      // Resource counts
      data.resources = {
        scripts: document.querySelectorAll('script').length,
        stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length,
        images: document.querySelectorAll('img').length,
        canvas: document.querySelectorAll('canvas').length
      };
      
      return data;
    });
    
    console.log('=== PERFORMANCE ANALYSIS ===');
    console.log(JSON.stringify(performanceData, null, 2));
    
    // Performance assertions
    if (performanceData.memory.used) {
      expect(performanceData.memory.used).toBeLessThan(200); // < 200MB
      console.log(`✅ Memory usage: ${performanceData.memory.used}MB (within limits)`);
    }
    
    if (performanceData.timing.pageLoad) {
      expect(performanceData.timing.pageLoad).toBeLessThan(10000); // < 10 seconds
      console.log(`✅ Page load time: ${performanceData.timing.pageLoad}ms (acceptable)`);
    }
    
    expect(performanceData.resources.scripts).toBeGreaterThan(0);
    expect(performanceData.resources.canvas).toBeGreaterThan(0);
    console.log('✅ Required resources loaded successfully');
  });

});