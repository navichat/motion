import { test, expect } from '@playwright/test';

test.describe('3D Ichika VRM Progress Screenshots', () => {

  test('Take comprehensive progress demonstration screenshots', async ({ page }) => {
    test.setTimeout(120000); // Shell timeout compliance
    
    console.log('🚀 Starting 3D Ichika VRM Progress Demonstration');
    
    // Test 1: Enhanced Classroom Demo
    console.log('📸 Capturing Enhanced Classroom Demo...');
    await page.goto('/demos/ichika_enhanced_classroom_demo.html');
    await page.waitForTimeout(4000); // Allow time for loading
    
    // Take screenshot of enhanced classroom
    await page.screenshot({ 
      path: 'test-results/progress-enhanced-classroom.png',
      fullPage: true 
    });
    
    // Validate basic elements are present
    const hasTitle = await page.locator('h1').first().isVisible();
    expect(hasTitle).toBe(true);
    
    console.log('✅ Enhanced Classroom Demo screenshot captured');
    
    // Test 2: Voice Conversation Demo
    console.log('📸 Capturing Voice Conversation Demo...');
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    await page.waitForTimeout(3000);
    
    // Try to interact with TTS
    const textInput = page.locator('input, textarea').first();
    if (await textInput.isVisible()) {
      await textInput.fill('This demonstrates our integrated 3D animated Ichika VRM system with voice interaction!');
      
      const sayButton = page.locator('button:text("Say")');
      if (await sayButton.isVisible()) {
        await sayButton.click();
        await page.waitForTimeout(2000); // Allow processing time
      }
    }
    
    await page.screenshot({ 
      path: 'test-results/progress-voice-conversation.png',
      fullPage: true 
    });
    
    console.log('✅ Voice Conversation Demo screenshot captured');
    
    // Test 3: VRM Orchestrator Demo
    console.log('📸 Capturing VRM Orchestrator Demo...');
    await page.goto('/demos/ichika_vrm_orchestrator_demo.html');
    await page.waitForTimeout(5000); // VRM systems need more time
    
    // Try to load VRM
    const loadButton = page.locator('button:text("Load Ichika VRM")');
    if (await loadButton.isVisible()) {
      await loadButton.click();
      await page.waitForTimeout(4000); // Allow VRM loading time
    }
    
    await page.screenshot({ 
      path: 'test-results/progress-vrm-orchestrator.png',
      fullPage: true 
    });
    
    console.log('✅ VRM Orchestrator Demo screenshot captured');
    
    // Test 4: Regular Classroom Demo for comparison
    console.log('📸 Capturing Regular Classroom Demo...');
    await page.goto('/demos/ichika_classroom_demo.html');
    await page.waitForTimeout(3000);
    
    // Try to trigger some animations
    const idleButton = page.locator('button:text("Start Idle")');
    if (await idleButton.isVisible()) {
      await idleButton.click();
      await page.waitForTimeout(2000);
    }
    
    const waveButton = page.locator('button:text("Wave")');
    if (await waveButton.isVisible()) {
      await waveButton.click();
      await page.waitForTimeout(1500);
    }
    
    await page.screenshot({ 
      path: 'test-results/progress-classroom-demo.png',
      fullPage: true 
    });
    
    console.log('✅ Classroom Demo screenshot captured');
    
    // Test 5: System Analysis
    console.log('🔍 Performing system integration analysis...');
    
    await page.goto('/demos/ichika_voice_conversation_demo.html');
    await page.waitForTimeout(3000);
    
    // Check what systems are available
    const systemCheck = await page.evaluate(() => {
      const systems = {};
      
      // Core 3D and rendering
      systems.threeJS = !!window.THREE;
      systems.webGL = (() => {
        try {
          const canvas = document.createElement('canvas');
          return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
          return false;
        }
      })();
      systems.webGPU = !!navigator.gpu;
      
      // VRM and animation systems
      systems.vrmLoader = !!window.VRM || !!window.VRMLoader;
      systems.bvhTimeline = !!window.bvhTimeline;
      systems.orchestrator = !!window.IchikaOrchestrator;
      
      // Audio and voice systems
      systems.speechSynthesis = !!window.speechSynthesis;
      systems.audioContext = !!(window.AudioContext || window.webkitAudioContext);
      systems.ttsSystem = !!window.ttsSystem;
      
      // UI elements
      systems.canvas = document.querySelectorAll('canvas').length;
      systems.buttons = document.querySelectorAll('button').length;
      
      return systems;
    });
    
    console.log('=== SYSTEM INTEGRATION ANALYSIS ===');
    console.log(JSON.stringify(systemCheck, null, 2));
    
    // Calculate progress
    const coreFeatures = [
      'threeJS', 'webGL', 'vrmLoader', 'bvhTimeline', 
      'speechSynthesis', 'audioContext'
    ];
    
    const workingFeatures = coreFeatures.filter(feature => systemCheck[feature]);
    const progressPercent = Math.round((workingFeatures.length / coreFeatures.length) * 100);
    
    console.log(`🎯 Core Systems Working: ${workingFeatures.length}/${coreFeatures.length} (${progressPercent}%)`);
    console.log(`✅ Working: ${workingFeatures.join(', ')}`);
    
    const notWorking = coreFeatures.filter(feature => !systemCheck[feature]);
    if (notWorking.length > 0) {
      console.log(`🔧 Needs Integration: ${notWorking.join(', ')}`);
    }
    
    // Final validation
    expect(systemCheck.canvas).toBeGreaterThan(0);
    expect(systemCheck.buttons).toBeGreaterThan(0);
    expect(workingFeatures.length).toBeGreaterThan(3); // At least 4/6 systems working
    
    console.log('✅ Progress demonstration test completed successfully!');
  });

});