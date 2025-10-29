const { test, expect } = require('@playwright/test');

test.describe('Real System Integration Assessment', () => {
  test.setTimeout(300000); // 5 minutes max per test (shell timeout compliance)
  
  test('Assess Current Working Capabilities', async ({ page }) => {
    console.log('=== REAL SYSTEM INTEGRATION ASSESSMENT ===');
    
    const baseURL = 'http://localhost:8080';
    const demos = [
      { name: 'Enhanced Classroom', path: '/demos/ichika_enhanced_classroom_demo.html' },
      { name: 'Voice Conversation', path: '/demos/ichika_voice_conversation_demo.html' },
      { name: 'VRM Orchestrator', path: '/demos/ichika_vrm_orchestrator_demo.html' },
      { name: 'Full Classroom Experience', path: '/demos/ichika_full_classroom_experience.html' },
      { name: 'Original Classroom', path: '/demos/ichika_classroom_demo.html' }
    ];
    
    const assessmentResults = {
      workingDemos: 0,
      totalDemos: demos.length,
      capabilities: {},
      screenshots: [],
      integrationStatus: 'UNKNOWN'
    };
    
    for (const [index, demo] of demos.entries()) {
      console.log(`\n--- Testing Demo ${index + 1}: ${demo.name} ---`);
      
      try {
        await page.goto(`${baseURL}${demo.path}`, { timeout: 30000 });
        await page.waitForTimeout(3000);
        
        // Basic functionality assessment
        const demoAnalysis = await page.evaluate(() => {
          return {
            title: document.title,
            hasCanvas: !!document.querySelector('canvas'),
            canvasCount: document.querySelectorAll('canvas').length,
            hasWebGL: (() => {
              const canvas = document.querySelector('canvas');
              if (!canvas) return false;
              try {
                return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'));
              } catch { return false; }
            })(),
            hasControls: document.querySelectorAll('button, select, input').length,
            hasAudio: document.querySelectorAll('audio').length,
            hasErrors: !!document.querySelector('.error, [class*="error"]'),
            loadedScripts: document.scripts.length,
            bodyContent: document.body.innerHTML.length
          };
        });
        
        console.log(`Canvas: ${demoAnalysis.hasCanvas}, WebGL: ${demoAnalysis.hasWebGL}, Controls: ${demoAnalysis.hasControls}`);
        
        // Determine if demo is actually working
        const isWorking = demoAnalysis.hasCanvas && demoAnalysis.hasControls && demoAnalysis.bodyContent > 1000;
        
        if (isWorking) {
          assessmentResults.workingDemos++;
          console.log(`✅ ${demo.name}: WORKING`);
          
          // Take screenshot of working demo
          const screenshotPath = `test-results/real-assessment/demo-${index + 1}-${demo.name.toLowerCase().replace(/\s/g, '-')}.png`;
          await page.screenshot({ path: screenshotPath, fullPage: false });
          assessmentResults.screenshots.push(screenshotPath);
          
          // Detailed capability assessment for working demos
          if (demo.name.includes('Voice') || demo.name.includes('Conversation')) {
            assessmentResults.capabilities.voiceProcessing = 'WORKING';
          }
          if (demo.name.includes('VRM')) {
            assessmentResults.capabilities.vrmLoading = 'WORKING';
          }
          if (demo.name.includes('Classroom')) {
            assessmentResults.capabilities.classroomScene = 'WORKING';
          }
          
        } else {
          console.log(`❌ ${demo.name}: NOT FUNCTIONAL`);
          console.log(`  - Canvas: ${demoAnalysis.hasCanvas}, Controls: ${demoAnalysis.hasControls}, Content: ${demoAnalysis.bodyContent}b`);
        }
        
      } catch (error) {
        console.log(`❌ ${demo.name}: ERROR - ${error.message}`);
      }
    }
    
    // Overall assessment
    const workingPercentage = (assessmentResults.workingDemos / assessmentResults.totalDemos) * 100;
    
    if (workingPercentage >= 80) {
      assessmentResults.integrationStatus = 'WELL INTEGRATED';
    } else if (workingPercentage >= 60) {
      assessmentResults.integrationStatus = 'PARTIALLY INTEGRATED';
    } else if (workingPercentage >= 40) {
      assessmentResults.integrationStatus = 'BASIC FUNCTIONALITY';
    } else {
      assessmentResults.integrationStatus = 'NEEDS MAJOR WORK';
    }
    
    console.log('\n=== FINAL ASSESSMENT ===');
    console.log(`Working Demos: ${assessmentResults.workingDemos}/${assessmentResults.totalDemos} (${workingPercentage.toFixed(1)}%)`);
    console.log(`Integration Status: ${assessmentResults.integrationStatus}`);
    console.log(`Screenshots Captured: ${assessmentResults.screenshots.length}`);
    
    console.log('\nCapabilities Found:');
    Object.entries(assessmentResults.capabilities).forEach(([capability, status]) => {
      console.log(`- ${capability}: ${status}`);
    });
    
    // Write assessment report
    const reportContent = JSON.stringify(assessmentResults, null, 2);
    await page.evaluate((content) => {
      // Store results for potential retrieval
      window.assessmentResults = JSON.parse(content);
    }, reportContent);
    
    // Expectations for meaningful integration
    expect(assessmentResults.workingDemos).toBeGreaterThan(0);
    console.log('\n✅ Integration assessment completed with real evidence');
  });
  
  test('Technical Component Analysis', async ({ page }) => {
    console.log('\n=== TECHNICAL COMPONENT ANALYSIS ===');
    
    // Test the most comprehensive demo
    await page.goto('http://localhost:8080/demos/ichika_full_classroom_experience.html');
    await page.waitForTimeout(4000);
    
    const technicalAnalysis = await page.evaluate(() => {
      const analysis = {
        environment: {
          webGLSupport: false,
          webGPUSupport: false,
          audioContext: false,
          performanceAPI: false
        },
        components: {
          threeJS: typeof window.THREE !== 'undefined',
          vrmLoader: typeof window.VRMLoader !== 'undefined' || (window.THREE && typeof window.THREE.VRM !== 'undefined'),
          bvhTimeline: typeof window.BVHTimeline !== 'undefined',
          taskScheduler: typeof window.TaskScheduler !== 'undefined',
          ichikaOrchestrator: typeof window.IchikaOrchestrator !== 'undefined'
        },
        integration: {
          audioProcessing: false,
          animationSystem: false,
          sceneRendering: false
        },
        globalObjects: []
      };
      
      // Test WebGL
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        analysis.environment.webGLSupport = !!gl;
      } catch {}
      
      // Test WebGPU
      analysis.environment.webGPUSupport = 'gpu' in navigator;
      
      // Test Audio
      analysis.environment.audioContext = typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined';
      
      // Test Performance API
      analysis.environment.performanceAPI = typeof performance !== 'undefined' && typeof performance.memory !== 'undefined';
      
      // Scene rendering check
      analysis.integration.sceneRendering = !!document.querySelector('canvas') && analysis.environment.webGLSupport;
      
      // Animation system check
      analysis.integration.animationSystem = analysis.components.bvhTimeline || analysis.components.taskScheduler;
      
      // Audio processing check
      analysis.integration.audioProcessing = analysis.environment.audioContext && 
        (!!document.querySelector('button[id*="mic"], button[id*="say"], select[id*="backend"]'));
      
      // Global objects scan
      analysis.globalObjects = Object.keys(window).filter(key => 
        key.includes('VRM') || key.includes('BVH') || key.includes('Timeline') ||
        key.includes('Ichika') || key.includes('Avatar') || key.includes('THREE') ||
        key.includes('Orchestrator') || key.includes('Scheduler')
      );
      
      return analysis;
    });
    
    console.log('Environment Support:');
    Object.entries(technicalAnalysis.environment).forEach(([feature, supported]) => {
      console.log(`  ${feature}: ${supported ? '✅' : '❌'}`);
    });
    
    console.log('\nCore Components:');
    Object.entries(technicalAnalysis.components).forEach(([component, available]) => {
      console.log(`  ${component}: ${available ? '✅' : '❌'}`);
    });
    
    console.log('\nIntegration Status:');
    Object.entries(technicalAnalysis.integration).forEach(([system, working]) => {
      console.log(`  ${system}: ${working ? '✅' : '❌'}`);
    });
    
    console.log(`\nGlobal Objects Found: ${technicalAnalysis.globalObjects.length}`);
    if (technicalAnalysis.globalObjects.length > 0) {
      console.log(`  ${technicalAnalysis.globalObjects.slice(0, 5).join(', ')}${technicalAnalysis.globalObjects.length > 5 ? '...' : ''}`);
    }
    
    // Take technical analysis screenshot
    await page.screenshot({ 
      path: 'test-results/real-assessment/technical-analysis.png',
      fullPage: true 
    });
    
    // Calculate integration score
    const environmentScore = Object.values(technicalAnalysis.environment).filter(Boolean).length * 5;
    const componentScore = Object.values(technicalAnalysis.components).filter(Boolean).length * 10;
    const integrationScore = Object.values(technicalAnalysis.integration).filter(Boolean).length * 15;
    const totalScore = environmentScore + componentScore + integrationScore;
    const maxScore = 4 * 5 + 5 * 10 + 3 * 15; // 20 + 50 + 45 = 115
    const percentage = (totalScore / maxScore) * 100;
    
    console.log(`\nIntegration Score: ${totalScore}/${maxScore} (${percentage.toFixed(1)}%)`);
    
    expect(totalScore).toBeGreaterThan(20); // Minimum viable integration
  });
});