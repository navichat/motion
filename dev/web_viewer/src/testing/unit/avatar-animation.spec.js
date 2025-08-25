/**
 * Avatar Animation Component Tests
 * 
 * Tests individual avatar animation components:
 * - Animation backends (WebNN/WebGPU/WASM)
 * - BVH animation processing
 * - Phase visualization
 * - Style control systems
 * 
 * Part of the reorganized WebNN/WebGPU/WASM avatar system testing infrastructure.
 */

import { test, expect } from '@playwright/test';

test.describe('Avatar Animation Components', () => {
  
  test('Animation Backends Test Suite', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/animation/backends-demo.html');
    
    // Test animation backend switching
    await expect(page.locator('h1')).toContainText('Animation Backends');
    
    // Test WebNN backend
    await page.locator('button:has-text("WebNN Backend")').click();
    await page.waitForTimeout(2000);
    
    let backendStatus = await page.locator('#backend-status').textContent();
    expect(backendStatus).toContain('WebNN');
    
    // Test WebGPU backend
    await page.locator('button:has-text("WebGPU Backend")').click();
    await page.waitForTimeout(2000);
    
    backendStatus = await page.locator('#backend-status').textContent();
    expect(backendStatus).toContain('WebGPU');
    
    // Test WASM backend
    await page.locator('button:has-text("WASM Backend")').click();
    await page.waitForTimeout(2000);
    
    backendStatus = await page.locator('#backend-status').textContent();
    expect(backendStatus).toContain('WASM');
  });

  test('BVH Animation Processing', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/animation/bvh-demo.html');
    
    // Test BVH animation loading and processing
    await expect(page.locator('h1')).toContainText('BVH Animation');
    
    // Load BVH file
    await page.locator('button:has-text("Load BVH")').click();
    await page.waitForTimeout(3000);
    
    // Verify BVH processing
    const bvhStatus = await page.locator('#bvh-status').textContent();
    expect(bvhStatus).toContain('loaded');
    
    // Test animation playback
    await page.locator('button:has-text("Play")').click();
    await page.waitForTimeout(1000);
    
    const playbackStatus = await page.locator('#playback-status').textContent();
    expect(playbackStatus).toContain('playing');
  });

  test('Phase Visualizer', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/animation/phase-visualizer-demo.html');
    
    // Test phase visualization
    await expect(page.locator('h1')).toContainText('Phase Visualizer');
    
    // Initialize phase visualization
    await page.locator('button:has-text("Initialize")').click();
    await page.waitForTimeout(2000);
    
    // Verify phase data
    const phaseData = await page.locator('#phase-data').isVisible();
    expect(phaseData).toBeTruthy();
    
    // Test phase updates
    await page.locator('button:has-text("Update Phase")').click();
    await page.waitForTimeout(1000);
    
    const phaseValue = await page.locator('#phase-value').textContent();
    expect(phaseValue).toBeTruthy();
  });

  test('Style Controller', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/animation/style-controller-demo.html');
    
    // Test animation style control
    await expect(page.locator('h1')).toContainText('Style Controller');
    
    // Test style transitions
    await page.locator('select#style-selector').selectOption('walking');
    await page.waitForTimeout(1000);
    
    let currentStyle = await page.locator('#current-style').textContent();
    expect(currentStyle).toContain('walking');
    
    await page.locator('select#style-selector').selectOption('running');
    await page.waitForTimeout(1000);
    
    currentStyle = await page.locator('#current-style').textContent();
    expect(currentStyle).toContain('running');
  });
});

test.describe('Animation Performance Testing', () => {
  
  test('Animation Frame Rate Test', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/animation/performance-demo.html');
    
    // Start performance monitoring
    await page.locator('button:has-text("Start Monitor")').click();
    await page.waitForTimeout(5000);
    
    // Check frame rate
    const frameRate = await page.locator('#frame-rate').textContent();
    const fps = parseFloat(frameRate);
    expect(fps).toBeGreaterThan(30); // Expect at least 30 FPS
  });

  test('Backend Performance Comparison', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/animation/backend-comparison-demo.html');
    
    // Run performance comparison
    await page.locator('button:has-text("Run Comparison")').click();
    await page.waitForTimeout(10000); // Allow time for all backends to be tested
    
    // Verify results
    const webnnTime = await page.locator('#webnn-time').textContent();
    const webgpuTime = await page.locator('#webgpu-time').textContent();
    const wasmTime = await page.locator('#wasm-time').textContent();
    
    expect(webnnTime).toBeTruthy();
    expect(webgpuTime).toBeTruthy();
    expect(wasmTime).toBeTruthy();
  });
});
