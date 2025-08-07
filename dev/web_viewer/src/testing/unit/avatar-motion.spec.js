/**
 * Avatar Motion Component Tests
 * 
 * Tests individual avatar motion processing components:
 * - Audio2Gesture BVH conversion
 * - RSMT motion processing  
 * - Motion analysis and capture
 * - Pathfinding and timeline integration
 * 
 * Part of the reorganized WebNN/WebGPU/WASM avatar system testing infrastructure.
 */

import { test, expect } from '@playwright/test';

test.describe('Avatar Motion Components', () => {
  
  test('Audio2Gesture BVH Converter', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/audio2gesture-demo.html');
    
    // Test component loading
    await expect(page.locator('h1')).toContainText('Audio2Gesture');
    
    // Test BVH conversion functionality
    await page.locator('button:has-text("Load Audio")').click();
    await page.waitForTimeout(2000);
    
    // Verify conversion results
    const conversionStatus = await page.locator('#conversion-status').textContent();
    expect(conversionStatus).toBeTruthy();
  });

  test('RSMT Motion Processing', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/rsmt-demo.html');
    
    // Test RSMT client functionality
    await expect(page.locator('h1')).toContainText('RSMT');
    
    // Test motion processing
    await page.locator('button:has-text("Process Motion")').click();
    await page.waitForTimeout(3000);
    
    // Verify processing completed
    const processingStatus = await page.locator('#processing-status').textContent();
    expect(processingStatus).toContain('completed');
  });

  test('Motion Analysis', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/motion-analyzer-demo.html');
    
    // Test motion analyzer
    await expect(page.locator('h1')).toContainText('Motion Analyzer');
    
    // Load sample motion data
    await page.locator('button:has-text("Load Sample")').click();
    await page.waitForTimeout(2000);
    
    // Verify analysis results
    const analysisResults = await page.locator('#analysis-results').isVisible();
    expect(analysisResults).toBeTruthy();
  });

  test('Pathfinding Timeline Integration', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/pathfinding-demo.html');
    
    // Test pathfinding functionality
    await expect(page.locator('h1')).toContainText('Pathfinding');
    
    // Start pathfinding
    await page.locator('button:has-text("Calculate Path")').click();
    await page.waitForTimeout(2000);
    
    // Verify path calculation
    const pathData = await page.locator('#path-data').textContent();
    expect(pathData).toBeTruthy();
  });
});

test.describe('WebNN/WebGPU/WASM Motion Backends', () => {
  
  test('WebNN Motion Processing', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/webnn-motion-demo.html');
    
    // Test WebNN backend
    await page.locator('button:has-text("Test WebNN")').click();
    await page.waitForTimeout(3000);
    
    const webnnStatus = await page.locator('#webnn-status').textContent();
    expect(webnnStatus).toContain('WebNN');
  });

  test('WebGPU Motion Processing', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/webgpu-motion-demo.html');
    
    // Test WebGPU backend
    await page.locator('button:has-text("Test WebGPU")').click();
    await page.waitForTimeout(3000);
    
    const webgpuStatus = await page.locator('#webgpu-status').textContent();
    expect(webgpuStatus).toContain('WebGPU');
  });

  test('WASM Motion Processing', async ({ page }) => {
    await page.goto('http://localhost:8080/dev/web_viewer/demos/motion/wasm-motion-demo.html');
    
    // Test WASM backend
    await page.locator('button:has-text("Test WASM")').click();
    await page.waitForTimeout(3000);
    
    const wasmStatus = await page.locator('#wasm-status').textContent();
    expect(wasmStatus).toContain('WASM');
  });
});
