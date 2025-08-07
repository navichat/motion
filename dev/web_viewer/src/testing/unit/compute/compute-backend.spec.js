/**
 * Compute Backend System Unit Test
 * Tests WebGPU, WebNN, and WASM compute capabilities
 */

import { test, expect } from '@playwright/test';

test.describe('Compute Backend System Tests', () => {
  test('should initialize WebGPU backend', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/compute/webgpu-backend-test.html');
    
    const result = await page.evaluate(async () => {
      try {
        const webgpuBackend = new window.WebGPUBackend();
        const initialized = await webgpuBackend.initialize();
        
        return {
          initialized: initialized,
          hasAdapter: !!webgpuBackend.adapter,
          hasDevice: !!webgpuBackend.device,
          deviceLabel: webgpuBackend.device ? webgpuBackend.device.label : null,
          supportedFeatures: webgpuBackend.device ? Array.from(webgpuBackend.device.features) : []
        };
      } catch (error) {
        return {
          initialized: false,
          error: error.message
        };
      }
    });
    
    if (result.initialized) {
      expect(result.hasAdapter).toBe(true);
      expect(result.hasDevice).toBe(true);
      expect(Array.isArray(result.supportedFeatures)).toBe(true);
    } else {
      console.log('WebGPU not supported:', result.error);
    }
  });

  test('should initialize WebNN backend', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/compute/webnn-backend-test.html');
    
    const result = await page.evaluate(async () => {
      try {
        const webnnBackend = new window.WebNNBackend();
        const initialized = await webnnBackend.initialize();
        
        return {
          initialized: initialized,
          hasML: !!window.navigator.ml,
          hasContext: !!webnnBackend.context,
          supportedTypes: webnnBackend.getSupportedTypes ? webnnBackend.getSupportedTypes() : [],
          deviceType: webnnBackend.deviceType || 'unknown'
        };
      } catch (error) {
        return {
          initialized: false,
          error: error.message
        };
      }
    });
    
    if (result.initialized) {
      expect(result.hasML).toBe(true);
      expect(result.hasContext).toBe(true);
      expect(Array.isArray(result.supportedTypes)).toBe(true);
    } else {
      console.log('WebNN not supported:', result.error);
    }
  });

  test('should execute WASM inference', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/compute/wasm-backend-test.html');
    
    const result = await page.evaluate(async () => {
      try {
        const wasmBackend = new window.WASMBackend();
        const initialized = await wasmBackend.initialize();
        
        if (!initialized) {
          throw new Error('Failed to initialize WASM backend');
        }
        
        // Test simple computation
        const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        const outputData = await wasmBackend.compute(inputData, {
          operation: 'identity',
          shape: [1, 4]
        });
        
        return {
          initialized: true,
          hasWasm: !!window.WebAssembly,
          inputLength: inputData.length,
          outputLength: outputData.length,
          dataMatches: inputData.every((val, idx) => Math.abs(val - outputData[idx]) < 0.001)
        };
      } catch (error) {
        return {
          initialized: false,
          error: error.message
        };
      }
    });
    
    expect(result.initialized).toBe(true);
    expect(result.hasWasm).toBe(true);
    expect(result.inputLength).toBe(4);
    expect(result.outputLength).toBe(4);
    expect(result.dataMatches).toBe(true);
  });

  test('should handle model inference with different backends', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/compute/inference-backend-test.html');
    
    const result = await page.evaluate(async () => {
      const backends = ['webgpu', 'webnn', 'wasm'];
      const results = {};
      
      for (const backendType of backends) {
        try {
          const backend = window.createBackend(backendType);
          const initialized = await backend.initialize();
          
          if (initialized) {
            // Test simple matrix multiplication
            const input = new Float32Array([1, 2, 3, 4]);
            const weights = new Float32Array([0.5, 0.5, 0.5, 0.5]);
            
            const output = await backend.matmul(input, weights, {
              inputShape: [1, 4],
              weightsShape: [4, 1],
              outputShape: [1, 1]
            });
            
            results[backendType] = {
              initialized: true,
              hasOutput: !!output,
              outputValue: output[0],
              expectedValue: 5.0, // 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5.0
              correct: Math.abs(output[0] - 5.0) < 0.001
            };
          } else {
            results[backendType] = {
              initialized: false,
              reason: 'Backend not supported'
            };
          }
        } catch (error) {
          results[backendType] = {
            initialized: false,
            error: error.message
          };
        }
      }
      
      return results;
    });
    
    // Check that at least one backend works
    const workingBackends = Object.keys(result).filter(backend => result[backend].initialized);
    expect(workingBackends.length).toBeGreaterThan(0);
    
    // Validate working backends produce correct results
    workingBackends.forEach(backend => {
      expect(result[backend].hasOutput).toBe(true);
      expect(result[backend].correct).toBe(true);
    });
  });
});
