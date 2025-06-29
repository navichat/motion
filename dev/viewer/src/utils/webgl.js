/**
 * WebGL Utility Functions
 * 
 * Helper functions for WebGL support detection and performance monitoring
 */

/**
 * Check WebGL support and capabilities
 */
export function checkWebGLSupport() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  
  if (!gl) {
    return {
      supported: false,
      version: null,
      extensions: []
    };
  }
  
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const extensions = gl.getSupportedExtensions() || [];
  
  return {
    supported: true,
    version: gl.getParameter(gl.VERSION),
    vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown',
    renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown',
    extensions: extensions
  };
}

/**
 * Get GPU information
 */
export function getGPUInfo(gl) {
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  
  if (!debugInfo) {
    return {
      vendor: 'Unknown',
      renderer: 'Unknown'
    };
  }
  
  return {
    vendor: gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL),
    renderer: gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
  };
}

/**
 * Frame rate monitoring utility
 */
export class FrameRateMonitor {
  constructor(sampleSize = 60) {
    this.sampleSize = sampleSize;
    this.frameTimes = [];
    this.lastFrameTime = performance.now();
  }
  
  recordFrame() {
    const now = performance.now();
    const deltaTime = now - this.lastFrameTime;
    
    this.frameTimes.push(deltaTime);
    
    if (this.frameTimes.length > this.sampleSize) {
      this.frameTimes.shift();
    }
    
    this.lastFrameTime = now;
  }
  
  getFPS() {
    if (this.frameTimes.length === 0) {
      return 0;
    }
    
    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length;
    return Math.round(1000 / avgFrameTime);
  }
  
  getAverageFrameTime() {
    if (this.frameTimes.length === 0) {
      return 0;
    }
    
    return this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length;
  }
  
  reset() {
    this.frameTimes = [];
    this.lastFrameTime = performance.now();
  }
}
