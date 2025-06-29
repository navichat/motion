import { describe, it, expect, vi } from 'vitest';

/**
 * Unit Tests for Utility Functions
 * 
 * Tests for helper functions, formatters, and utility classes
 */

describe('Animation Utils', () => {
  it('should validate JSON animation format', async () => {
    const { validateAnimationFormat } = await import('../../../viewer/src/utils/animation.js');
    
    const validAnimation = {
      format: 'json',
      name: 'test-animation',
      tracks: [
        {
          name: 'bone.position',
          type: 'vector',
          times: [0, 1],
          values: [0, 0, 0, 1, 1, 1]
        }
      ]
    };
    
    expect(validateAnimationFormat(validAnimation)).toBe(true);
    
    const invalidAnimation = {
      format: 'json',
      // missing name and tracks
    };
    
    expect(validateAnimationFormat(invalidAnimation)).toBe(false);
  });

  it('should convert BVH to JSON format', async () => {
    const { convertBVHToJSON } = await import('../../../viewer/src/utils/animation.js');
    
    const mockBVHData = `
HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.00 5.36 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.00 5.36 0.00
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
1.00 1.00 1.00 10.00 10.00 10.00 5.00 5.00 5.00
`;
    
    const jsonAnimation = convertBVHToJSON(mockBVHData);
    
    expect(jsonAnimation.format).toBe('json');
    expect(jsonAnimation.tracks).toBeDefined();
    expect(jsonAnimation.tracks.length).toBeGreaterThan(0);
  });

  it('should calculate animation duration', async () => {
    const { calculateAnimationDuration } = await import('../../../viewer/src/utils/animation.js');
    
    const animation = {
      tracks: [
        { times: [0, 1, 2, 3] },
        { times: [0, 0.5, 1.5, 2.5] },
        { times: [0, 2, 4] }
      ]
    };
    
    const duration = calculateAnimationDuration(animation);
    expect(duration).toBe(4); // Maximum time across all tracks
  });

  it('should interpolate animation values', async () => {
    const { interpolateValues } = await import('../../../viewer/src/utils/animation.js');
    
    const values = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const times = [0, 1, 2];
    
    // Test interpolation at t=0.5 (halfway between frames 0 and 1)
    const result = interpolateValues(values, times, 0.5, 3);
    expect(result).toEqual([0.5, 0.5, 0.5]);
  });
});

describe('WebGL Utils', () => {
  it('should check WebGL support', async () => {
    const { checkWebGLSupport } = await import('../../../viewer/src/utils/webgl.js');
    
    // Mock canvas and context
    const mockCanvas = {
      getContext: vi.fn(() => ({
        getSupportedExtensions: vi.fn(() => ['WEBGL_debug_renderer_info']),
        getParameter: vi.fn(() => 'Test GPU')
      }))
    };
    
    global.document = {
      createElement: vi.fn(() => mockCanvas)
    };
    
    const support = checkWebGLSupport();
    
    expect(support.supported).toBe(true);
    expect(support.version).toBeDefined();
  });

  it('should get GPU info', async () => {
    const { getGPUInfo } = await import('../../../viewer/src/utils/webgl.js');
    
    const mockContext = {
      getExtension: vi.fn(() => ({
        UNMASKED_VENDOR_WEBGL: 37445,
        UNMASKED_RENDERER_WEBGL: 37446
      })),
      getParameter: vi.fn((param) => {
        if (param === 37445) return 'Test Vendor';
        if (param === 37446) return 'Test Renderer';
        return null;
      })
    };
    
    const info = getGPUInfo(mockContext);
    
    expect(info.vendor).toBe('Test Vendor');
    expect(info.renderer).toBe('Test Renderer');
  });

  it('should measure frame rate', async () => {
    const { FrameRateMonitor } = await import('../../../viewer/src/utils/webgl.js');
    
    const monitor = new FrameRateMonitor();
    
    // Simulate frames
    monitor.recordFrame();
    monitor.recordFrame();
    monitor.recordFrame();
    
    const fps = monitor.getFPS();
    expect(fps).toBeGreaterThanOrEqual(0);
  });
});

describe('File Utils', () => {
  it('should detect file type from extension', async () => {
    const { getFileType } = await import('../../../viewer/src/utils/file.js');
    
    expect(getFileType('test.vrm')).toBe('vrm');
    expect(getFileType('test.gltf')).toBe('gltf');
    expect(getFileType('test.bvh')).toBe('bvh');
    expect(getFileType('test.json')).toBe('json');
    expect(getFileType('test.unknown')).toBe('unknown');
  });

  it('should validate file size', async () => {
    const { validateFileSize } = await import('../../../viewer/src/utils/file.js');
    
    const smallFile = { size: 1024 * 1024 }; // 1MB
    const largeFile = { size: 100 * 1024 * 1024 }; // 100MB
    
    expect(validateFileSize(smallFile, 50 * 1024 * 1024)).toBe(true);
    expect(validateFileSize(largeFile, 50 * 1024 * 1024)).toBe(false);
  });

  it('should read file as text', async () => {
    const { readFileAsText } = await import('../../../viewer/src/utils/file.js');
    
    const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
    
    // Mock FileReader
    global.FileReader = vi.fn(() => ({
      readAsText: vi.fn(function() {
        this.result = 'test content';
        this.onload?.();
      }),
      result: null,
      onload: null,
      onerror: null
    }));
    
    const content = await readFileAsText(mockFile);
    expect(content).toBe('test content');
  });
});

describe('Math Utils', () => {
  it('should clamp values correctly', async () => {
    const { clamp } = await import('../../../viewer/src/utils/math.js');
    
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  it('should lerp between values', async () => {
    const { lerp } = await import('../../../viewer/src/utils/math.js');
    
    expect(lerp(0, 10, 0.5)).toBe(5);
    expect(lerp(0, 10, 0)).toBe(0);
    expect(lerp(0, 10, 1)).toBe(10);
  });

  it('should convert degrees to radians', async () => {
    const { degToRad } = await import('../../../viewer/src/utils/math.js');
    
    expect(degToRad(180)).toBeCloseTo(Math.PI, 5);
    expect(degToRad(90)).toBeCloseTo(Math.PI / 2, 5);
    expect(degToRad(360)).toBeCloseTo(2 * Math.PI, 5);
  });

  it('should convert radians to degrees', async () => {
    const { radToDeg } = await import('../../../viewer/src/utils/math.js');
    
    expect(radToDeg(Math.PI)).toBeCloseTo(180, 5);
    expect(radToDeg(Math.PI / 2)).toBeCloseTo(90, 5);
    expect(radToDeg(2 * Math.PI)).toBeCloseTo(360, 5);
  });
});

describe('Storage Utils', () => {
  beforeEach(() => {
    // Mock localStorage
    const mockStorage = {};
    global.localStorage = {
      getItem: vi.fn(key => mockStorage[key] || null),
      setItem: vi.fn((key, value) => { mockStorage[key] = value; }),
      removeItem: vi.fn(key => { delete mockStorage[key]; }),
      clear: vi.fn(() => { Object.keys(mockStorage).forEach(key => delete mockStorage[key]); })
    };
  });

  it('should save and load preferences', async () => {
    const { savePreferences, loadPreferences } = await import('../../../viewer/src/utils/storage.js');
    
    const preferences = {
      environment: 'classroom',
      quality: 'high',
      autoPlay: true
    };
    
    savePreferences(preferences);
    const loaded = loadPreferences();
    
    expect(loaded).toEqual(preferences);
  });

  it('should handle invalid JSON in storage', async () => {
    const { loadPreferences } = await import('../../../viewer/src/utils/storage.js');
    
    localStorage.setItem('viewer-preferences', 'invalid json');
    
    const loaded = loadPreferences();
    expect(loaded).toEqual({}); // Should return default empty object
  });

  it('should cache animation data', async () => {
    const { cacheAnimation, getCachedAnimation } = await import('../../../viewer/src/utils/storage.js');
    
    const animation = {
      name: 'test-animation',
      data: { tracks: [] }
    };
    
    cacheAnimation('test-key', animation);
    const cached = getCachedAnimation('test-key');
    
    expect(cached).toEqual(animation);
  });
});

describe('Event Utils', () => {
  it('should create event emitter', async () => {
    const { EventEmitter } = await import('../../../viewer/src/utils/events.js');
    
    const emitter = new EventEmitter();
    const handler = vi.fn();
    
    emitter.on('test', handler);
    emitter.emit('test', 'data');
    
    expect(handler).toHaveBeenCalledWith('data');
  });

  it('should remove event listeners', async () => {
    const { EventEmitter } = await import('../../../viewer/src/utils/events.js');
    
    const emitter = new EventEmitter();
    const handler = vi.fn();
    
    emitter.on('test', handler);
    emitter.off('test', handler);
    emitter.emit('test', 'data');
    
    expect(handler).not.toHaveBeenCalled();
  });

  it('should handle once listeners', async () => {
    const { EventEmitter } = await import('../../../viewer/src/utils/events.js');
    
    const emitter = new EventEmitter();
    const handler = vi.fn();
    
    emitter.once('test', handler);
    emitter.emit('test', 'data1');
    emitter.emit('test', 'data2');
    
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith('data1');
  });
});

describe('Validation Utils', () => {
  it('should validate avatar configuration', async () => {
    const { validateAvatarConfig } = await import('../../../viewer/src/utils/validation.js');
    
    const validConfig = {
      name: 'Test Avatar',
      url: '/test.vrm',
      scale: 1.0,
      position: [0, 0, 0]
    };
    
    expect(validateAvatarConfig(validConfig)).toBe(true);
    
    const invalidConfig = {
      // missing required fields
      scale: 'invalid'
    };
    
    expect(validateAvatarConfig(invalidConfig)).toBe(false);
  });

  it('should validate environment configuration', async () => {
    const { validateEnvironmentConfig } = await import('../../../viewer/src/utils/validation.js');
    
    const validConfig = {
      type: 'classroom',
      lighting: {
        ambient: [0.3, 0.3, 0.3],
        directional: {
          color: [1, 1, 1],
          position: [10, 10, 10]
        }
      }
    };
    
    expect(validateEnvironmentConfig(validConfig)).toBe(true);
    
    const invalidConfig = {
      type: 'invalid-type'
    };
    
    expect(validateEnvironmentConfig(invalidConfig)).toBe(false);
  });

  it('should validate color values', async () => {
    const { validateColor } = await import('../../../viewer/src/utils/validation.js');
    
    expect(validateColor([1, 0.5, 0])).toBe(true);
    expect(validateColor([0, 0, 0])).toBe(true);
    expect(validateColor([1, 1, 1])).toBe(true);
    
    expect(validateColor([2, 0, 0])).toBe(false); // > 1
    expect(validateColor([-1, 0, 0])).toBe(false); // < 0
    expect(validateColor([1, 1])).toBe(false); // wrong length
  });
});
