import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { JSDOM } from 'jsdom';

/**
 * Unit Tests for 3D Viewer Components
 * 
 * These tests verify individual component functionality without requiring
 * a full browser environment.
 */

// Mock Three.js for unit testing
vi.mock('three', () => ({
  Scene: vi.fn(() => ({
    add: vi.fn(),
    remove: vi.fn(),
    clear: vi.fn()
  })),
  PerspectiveCamera: vi.fn(() => ({
    position: { set: vi.fn(), copy: vi.fn() },
    lookAt: vi.fn(),
    updateProjectionMatrix: vi.fn(),
    aspect: 1
  })),
  WebGLRenderer: vi.fn(() => ({
    setSize: vi.fn(),
    setPixelRatio: vi.fn(),
    render: vi.fn(),
    dispose: vi.fn()
  })),
  Clock: vi.fn(() => ({
    getDelta: vi.fn(() => 0.016) // ~60 FPS
  })),
  AnimationMixer: vi.fn(() => ({
    update: vi.fn(),
    stopAllAction: vi.fn(),
    clipAction: vi.fn(() => ({
      setLoop: vi.fn(),
      reset: vi.fn(),
      play: vi.fn(),
      stop: vi.fn(),
      fadeIn: vi.fn(),
      fadeOut: vi.fn(),
      getClip: vi.fn(() => ({ duration: 5.0 })),
      time: 0,
      timeScale: 1.0,
      paused: false
    })),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  })),
  Vector3: vi.fn(() => ({
    copy: vi.fn(),
    set: vi.fn(),
    setScalar: vi.fn()
  })),
  AmbientLight: vi.fn(),
  DirectionalLight: vi.fn(() => ({
    position: { set: vi.fn() },
    castShadow: true
  })),
  LoopRepeat: 'LoopRepeat',
  LoopOnce: 'LoopOnce'
}));

describe('Player', () => {
  let mockCanvas;
  let dom;

  beforeEach(() => {
    // Set up JSDOM
    dom = new JSDOM('<!DOCTYPE html><html><body><canvas></canvas></body></html>');
    global.window = dom.window;
    global.document = dom.window.document;
    global.requestAnimationFrame = vi.fn(cb => setTimeout(cb, 16));
    global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
    
    mockCanvas = document.querySelector('canvas');
    mockCanvas.getContext = vi.fn(() => ({
      isContextLost: vi.fn(() => false)
    }));
    mockCanvas.getBoundingClientRect = vi.fn(() => ({
      width: 800,
      height: 600
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with canvas element', async () => {
    const { Player } = await import('../../../viewer/src/Player.js');
    
    const player = new Player(mockCanvas);
    
    expect(player.canvas).toBe(mockCanvas);
    expect(player.scene).toBeDefined();
    expect(player.camera).toBeDefined();
    expect(player.renderer).toBeDefined();
  });

  it('should handle canvas resize correctly', async () => {
    const { Player } = await import('../../../viewer/src/Player.js');
    
    const player = new Player(mockCanvas);
    
    // Mock renderer methods
    player.renderer.setSize = vi.fn();
    player.renderer.setPixelRatio = vi.fn();
    player.camera.updateProjectionMatrix = vi.fn();
    
    // Simulate resize
    mockCanvas.getBoundingClientRect = vi.fn(() => ({
      width: 1200,
      height: 800
    }));
    
    player.handleCanvasResize();
    
    expect(player.renderer.setSize).toHaveBeenCalledWith(1200, 800);
    expect(player.camera.updateProjectionMatrix).toHaveBeenCalled();
  });

  it('should emit tick events', async () => {
    const { Player } = await import('../../../viewer/src/Player.js');
    
    const player = new Player(mockCanvas);
    const tickHandler = vi.fn();
    
    player.on('tick', tickHandler);
    player.emit('tick', 0.016);
    
    expect(tickHandler).toHaveBeenCalledWith(0.016);
  });

  it('should dispose resources correctly', async () => {
    const { Player } = await import('../../../viewer/src/Player.js');
    
    const player = new Player(mockCanvas);
    player.renderer.dispose = vi.fn();
    player.scene.clear = vi.fn();
    
    player.dispose();
    
    expect(player.renderer.dispose).toHaveBeenCalled();
    expect(player.scene.clear).toHaveBeenCalled();
  });
});

describe('AnimationController', () => {
  let mockAvatarInstance;
  let mockMixer;

  beforeEach(() => {
    mockMixer = {
      update: vi.fn(),
      stopAllAction: vi.fn(),
      clipAction: vi.fn(() => ({
        setLoop: vi.fn(),
        reset: vi.fn(),
        play: vi.fn(),
        stop: vi.fn(),
        fadeIn: vi.fn(),
        fadeOut: vi.fn(),
        getClip: vi.fn(() => ({ duration: 5.0, name: 'test-animation' })),
        time: 0,
        timeScale: 1.0,
        paused: false
      })),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    };

    mockAvatarInstance = {
      mixer: mockMixer,
      scene: { name: 'test-avatar' }
    };
  });

  it('should initialize with avatar instance', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    expect(controller.avatarInstance).toBe(mockAvatarInstance);
    expect(controller.mixer).toBe(mockMixer);
  });

  it('should play JSON animation correctly', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    const mockAnimation = {
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

    const action = controller.playAnimation(mockAnimation);
    
    expect(mockMixer.clipAction).toHaveBeenCalled();
    expect(action).toBeDefined();
  });

  it('should handle playback controls', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    // Mock current action
    const mockAction = {
      paused: false,
      stop: vi.fn(),
      time: 0,
      timeScale: 1.0,
      getClip: vi.fn(() => ({ duration: 5.0 }))
    };
    controller.currentAction = mockAction;
    
    // Test pause
    controller.pause();
    expect(mockAction.paused).toBe(true);
    
    // Test resume
    controller.resume();
    expect(mockAction.paused).toBe(false);
    
    // Test stop
    controller.stop();
    expect(mockAction.stop).toHaveBeenCalled();
    
    // Test speed control
    controller.setSpeed(2.0);
    expect(mockAction.timeScale).toBe(2.0);
  });

  it('should queue animations correctly', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    const animations = [
      { name: 'anim1', format: 'json', tracks: [] },
      { name: 'anim2', format: 'json', tracks: [] },
      { name: 'anim3', format: 'json', tracks: [] }
    ];
    
    controller.queueAnimations(animations);
    
    expect(controller.animationQueue).toHaveLength(3);
  });

  it('should emit events correctly', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    const eventHandler = vi.fn();
    controller.on('animationStarted', eventHandler);
    
    controller.emit('animationStarted', { test: 'data' });
    
    expect(eventHandler).toHaveBeenCalledWith({ test: 'data' });
  });

  it('should update correctly', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    controller.update(0.016);
    
    expect(mockMixer.update).toHaveBeenCalledWith(0.016);
  });

  it('should dispose resources correctly', async () => {
    const { AnimationController } = await import('../../../viewer/src/AnimationController.js');
    
    const controller = new AnimationController(mockAvatarInstance);
    
    controller.dispose();
    
    expect(mockMixer.stopAllAction).toHaveBeenCalled();
    expect(controller.animationQueue).toHaveLength(0);
  });
});

describe('AvatarLoader', () => {
  beforeEach(() => {
    // Mock GLTFLoader
    global.fetch = vi.fn();
  });

  it('should initialize correctly', async () => {
    const { AvatarLoader } = await import('../../../viewer/src/AvatarLoader.js');
    
    const loader = new AvatarLoader();
    
    expect(loader.cache).toBeDefined();
    expect(loader.loader).toBeDefined();
  });

  it('should create avatar config correctly', async () => {
    const { AvatarLoader } = await import('../../../viewer/src/AvatarLoader.js');
    
    const config = AvatarLoader.createAvatarConfig('Test Avatar', '/test.vrm', {
      scale: 1.5
    });
    
    expect(config.name).toBe('Test Avatar');
    expect(config.url).toBe('/test.vrm');
    expect(config.scale).toBe(1.5);
  });

  it('should get default avatars when index fails', async () => {
    const { AvatarLoader } = await import('../../../viewer/src/AvatarLoader.js');
    
    global.fetch.mockRejectedValue(new Error('Network error'));
    
    const loader = new AvatarLoader();
    const avatars = await loader.getAvailableAvatars();
    
    expect(avatars).toHaveLength(1);
    expect(avatars[0].name).toBe('Default Avatar');
  });

  it('should dispose correctly', async () => {
    const { AvatarLoader } = await import('../../../viewer/src/AvatarLoader.js');
    
    const loader = new AvatarLoader();
    loader.cache.set('test', { data: 'test' });
    
    loader.dispose();
    
    expect(loader.cache.size).toBe(0);
  });
});

describe('SceneManager', () => {
  let mockScene;

  beforeEach(() => {
    mockScene = {
      add: vi.fn(),
      remove: vi.fn()
    };
  });

  it('should initialize with scene', async () => {
    const { SceneManager } = await import('../../../viewer/src/SceneManager.js');
    
    const manager = new SceneManager(mockScene);
    
    expect(manager.scene).toBe(mockScene);
    expect(manager.environments).toBeDefined();
    expect(manager.avatars).toBeDefined();
  });

  it('should create classroom environment', async () => {
    const { SceneManager } = await import('../../../viewer/src/SceneManager.js');
    
    const manager = new SceneManager(mockScene);
    const environment = manager.createClassroomEnvironment();
    
    expect(environment.type).toBe('classroom');
    expect(environment.objects.length).toBeGreaterThan(0);
    expect(environment.lights.length).toBeGreaterThan(0);
  });

  it('should switch environments correctly', async () => {
    const { SceneManager } = await import('../../../viewer/src/SceneManager.js');
    
    const manager = new SceneManager(mockScene);
    
    // Create and set initial environment
    const classroom = manager.createClassroomEnvironment();
    manager.environments.set('classroom', classroom);
    manager.currentEnvironment = classroom;
    
    // Create and switch to stage
    const stage = manager.createStageEnvironment();
    manager.environments.set('stage', stage);
    
    manager.switchToEnvironment('stage');
    
    expect(manager.currentEnvironment).toBe(stage);
  });

  it('should manage avatars correctly', async () => {
    const { SceneManager } = await import('../../../viewer/src/SceneManager.js');
    
    const manager = new SceneManager(mockScene);
    
    const mockAvatar = {
      scene: {
        position: { copy: vi.fn() }
      }
    };
    
    manager.addAvatar('test-avatar', mockAvatar);
    
    expect(manager.avatars.has('test-avatar')).toBe(true);
    expect(mockScene.add).toHaveBeenCalledWith(mockAvatar.scene);
    
    manager.removeAvatar('test-avatar');
    
    expect(manager.avatars.has('test-avatar')).toBe(false);
    expect(mockScene.remove).toHaveBeenCalledWith(mockAvatar.scene);
  });

  it('should get available environments', async () => {
    const { SceneManager } = await import('../../../viewer/src/SceneManager.js');
    
    const manager = new SceneManager(mockScene);
    const environments = manager.getAvailableEnvironments();
    
    expect(environments).toContain('classroom');
    expect(environments).toContain('stage');
    expect(environments).toContain('studio');
    expect(environments).toContain('outdoor');
  });

  it('should dispose correctly', async () => {
    const { SceneManager } = await import('../../../viewer/src/SceneManager.js');
    
    const manager = new SceneManager(mockScene);
    
    // Add some test data
    const mockEnvironment = { objects: [], lights: [] };
    manager.environments.set('test', mockEnvironment);
    
    const mockAvatar = { scene: {} };
    manager.avatars.set('test', mockAvatar);
    
    manager.dispose();
    
    expect(manager.environments.size).toBe(0);
    expect(manager.avatars.size).toBe(0);
  });
});
