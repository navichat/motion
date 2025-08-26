/**
 * ClassroomAvatarIntegration.js
 * 
 * Integrates 3D VRM avatar into classroom environment with:
 * - Classroom scene loading (classroom.glb)  
 * - VRM avatar positioning and setup
 * - Animation and interaction systems
 * - WebGL/WebGPU 3D rendering pipeline
 */

class ClassroomAvatarIntegration {
  constructor(containerElement = null) {
    this.container = containerElement || document.body;
    this.scene = null;
    this.renderer = null;
    this.camera = null;
    this.avatar = null;
    this.classroom = null;
    this.mixer = null;
    
    this.initialized = false;
    this.renderLoop = null;
    
    // Three.js integration
    this.clock = null;
    this.controls = null;
    
    // VRM and animation systems
    this.vrmLoader = null;
    this.animationSystem = null;
  }

  /**
   * Initialize complete 3D scene with classroom and avatar
   */
  async initializeScene() {
    try {
      console.log('ClassroomAvatarIntegration: Initializing 3D scene...');
      
      // Setup Three.js scene
      await this.setupThreeJSScene();
      
      // Load classroom environment
      await this.loadClassroomEnvironment();
      
      // Load and position Ichika VRM avatar
      await this.loadAndPositionAvatar();
      
      // Setup animation systems
      await this.setupAnimationSystems();
      
      // Start render loop
      this.startRenderLoop();
      
      this.initialized = true;
      console.log('ClassroomAvatarIntegration: Scene initialization complete');
      
      return true;
      
    } catch (error) {
      console.error('ClassroomAvatarIntegration: Scene initialization failed:', error);
      throw error;
    }
  }

  /**
   * Setup Three.js scene, camera, renderer
   */
  async setupThreeJSScene() {
    // Import Three.js (if available globally)
    if (typeof THREE === 'undefined') {
      throw new Error('Three.js is required for 3D scene');
    }
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf0f0f0);
    
    // Setup camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 1.6, 3);
    
    // Setup renderer with WebGL/WebGPU support
    if (typeof THREE.WebGPURenderer !== 'undefined' && navigator.gpu) {
      try {
        this.renderer = new THREE.WebGPURenderer({ antialias: true });
        console.log('ClassroomAvatarIntegration: Using WebGPU renderer');
      } catch (e) {
        console.warn('WebGPU failed, falling back to WebGL:', e);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
      }
    } else {
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
      console.log('ClassroomAvatarIntegration: Using WebGL renderer');
    }
    
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    
    // Add to container - safely remove existing canvas
    const existingCanvas = this.container.querySelector('canvas');
    if (existingCanvas) {
      try {
        // Check if the canvas is actually a child of this container
        if (existingCanvas.parentNode === this.container) {
          this.container.removeChild(existingCanvas);
        }
      } catch (error) {
        console.warn('ClassroomAvatarIntegration: Failed to remove existing canvas:', error);
        // Continue anyway - might be already removed
      }
    }
    this.container.appendChild(this.renderer.domElement);
    
    // Setup lighting
    this.setupLighting();
    
    // Setup controls (if available)
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.25;
    }
    
    // Setup clock for animations
    this.clock = new THREE.Clock();
    
    // Handle window resize
    window.addEventListener('resize', () => this.handleResize());
  }

  /**
   * Setup scene lighting for classroom
   */
  setupLighting() {
    // Main directional light (sunlight)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    this.scene.add(directionalLight);
    
    // Ambient light for overall brightness
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);
    
    // Point lights for classroom atmosphere
    const pointLight1 = new THREE.PointLight(0xffffff, 0.8, 10);
    pointLight1.position.set(-3, 3, 2);
    this.scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0xffffff, 0.8, 10);
    pointLight2.position.set(3, 3, 2);
    this.scene.add(pointLight2);
  }

  /**
   * Load classroom environment (classroom.glb)
   */
  async loadClassroomEnvironment() {
    return new Promise((resolve, reject) => {
      // Check for GLTFLoader availability
      if (typeof THREE.GLTFLoader === 'undefined') {
        console.warn('GLTFLoader not available, creating simple classroom');
        this.createSimpleClassroom();
        resolve();
        return;
      }
      
      const loader = new THREE.GLTFLoader();
      const classroomPath = './assets/classroom.glb';
      
      loader.load(
        classroomPath,
        (gltf) => {
          console.log('ClassroomAvatarIntegration: Classroom loaded successfully');
          this.classroom = gltf.scene;
          
          // Position and scale classroom
          this.classroom.position.set(0, 0, 0);
          this.classroom.scale.setScalar(1);
          
          // Enable shadows
          this.classroom.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
            }
          });
          
          this.scene.add(this.classroom);
          resolve(gltf);
        },
        (progress) => {
          console.log('ClassroomAvatarIntegration: Classroom loading progress:', 
            (progress.loaded / progress.total * 100) + '%');
        },
        (error) => {
          console.warn('Failed to load classroom.glb, creating simple classroom:', error);
          this.createSimpleClassroom();
          resolve();
        }
      );
    });
  }

  /**
   * Create simple classroom if glb file unavailable
   */
  createSimpleClassroom() {
    const classroom = new THREE.Group();
    
    // Floor
    const floorGeometry = new THREE.PlaneGeometry(10, 10);
    const floorMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    classroom.add(floor);
    
    // Walls
    const wallMaterial = new THREE.MeshLambertMaterial({ color: 0xE6E6FA });
    
    // Back wall
    const backWallGeometry = new THREE.PlaneGeometry(10, 5);
    const backWall = new THREE.Mesh(backWallGeometry, wallMaterial);
    backWall.position.set(0, 2.5, -5);
    classroom.add(backWall);
    
    // Side walls
    const sideWallGeometry = new THREE.PlaneGeometry(10, 5);
    const leftWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    leftWall.position.set(-5, 2.5, 0);
    leftWall.rotation.y = Math.PI / 2;
    classroom.add(leftWall);
    
    const rightWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    rightWall.position.set(5, 2.5, 0);
    rightWall.rotation.y = -Math.PI / 2;
    classroom.add(rightWall);
    
    // Simple desk
    const deskGeometry = new THREE.BoxGeometry(2, 0.1, 1);
    const deskMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
    const desk = new THREE.Mesh(deskGeometry, deskMaterial);
    desk.position.set(0, 1, -2);
    desk.castShadow = true;
    classroom.add(desk);
    
    this.classroom = classroom;
    this.scene.add(this.classroom);
    
    console.log('ClassroomAvatarIntegration: Simple classroom created');
  }

  /**
   * Load and position Ichika VRM avatar
   */
  async loadAndPositionAvatar() {
    return new Promise((resolve, reject) => {
      // Check for VRM/GLTF loader availability  
      if (typeof THREE.VRMLoaderPlugin === 'undefined' || typeof THREE.GLTFLoader === 'undefined') {
        console.warn('VRM/GLTF loader not available, creating simple avatar');
        this.createSimpleAvatar();
        resolve();
        return;
      }

      // Setup VRM loader
      const loader = new THREE.GLTFLoader();
      
      if (typeof THREE.VRMLoaderPlugin !== 'undefined') {
        loader.register((parser) => new THREE.VRMLoaderPlugin(parser));
      }
      
      // Try to load Ichika VRM with fallbacks
      const vrmPaths = [
        './assets/ichika.vrm',
        './assets/buny.vrm', 
        './assets/kaede.vrm'
      ];
      
      this.loadVRMWithFallback(loader, vrmPaths, 0, resolve, reject);
    });
  }

  /**
   * Load VRM with fallback chain
   */
  loadVRMWithFallback(loader, paths, index, resolve, reject) {
    if (index >= paths.length) {
      console.warn('All VRM paths failed, creating simple avatar');
      this.createSimpleAvatar();
      resolve();
      return;
    }
    
    const path = paths[index];
    console.log(`Attempting to load VRM: ${path}`);
    
    loader.load(
      path,
      (gltf) => {
        console.log(`VRM loaded successfully: ${path}`);
        
        // Extract VRM from gltf
        const vrm = gltf.userData.vrm || gltf.scene;
        this.avatar = vrm;
        
        // Position avatar in classroom
        this.positionAvatarInClassroom();
        
        // Setup avatar for animation
        this.setupAvatarAnimation();
        
        resolve(gltf);
      },
      (progress) => {
        console.log(`VRM loading progress (${path}):`, 
          (progress.loaded / progress.total * 100) + '%');
      },
      (error) => {
        console.warn(`Failed to load VRM ${path}:`, error);
        // Try next path in fallback chain
        this.loadVRMWithFallback(loader, paths, index + 1, resolve, reject);
      }
    );
  }

  /**
   * Position avatar appropriately in classroom
   */
  positionAvatarInClassroom() {
    if (!this.avatar) return;
    
    // Position avatar at teacher position
    this.avatar.position.set(0, 0, -1.5);
    this.avatar.rotation.y = 0; // Face forward
    this.avatar.scale.setScalar(1);
    
    // Enable shadows
    this.avatar.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
    
    this.scene.add(this.avatar);
    console.log('Avatar positioned in classroom');
  }

  /**
   * Setup avatar animation system
   */
  setupAvatarAnimation() {
    if (!this.avatar) return;
    
    // Create animation mixer
    this.mixer = new THREE.AnimationMixer(this.avatar);
    
    // Setup basic idle animation if available
    if (this.avatar.animations && this.avatar.animations.length > 0) {
      const idleAnimation = this.mixer.clipAction(this.avatar.animations[0]);
      idleAnimation.play();
    }
  }

  /**
   * Create simple avatar if VRM loading fails
   */
  createSimpleAvatar() {
    try {
      const avatar = new THREE.Group();
      
      // Simple head
      const headGeometry = new THREE.SphereGeometry(0.15);
      const headMaterial = new THREE.MeshLambertMaterial({ color: 0xFFDBB3 });
      const head = new THREE.Mesh(headGeometry, headMaterial);
      head.position.set(0, 1.65, 0);
      head.castShadow = true;
      avatar.add(head);
      
      // Simple body
      const bodyGeometry = new THREE.CylinderGeometry(0.1, 0.15, 0.6);
      const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0x4169E1 });
      const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
      body.position.set(0, 1.2, 0);
      body.castShadow = true;
      avatar.add(body);
      
      // Simple arms
      const armGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.4);
      const armMaterial = new THREE.MeshLambertMaterial({ color: 0xFFDBB3 });
      
      const leftArm = new THREE.Mesh(armGeometry, armMaterial);
      leftArm.position.set(-0.2, 1.3, 0);
      leftArm.rotation.z = Math.PI / 6;
      avatar.add(leftArm);
      
      const rightArm = new THREE.Mesh(armGeometry, armMaterial);
      rightArm.position.set(0.2, 1.3, 0);
      rightArm.rotation.z = -Math.PI / 6;
      avatar.add(rightArm);
      
      // Mark as Ichika avatar for identification
      avatar.name = 'IchikaAvatar';
      avatar.userData = { type: 'simple_avatar', character: 'ichika' };
      
      this.avatar = avatar;
      this.positionAvatarInClassroom();
      
      console.log('✅ Simple Ichika avatar created successfully');
      
    } catch (error) {
      console.error('Failed to create simple avatar:', error);
      // Create minimal fallback
      this.avatar = new THREE.Group();
      this.avatar.name = 'MinimalAvatar';
      this.positionAvatarInClassroom();
    }
  }

  /**
   * Setup animation systems integration
   */
  async setupAnimationSystems() {
    try {
      // Initialize BVH animation system if available
      if (typeof BVHTimeline !== 'undefined') {
        this.animationSystem = new BVHTimeline();
        await this.animationSystem.initialize();
        console.log('BVH animation system initialized');
      }
      
      // Setup speech gesture synchronization
      this.setupSpeechGestureSync();
      
    } catch (error) {
      console.warn('Animation system setup failed:', error);
    }
  }

  /**
   * Setup speech-gesture synchronization
   */
  setupSpeechGestureSync() {
    if (!this.avatar) return;
    
    // Create avatar controller with speech capabilities
    this.avatar.speak = async (text) => {
      console.log('Avatar speaking:', text);
      
      // Trigger speaking animation if available
      if (this.mixer && this.avatar.speakingAnimation) {
        const speakAction = this.mixer.clipAction(this.avatar.speakingAnimation);
        speakAction.reset().play();
      } else {
        // Simple head bob animation for speaking
        this.animateSimpleSpeaking();
      }
      
      // Use TTS system if available
      if (typeof MultiEngineTTSManager !== 'undefined') {
        const tts = new MultiEngineTTSManager();
        await tts.speak(text);
      } else {
        // Fallback to Web Speech API
        return new Promise((resolve) => {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.onend = resolve;
          speechSynthesis.speak(utterance);
        });
      }
    };
    
    // Add simple speaking animation method for testing
    this.avatar.animateSimpleSpeaking = () => {
      this.animateSimpleSpeaking();
    };
    
    this.avatar.stop = () => {
      if (this.mixer) {
        this.mixer.stopAllActions();
      }
    };
    
    console.log('✅ Avatar speech capabilities initialized');
  }

  /**
   * Simple speaking animation for fallback avatar
   */
  animateSimpleSpeaking() {
    if (!this.avatar || !this.avatar.children) return;
    
    let bobCount = 0;
    const bobDuration = 200; // ms
    const maxBobs = 5;
    
    const bobInterval = setInterval(() => {
      if (bobCount >= maxBobs) {
        clearInterval(bobInterval);
        return;
      }
      
      try {
        // Simple head movement - find head by geometry type
        const head = this.avatar.children.find(child => 
          child.geometry && child.geometry.type === 'SphereGeometry'
        );
        
        if (head) {
          head.rotation.x = Math.sin(bobCount * Math.PI / 4) * 0.1;
          head.position.y = 1.65 + Math.sin(bobCount * Math.PI / 2) * 0.02;
        } else {
          // Fallback: just rotate the whole avatar slightly
          this.avatar.rotation.y = Math.sin(bobCount * Math.PI / 6) * 0.05;
        }
      } catch (error) {
        console.warn('Simple animation error (non-critical):', error);
        // Continue animation even if one frame fails
      }
      
      bobCount++;
    }, bobDuration);
  }

  /**
   * Start render loop
   */
  startRenderLoop() {
    const animate = () => {
      this.renderLoop = requestAnimationFrame(animate);
      
      const deltaTime = this.clock.getDelta();
      
      // Update controls
      if (this.controls) {
        this.controls.update();
      }
      
      // Update animation mixer
      if (this.mixer) {
        this.mixer.update(deltaTime);
      }
      
      // Update animation system
      if (this.animationSystem && this.animationSystem.update) {
        this.animationSystem.update(deltaTime);
      }
      
      // Render scene
      this.renderer.render(this.scene, this.camera);
    };
    
    animate();
    console.log('Render loop started');
  }

  /**
   * Handle window resize
   */
  handleResize() {
    if (!this.camera || !this.renderer) return;
    
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  /**
   * Cleanup and dispose resources
   */
  dispose() {
    console.log('ClassroomAvatarIntegration: Disposing resources...');
    
    // Stop render loop
    if (this.renderLoop) {
      cancelAnimationFrame(this.renderLoop);
      this.renderLoop = null;
    }
    
    // Dispose of Three.js resources
    if (this.scene) {
      this.scene.clear();
    }
    
    if (this.renderer) {
      this.renderer.dispose();
    }
    
    // Remove event listeners
    window.removeEventListener('resize', () => this.handleResize());
    
    this.initialized = false;
  }

  /**
   * Get current integration status
   */
  getStatus() {
    return {
      initialized: this.initialized,
      hasClassroom: !!this.classroom,
      hasAvatar: !!this.avatar,
      hasAnimationSystem: !!this.animationSystem,
      rendererType: this.renderer?.constructor.name,
      avatarPosition: this.avatar?.position
    };
  }
}

// Export for use in browser environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ClassroomAvatarIntegration;
} else if (typeof window !== 'undefined') {
  window.ClassroomAvatarIntegration = ClassroomAvatarIntegration;
}