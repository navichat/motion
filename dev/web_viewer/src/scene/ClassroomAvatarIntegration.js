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
   * Load and position Ichika VRM avatar using working VRM system
   */
  async loadAndPositionAvatar() {
    try {
      console.log('ClassroomAvatarIntegration: Loading VRM avatar with working VRM system...');
      
      // Use the working VRMLoaderLite pattern from ichika_voice_conversation_demo.html
      if (typeof window.VRMLoaderLite !== 'undefined') {
        console.log('✅ VRMLoaderLite is available, attempting VRM load...');
        
        const vrmPaths = [
          '../assets/avatars/ichika.vrm',
          '../assets/avatars/buny.vrm', 
          '../assets/avatars/kaede.vrm'
        ];
        
        for (const vrmPath of vrmPaths) {
          try {
            console.log(`Loading VRM with VRMLoaderLite: ${vrmPath}`);
            
            // Fetch VRM file as array buffer (working pattern)
            const response = await fetch(vrmPath);
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}`);
            }
            
            const buffer = await response.arrayBuffer();
            console.log(`✅ VRM file fetched (${buffer.byteLength} bytes)`);
            
            // Load VRM using VRMLoaderLite
            const loader = new window.VRMLoaderLite();
            const result = await loader.loadFromArrayBuffer(buffer, 'ichika.vrm');
            
            if (result && result.vrm) {
              console.log('✅ VRM loaded successfully!');
              
              this.vrm = result.vrm;
              this.vrmModel = result;
              this.avatar = result;
              this.vrmReady = true;
              
              // Add VRM scene to main scene
              if (result.vrm.scene) {
                this.scene.add(result.vrm.scene);
                console.log('✅ VRM scene added to main scene');
              }
              
              // Position avatar in classroom
              this.positionAvatarInClassroom();
              
              // Setup VRM animation system with working pattern
              await this.setupWorkingVRMAnimationSystem();
              
              console.log('🎉 VRM avatar loaded successfully with VRMLoaderLite!');
              return;
            }
          } catch (error) {
            console.warn(`Failed to load VRM ${vrmPath} with VRMLoaderLite:`, error.message);
            continue;
          }
        }
      }
      
      // Fallback to manual VRM loading only if VRMLoaderLite is not available
      console.warn('VRMLoaderLite not available, trying manual loading...');
      await this.loadVRMManually();
      
    } catch (error) {
      console.warn('VRM loading failed completely, creating simple avatar:', error);
      this.createSimpleAvatar();
    }
  }

  /**
   * Manual VRM loading fallback
   */
  async loadVRMManually() {
    // Check if dynamic import of Three.js modules is available
    let GLTFLoader, VRMLoaderPlugin;
    
    try {
      // Try to import Three.js modules dynamically
      if (typeof window !== 'undefined' && window.THREE) {
        GLTFLoader = window.THREE.GLTFLoader;
        VRMLoaderPlugin = window.THREE.VRMLoaderPlugin;
      }
      
      // If not available globally, try dynamic import
      if (!GLTFLoader) {
        const threeModule = await import('three/addons/loaders/GLTFLoader.js');
        GLTFLoader = threeModule.GLTFLoader;
      }
      
      if (!VRMLoaderPlugin) {
        const vrmModule = await import('@pixiv/three-vrm');
        VRMLoaderPlugin = vrmModule.VRMLoaderPlugin;
      }
      
    } catch (error) {
      console.warn('Failed to load VRM modules dynamically:', error);
      throw error;
    }
    
    if (!GLTFLoader || !VRMLoaderPlugin) {
      throw new Error('VRM loading modules not available');
    }
    
    // Setup loader
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    
    const vrmPaths = [
      '../assets/avatars/ichika.vrm',
      '../assets/avatars/buny.vrm', 
      '../assets/avatars/kaede.vrm'
    ];
    
    for (const vrmPath of vrmPaths) {
      try {
        console.log(`Manually loading VRM: ${vrmPath}`);
        
        const gltf = await new Promise((resolve, reject) => {
          loader.load(vrmPath, resolve, 
            (progress) => console.log(`VRM loading progress: ${(progress.loaded / progress.total * 100).toFixed(1)}%`),
            reject);
        });
        
        const vrm = gltf.userData.vrm;
        if (vrm) {
          await vrm.ready;
          
          this.vrmModel = { vrm, scene: vrm.scene, gltf };
          this.avatar = this.vrmModel;
          this.vrmReady = true;
          
          this.scene.add(vrm.scene);
          
          // Position avatar in classroom
          this.positionAvatarInClassroom();
          
          // Setup VRM animation system
          await this.setupVRMAnimationSystem();
          
          console.log('✅ VRM loaded manually and positioned in classroom');
          return;
        }
      } catch (error) {
        console.warn(`Failed to manually load VRM ${vrmPath}:`, error);
        continue;
      }
    }
    
    throw new Error('All VRM loading attempts failed');
  }

  /**
   * Setup VRM animation system using working pattern
   */
  async setupWorkingVRMAnimationSystem() {
    if (!this.vrm) {
      console.warn('VRM not available for animation setup');
      return;
    }
    
    try {
      console.log('Setting up working VRM animation system...');
      
      // Initialize avatar binder (working pattern)
      if (typeof window.AvatarBinder !== 'undefined') {
        this.binder = new window.AvatarBinder(this.vrm);
        console.log('✅ AvatarBinder initialized with VRM');
      } else {
        console.warn('AvatarBinder not available, using stub');
        this.binder = new (class { 
          constructor() { this.stub = true; this.records = []; this.blendshapeRecords = []; } 
          updateBone() {} 
          update() {} 
          updateBlendshape(n, w) { this.blendshapeRecords.push({ name: n, weight: w }); } 
          getStats() { return { applied: this.records.length, expressions: this.blendshapeRecords.length, stub: true }; } 
        })();
      }
      
      // Load BVH animations
      await this.loadWorkingBVHAnimations();
      
      // Setup BVH timeline (working pattern)
      if (typeof window.BVHTimeline !== 'undefined') {
        const BVHTLCtor = (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
        this.bvhTimeline = new BVHTLCtor({ 
          framerate: 30, 
          lookaheadFrames: 30, 
          onFrameUpdate: (frame, t) => {
            // Apply frame updates to VRM
            if (this.binder && frame) {
              this.binder.update();
            }
          }
        });
        console.log('✅ BVH Timeline initialized');
      }
      
      // Setup VRM-BVH integration (working pattern)
      if (typeof window.BVHTimelineVRMIntegration !== 'undefined' && this.binder) {
        this.vrmIntegration = new window.BVHTimelineVRMIntegration(this.binder);
        if (this.bvhTimeline) {
          this.vrmIntegration.connectTimeline(this.bvhTimeline);
        }
        console.log('✅ VRM-BVH integration connected');
      }
      
      // Start simple idle animations
      this.startIdleAnimations();
      
      this.animationsReady = true;
      console.log('🎉 Working VRM animation system setup complete!');
      
    } catch (error) {
      console.warn('Working VRM animation system setup failed:', error);
      // Continue without animations
    }
  }

  /**
   * Load BVH animations using working approach
   */
  async loadWorkingBVHAnimations() {
    const bvhPaths = [
      '../assets/bvh/minimal_idle.bvh',
      '../assets/animations/neutral_reference.bvh'
    ];
    
    for (const bvhPath of bvhPaths) {
      try {
        console.log(`Loading BVH animation: ${bvhPath}`);
        
        const response = await fetch(bvhPath);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        
        const bvhData = await response.text();
        console.log(`✅ BVH animation loaded: ${bvhPath} (${bvhData.length} chars)`);
        
        this.currentBVHData = bvhData;
        return; // Use first successful load
        
      } catch (error) {
        console.warn(`Failed to load BVH ${bvhPath}:`, error.message);
        continue;
      }
    }
    
    console.log('⚠️ No BVH files loaded, continuing without BVH data');
  }

  /**
   * Start simple idle animations for the VRM
   */
  startIdleAnimations() {
    if (!this.vrm || !this.vrm.humanoid) {
      console.warn('VRM humanoid not available for idle animations');
      return;
    }

    console.log('🎭 Starting VRM idle animations...');
    
    let animationFrame = 0;
    const startTime = performance.now();
    
    const idleAnimation = () => {
      if (!this.vrm || !this.vrm.humanoid) return;
      
      const time = (performance.now() - startTime) * 0.001;
      
      try {
        // Breathing animation
        const chest = this.vrm.humanoid.getNormalizedBoneNode('chest');
        if (chest) {
          chest.scale.y = 1 + Math.sin(time * 0.5) * 0.02;
        }
        
        // Head movement
        const head = this.vrm.humanoid.getNormalizedBoneNode('head');
        if (head) {
          head.rotation.y = Math.sin(time * 0.3) * 0.05;
          head.rotation.x = Math.sin(time * 0.2) * 0.02;
        }
        
        // Subtle blinking (blend shapes)
        if (this.vrm.expressionManager) {
          const blinkValue = Math.max(0, Math.sin(time * 2) * 0.1);
          this.vrm.expressionManager.setValue('blink', blinkValue);
        }
        
      } catch (error) {
        // Silently handle animation errors
      }
      
      animationFrame++;
      if (animationFrame % 60 === 0) {
        console.log('🎭 VRM idle animation running...');
      }
      
      requestAnimationFrame(idleAnimation);
    };
    
    idleAnimation();
    console.log('✅ VRM idle animations started');
  }

  /**
   * Load BVH animation data
   */
  async loadBVHAnimations() {
    const bvhPaths = [
      '../assets/bvh/minimal_idle.bvh',
      '../assets/animations/neutral_reference.bvh',
      '../assets/animations/test_neutral.bvh'
    ];
    
    for (const bvhPath of bvhPaths) {
      try {
        console.log(`Loading BVH animation: ${bvhPath}`);
        
        const response = await fetch(bvhPath);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        
        const bvhData = await response.text();
        
        // Parse BVH data (simple parsing for now)
        this.currentBVHData = this.parseBVHData(bvhData);
        
        console.log('✅ BVH animation loaded:', bvhPath);
        return; // Use first successful load
        
      } catch (error) {
        console.warn(`Failed to load BVH ${bvhPath}:`, error);
        continue;
      }
    }
    
    // Create minimal BVH skeleton if no files load
    this.currentBVHData = this.createMinimalBVHSkeleton();
    console.log('✅ Using minimal BVH skeleton');
  }

  /**
   * Simple BVH data parser
   */
  parseBVHData(bvhText) {
    // Very basic BVH parsing - in a real implementation this would be more robust
    const lines = bvhText.split('\n');
    const bones = [];
    
    // Extract basic bone structure
    let inHierarchy = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed === 'HIERARCHY') {
        inHierarchy = true;
        continue;
      }
      if (trimmed === 'MOTION') {
        break;
      }
      if (inHierarchy && (trimmed.startsWith('ROOT') || trimmed.startsWith('JOINT'))) {
        const boneName = trimmed.split(' ')[1];
        bones.push(boneName);
      }
    }
    
    return {
      bones: bones,
      frameRate: 30,
      frames: [], // Would contain actual frame data in full implementation
      skeleton: { bones: bones.map(name => ({ name, parent: null, position: [0,0,0], rotation: [0,0,0] })) }
    };
  }

  /**
   * Create minimal BVH skeleton for animation
   */
  createMinimalBVHSkeleton() {
    const basicBones = ['Hips', 'Spine', 'Head', 'LeftArm', 'RightArm', 'LeftLeg', 'RightLeg'];
    
    return {
      bones: basicBones,
      frameRate: 30,
      frames: [],
      skeleton: {
        bones: basicBones.map(name => ({
          name: name,
          parent: name === 'Hips' ? null : 'Hips',
          position: [0, 0, 0],
          rotation: [0, 0, 0]
        }))
      }
    };
  }

  /**
   * Position avatar appropriately in classroom
   */
  positionAvatarInClassroom() {
    if (!this.avatar) return;
    
    // Handle VRM positioning (working pattern)
    if (this.vrm && this.vrm.scene) {
      console.log('Positioning VRM avatar in classroom...');
      
      // Position VRM scene at teacher position
      this.vrm.scene.position.set(0, 0, -1.5);
      this.vrm.scene.rotation.y = 0; // Face forward
      this.vrm.scene.scale.setScalar(1);
      
      // Enable shadows for VRM
      this.vrm.scene.traverse((child) => {
        if (child.isMesh) {
          try {
            child.castShadow = true;
            child.receiveShadow = true;
          } catch (error) {
            console.warn('Shadow setup failed for VRM mesh, continuing without shadows');
          }
        }
      });
      
      console.log('✅ VRM avatar positioned in classroom');
      
    } else {
      // Handle simple avatar or other avatar types
      const avatarScene = this.avatar.vrm ? this.avatar.vrm.scene : 
                        this.avatar.scene ? this.avatar.scene : this.avatar;
      
      if (avatarScene) {
        // Position avatar at teacher position in classroom
        avatarScene.position.set(0, 0, -1.5); // Teacher position
        avatarScene.rotation.y = 0; // Face forward towards students
        avatarScene.scale.setScalar(1); // Normal scale
        
        // Enable shadows for avatar (safe fallback)
        avatarScene.traverse((child) => {
          if (child.isMesh) {
            try {
              child.castShadow = true;
              child.receiveShadow = true;
            } catch (error) {
              console.warn('Shadow setup failed for avatar mesh, continuing without shadows');
            }
          }
        });
        
        console.log('✅ Avatar positioned in classroom at teacher position');
      }
    }
    
    // Set avatar ready state
    this.avatarReady = true;
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
   * Start render loop with VRM animation support
   */
  startRenderLoop() {
    const animate = () => {
      this.renderLoop = requestAnimationFrame(animate);
      
      const deltaTime = this.clock.getDelta();
      
      // Update controls
      if (this.controls) {
        this.controls.update();
      }
      
      // Update VRM animations (working pattern)
      if (this.vrm) {
        try {
          // Update VRM internal systems
          this.vrm.update(deltaTime);
        } catch (error) {
          // Ignore VRM update errors for compatibility
        }
      }
      
      // Update avatar binder
      if (this.binder && this.binder.update) {
        try {
          this.binder.update();
        } catch (error) {
          // Ignore binder update errors for compatibility
        }
      }
      
      // Update BVH adapter animations
      if (this.bvhAdapter && this.bvhAdapter.tick) {
        try {
          this.bvhAdapter.tick(deltaTime);
        } catch (error) {
          // Ignore BVH adapter errors for compatibility
        }
      }
      
      // Update BVH timeline
      if (this.bvhTimeline && this.bvhTimeline.update) {
        try {
          this.bvhTimeline.update(deltaTime);
        } catch (error) {
          // Ignore timeline errors for compatibility
        }
      }
      
      // Update animation mixer (Three.js animations)
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
    console.log('✅ Render loop started with VRM animation support');
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