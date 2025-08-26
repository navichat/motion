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
      console.log('ClassroomAvatarIntegration: Loading VRM avatar with existing infrastructure...');
      
      // First priority: Use the working VRMLoaderLite pattern
      if (typeof window.VRMLoaderLite !== 'undefined') {
        console.log('✅ VRMLoaderLite is available, attempting VRM load...');
        
        const vrmPaths = [
          './assets/avatars/ichika.vrm',
          './assets/avatars/buny.vrm', 
          './assets/avatars/kaede.vrm'
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
              
              // Wait for VRM to be fully ready
              if (result.vrm.ready) {
                await result.vrm.ready;
                console.log('✅ VRM ready promise resolved');
              }
              
              this.vrm = result.vrm;
              this.vrmModel = result;
              this.avatar = result.vrm.scene; // Use VRM scene as avatar
              this.vrmReady = true;
              
              // Add VRM scene to main scene
              if (result.vrm.scene) {
                this.scene.add(result.vrm.scene);
                console.log('✅ VRM scene added to main scene');
                
                // Log VRM scene details for debugging
                result.vrm.scene.traverse((child) => {
                  if (child.isMesh) {
                    console.log(`VRM mesh found: ${child.name}, geometry: ${child.geometry.constructor.name}`);
                  }
                });
                
                // Make sure VRM is visible and properly scaled
                result.vrm.scene.visible = true;
                result.vrm.scene.scale.setScalar(1.0);
              }
              
              // Position avatar in classroom
              this.positionAvatarInClassroom();
              
              // Setup VRM animation system with existing infrastructure
              await this.setupRealVRMAnimationSystem();
              
              console.log('🎉 Real VRM avatar loaded successfully - no geometric fallbacks!');
              return;
            }
          } catch (error) {
            console.warn(`Failed to load VRM ${vrmPath} with VRMLoaderLite:`, error.message);
            continue;
          }
        }
      }
      
      // Second priority: Use Three.js VRM loading directly 
      console.log('Attempting Three.js VRM loading with proper infrastructure...');
      await this.loadVRMWithThreeJS();
      
    } catch (error) {
      console.error('❌ VRM loading failed completely - this should not happen:', error);
      // Instead of geometric fallback, throw error to show the problem
      throw new Error(`VRM loading system failure: ${error.message}`);
    }
  }

  /**
   * Load VRM using Three.js modules with proper infrastructure
   */
  async loadVRMWithThreeJS() {
    // Check if Three.js VRM modules are available
    if (!window.THREE || !window.THREE.GLTFLoader || !window.THREE.VRMLoaderPlugin) {
      throw new Error('Required Three.js VRM modules are not available');
    }
    
    // Setup Three.js VRM loader
    const loader = new window.THREE.GLTFLoader();
    loader.register((parser) => new window.THREE.VRMLoaderPlugin(parser));
    
    const vrmPaths = [
      './assets/avatars/ichika.vrm',
      './assets/avatars/buny.vrm', 
      './assets/avatars/kaede.vrm'
    ];
    
    for (const vrmPath of vrmPaths) {
      try {
        console.log(`Loading VRM with Three.js: ${vrmPath}`);
        
        const gltf = await new Promise((resolve, reject) => {
          loader.load(vrmPath, resolve, 
            (progress) => {
              const percent = (progress.loaded / progress.total * 100).toFixed(1);
              console.log(`VRM loading progress: ${percent}%`);
            },
            reject);
        });
        
        const vrm = gltf.userData.vrm;
        if (vrm) {
          // Wait for VRM ready state
          if (vrm.ready) {
            await vrm.ready;
          }
          
          this.vrm = vrm;
          this.vrmModel = { vrm, scene: vrm.scene, gltf };
          this.avatar = vrm.scene;
          this.vrmReady = true;
          
          // Add to scene
          this.scene.add(vrm.scene);
          
          // Log VRM details for debugging  
          console.log(`✅ Three.js VRM loaded: ${vrm.scene.children.length} children`);
          vrm.scene.traverse((child) => {
            if (child.isMesh) {
              console.log(`VRM mesh: ${child.name}, visible: ${child.visible}`);
            }
          });
          
          // Position avatar in classroom
          this.positionAvatarInClassroom();
          
          // Setup animation system with existing infrastructure
          await this.setupRealVRMAnimationSystem();
          
          console.log('🎉 Three.js VRM loaded successfully!');
          return;
        }
      } catch (error) {
        console.warn(`Failed to load VRM ${vrmPath} with Three.js:`, error);
        continue;
      }
    }
    
    throw new Error('All Three.js VRM loading attempts failed');
  }

  /**
   * Setup VRM animation system using existing infrastructure 
   */
  async setupRealVRMAnimationSystem() {
    if (!this.vrm) {
      console.warn('VRM not available for animation setup');
      return;
    }
    
    try {
      console.log('Setting up real VRM animation system with existing infrastructure...');
      
      // Initialize AvatarBinder with the loaded VRM
      if (typeof window.AvatarBinder !== 'undefined') {
        this.binder = new window.AvatarBinder(this.vrm);
        console.log('✅ AvatarBinder initialized with real VRM');
      } else {
        console.error('❌ AvatarBinder not available - required for VRM animation');
        throw new Error('AvatarBinder not available');
      }
      
      // Setup BVH Timeline for animation composition
      if (typeof window.BVHTimeline !== 'undefined') {
        // Use the BVHTimeline constructor properly
        const BVHTLCtor = (window.BVHTimeline && window.BVHTimeline.BVHTimeline) || window.BVHTimeline;
        this.bvhTimeline = new BVHTLCtor({ 
          framerate: 30, 
          lookaheadFrames: 30, 
          onFrameUpdate: (frame, time) => {
            // Apply BVH frame updates to VRM through binder
            if (this.binder && frame) {
              this.binder.update(time);
            }
          }
        });
        console.log('✅ BVH Timeline initialized for animation composition');
      } else {
        console.error('❌ BVHTimeline not available - required for animation system');
        throw new Error('BVHTimeline not available');
      }
      
      // Setup VRM-BVH integration using existing system
      if (typeof window.BVHTimelineVRMIntegration !== 'undefined' && this.binder && this.bvhTimeline) {
        this.vrmIntegration = new window.BVHTimelineVRMIntegration(this.binder);
        this.vrmIntegration.connectTimeline(this.bvhTimeline);
        console.log('✅ VRM-BVH integration connected with existing infrastructure');
      } else {
        console.error('❌ BVHTimelineVRMIntegration not available - required for VRM animation');
        throw new Error('BVHTimelineVRMIntegration not available');
      }
      
      // Load real BVH animation data
      await this.loadRealBVHAnimations();
      
      // Start VRM humanoid animations using Three-VRM capabilities
      this.startVRMHumanoidAnimations();
      
      this.animationsReady = true;
      console.log('🎉 Real VRM animation system setup complete with existing infrastructure!');
      
    } catch (error) {
      console.error('❌ Real VRM animation system setup failed:', error);
      throw error; // Don't continue with broken animation system
    }
  }

  /**
   * Load real BVH animations using existing infrastructure
   */
  async loadRealBVHAnimations() {
    const bvhPaths = [
      './assets/bvh/minimal_idle.bvh',
      './assets/animations/neutral_reference.bvh',
      './assets/animations/test_neutral.bvh'
    ];
    
    for (const bvhPath of bvhPaths) {
      try {
        console.log(`Loading real BVH animation: ${bvhPath}`);
        
        const response = await fetch(bvhPath);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        
        const bvhData = await response.text();
        console.log(`✅ Real BVH animation loaded: ${bvhPath} (${bvhData.length} chars)`);
        
        // Parse BVH data using existing infrastructure
        if (typeof window.BVHFileLoader !== 'undefined') {
          const loader = new window.BVHFileLoader();
          this.currentBVHData = await loader.loadBvhFile(bvhData);
          console.log('✅ BVH data parsed with existing BVHFileLoader');
        } else {
          // Fallback to basic parsing
          this.currentBVHData = this.parseRealBVHData(bvhData);
          console.log('✅ BVH data parsed with fallback parser');
        }
        
        // Add to BVH timeline if available
        if (this.bvhTimeline && this.currentBVHData) {
          // Try to add BVH clip to timeline
          try {
            this.bvhTimeline.tracks.base.addClip({
              name: `bvh_${path.basename(bvhPath, '.bvh')}`,
              data: this.currentBVHData,
              startTime: 0,
              loop: true
            });
            console.log('✅ BVH clip added to timeline');
          } catch (clipError) {
            console.warn('Could not add BVH clip to timeline:', clipError);
          }
        }
        
        return; // Use first successful load
        
      } catch (error) {
        console.warn(`Failed to load real BVH ${bvhPath}:`, error.message);
        continue;
      }
    }
    
    console.warn('⚠️ No real BVH files loaded, VRM will use minimal animations only');
  }

  /**
   * Start VRM humanoid animations using Three-VRM capabilities
   */
  startVRMHumanoidAnimations() {
    if (!this.vrm || !this.vrm.humanoid) {
      console.warn('VRM humanoid not available for animations');
      return;
    }

    console.log('🎭 Starting VRM humanoid animations...');
    
    let animationFrame = 0;
    const startTime = performance.now();
    
    const vrmAnimation = () => {
      if (!this.vrm || !this.vrm.humanoid) return;
      
      const time = (performance.now() - startTime) * 0.001;
      
      try {
        // Use VRM humanoid bone nodes for proper animation
        const humanoid = this.vrm.humanoid;
        
        // Natural breathing animation
        const spine = humanoid.getNormalizedBoneNode('spine');
        const chest = humanoid.getNormalizedBoneNode('chest');
        if (spine || chest) {
          const breathingTarget = chest || spine;
          const breathingIntensity = 0.015 + Math.sin(time * 0.8) * 0.005;
          breathingTarget.scale.y = 1 + Math.sin(time * 0.6) * breathingIntensity;
          breathingTarget.scale.z = 1 + Math.sin(time * 0.6) * breathingIntensity * 0.5;
        }
        
        // Natural head movement with personality
        const head = humanoid.getNormalizedBoneNode('head');
        const neck = humanoid.getNormalizedBoneNode('neck');
        if (head) {
          // Subtle head rotation for natural look
          head.rotation.y = Math.sin(time * 0.3) * 0.08 + Math.sin(time * 0.13) * 0.02;
          head.rotation.x = Math.sin(time * 0.25) * 0.03 + Math.sin(time * 0.17) * 0.01;
          head.rotation.z = Math.sin(time * 0.2) * 0.02;
        }
        if (neck) {
          // Neck supports head movement
          neck.rotation.y = Math.sin(time * 0.28) * 0.02;
          neck.rotation.x = Math.sin(time * 0.22) * 0.015;
        }
        
        // Subtle shoulder animation
        const leftShoulder = humanoid.getNormalizedBoneNode('leftShoulder');
        const rightShoulder = humanoid.getNormalizedBoneNode('rightShoulder');
        if (leftShoulder) {
          leftShoulder.rotation.z = Math.sin(time * 0.4 + Math.PI) * 0.02;
        }
        if (rightShoulder) {
          rightShoulder.rotation.z = Math.sin(time * 0.4) * 0.02;
        }
        
        // Natural arm sway
        const leftUpperArm = humanoid.getNormalizedBoneNode('leftUpperArm');
        const rightUpperArm = humanoid.getNormalizedBoneNode('rightUpperArm');
        if (leftUpperArm) {
          leftUpperArm.rotation.x = Math.sin(time * 0.35) * 0.05;
          leftUpperArm.rotation.z = Math.sin(time * 0.3) * 0.03;
        }
        if (rightUpperArm) {
          rightUpperArm.rotation.x = Math.sin(time * 0.35 + Math.PI * 0.7) * 0.05;
          rightUpperArm.rotation.z = Math.sin(time * 0.3 + Math.PI * 0.7) * 0.03;
        }
        
        // Hip sway for natural posture
        const hips = humanoid.getNormalizedBoneNode('hips');
        if (hips) {
          hips.rotation.y = Math.sin(time * 0.2) * 0.01;
          hips.rotation.z = Math.sin(time * 0.18) * 0.005;
        }
        
        // Facial expressions using VRM expression system
        if (this.vrm.expressionManager) {
          // Natural blinking
          const blinkCycle = time * 3.2;
          const blinkValue = Math.max(0, Math.sin(blinkCycle) * 0.1 + Math.sin(blinkCycle * 4.7) * 0.05);
          this.vrm.expressionManager.setValue('blink', Math.max(0, Math.min(1, blinkValue)));
          
          // Subtle smile expression
          const smileValue = 0.1 + Math.sin(time * 0.1) * 0.05;
          this.vrm.expressionManager.setValue('happy', Math.max(0, Math.min(1, smileValue)));
        }
        
        // Use binder to apply additional BVH-driven animations if available
        if (this.binder && this.binder.update) {
          this.binder.update(time);
        }
        
      } catch (error) {
        // Log animation errors but don't break the loop
        if (animationFrame % 300 === 0) { // Log every 10 seconds at 30fps
          console.warn('VRM animation warning (non-critical):', error.message);
        }
      }
      
      animationFrame++;
      if (animationFrame % 90 === 0) { // Log every 3 seconds at 30fps
        console.log('🎭 VRM humanoid animation running...');
        
        // Log VRM state for debugging
        if (this.binder && this.binder.getStats) {
          const stats = this.binder.getStats();
          console.log(`VRM Binder Stats:`, stats);
        }
      }
      
      requestAnimationFrame(vrmAnimation);
    };
    
    vrmAnimation();
    console.log('✅ VRM humanoid animations started with natural movement patterns');
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
   * Parse BVH data with real structure understanding
   */
  parseRealBVHData(bvhText) {
    const lines = bvhText.split('\n');
    const bones = [];
    const hierarchy = {};
    const channels = [];
    let frameTime = 1/30; // Default 30 FPS
    let numFrames = 0;
    
    // Parse hierarchy section
    let inHierarchy = false;
    let inMotion = false;
    let currentBone = null;
    let boneStack = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      if (line === 'HIERARCHY') {
        inHierarchy = true;
        continue;
      }
      
      if (line === 'MOTION') {
        inHierarchy = false;
        inMotion = true;
        continue;
      }
      
      if (inHierarchy) {
        // Parse bone hierarchy
        if (line.startsWith('ROOT') || line.startsWith('JOINT')) {
          const boneName = line.split(/\s+/)[1];
          bones.push(boneName);
          currentBone = boneName;
          
          hierarchy[boneName] = {
            name: boneName,
            parent: boneStack.length > 0 ? boneStack[boneStack.length - 1] : null,
            children: [],
            offset: [0, 0, 0],
            channels: []
          };
          
          if (boneStack.length > 0) {
            hierarchy[boneStack[boneStack.length - 1]].children.push(boneName);
          }
        } else if (line.startsWith('OFFSET')) {
          const values = line.split(/\s+/).slice(1).map(parseFloat);
          if (currentBone && values.length >= 3) {
            hierarchy[currentBone].offset = values;
          }
        } else if (line.startsWith('CHANNELS')) {
          const parts = line.split(/\s+/);
          const numChannels = parseInt(parts[1]);
          const channelNames = parts.slice(2, 2 + numChannels);
          
          if (currentBone) {
            hierarchy[currentBone].channels = channelNames;
            channels.push(...channelNames.map(name => ({ bone: currentBone, channel: name })));
          }
        } else if (line === '{') {
          if (currentBone) {
            boneStack.push(currentBone);
          }
        } else if (line === '}') {
          if (boneStack.length > 0) {
            boneStack.pop();
          }
          if (line.startsWith('End Site')) {
            // Handle end sites if needed
          }
        }
      } else if (inMotion) {
        // Parse motion data
        if (line.startsWith('Frames:')) {
          numFrames = parseInt(line.split(':')[1].trim());
        } else if (line.startsWith('Frame Time:')) {
          frameTime = parseFloat(line.split(':')[1].trim());
        }
        // Frame data parsing would go here for full implementation
      }
    }
    
    return {
      bones: bones,
      hierarchy: hierarchy,
      channels: channels,
      frameRate: 1.0 / frameTime,
      numFrames: numFrames,
      frameTime: frameTime,
      // This would contain actual frame data in full implementation
      frames: [], 
      skeleton: { 
        bones: bones.map(name => ({
          name: name,
          parent: hierarchy[name]?.parent,
          offset: hierarchy[name]?.offset || [0, 0, 0],
          channels: hierarchy[name]?.channels || []
        }))
      }
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
   * This method has been removed - we only use real VRM avatars now
   * No more geometric fallbacks (pink sphere + blue rectangle)
   */
  createSimpleAvatar() {
    console.error('❌ createSimpleAvatar() called - this should not happen!');
    console.error('The system should load real VRM models, not geometric fallbacks');
    throw new Error('VRM loading system failed - geometric fallbacks disabled');
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