/**
 * VRM Mesh Skeleton - Standalone VRM Character Loading and Rendering
 * Extracted from rsmt_showcase.html for standalone use
 * Applies Ichika or other VRM meshes to skeleton systems
 */

class VRMMeshSkeleton {
    constructor(scene, camera, renderer) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.vrmModel = null;
        this.characterPath = null;
        this.debugMode = true;
        this.isLoaded = false;
        
        // Character mesh cache
        this.loadedVRMs = new Map();
        
        // Default character options
        this.characters = {
            'ichika.vrm': 'Ichika (Student)',
            'kaede.vrm': 'Kaede (School Girl)', 
            'buny.vrm': 'Buny (Teacher)'
        };
        
        console.log('🎭 VRMMeshSkeleton initialized');
    }

    /**
     * Setup professional lighting optimized for anime VRM characters
     */
    setupProfessionalLighting() {
        console.log('🌟 Setting up professional anime character lighting...');
        
        // Clear existing lights
        const existingLights = [];
        this.scene.traverse((child) => {
            if (child.isLight) {
                existingLights.push(child);
            }
        });
        existingLights.forEach(light => this.scene.remove(light));
        
        // Beautiful gradient background for anime characters
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, '#87CEEB'); // Sky blue
        gradient.addColorStop(0.5, '#98D8E8'); // Light blue
        gradient.addColorStop(1, '#F0F8FF'); // Alice blue
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const bgTexture = new THREE.CanvasTexture(canvas);
        this.scene.background = bgTexture;
        
        // High-intensity ambient light for soft overall illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        this.scene.add(ambientLight);
        
        // Main key light (strong directional light from front-top)
        this.keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
        this.keyLight.position.set(2, 3, 2);
        this.keyLight.castShadow = false; // Avoid VRM shader conflicts
        this.scene.add(this.keyLight);
        
        // Fill light from left (softer, warmer)
        this.fillLight = new THREE.DirectionalLight(0xffeeaa, 1.0);
        this.fillLight.position.set(-2, 2, 1);
        this.scene.add(this.fillLight);
        
        // Rim light for character outline
        this.rimLight = new THREE.DirectionalLight(0xaaccff, 0.8);
        this.rimLight.position.set(0, 2, -3);
        this.scene.add(this.rimLight);
        
        // Character face lighting (important for anime style)
        this.faceLight = new THREE.SpotLight(0xffffff, 1.2);
        this.faceLight.position.set(0, 2, 1.5);
        this.faceLight.angle = Math.PI / 6;
        this.faceLight.penumbra = 0.2;
        this.faceLight.decay = 2;
        this.faceLight.distance = 5;
        this.scene.add(this.faceLight);
        
        // Soft environmental lights
        const envLight1 = new THREE.PointLight(0xffffff, 0.6, 8);
        envLight1.position.set(2, 2, -1);
        this.scene.add(envLight1);
        
        const envLight2 = new THREE.PointLight(0xffffcc, 0.6, 8);
        envLight2.position.set(-2, 2, -1);
        this.scene.add(envLight2);
        
        // Configure renderer for VRM compatibility
        this.renderer.shadowMap.enabled = false; // Disabled for VRM compatibility 
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0; // Reduced for better anime lighting
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        
        console.log('✅ Professional anime lighting setup complete');
    }

    /**
     * Load VRM character from path
     */
    async loadVRMCharacter(characterPath) {
        try {
            console.log('🎭 Loading VRM character:', characterPath);
            
            // Check cache first
            if (this.loadedVRMs.has(characterPath)) {
                console.log('📦 Using cached VRM:', characterPath);
                const cachedData = this.loadedVRMs.get(characterPath);
                this.applyVRMToScene(cachedData.vrm);
                return cachedData.vrm;
            }
            
            // Ensure THREE.js modules are available
            if (!window.THREE) {
                throw new Error('THREE.js not available');
            }
            
            // Import VRM loader modules
            const { GLTFLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.177.0/examples/jsm/loaders/GLTFLoader.js');
            const { VRMLoaderPlugin } = await import('https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.6/lib/three-vrm.module.js');
            
            const loader = new GLTFLoader();
            loader.register((parser) => new VRMLoaderPlugin(parser));
            
            // Load VRM model with progress tracking
            const gltf = await new Promise((resolve, reject) => {
                loader.load(
                    characterPath,
                    (gltf) => {
                        console.log('✅ VRM file loaded successfully');
                        resolve(gltf);
                    },
                    (progress) => {
                        const percent = (progress.loaded / progress.total * 100).toFixed(1);
                        if (this.debugMode && Math.random() < 0.2) {
                            console.log('📈 VRM loading progress:', percent + '%');
                        }
                    },
                    (error) => {
                        console.error('❌ Failed to load VRM file:', error);
                        reject(error);
                    }
                );
            });
            
            // Extract VRM data
            const vrm = gltf.userData.vrm;
            if (!vrm) {
                throw new Error('No VRM data found in file');
            }
            
            // Wait for VRM to be ready
            if (vrm.ready) {
                await vrm.ready;
            }
            
            // Configure VRM for optimal rendering
            this.configureVRMForRendering(vrm);
            
            // Cache the VRM
            this.loadedVRMs.set(characterPath, { gltf, vrm });
            
            // Apply to scene
            this.applyVRMToScene(vrm);
            
            this.characterPath = characterPath;
            this.isLoaded = true;
            
            console.log('🎉 VRM character loaded and applied!');
            return vrm;
            
        } catch (error) {
            console.error('❌ Failed to load VRM character:', error);
            throw error;
        }
    }
    
    /**
     * Configure VRM for optimal rendering and compatibility
     */
    configureVRMForRendering(vrm) {
        console.log('⚙️ Configuring VRM for optimal rendering...');
        
        if (!vrm.scene) {
            console.warn('⚠️ VRM has no scene object');
            return;
        }
        
        // Fix all materials to prevent shader errors
        this.fixAllVRMMaterials(vrm.scene);
        
        // Configure VRM scene properties
        vrm.scene.traverse((child) => {
            if (child.isMesh) {
                // Ensure mesh visibility and prevent culling issues
                child.visible = true;
                child.frustumCulled = false;
                
                // Disable shadows for VRM compatibility
                child.castShadow = false;
                child.receiveShadow = false;
                
                // Configure material properties
                if (child.material) {
                    const materials = Array.isArray(child.material) ? child.material : [child.material];
                    materials.forEach(material => {
                        if (material.isMeshStandardMaterial || material.isMeshPhongMaterial) {
                            // Optimize for anime-style rendering
                            material.roughness = 0.8; // Slightly rough for anime look
                            material.metalness = 0.0; // Non-metallic for anime characters
                        }
                    });
                }
            }
        });
        
        // Set up idle animations
        this.setupIdleAnimations(vrm);
        
        console.log('✅ VRM configured for rendering');
    }
    
    /**
     * Fix VRM materials to prevent WebGL shader errors
     */
    fixAllVRMMaterials(root) {
        console.log('🔧 Fixing VRM materials for compatibility...');
        
        let fixedCount = 0;
        root.traverse((child) => {
            if (child.isMesh && child.material) {
                const materials = Array.isArray(child.material) ? child.material : [child.material];
                let materialFixed = false;
                
                for (let i = 0; i < materials.length; i++) {
                    const mat = materials[i];
                    
                    // Replace problematic materials with safe MeshBasicMaterial
                    if (mat && !mat.isMeshBasicMaterial && this.needsMaterialFix(mat)) {
                        const safeMaterial = new THREE.MeshBasicMaterial({
                            color: mat.color || 0xffffff,
                            map: mat.map || null,
                            skinning: !!mat.skinning,
                            transparent: !!mat.transparent,
                            opacity: (typeof mat.opacity === 'number') ? mat.opacity : 1.0,
                            side: mat.side || THREE.FrontSide
                        });
                        
                        if (mat.name) safeMaterial.name = mat.name + '_safe';
                        materials[i] = safeMaterial;
                        materialFixed = true;
                    }
                }
                
                if (materialFixed) {
                    child.material = Array.isArray(child.material) ? materials : materials[0];
                    fixedCount++;
                }
            }
        });
        
        console.log(`✅ Fixed ${fixedCount} VRM materials`);
    }
    
    /**
     * Check if material needs fixing to prevent shader errors
     */
    needsMaterialFix(material) {
        // Fix materials that commonly cause shader errors in VRM
        return material.isMeshStandardMaterial || 
               material.isMeshPhongMaterial ||
               material.isMeshLambertMaterial ||
               (material.uniforms && Object.keys(material.uniforms).length > 10);
    }
    
    /**
     * Setup idle animations for VRM character
     */
    setupIdleAnimations(vrm) {
        console.log('😊 Setting up idle animations...');
        
        if (!vrm.humanoid) {
            console.warn('⚠️ VRM has no humanoid data for animations');
            return;
        }
        
        // Enable auto-update for VRM systems
        vrm.humanoid.autoUpdateHumanBones = true;
        
        if (vrm.lookAt) {
            vrm.lookAt.autoUpdate = true;
        }
        
        // Simple breathing animation
        this.setupBreathingAnimation(vrm);
        
        console.log('✅ Idle animations configured');
    }
    
    /**
     * Setup breathing animation for character
     */
    setupBreathingAnimation(vrm) {
        if (!vrm.humanoid) return;
        
        const chestBone = vrm.humanoid.getNormalizedBoneNode('chest');
        if (chestBone) {
            this.breathingAnimation = {
                bone: chestBone,
                originalRotation: chestBone.rotation.clone(),
                amplitude: 0.01,
                frequency: 0.3,
                time: 0
            };
        }
    }
    
    /**
     * Apply VRM to scene
     */
    applyVRMToScene(vrm) {
        console.log('🎬 Applying VRM to scene...');
        
        // Remove existing VRM if present
        if (this.vrmModel && this.vrmModel.scene) {
            this.scene.remove(this.vrmModel.scene);
        }
        
        // Add new VRM to scene
        this.scene.add(vrm.scene);
        this.vrmModel = vrm;
        
        // Position character
        vrm.scene.position.set(0, 0, 0);
        vrm.scene.rotation.set(0, 0, 0);
        vrm.scene.scale.set(1, 1, 1);
        
        console.log('✅ VRM applied to scene');
    }
    
    /**
     * Load Ichika character specifically
     */
    async loadIchika() {
        return this.loadVRMCharacter('./assets/avatars/ichika.vrm');
    }
    
    /**
     * Load Kaede character
     */
    async loadKaede() {
        return this.loadVRMCharacter('./assets/avatars/kaede.vrm');
    }
    
    /**
     * Load Buny character
     */
    async loadBuny() {
        return this.loadVRMCharacter('./assets/avatars/buny.vrm');
    }
    
    /**
     * Update VRM animations (call in render loop)
     */
    update(deltaTime = 0.016) {
        if (!this.vrmModel) return;
        
        // Update VRM internal systems
        if (this.vrmModel.update) {
            this.vrmModel.update(deltaTime);
        }
        
        // Update breathing animation
        if (this.breathingAnimation) {
            this.breathingAnimation.time += deltaTime;
            const breathOffset = Math.sin(this.breathingAnimation.time * this.breathingAnimation.frequency) * this.breathingAnimation.amplitude;
            this.breathingAnimation.bone.rotation.x = this.breathingAnimation.originalRotation.x + breathOffset;
        }
    }
    
    /**
     * Focus camera on character face
     */
    focusOnFace() {
        if (!this.vrmModel || !this.vrmModel.humanoid) {
            console.warn('⚠️ No VRM character loaded for face focus');
            return;
        }
        
        const headBone = this.vrmModel.humanoid.getNormalizedBoneNode('head');
        if (headBone) {
            const headPosition = new THREE.Vector3();
            headBone.getWorldPosition(headPosition);
            
            // Position camera to focus on face
            this.camera.position.set(
                headPosition.x + 0.5,
                headPosition.y + 0.2,
                headPosition.z + 1.0
            );
            this.camera.lookAt(headPosition);
            this.camera.updateProjectionMatrix();
            
            console.log('👤 Camera focused on character face');
        }
    }
    
    /**
     * Set full body view
     */
    fullBodyView() {
        this.camera.position.set(0, 1.6, 2.5);
        this.camera.lookAt(0, 1.0, 0);
        this.camera.fov = 40;
        this.camera.updateProjectionMatrix();
        console.log('📷 Full body view set');
    }
    
    /**
     * Get current loaded character info
     */
    getCharacterInfo() {
        if (!this.isLoaded || !this.characterPath) {
            return null;
        }
        
        const filename = this.characterPath.split('/').pop();
        return {
            path: this.characterPath,
            filename: filename,
            name: this.characters[filename] || filename.replace('.vrm', ''),
            vrm: this.vrmModel
        };
    }
    
    /**
     * Remove current character from scene
     */
    removeCharacter() {
        if (this.vrmModel && this.vrmModel.scene) {
            this.scene.remove(this.vrmModel.scene);
            this.vrmModel = null;
            this.characterPath = null;
            this.isLoaded = false;
            console.log('🗑️ Character removed from scene');
        }
    }
    
    /**
     * Enhanced lighting specifically for VRM characters
     */
    enhanceVRMLighting() {
        console.log('✨ Applying enhanced VRM lighting...');
        
        // Boost key light for better character visibility
        if (this.keyLight) {
            this.keyLight.intensity = 2.0;
        }
        
        // Enhance face light for anime style
        if (this.faceLight) {
            this.faceLight.intensity = 1.5;
            this.faceLight.distance = 3;
        }
        
        // Adjust fill light for better balance
        if (this.fillLight) {
            this.fillLight.intensity = 1.2;
        }
        
        // Boost tone mapping exposure for anime brightness
        this.renderer.toneMappingExposure = 1.2;
        
        console.log('✨ Enhanced VRM lighting applied!');
    }
    
    /**
     * Create a simple test scene with ground and axes
     */
    createTestScene() {
        console.log('🎬 Creating test scene for VRM character...');
        
        // Ground plane (circular for anime aesthetic)
        const groundGeometry = new THREE.CircleGeometry(5, 32);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xf0f0f0,
            opacity: 0.8,
            transparent: true
        });
        const groundPlane = new THREE.Mesh(groundGeometry, groundMaterial);
        groundPlane.rotation.x = -Math.PI / 2;
        groundPlane.position.y = -0.01;
        this.scene.add(groundPlane);
        
        // Subtle circular grid
        const gridHelper = new THREE.PolarGridHelper(3, 8, 8, 64, 0x888888, 0x444444);
        gridHelper.position.y = 0.01;
        gridHelper.material.opacity = 0.2;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
        
        // Small axes helper
        const axesHelper = new THREE.AxesHelper(0.5);
        this.scene.add(axesHelper);
        
        console.log('✅ Test scene created');
    }
}

// Global export for browser use
if (typeof window !== 'undefined') {
    window.VRMMeshSkeleton = VRMMeshSkeleton;
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VRMMeshSkeleton;
}
