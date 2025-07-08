/**
 * Classroom Environment - Creates and manages a 3D classroom scene
 * Adapted from the chat folder implementation for the modern viewer
 */
class ClassroomEnvironment {
    constructor(scene) {
        console.log('🏫 ClassroomEnvironment constructor called with scene:', scene);
        console.log('🏫 Scene type:', typeof scene);
        console.log('🏫 Scene is valid THREE.Scene:', scene instanceof THREE.Scene);
        
        this.scene = scene;
        this.environment = null;
        this.objects = [];
        this.lights = [];
        this.isLoaded = false;
        console.log('🏫 ClassroomEnvironment initialized');
    }
    
    // Load the classroom environment
    async loadClassroom(options = {}) {
        console.log('Loading classroom environment...');
        
        try {
            // First try to load the classroom GLB file from assets
            const classroomPath = './assets/scenes/classroom.glb';
            const environment = await this.loadClassroomGLB(classroomPath);
            
            if (environment) {
                this.addEnvironmentToScene(environment);
                this.isLoaded = true;
                console.log('Classroom GLB loaded successfully');
                return;
            }
        } catch (error) {
            console.warn('Failed to load classroom GLB, creating procedural classroom:', error);
        }
        
        // Fallback: create procedural classroom
        this.createProceduralClassroom(options);
        this.isLoaded = true;
        console.log('Procedural classroom created');
    }
    
    // Load classroom from GLB file
    async loadClassroomGLB(path) {
        const loader = new window.THREELoaders.GLTFLoader();
        
        return new Promise((resolve, reject) => {
            loader.load(
                path,
                (gltf) => {
                    console.log('Classroom GLB loaded');
                    
                    // Process the loaded scene
                    const environment = {
                        type: 'classroom_glb',
                        scene: gltf.scene,
                        animations: gltf.animations || []
                    };
                    
                    // Set up shadows and materials
                    this.setupGLBEnvironment(gltf.scene);
                    
                    resolve(environment);
                },
                (progress) => {
                    if (progress.total > 0) {
                        console.log(`Classroom loading progress: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
                    } else {
                        console.log(`Classroom loading progress: ${progress.loaded} bytes loaded`);
                    }
                },
                (error) => {
                    console.error('Failed to load classroom GLB:', error);
                    reject(error);
                }
            );
        });
    }
    
    setupGLBEnvironment(scene) {
        // Traverse and set up materials and shadows
        scene.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
                
                // Optimize materials
                if (child.material) {
                    if (child.material.map) {
                        child.material.map.flipY = false;
                    }
                    child.material.needsUpdate = true;
                }
            }
        });
        
        // Scale the classroom if needed
        scene.scale.set(1, 1, 1);
        scene.position.set(0, 0, 0);
    }
    
    // Create procedural classroom when GLB is not available
    createProceduralClassroom(options = {}) {
        console.log('Creating procedural classroom...');
        
        this.environment = {
            type: 'classroom_procedural',
            objects: [],
            lights: []
        };
        
        // Create floor
        this.createFloor();
        
        // Create walls
        this.createWalls();
        
        // Create whiteboard
        this.createWhiteboard();
        
        // Create desks and chairs
        this.createFurniture(options);
        
        // Create ceiling
        this.createCeiling();
        
        // Create lighting
        this.createLighting();
        
        // Add all objects to scene
        this.addEnvironmentToScene(this.environment);
    }
    
    createFloor() {
        const floorGeometry = new window.THREE.PlaneGeometry(25, 20);
        const floorMaterial = new window.THREE.MeshLambertMaterial({ 
            color: 0x8B7355, // Wooden floor color
            transparent: false
        });
        
        const floor = new window.THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        floor.position.set(0, 0, 0);
        
        this.environment.objects.push(floor);
        console.log('Floor created');
    }
    
    createWalls() {
        const wallMaterial = new window.THREE.MeshLambertMaterial({ 
            color: 0xF5F5DC // Beige wall color
        });
        
        // Back wall
        const backWallGeometry = new window.THREE.PlaneGeometry(25, 10);
        const backWall = new window.THREE.Mesh(backWallGeometry, wallMaterial);
        backWall.position.set(0, 5, -10);
        backWall.receiveShadow = true;
        this.environment.objects.push(backWall);
        
        // Left wall
        const leftWallGeometry = new window.THREE.PlaneGeometry(20, 10);
        const leftWall = new window.THREE.Mesh(leftWallGeometry, wallMaterial);
        leftWall.rotation.y = Math.PI / 2;
        leftWall.position.set(-12.5, 5, 0);
        leftWall.receiveShadow = true;
        this.environment.objects.push(leftWall);
        
        // Right wall
        const rightWall = new window.THREE.Mesh(leftWallGeometry, wallMaterial);
        rightWall.rotation.y = -Math.PI / 2;
        rightWall.position.set(12.5, 5, 0);
        rightWall.receiveShadow = true;
        this.environment.objects.push(rightWall);
        
        console.log('Walls created');
    }
    
    createCeiling() {
        const ceilingGeometry = new window.THREE.PlaneGeometry(25, 20);
        const ceilingMaterial = new window.THREE.MeshLambertMaterial({ 
            color: 0xFFFFFF
        });
        
        const ceiling = new window.THREE.Mesh(ceilingGeometry, ceilingMaterial);
        ceiling.rotation.x = Math.PI / 2;
        ceiling.position.set(0, 10, 0);
        ceiling.receiveShadow = true;
        
        this.environment.objects.push(ceiling);
        console.log('Ceiling created');
    }
    
    createWhiteboard() {
        // Whiteboard frame
        const frameGeometry = new window.THREE.BoxGeometry(8, 4, 0.1);
        const frameMaterial = new window.THREE.MeshLambertMaterial({ color: 0x333333 });
        const frame = new window.THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.set(0, 4, -9.9);
        frame.castShadow = true;
        this.environment.objects.push(frame);
        
        // Whiteboard surface
        const boardGeometry = new window.THREE.PlaneGeometry(7.8, 3.8);
        const boardMaterial = new window.THREE.MeshLambertMaterial({ color: 0xFFFFFF });
        const board = new window.THREE.Mesh(boardGeometry, boardMaterial);
        board.position.set(0, 4, -9.8);
        this.environment.objects.push(board);
        
        console.log('Whiteboard created');
    }
    
    createFurniture(options) {
        const deskCount = options.deskCount || 15;
        const rows = options.rows || 3;
        const desksPerRow = Math.ceil(deskCount / rows);
        
        for (let row = 0; row < rows; row++) {
            for (let desk = 0; desk < desksPerRow && (row * desksPerRow + desk) < deskCount; desk++) {
                this.createDeskAndChair({
                    x: (desk - desksPerRow / 2) * 4 + 2,
                    z: row * 4 - 4,
                    y: 0
                });
            }
        }
        
        // Teacher's desk
        this.createTeacherDesk();
        
        console.log(`Created ${deskCount} student desks and teacher desk`);
    }
    
    createDeskAndChair(position) {
        // Desk
        const deskGeometry = new window.THREE.BoxGeometry(1.2, 0.8, 0.8);
        const deskMaterial = new window.THREE.MeshLambertMaterial({ color: 0x8B4513 });
        const desk = new window.THREE.Mesh(deskGeometry, deskMaterial);
        desk.position.set(position.x, 0.4, position.z);
        desk.castShadow = true;
        desk.receiveShadow = true;
        this.environment.objects.push(desk);
        
        // Chair
        const chairSeatGeometry = new window.THREE.BoxGeometry(0.5, 0.1, 0.5);
        const chairMaterial = new window.THREE.MeshLambertMaterial({ color: 0x654321 });
        const chairSeat = new window.THREE.Mesh(chairSeatGeometry, chairMaterial);
        chairSeat.position.set(position.x, 0.45, position.z + 0.8);
        chairSeat.castShadow = true;
        chairSeat.receiveShadow = true;
        this.environment.objects.push(chairSeat);
        
        // Chair back
        const chairBackGeometry = new window.THREE.BoxGeometry(0.5, 0.8, 0.1);
        const chairBack = new window.THREE.Mesh(chairBackGeometry, chairMaterial);
        chairBack.position.set(position.x, 0.85, position.z + 1.05);
        chairBack.castShadow = true;
        chairBack.receiveShadow = true;
        this.environment.objects.push(chairBack);
    }
    
    createTeacherDesk() {
        const deskGeometry = new window.THREE.BoxGeometry(2, 1, 1.2);
        const deskMaterial = new window.THREE.MeshLambertMaterial({ color: 0x654321 });
        const desk = new window.THREE.Mesh(deskGeometry, deskMaterial);
        desk.position.set(-6, 0.5, -7);
        desk.castShadow = true;
        desk.receiveShadow = true;
        this.environment.objects.push(desk);
    }
    
    createLighting() {
        // Ambient light
        const ambientLight = new window.THREE.AmbientLight(0x404040, 0.6);
        this.environment.lights.push(ambientLight);
        
        // Single Directional Light
        const directionalLight = new window.THREE.DirectionalLight(0xFFFFFF, 0.8);
        directionalLight.position.set(5, 10, 7);
        directionalLight.target.position.set(0, 0, 0);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 1024;
        directionalLight.shadow.mapSize.height = 1024;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 20;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        directionalLight.add(directionalLight.target); // Add target as a child of the light
        this.environment.lights.push(directionalLight);
        
        console.log('Classroom lighting created');
    }
    
    addEnvironmentToScene(environment) {
        console.log('🏫 Adding environment to scene...');
        console.log('🏫 this.scene:', this.scene);
        console.log('🏫 this.scene type:', typeof this.scene);
        console.log('🏫 environment:', environment);
        
        if (environment.scene) {
            // GLB-based environment
            console.log('🏫 Adding GLB environment scene');
            this.scene.add(environment.scene);
        } else {
            // Procedural environment
            console.log('🏫 Adding procedural environment objects:', environment.objects.length);
            console.log('🏫 Adding procedural environment lights:', environment.lights.length);
            environment.objects.forEach(obj => {
                console.log('🏫 Adding object:', obj);
                this.scene.add(obj);
            });
            environment.lights.forEach(light => {
                console.log('🏫 Adding light:', light);
                this.scene.add(light);
            });
        }
        
        console.log('Environment added to scene');
    }
    
    removeEnvironmentFromScene() {
        if (!this.environment) return;
        
        if (this.environment.scene) {
            this.scene.remove(this.environment.scene);
        } else {
            this.environment.objects.forEach(obj => this.scene.remove(obj));
            this.environment.lights.forEach(light => this.scene.remove(light));
        }
        
        console.log('Environment removed from scene');
    }
    
    // Get optimal camera position for classroom view
    getOptimalCameraPosition() {
        return {
            position: new window.THREE.Vector3(8, 3, 8),
            target: new window.THREE.Vector3(0, 1, 0)
        };
    }
    
    // Animation helpers
    getWalkingArea() {
        return {
            minX: -10,
            maxX: 10,
            minZ: -8,
            maxZ: 6,
            floorY: 0
        };
    }
    
    dispose() {
        if (this.environment) {
            this.removeEnvironmentFromScene();
            this.environment = null;
        }
        this.objects = [];
        this.lights = [];
        this.isLoaded = false;
        console.log('Classroom environment disposed');
    }
    
    // Set visibility of all classroom objects
    setVisible(visible = true) {
        console.log(`🏫 Setting classroom visibility to: ${visible}`);
        
        if (!this.environment) {
            console.warn('⚠️ No classroom environment loaded to set visibility');
            return false;
        }
        
        // Set visibility for all objects
        if (this.environment.objects) {
            this.environment.objects.forEach(obj => {
                if (obj && obj.visible !== undefined) {
                    obj.visible = visible;
                }
            });
            console.log(`✅ Set visibility for ${this.environment.objects.length} classroom objects`);
        }
        
        // Set visibility for all lights  
        if (this.environment.lights) {
            this.environment.lights.forEach(light => {
                if (light && light.visible !== undefined) {
                    light.visible = visible;
                }
            });
            console.log(`✅ Set visibility for ${this.environment.lights.length} classroom lights`);
        }
        
        return true;
    }
    
    // Get current visibility status
    isVisible() {
        if (!this.environment || !this.environment.objects || this.environment.objects.length === 0) {
            return false;
        }
        
        // Check if at least one object is visible
        return this.environment.objects.some(obj => obj && obj.visible);
    }
    
    // Toggle classroom visibility
    toggleVisibility() {
        const currentlyVisible = this.isVisible();
        this.setVisible(!currentlyVisible);
        console.log(`🏫 Classroom visibility toggled: ${!currentlyVisible ? 'VISIBLE' : 'HIDDEN'}`);
        return !currentlyVisible;
    }
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ClassroomEnvironment };
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.ClassroomEnvironment = ClassroomEnvironment;
}
