/**
 * ClassroomGLBLoader - Loads and manages classroom GLB file with collision detection
 * Reuses RSMT showcase patterns for loading and rendering 3D environments
 */

class ClassroomGLBLoader {
    constructor(scene) {
        this.scene = scene;
        this.classroomModel = null;
        this.collisionMeshes = [];
        this.floorBounds = null;
        this.wallBounds = [];
        this.initialized = false;
        this.loader = null;
        
        // Default classroom GLB path (can be overridden)
        this.classroomPath = './models/classroom.glb';
        
        console.log('🏫 ClassroomGLBLoader initialized');
    }

    async initialize(classroomPath = null) {
        console.log('🏗️ Initializing ClassroomGLBLoader...');
        
        try {
            // Initialize GLTFLoader following RSMT pattern
            if (!window.THREE) {
                throw new Error('THREE.js not available');
            }
            
            if (!window.GLTFLoader) {
                throw new Error('GLTFLoader not available');
            }
            
            this.loader = new window.GLTFLoader();
            
            this.initialized = true;
            console.log('✅ ClassroomGLBLoader ready for GLB loading');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize ClassroomGLBLoader:', error);
            return false;
        }
    }

    async loadClassroomGLB(classroomPath = null) {
        if (classroomPath) {
            this.classroomPath = classroomPath;
        }
        
        return new Promise((resolve, reject) => {
            console.log(`📂 Loading GLB from: ${this.classroomPath}`);
            
            this.loader.load(
                this.classroomPath,
                (gltf) => {
                    console.log('✅ GLB loaded successfully');
                    this.processClassroomModel(gltf);
                    this.setupCollisionDetection();
                    this.optimizeClassroomMaterials();
                    resolve(true);
                },
                (progress) => {
                    const percent = (progress.loaded / progress.total * 100).toFixed(1);
                    console.log(`📊 Loading progress: ${percent}%`);
                },
                (error) => {
                    console.error('❌ GLB loading failed:', error);
                    resolve(false);
                }
            );
        });
    }

    processClassroomModel(gltf) {
        console.log('🔧 Processing classroom model...');
        
        this.classroomModel = gltf.scene;
        this.classroomModel.name = 'ClassroomGLB';
        
        // Scale and position the classroom if needed
        this.classroomModel.scale.setScalar(1);
        this.classroomModel.position.set(0, 0, 0);
        this.classroomModel.rotation.set(0, 0, 0);
        
        // Add to scene
        this.scene.add(this.classroomModel);
        
        // Analyze the model structure for collision detection
        this.analyzeClassroomStructure();
        
        console.log('✅ Classroom model processed and added to scene');
    }

    analyzeClassroomStructure() {
        console.log('🔍 Analyzing classroom structure for collision detection...');
        
        const meshCount = { floors: 0, walls: 0, furniture: 0, other: 0 };
        
        this.classroomModel.traverse((child) => {
            if (child.isMesh) {
                const name = (child.name || '').toLowerCase();
                const parentName = (child.parent?.name || '').toLowerCase();
                
                // Categorize meshes based on naming conventions
                let meshType = 'other';
                let isCollision = false;
                
                if (name.includes('floor') || parentName.includes('floor')) {
                    meshType = 'floor';
                    isCollision = true;
                    this.setupFloorBounds(child);
                    meshCount.floors++;
                } else if (name.includes('wall') || parentName.includes('wall')) {
                    meshType = 'wall';
                    isCollision = true;
                    meshCount.walls++;
                } else if (name.includes('desk') || name.includes('chair') || 
                          name.includes('table') || name.includes('furniture') ||
                          parentName.includes('desk') || parentName.includes('chair')) {
                    meshType = 'furniture';
                    isCollision = true;
                    meshCount.furniture++;
                } else if (name.includes('door') || name.includes('blackboard') || 
                          name.includes('board') || parentName.includes('blackboard')) {
                    meshType = 'furniture';
                    isCollision = true;
                    meshCount.other++;
                }
                
                // Set collision properties
                child.userData.meshType = meshType;
                child.userData.collision = isCollision;
                
                if (isCollision) {
                    this.collisionMeshes.push(child);
                }
                
                // Enable shadows for better visual quality
                child.castShadow = true;
                child.receiveShadow = true;
                
                console.log(`📦 Mesh: ${child.name || 'unnamed'} -> ${meshType} (collision: ${isCollision})`);
            }
        });
        
        console.log('📊 Classroom structure analysis complete:');
        console.log(`  Floors: ${meshCount.floors}`);
        console.log(`  Walls: ${meshCount.walls}`);
        console.log(`  Furniture: ${meshCount.furniture}`);
        console.log(`  Other collision objects: ${meshCount.other}`);
        console.log(`  Total collision meshes: ${this.collisionMeshes.length}`);
    }

    setupFloorBounds(floorMesh) {
        // Calculate floor bounds from the actual floor mesh
        const box = new window.THREE.Box3().setFromObject(floorMesh);
        
        this.floorBounds = {
            minX: box.min.x,
            maxX: box.max.x,
            minZ: box.min.z,
            maxZ: box.max.z,
            y: box.max.y // Use the top of the floor as the walking surface
        };
        
        console.log('🏠 Floor bounds established:', this.floorBounds);
    }

    setupCollisionDetection() {
        console.log('🛡️ Setting up collision detection...');
        
        // Create bounding boxes for all collision meshes
        this.collisionMeshes.forEach((mesh, index) => {
            try {
                const box = new window.THREE.Box3().setFromObject(mesh);
                mesh.userData.boundingBox = box;
                
                console.log(`📦 Collision box ${index}: ${mesh.name || 'unnamed'} - Type: ${mesh.userData.meshType}`);
            } catch (error) {
                console.warn(`⚠️ Failed to create bounding box for mesh ${mesh.name}:`, error);
            }
        });
        
        console.log(`✅ Collision detection ready for ${this.collisionMeshes.length} objects`);
    }

    optimizeClassroomMaterials() {
        console.log('🎨 Optimizing classroom materials...');
        
        let materialCount = 0;
        
        this.classroomModel.traverse((child) => {
            if (child.isMesh && child.material) {
                materialCount++;
                
                // Optimize materials following RSMT patterns
                if (Array.isArray(child.material)) {
                    child.material.forEach(mat => this.optimizeMaterial(mat));
                } else {
                    this.optimizeMaterial(child.material);
                }
                
                // Enable proper lighting
                if (child.material.type === 'MeshStandardMaterial' || 
                    child.material.type === 'MeshPhysicalMaterial') {
                    child.material.needsUpdate = true;
                }
            }
        });
        
        console.log(`✅ Optimized ${materialCount} materials`);
    }

    optimizeMaterial(material) {
        // Apply RSMT-style material optimizations
        if (material.isMaterial) {
            // Ensure proper color space
            if (material.map) {
                material.map.colorSpace = window.THREE.SRGBColorSpace;
            }
            
            // Optimize for VRM compatibility
            if (material.type === 'MeshStandardMaterial') {
                material.roughness = material.roughness || 0.8;
                material.metalness = material.metalness || 0.1;
            }
            
            material.needsUpdate = true;
        }
    }

    // Collision detection methods
    checkCollision(position, radius = 0.3) {
        if (!this.initialized) {
            return null; // No collision if not initialized
        }
        
        // If no floor bounds (no classroom loaded), allow free movement
        if (!this.floorBounds) {
            return null;
        }
        
        // Check floor bounds
        if (!this.isWithinFloorBounds(position)) {
            return {
                type: 'floor_boundary',
                correctedPosition: this.clampToFloorBounds(position)
            };
        }
        
    // Check collision with objects
        const characterBounds = new window.THREE.Box3(
            new window.THREE.Vector3(position.x - radius, position.y - 0.1, position.z - radius),
            new window.THREE.Vector3(position.x + radius, position.y + 1.8, position.z + radius)
        );
        
        for (const mesh of this.collisionMeshes) {
            if (mesh.userData.collision && mesh.userData.boundingBox) {
                if (characterBounds.intersectsBox(mesh.userData.boundingBox)) {
                    return {
                        type: 'object_collision',
                        object: mesh,
                        correctedPosition: this.resolveCollision(position, mesh.userData.boundingBox, radius)
                    };
                }
            }
        }
        
        return null;
    }

    isWithinFloorBounds(position) {
        if (!this.floorBounds) return true; // No bounds means anywhere is valid
        return position.x >= this.floorBounds.minX && 
               position.x <= this.floorBounds.maxX &&
               position.z >= this.floorBounds.minZ && 
               position.z <= this.floorBounds.maxZ;
    }

    clampToFloorBounds(position) {
        if (!this.floorBounds) return position.clone(); // No bounds to clamp to
        return new window.THREE.Vector3(
            Math.max(this.floorBounds.minX + 0.3, Math.min(this.floorBounds.maxX - 0.3, position.x)),
            position.y,
            Math.max(this.floorBounds.minZ + 0.3, Math.min(this.floorBounds.maxZ - 0.3, position.z))
        );
    }

    resolveCollision(position, objectBounds, characterRadius) {
        const characterCenter = new window.THREE.Vector3(position.x, position.y + 0.9, position.z);
        const objectCenter = objectBounds.getCenter(new window.THREE.Vector3());
        
        // Calculate push direction (away from object center)
        const pushDirection = characterCenter.clone().sub(objectCenter).normalize();
        
        // Expand object bounds by character radius
        const expandedBounds = objectBounds.clone().expandByScalar(characterRadius);
        
        // Find the closest point outside the expanded bounds
        const correctedPosition = position.clone();
        
        if (pushDirection.x !== 0) {
            correctedPosition.x = pushDirection.x > 0 ? expandedBounds.max.x : expandedBounds.min.x;
        }
        if (pushDirection.z !== 0) {
            correctedPosition.z = pushDirection.z > 0 ? expandedBounds.max.z : expandedBounds.min.z;
        }
        
        return correctedPosition;
    }

    // Utility methods
    getFloorHeight() {
        return this.floorBounds ? this.floorBounds.y : 0;
    }

    // Get a random safe position (with fallback for empty classroom)
    getRandomSafePosition() {
        // If no floor bounds are set, return center position
        if (!this.floorBounds) {
            return new window.THREE.Vector3(0, 0, 0);
        }
        
        const maxAttempts = 50;
        let attempts = 0;
        
        while (attempts < maxAttempts) {
            const x = window.THREE.MathUtils.randFloat(this.floorBounds.minX + 1, this.floorBounds.maxX - 1);
            const z = window.THREE.MathUtils.randFloat(this.floorBounds.minZ + 1, this.floorBounds.maxZ - 1);
            const position = new window.THREE.Vector3(x, this.getFloorHeight(), z);
            
            if (!this.checkCollision(position)) {
                return position;
            }
            
            attempts++;
        }
        
        // Fallback to center if no safe position found
        return new window.THREE.Vector3(0, this.getFloorHeight(), 0);
    }

    // Toggle classroom visibility
    setVisible(visible) {
        if (this.classroomModel) {
            this.classroomModel.visible = visible;
            console.log(`🏫 Classroom ${visible ? 'shown' : 'hidden'}`);
        }
    }

    // Get classroom info for debugging
    getClassroomInfo() {
        return {
            initialized: this.initialized,
            modelLoaded: !!this.classroomModel,
            collisionMeshCount: this.collisionMeshes.length,
            floorBounds: this.floorBounds,
            path: this.classroomPath
        };
    }

    // Cleanup
    dispose() {
        if (this.classroomModel) {
            this.scene.remove(this.classroomModel);
            
            // Dispose of geometries and materials
            this.classroomModel.traverse(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(mat => {
                            if (mat.map) mat.map.dispose();
                            mat.dispose();
                        });
                    } else {
                        if (child.material.map) child.material.map.dispose();
                        child.material.dispose();
                    }
                }
            });
        }
        
        this.collisionMeshes = [];
        this.initialized = false;
        
        console.log('🧹 Classroom GLB disposed');
    }
}

// Export for use in other modules (following RSMT pattern)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClassroomGLBLoader;
} else if (typeof window !== 'undefined') {
    window.ClassroomGLBLoader = ClassroomGLBLoader;
}
