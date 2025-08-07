/**
 * VRM Material Extractor - Extracts textures and materials from VRM files
 * and applies them to BVH skeleton bones
 */
class VRMMaterialExtractor {
    constructor() {
        this.vrmData = null;
        this.extractedMaterials = new Map();
        this.boneToMaterialMap = new Map();
        this.setupBoneMaterialMapping();
        console.log('🎨 VRMMaterialExtractor initialized');
    }
    
    // Setup mapping between bone names and VRM material parts
    setupBoneMaterialMapping() {
        this.boneToMaterialMap.set('Hips', 'body');
        this.boneToMaterialMap.set('Spine', 'body');
        this.boneToMaterialMap.set('Spine1', 'body');
        this.boneToMaterialMap.set('Spine2', 'body');
        this.boneToMaterialMap.set('Neck', 'body');
        this.boneToMaterialMap.set('Head', 'head');
        
        // Arms
        this.boneToMaterialMap.set('LeftShoulder', 'body');
        this.boneToMaterialMap.set('LeftArm', 'body');
        this.boneToMaterialMap.set('LeftForeArm', 'body');
        this.boneToMaterialMap.set('LeftHand', 'body');
        this.boneToMaterialMap.set('RightShoulder', 'body');
        this.boneToMaterialMap.set('RightArm', 'body');
        this.boneToMaterialMap.set('RightForeArm', 'body');
        this.boneToMaterialMap.set('RightHand', 'body');
        
        // Legs
        this.boneToMaterialMap.set('LeftUpLeg', 'body');
        this.boneToMaterialMap.set('LeftLeg', 'body');
        this.boneToMaterialMap.set('LeftFoot', 'body');
        this.boneToMaterialMap.set('RightUpLeg', 'body');
        this.boneToMaterialMap.set('RightLeg', 'body');
        this.boneToMaterialMap.set('RightFoot', 'body');
        
        // Fingers (optional)
        for (let side of ['Left', 'Right']) {
            for (let finger of ['Thumb', 'Index', 'Middle', 'Ring', 'Little']) {
                for (let joint of ['1', '2', '3']) {
                    this.boneToMaterialMap.set(`${side}Hand${finger}${joint}`, 'body');
                }
            }
        }
    }
    
    // Load VRM file and extract materials
    async loadVRM(vrmPath) {
        try {
            console.log(`Loading VRM for material extraction: ${vrmPath}`);
            
            // Check if THREE.js is available
            if (!window.THREE) {
                throw new Error('THREE.js not available globally');
            }
            
            // Import required THREE.js modules if not available
            let GLTFLoader = window.THREELoaders?.GLTFLoader || window.THREE?.GLTFLoader;
            
            if (!GLTFLoader) {
                console.log('Loading GLTFLoader via import map...');
                try {
                    const module = await import('three/examples/jsm/loaders/GLTFLoader.js');
                    GLTFLoader = module.GLTFLoader;
                    // Store it in our loaders object
                    if (!window.THREELoaders) window.THREELoaders = {};
                    window.THREELoaders.GLTFLoader = GLTFLoader;
                    console.log('✅ GLTFLoader loaded in VRMMaterialExtractor');
                } catch (importError) {
                    console.warn('Failed to import GLTFLoader in VRMMaterialExtractor:', importError);
                    throw new Error('GLTFLoader not available');
                }
            }
            
            let VRMLoaderPlugin = window.THREELoaders?.VRMLoaderPlugin || window.VRMLoaderPlugin;
            
            if (!VRMLoaderPlugin) {
                console.log('Loading VRMLoaderPlugin via import map...');
                try {
                    const module = await import('@pixiv/three-vrm');
                    VRMLoaderPlugin = module.VRMLoaderPlugin;
                    // Store it in our loaders object
                    if (!window.THREELoaders) window.THREELoaders = {};
                    window.THREELoaders.VRMLoaderPlugin = VRMLoaderPlugin;
                    console.log('✅ VRMLoaderPlugin loaded in VRMMaterialExtractor');
                } catch (vrmError) {
                    console.warn('Failed to load VRM plugin in VRMMaterialExtractor:', vrmError);
                    console.log('Will continue with basic GLTF loading...');
                }
            }
            
            const loader = new GLTFLoader();
            
            // Only register VRM plugin if available
            if (VRMLoaderPlugin) {
                loader.register((parser) => new VRMLoaderPlugin(parser));
                console.log('VRM plugin registered with loader');
            } else {
                console.log('Loading as standard GLTF without VRM support');
            }
            
            return new Promise((resolve, reject) => {
                loader.load(
                    vrmPath,
                    (gltf) => {
                        console.log('GLTF/VRM loaded successfully:', gltf);
                        this.vrmData = gltf;
                        this.extractMaterialsFromVRM(gltf);
                        resolve(gltf);
                    },
                    (progress) => {
                        if (progress.total > 0) {
                            console.log(`Loading progress: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
                        } else {
                            console.log(`Loading progress: ${progress.loaded} bytes loaded`);
                        }
                    },
                    (error) => {
                        console.error('Failed to load GLTF/VRM:', error);
                        reject(error);
                    }
                );
            });
        } catch (error) {
            console.error('Error in loadVRM:', error);
            throw error;
        }
    }
    
    // Extract materials and textures from the loaded GLTF/VRM
    extractMaterialsFromVRM(gltf) {
        console.log('Extracting materials from GLTF/VRM...');
        
        if (!gltf.scene) {
            console.warn('No scene found in GLTF/VRM');
            return;
        }
        
        // First pass: Replace all problematic materials in the actual VRM scene
        gltf.scene.traverse((child) => {
            if (child.isMesh && child.material) {
                // Replace any VRM-specific materials with safe ones
                const materials = Array.isArray(child.material) ? child.material : [child.material];
                
                materials.forEach((material, index) => {
                    if (material.type && (material.type.includes('VRM') || material.type.includes('MToon'))) {
                        console.log(`Replacing VRM material ${material.type} with MeshBasicMaterial for ${child.name}`);
                        const safeMaterial = new window.THREE.MeshBasicMaterial({
                            color: material.color || new window.THREE.Color(0xFFFFFF),
                            map: material.map || null,
                            transparent: material.transparent || false,
                            opacity: material.opacity !== undefined ? material.opacity : 1.0,
                            side: window.THREE.FrontSide
                        });
                        
                        if (Array.isArray(child.material)) {
                            child.material[index] = safeMaterial;
                        } else {
                            child.material = safeMaterial;
                        }
                    }
                });
                
                // Also disable shadows for all VRM meshes
                child.castShadow = false;
                child.receiveShadow = false;
            }
        });
        
        // Second pass: Extract materials for our skeleton
        gltf.scene.traverse((child) => {
            if (child.isMesh && child.material) {
                this.processMeshMaterial(child);
            }
        });
        
        // If no materials were found, create default ones
        if (this.extractedMaterials.size === 0) {
            console.log('No materials found, creating default materials...');
            this.createDefaultMaterials();
        }
        
        console.log(`Extracted ${this.extractedMaterials.size} material categories from GLTF/VRM`);
    }
    
    // Validate and fix material properties to prevent shader errors
    validateMaterial(material) {
        if (!material) return null;
        
        // If it's a VRM material, replace it immediately
        if (material.type && (material.type.includes('VRM') || material.type.includes('MToon'))) {
            console.log(`Replacing problematic ${material.type} material with MeshBasicMaterial`);
            return new window.THREE.MeshBasicMaterial({
                color: material.color || new window.THREE.Color(0xFFFFFF),
                map: material.map || null,
                transparent: material.transparent || false,
                opacity: material.opacity !== undefined ? material.opacity : 1.0,
                side: window.THREE.FrontSide
            });
        }
        
        // Ensure color is valid
        if (material.color && !material.color.isColor) {
            if (typeof material.color === 'number') {
                material.color = new window.THREE.Color(material.color);
            } else if (typeof material.color === 'string') {
                material.color = new window.THREE.Color(material.color);
            } else {
                material.color = new window.THREE.Color(0xFFFFFF);
            }
        }
        
        // Ensure emissive is valid
        if (material.emissive && !material.emissive.isColor) {
            if (typeof material.emissive === 'number') {
                material.emissive = new window.THREE.Color(material.emissive);
            } else if (typeof material.emissive === 'string') {
                material.emissive = new window.THREE.Color(material.emissive);
            } else {
                material.emissive = new window.THREE.Color(0x000000);
            }
        }
        
        // Validate uniforms if present - if we find any, replace the material
        if (material.uniforms) {
            console.log('Material has uniforms - replacing with MeshBasicMaterial for safety');
            return new window.THREE.MeshBasicMaterial({
                color: material.color || new window.THREE.Color(0xFFFFFF),
                map: material.map || null,
                transparent: material.transparent || false,
                opacity: material.opacity !== undefined ? material.opacity : 1.0,
                side: window.THREE.FrontSide
            });
        }
        
        return material;
    }

    processMeshMaterial(mesh) {
        let material = mesh.material;
        // Validate and fix material properties before processing
        material = this.validateMaterial(material);

        let extractedMaterial;
        // Always create a new MeshBasicMaterial to ensure shader compatibility
        extractedMaterial = new window.THREE.MeshBasicMaterial({
            color: new window.THREE.Color(material.color && material.color.isColor ? material.color.getHex() : (typeof material.color === 'number' ? material.color : 0xFFFFFF)), // Ensure valid color
            map: material.map || null,
            transparent: material.transparent || false,
            opacity: material.opacity !== undefined ? material.opacity : 1.0,
            side: window.THREE.FrontSide // Ensure proper rendering
        });
        console.log(`Created new MeshBasicMaterial for ${mesh.name} to prevent shader errors.` );
        
        // Determine material category based on mesh name
        let category = 'body'; // default
        const meshName = mesh.name.toLowerCase();
        if (meshName.includes('hair')) {
            category = 'hair';
        } else if (meshName.includes('face') || meshName.includes('head')) {
            category = 'head';
        } else if (meshName.includes('eye')) {
            category = 'eyes';
        } else if (meshName.includes('cloth') || meshName.includes('dress') || meshName.includes('shirt')) {
            category = 'clothing';
        }

        // Store the extracted material
        if (!this.extractedMaterials.has(category)) {
            this.extractedMaterials.set(category, []);
        }
        
        this.extractedMaterials.get(category).push({
            name: mesh.name,
            material: extractedMaterial,
            originalMesh: mesh,
            geometry: mesh.geometry
        });
        
        console.log(`Extracted material for ${category}: ${mesh.name}`);
    }
    
    // Apply VRM materials to BVH skeleton bones
    applyMaterialsToSkeleton(skeleton) {
        if (!skeleton) {
            console.warn('Cannot apply materials: skeleton is null/undefined');
            return;
        }
        
        if (!this.extractedMaterials || this.extractedMaterials.size === 0) {
            console.warn('Cannot apply materials: no extracted materials available');
            return;
        }
        
        console.log('Applying VRM materials to BVH skeleton...');
        console.log('Skeleton type:', skeleton.constructor.name);
        console.log('Skeleton properties:', Object.keys(skeleton));
        
        // Create body mesh geometry (basic capsule/cylinder for each bone)
        const bodyMaterial = this.getMaterialForCategory('body') || this.createDefaultMaterial();
        const headMaterial = this.getMaterialForCategory('head') || bodyMaterial;
        
        
        // Iterate over the children of the skeleton group (which are the joint meshes)
        skeleton.traverse((child) => {
            if (child.isMesh && child.userData.isJoint) { // Check if it's a joint mesh
                const boneName = child.userData.bvhJointName;
                if (boneName && this.shouldCreateMeshForBone(boneName)) {
                    this.createMeshForBone(child, bodyMaterial, headMaterial); // Pass the joint mesh as 'bone'
                }
            }
        });
        
        console.log('Applied VRM materials to skeleton bones');
    }
    
    getMaterialForCategory(category) {
        const materials = this.extractedMaterials.get(category);
        if (materials && materials.length > 0) {
            return materials[0].material;
        }
        return null;
    }
    
    shouldCreateMeshForBone(boneName) {
        // Only create meshes for main body bones
        const mainBones = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ];
        
        return mainBones.includes(boneName);
    }
    
    createMeshForBone(bone, bodyMaterial, headMaterial) {
        let geometry;
        // Use a simple MeshBasicMaterial for absolute shader compatibility
        // Choose geometry based on bone type
        if (bone.name.includes('Head')) {
            geometry = new window.THREE.SphereGeometry(0.12, 16, 16);
        } else if (bone.name.includes('Hips') || bone.name.includes('Spine')) {
            geometry = new window.THREE.CylinderGeometry(0.15, 0.15, 0.3, 12);
        } else if (bone.name.includes('Arm') || bone.name.includes('Leg')) {
            geometry = new window.THREE.CylinderGeometry(0.08, 0.08, 0.4, 8);
        } else {
            geometry = new window.THREE.CylinderGeometry(0.06, 0.06, 0.25, 8);
        }
        
        // Use MeshBasicMaterial for maximum shader compatibility
        const material = new window.THREE.MeshBasicMaterial({ 
            color: new window.THREE.Color(bone.name.includes('Head') ? (headMaterial.color && headMaterial.color.isColor ? headMaterial.color.getHex() : 0xFFCCCC) : (bodyMaterial.color && bodyMaterial.color.isColor ? bodyMaterial.color.getHex() : 0xFFCCCC)),
            transparent: true,
            opacity: 0.9,
            side: window.THREE.FrontSide
        });
        
        // Create mesh and attach to bone
        const mesh = new window.THREE.Mesh(geometry, material);
        mesh.castShadow = false; // Disable shadows to prevent shader issues
        mesh.receiveShadow = false;
        
        // Position mesh relative to bone
        if (bone.name.includes('Head')) {
            mesh.position.set(0, 0.1, 0);
        } else {
            mesh.position.set(0, 0.15, 0);
        }
        
        bone.add(mesh);
        
        console.log(`Created mesh for bone: ${bone.name}`);
    }
    
    createDefaultMaterial() {
        return new window.THREE.MeshBasicMaterial({
            color: 0xFFCCCC, // Skin-like color
            transparent: true,
            opacity: 0.9,
            side: window.THREE.FrontSide
        });
    }
    
    // Create default materials when none are extracted
    createDefaultMaterials() {
        const bodyMaterial = new window.THREE.MeshBasicMaterial({
            color: 0xFFDDBB, // Light skin tone
            transparent: true,
            opacity: 0.9,
            side: window.THREE.FrontSide
        });
        
        const headMaterial = new window.THREE.MeshBasicMaterial({
            color: 0xFFE4C4, // Slightly lighter for face
            transparent: true,
            opacity: 0.9,
            side: window.THREE.FrontSide
        });
        
        const clothingMaterial = new window.THREE.MeshBasicMaterial({
            color: 0x4169E1, // Blue clothing
            transparent: true,
            opacity: 0.9,
            side: window.THREE.FrontSide
        });
        
        this.extractedMaterials.set('body', [{
            name: 'default_body',
            material: bodyMaterial,
            originalMesh: null,
            geometry: null
        }]);
        
        this.extractedMaterials.set('head', [{
            name: 'default_head',
            material: headMaterial,
            originalMesh: null,
            geometry: null
        }]);
        
        this.extractedMaterials.set('clothing', [{
            name: 'default_clothing',
            material: clothingMaterial,
            originalMesh: null,
            geometry: null
        }]);
        
        console.log('Created default MeshBasicMaterials for body, head, and clothing');
    }
    
    // Get all extracted materials for external use
    getExtractedMaterials() {
        return this.extractedMaterials;
    }
    
    // Apply clothing/accessory meshes as separate objects
    applyClothingMeshes(scene) {
        const clothingMaterials = this.extractedMaterials.get('clothing');
        if (!clothingMaterials || clothingMaterials.length === 0) {
            console.log('No clothing materials to apply');
            return;
        }
        
        clothingMaterials.forEach((clothingData) => {
            try {
                // Check if geometry and material exist
                if (!clothingData.geometry || !clothingData.material) {
                    console.warn(`Skipping clothing mesh ${clothingData.name}: missing geometry or material`);
                    return;
                }
                
                const geometry = clothingData.geometry.clone ? clothingData.geometry.clone() : clothingData.geometry;
                const material = clothingData.material.clone ? clothingData.material.clone() : clothingData.material;
                
                const mesh = new window.THREE.Mesh(geometry, material);
                
                mesh.scale.set(1.1, 1.1, 1.1); // Slightly larger to fit over body
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                
                scene.add(mesh);
                console.log(`Added clothing mesh: ${clothingData.name}`);
            } catch (clothingError) {
                console.warn(`Error adding clothing mesh ${clothingData.name}:`, clothingError);
            }
        });
    }
    
    dispose() {
        this.extractedMaterials.clear();
        this.boneToMaterialMap.clear();
        this.vrmData = null;
        console.log('VRM Material Extractor disposed');
    }
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VRMMaterialExtractor };
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.VRMMaterialExtractor = VRMMaterialExtractor;
}
