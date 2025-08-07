/**
 * VRM Visibility Fix - Fixes missing body parts and rendering issues
 */

// Global VRM visibility fix function
window.fixVRMVisibility = function() {
    console.log('👁️ === FIXING VRM CHARACTER VISIBILITY ISSUES ===');
    
    if (!window.vrmCharacter?.vrm?.scene) {
        console.log('❌ No VRM character loaded');
        return 'No VRM character to fix';
    }
    
    let fixedCount = 0;
    let meshCount = 0;
    const vrmScene = window.vrmCharacter.vrm.scene;
    const bodyParts = [];
    
    // Force the entire VRM scene to be visible
    vrmScene.visible = true;
    console.log('✅ VRM scene visibility forced to true');
    
    vrmScene.traverse((child) => {
        if (child.isMesh) {
            meshCount++;
            const name = child.name || 'unnamed';
            
            // Track what body parts we find
            bodyParts.push({
                name: name,
                originallyVisible: child.visible,
                hasGeometry: !!child.geometry,
                hasMaterial: !!child.material,
                vertices: child.geometry?.attributes?.position?.count || 0
            });
            
            // Force mesh visibility
            child.visible = true;
            child.frustumCulled = false;
            
            // Check if geometry exists
            if (!child.geometry) {
                console.warn('⚠️ Mesh has no geometry:', name);
                return;
            }
            
            // Fix materials
            if (child.material) {
                const materials = Array.isArray(child.material) ? child.material : [child.material];
                
                materials.forEach((material, index) => {
                    // Force material visibility
                    material.visible = true;
                    material.transparent = false;
                    material.opacity = 1.0;
                    material.alphaTest = 0.001;
                    material.side = THREE.DoubleSide;
                    
                    // Ensure material has color
                    if (!material.color) {
                        material.color = new THREE.Color(0xffffff);
                    }
                    
                    // Fix specific VRM material issues
                    if (material.isShaderMaterial || material.name?.includes('VRM') || material.name?.includes('MToon')) {
                        console.log('🔧 VRM material detected, applying fixes:', material.name);
                        
                        // For VRM materials, ensure proper rendering
                        material.alphaTest = 0.001;
                        material.depthWrite = true;
                        material.depthTest = true;
                    }
                    
                    material.needsUpdate = true;
                    fixedCount++;
                });
                
                console.log(`✅ Fixed: ${name} (visible: ${child.visible}, materials: ${materials.length})`);
            } else {
                console.warn('⚠️ Mesh has no material:', name);
            }
        }
    });
    
    // Report findings
    console.log(`📊 VRM Analysis: ${meshCount} meshes found`);
    console.log('📝 Body parts inventory:');
    bodyParts.forEach(part => {
        const status = part.originallyVisible ? '✅' : '❌';
        console.log(`  ${status} ${part.name}: visible=${part.originallyVisible}, geometry=${part.hasGeometry}, material=${part.hasMaterial}, vertices=${part.vertices}`);
    });
    
    // Check for missing common body parts
    const commonParts = ['hair', 'head', 'body', 'torso', 'chest', 'skirt', 'dress', 'arm', 'hand'];
    const missingParts = commonParts.filter(part => 
        !bodyParts.some(found => found.name.toLowerCase().includes(part))
    );
    
    if (missingParts.length > 0) {
        console.warn('⚠️ Potentially missing body parts:', missingParts);
    }
    
    console.log(`👁️ VRM visibility fix completed: ${fixedCount} materials fixed on ${meshCount} meshes`);
    console.log('👁️ === VRM VISIBILITY FIX COMPLETE ===');
    
    return `VRM visibility fixed for ${fixedCount} materials on ${meshCount} meshes`;
};

// Global VRM debug function
window.debugVRM = function() {
    console.log('🔍 === VRM DEBUG INFORMATION ===');
    
    if (!window.vrmCharacter?.vrm) {
        console.log('❌ No VRM character loaded');
        return;
    }
    
    const vrm = window.vrmCharacter.vrm;
    
    console.log('VRM Object:', vrm);
    console.log('VRM Scene:', vrm.scene);
    console.log('VRM Humanoid:', vrm.humanoid);
    console.log('VRM Expression Manager:', vrm.expressionManager);
    
    if (vrm.scene) {
        let meshes = [];
        vrm.scene.traverse((child) => {
            if (child.isMesh) {
                meshes.push({
                    name: child.name,
                    visible: child.visible,
                    material: child.material?.type || 'none',
                    vertices: child.geometry?.attributes?.position?.count || 0
                });
            }
        });
        console.log('VRM Meshes:', meshes);
    }
    
    if (vrm.humanoid) {
        console.log('Available bones:', Object.keys(vrm.humanoid.normalizedHumanBones || {}));
    }
    
    console.log('🔍 === VRM DEBUG COMPLETE ===');
};

// Auto-fix function for missing parts
window.fixMissingVRMParts = function() {
    console.log('🔧 Attempting to fix missing VRM parts...');
    
    if (!window.vrmCharacter?.vrm?.scene) {
        return 'No VRM character loaded';
    }
    
    // First, apply visibility fix
    window.fixVRMVisibility();
    
    // Then check for material issues and apply emergency fix if needed
    let hasIssues = false;
    window.vrmCharacter.vrm.scene.traverse((child) => {
        if (child.isMesh && child.material) {
            const materials = Array.isArray(child.material) ? child.material : [child.material];
            materials.forEach(mat => {
                if (mat.isShaderMaterial && mat.uniforms) {
                    // Check for problematic uniforms
                    Object.keys(mat.uniforms).forEach(key => {
                        const uniform = mat.uniforms[key];
                        if (uniform && uniform.value) {
                            if (uniform.value.isVector3 && 
                                (isNaN(uniform.value.x) || isNaN(uniform.value.y) || isNaN(uniform.value.z))) {
                                hasIssues = true;
                            }
                        }
                    });
                }
            });
        }
    });
    
    if (hasIssues) {
        console.log('🚨 Detected shader issues, applying emergency material fix...');
        if (window.emergencyMaterialFix) {
            window.emergencyMaterialFix();
        }
    }
    
    return 'VRM parts fix completed';
};

console.log('✅ VRM Visibility Fix module loaded');
