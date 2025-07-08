/**
 * VRM Diagnostics - Comprehensive debugging and analysis tools
 */

// Global diagnostic functions for VRM character debugging
window.runVRMDiagnostics = function() {
    console.log('🔍 === COMPREHENSIVE VRM DIAGNOSTICS ===');
    
    const results = {
        vrmLoaded: false,
        animationActive: false,
        visibilityIssues: [],
        coordinateIssues: [],
        materialIssues: [],
        recommendations: []
    };
    
    // Check VRM character status
    if (window.vrmCharacter?.vrm?.scene) {
        results.vrmLoaded = true;
        console.log('✅ VRM character loaded');
        
        const vrmScene = window.vrmCharacter.vrm.scene;
        console.log('📊 VRM Scene Analysis:');
        console.log('  - Position:', vrmScene.position);
        console.log('  - Rotation:', vrmScene.rotation);
        console.log('  - Scale:', vrmScene.scale);
        console.log('  - Visible:', vrmScene.visible);
        
        // Check bounding box for positioning
        const boundingBox = new THREE.Box3().setFromObject(vrmScene);
        console.log('  - Bounding Box:', boundingBox);
        console.log('  - Character Height:', (boundingBox.max.y - boundingBox.min.y).toFixed(2), 'units');
        console.log('  - Ground Level:', boundingBox.min.y.toFixed(2), 'units');
        
        // Analyze mesh visibility
        let totalMeshes = 0;
        let visibleMeshes = 0;
        let meshDetails = [];
        
        vrmScene.traverse((child) => {
            if (child.isMesh) {
                totalMeshes++;
                const isVisible = child.visible && child.material && 
                    (!child.material.transparent || child.material.opacity > 0);
                if (isVisible) visibleMeshes++;
                
                meshDetails.push({
                    name: child.name || 'unnamed',
                    visible: child.visible,
                    hasGeometry: !!child.geometry,
                    hasMaterial: !!child.material,
                    materialType: child.material?.type || 'none',
                    vertices: child.geometry?.attributes?.position?.count || 0,
                    opacity: child.material?.opacity || 0
                });
                
                // Check for visibility issues
                if (!child.visible) {
                    results.visibilityIssues.push(`Mesh "${child.name}" is not visible`);
                }
                if (child.material?.transparent && child.material?.opacity < 0.1) {
                    results.visibilityIssues.push(`Mesh "${child.name}" has very low opacity: ${child.material.opacity}`);
                }
            }
        });
        
        console.log('👁️ Mesh Visibility Analysis:');
        console.log(`  - Total meshes: ${totalMeshes}`);
        console.log(`  - Visible meshes: ${visibleMeshes}`);
        console.log(`  - Visibility rate: ${(visibleMeshes/totalMeshes*100).toFixed(1)}%`);
        
        if (visibleMeshes < totalMeshes * 0.7) {
            results.recommendations.push('Apply VRM visibility fix - many meshes are hidden');
        }
        
        // Log mesh details
        console.log('📋 Mesh Details:');
        meshDetails.forEach((mesh, index) => {
            const status = mesh.visible ? '✅' : '❌';
            console.log(`  ${status} ${mesh.name}: ${mesh.materialType}, ${mesh.vertices} vertices, opacity: ${mesh.opacity}`);
        });
        
    } else {
        console.log('❌ No VRM character loaded');
        results.recommendations.push('Load a VRM character first');
    }
    
    // Check animation status
    if (window.currentVRMAdapter) {
        console.log('🎭 Animation System Analysis:');
        console.log('  - VRM Adapter initialized:', window.currentVRMAdapter.initialized);
        console.log('  - Available bones:', window.currentVRMAdapter.getAvailableBones?.().length || 'unknown');
        console.log('  - Debug mode:', window.currentVRMAdapter.debugMode);
        
        if (window.currentVRMAdapter.initialized) {
            results.animationActive = true;
            
            // Test frame application
            if (window.bvhData && window.currentAnimation && window.bvhData[window.currentAnimation]?.frames?.length > 0) {
                console.log('🧪 Testing frame application...');
                const testFrame = window.bvhData[window.currentAnimation].frames[0];
                const success = window.currentVRMAdapter.applyBVHFrameToVRM(testFrame);
                console.log('  - Frame application test:', success ? 'SUCCESS' : 'FAILED');
                
                if (!success) {
                    results.recommendations.push('Animation system needs debugging - frame application failed');
                }
            }
        } else {
            results.recommendations.push('Initialize VRM-BVH adapter for animation');
        }
    } else {
        console.log('❌ No VRM animation adapter found');
        results.recommendations.push('Create VRM-BVH adapter for animation support');
    }
    
    // Check coordinate system alignment
    if (window.vrmCharacter?.vrm?.scene) {
        const pos = window.vrmCharacter.vrm.scene.position;
        const rot = window.vrmCharacter.vrm.scene.rotation;
        
        console.log('📐 Coordinate System Analysis:');
        console.log('  - Position offset from origin:', Math.sqrt(pos.x*pos.x + pos.z*pos.z).toFixed(2));
        console.log('  - Y position (ground level):', pos.y.toFixed(2));
        console.log('  - Rotation angles (deg):', 
            (rot.x * 180/Math.PI).toFixed(1), 
            (rot.y * 180/Math.PI).toFixed(1), 
            (rot.z * 180/Math.PI).toFixed(1));
        
        if (pos.y < -0.5 || pos.y > 0.5) {
            results.coordinateIssues.push('Character may not be properly grounded');
        }
        
        if (Math.abs(rot.x) > 0.1 || Math.abs(rot.z) > 0.1) {
            results.coordinateIssues.push('Character has unexpected tilt rotation');
        }
    }
    
    // Check for WebGL/rendering issues
    console.log('🖥️ Rendering System Analysis:');
    if (window.renderer) {
        console.log('  - Renderer type:', window.renderer.constructor.name);
        console.log('  - Shadow mapping:', window.renderer.shadowMap.enabled);
        console.log('  - Tone mapping:', window.renderer.toneMapping);
        console.log('  - Pixel ratio:', window.renderer.getPixelRatio());
        
        // Check for WebGL errors
        const gl = window.renderer.getContext();
        const error = gl.getError();
        if (error !== gl.NO_ERROR) {
            results.materialIssues.push(`WebGL error detected: ${error}`);
        }
    }
    
    // Generate recommendations
    if (results.visibilityIssues.length > 0) {
        results.recommendations.push('Run window.fixVRMVisibility() to fix visibility issues');
    }
    
    if (results.coordinateIssues.length > 0) {
        results.recommendations.push('Run coordinate system fixes for proper positioning');
    }
    
    if (results.materialIssues.length > 0) {
        results.recommendations.push('Run window.emergencyMaterialFix() for rendering issues');
    }
    
    // Final summary
    console.log('📊 === DIAGNOSTIC SUMMARY ===');
    console.log('VRM Loaded:', results.vrmLoaded ? '✅' : '❌');
    console.log('Animation Active:', results.animationActive ? '✅' : '❌');
    console.log('Visibility Issues:', results.visibilityIssues.length);
    console.log('Coordinate Issues:', results.coordinateIssues.length);
    console.log('Material Issues:', results.materialIssues.length);
    
    if (results.recommendations.length > 0) {
        console.log('🔧 RECOMMENDATIONS:');
        results.recommendations.forEach((rec, index) => {
            console.log(`  ${index + 1}. ${rec}`);
        });
    } else {
        console.log('✅ No issues detected - system appears to be working correctly');
    }
    
    console.log('🔍 === END DIAGNOSTICS ===');
    
    return results;
};

// Quick fix function that applies all common solutions
window.quickFixVRM = function() {
    console.log('🚀 === QUICK VRM FIX SEQUENCE ===');
    
    let fixesApplied = [];
    
    // 1. Apply visibility fixes
    if (typeof window.fixVRMVisibility === 'function') {
        console.log('1️⃣ Applying VRM visibility fixes...');
        window.fixVRMVisibility();
        fixesApplied.push('Visibility fix');
    }
    
    // 2. Apply material fixes
    if (typeof window.emergencyMaterialFix === 'function') {
        console.log('2️⃣ Applying emergency material fixes...');
        window.emergencyMaterialFix();
        fixesApplied.push('Material fix');
    }
    
    // 2b. Apply VRM-specific shader fixes
    if (typeof window.fixVRMShaderErrors === 'function') {
        console.log('2️⃣b Applying VRM shader fixes...');
        window.fixVRMShaderErrors();
        fixesApplied.push('VRM shader fix');
    }
    
    // 3. Reset character positioning
    if (window.vrmCharacter?.vrm?.scene) {
        console.log('3️⃣ Resetting character position...');
        const vrmScene = window.vrmCharacter.vrm.scene;
        const boundingBox = new THREE.Box3().setFromObject(vrmScene);
        const groundOffset = -boundingBox.min.y;
        
        vrmScene.scale.setScalar(1.0);
        vrmScene.position.set(0, groundOffset * 0.98, 0);
        vrmScene.rotation.set(0, 0, 0);
        fixesApplied.push('Position reset');
    }
    
    // 4. Force render update
    if (window.renderer && window.scene && window.camera) {
        console.log('4️⃣ Forcing render update...');
        window.renderer.render(window.scene, window.camera);
        fixesApplied.push('Render update');
    }
    
    // 5. Check animation system
    if (window.currentVRMAdapter && !window.currentVRMAdapter.initialized) {
        console.log('5️⃣ Reinitializing animation system...');
        window.currentVRMAdapter.initialize();
        fixesApplied.push('Animation reinitialization');
    }
    
    console.log('✅ Quick fix completed. Applied:', fixesApplied.join(', '));
    console.log('🔍 Run window.runVRMDiagnostics() to verify fixes');
    
    return `Applied ${fixesApplied.length} fixes: ${fixesApplied.join(', ')}`;
};

// Function to reset everything and start fresh
window.resetVRMSystem = function() {
    console.log('🔄 === RESETTING VRM SYSTEM ===');
    
    // 1. Stop any running animations
    if (window.animationRunning) {
        window.animationRunning = false;
    }
    
    // 2. Reset VRM character position and properties
    if (window.vrmCharacter?.vrm?.scene) {
        const vrmScene = window.vrmCharacter.vrm.scene;
        vrmScene.position.set(0, 0, 0);
        vrmScene.rotation.set(0, 0, 0);
        vrmScene.scale.setScalar(1.0);
        vrmScene.visible = true;
    }
    
    // 3. Reset animation adapter
    if (window.currentVRMAdapter) {
        window.currentVRMAdapter.debugMode = true;
        window.currentVRMAdapter.initialize();
    }
    
    // 4. Apply all fixes
    window.quickFixVRM();
    
    console.log('✅ VRM system reset complete');
    return 'VRM system reset and fixes applied';
};

// Advanced VRM material fix specifically for shader errors
window.fixVRMShaderErrors = function() {
    console.log('🔧 === ADVANCED VRM SHADER FIX ===');
    
    let fixedCount = 0;
    const problematicUniforms = ['color', 'emissive', 'diffuse', 'specular'];
    
    // Fix VRM character materials
    if (window.vrmCharacter?.vrm?.scene) {
        window.vrmCharacter.vrm.scene.traverse((child) => {
            if (child.isMesh && child.material) {
                const materials = Array.isArray(child.material) ? child.material : [child.material];
                
                materials.forEach((material, index) => {
                    if (material.uniforms) {
                        // Check for problematic uniforms
                        let hasIssues = false;
                        
                        problematicUniforms.forEach(uniformName => {
                            if (material.uniforms[uniformName] && 
                                material.uniforms[uniformName].value && 
                                !Array.isArray(material.uniforms[uniformName].value) &&
                                typeof material.uniforms[uniformName].value !== 'number') {
                                hasIssues = true;
                            }
                        });
                        
                        if (hasIssues) {
                            // Replace with safe MeshBasicMaterial
                            const newMaterial = new THREE.MeshBasicMaterial({
                                color: new THREE.Color(0.8, 0.8, 0.8),
                                transparent: material.transparent || false,
                                opacity: material.opacity || 1.0,
                                side: material.side || THREE.FrontSide,
                                map: material.map || null
                            });
                            
                            if (Array.isArray(child.material)) {
                                child.material[index] = newMaterial;
                            } else {
                                child.material = newMaterial;
                            }
                            
                            fixedCount++;
                            console.log(`🔧 Fixed shader material for: ${child.name || 'unnamed mesh'}`);
                        }
                    }
                });
            }
        });
    }
    
    // Also fix any scene materials
    if (window.scene) {
        window.scene.traverse((child) => {
            if (child.isMesh && child.material && child.name && child.name.includes('vrm')) {
                const materials = Array.isArray(child.material) ? child.material : [child.material];
                
                materials.forEach((material, index) => {
                    if (material.type === 'ShaderMaterial' || material.uniforms) {
                        const newMaterial = new THREE.MeshBasicMaterial({
                            color: material.color || new THREE.Color(0.8, 0.8, 0.8),
                            transparent: material.transparent || false,
                            opacity: material.opacity || 1.0,
                            map: material.map || null
                        });
                        
                        if (Array.isArray(child.material)) {
                            child.material[index] = newMaterial;
                        } else {
                            child.material = newMaterial;
                        }
                        
                        fixedCount++;
                    }
                });
            }
        });
    }
    
    console.log(`✅ Fixed ${fixedCount} problematic VRM shader materials`);
    
    // Force material update
    if (window.renderer && window.scene && window.camera) {
        window.renderer.render(window.scene, window.camera);
    }
    
    return `Fixed ${fixedCount} VRM shader materials`;
};

// Quick diagnostic function for immediate use
window.quickVRMStatus = function() {
    console.log('🔍 === QUICK VRM STATUS ===');
    
    const status = {
        vrmLoaded: !!window.vrmCharacter,
        adapterAvailable: !!window.currentVRMAdapter,
        adapterInitialized: window.currentVRMAdapter?.initialized || false,
        sceneVRM: !!window.vrmCharacter?.vrm?.scene,
        humanoidAvailable: !!window.vrmCharacter?.vrm?.humanoid
    };
    
    console.log('VRM Character:', status.vrmLoaded ? '✅' : '❌');
    console.log('VRM Adapter:', status.adapterAvailable ? '✅' : '❌');
    console.log('Adapter Initialized:', status.adapterInitialized ? '✅' : '❌');
    console.log('VRM Scene:', status.sceneVRM ? '✅' : '❌');
    console.log('VRM Humanoid:', status.humanoidAvailable ? '✅' : '❌');
    
    if (window.vrmCharacter?.vrm?.humanoid) {
        const humanoid = window.vrmCharacter.vrm.humanoid;
        console.log('Available VRM bones:', Object.keys(humanoid.normalizedHumanBones || {}).length);
    }
    
    if (window.currentVRMAdapter && !window.currentVRMAdapter.initialized) {
        console.log('🔧 Attempting to manually initialize VRM adapter...');
        try {
            const result = window.currentVRMAdapter.initialize();
            console.log('Manual initialization result:', result ? '✅' : '❌');
        } catch (error) {
            console.error('Manual initialization failed:', error.message);
        }
    }
    
    console.log('🔍 === END QUICK STATUS ===');
    return status;
};

// Force create VRM adapter if missing
window.forceCreateVRMAdapter = function() {
    console.log('🔧 === FORCE CREATE VRM ADAPTER ===');
    
    if (!window.vrmCharacter) {
        console.error('❌ No VRM character available');
        return false;
    }
    
    if (!window.scene) {
        console.error('❌ No scene available');
        return false;
    }
    
    try {
        console.log('🔗 Creating Enhanced VRM adapter manually...');
        console.log('VRM object:', window.vrmCharacter);
        console.log('VRM type:', typeof window.vrmCharacter);
        console.log('VRM.vrm available:', !!window.vrmCharacter.vrm);
        console.log('VRM.vrm.humanoid available:', !!window.vrmCharacter.vrm?.humanoid);
        
        // Create the adapter
        window.currentVRMAdapter = new window.EnhancedVRMBVHAdapter(window.vrmCharacter.vrm, window.scene);
        
        console.log('✅ VRM adapter created successfully');
        console.log('Adapter initialized:', window.currentVRMAdapter.initialized);
        console.log('Available bones:', window.currentVRMAdapter.getAvailableBones().length);
        
        return true;
        
    } catch (error) {
        console.error('❌ Failed to force create VRM adapter:', error.message);
        console.error('Full error:', error);
        return false;
    }
};

console.log('✅ VRM Diagnostics module loaded');
console.log('🔧 Available commands:');
console.log('  - window.runVRMDiagnostics() - Full system analysis');
console.log('  - window.quickFixVRM() - Apply all common fixes');
console.log('  - window.resetVRMSystem() - Reset everything and start fresh');
console.log('  - window.fixVRMShaderErrors() - Fix shader uniform errors');
console.log('  - window.quickVRMStatus() - Quick status check and manual init');
console.log('  - window.forceCreateVRMAdapter() - Manually create VRM adapter');
