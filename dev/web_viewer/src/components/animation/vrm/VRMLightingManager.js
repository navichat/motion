/**
 * VRMLightingManager - Optimized lighting setup for VRM characters and classroom environments
 * Provides shader-safe lighting configurations that work well with VRM models
 */

class VRMLightingManager {
    constructor(scene, renderer) {
        this.scene = scene;
        this.renderer = renderer;
        this.lights = {};
        this.lightingMode = 'soft'; // 'bright', 'balanced', 'dramatic', 'soft'
        this.environmentType = 'studio'; // 'indoor', 'outdoor', 'studio'
        this.initialized = false;
        
        // Lighting configuration presets
        this.presets = {
            classroom: {
                ambient: { color: 0x404040, intensity: 0.8 },
                directional: { color: 0xffffff, intensity: 1.0, position: [10, 10, 5] },
                fill: { color: 0xffffff, intensity: 0.5, position: [-5, 5, -5] },
                accent: { color: 0x87ceeb, intensity: 0.5, position: [0, 8, 0] }
            },
            studio: {
                ambient: { color: 0x202020, intensity: 0.4 },
                directional: { color: 0xffffff, intensity: 1.0, position: [5, 8, 3] },
                fill: { color: 0xffffff, intensity: 0.5, position: [-3, 6, -2] },
                accent: { color: 0xffffff, intensity: 0.3, position: [0, 10, 0] }
            },
            dramatic: {
                ambient: { color: 0x101010, intensity: 0.2 },
                directional: { color: 0xffffff, intensity: 1.2, position: [8, 12, 4] },
                fill: { color: 0x4080ff, intensity: 0.4, position: [-4, 4, -3] },
                accent: { color: 0xff8040, intensity: 0.6, position: [2, 6, 2] }
            },
            soft: {
                ambient: { color: 0x606060, intensity: 1},
                directional: { color: 0xffffff, intensity: 1, position: [6, 8, 4] },
                fill: { color: 0xffffff, intensity: 1, position: [-3, 5, -2] },
                accent: { color: 0xffeecc, intensity: 0.3, position: [0, 6, 0] }
            }
        };
        
        console.log('💡 VRMLightingManager initialized');
    }

    async initialize(preset = 'classroom') {
        console.log(`🏗️ Initializing VRM lighting with preset: ${preset}`);
        
        try {
            // Clear existing lights first
            this.clearLights();
            
            // Configure renderer for VRM compatibility
            this.configureRenderer();
            
            // Setup lighting based on preset
            await this.setupLighting(preset);
            this.setPreset(preset);
            this.initialized = true;
            console.log('✅ VRM lighting system ready');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize VRM lighting:', error);
            return false;
        }
    }

    configureRenderer() {
        console.log('🔧 Configuring renderer for VRM lighting...');
        
        // VRM-safe renderer configuration
        this.renderer.shadowMap.enabled = false; // Disable shadows to prevent VRM shader issues
        this.renderer.shadowMap.type = window.THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = window.THREE.SRGBColorSpace;
        this.renderer.toneMapping = window.THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        this.renderer.physicallyCorrectLights = false; // Disable for VRM compatibility
        
        console.log('✅ Renderer configured for VRM compatibility');
    }

    async setupLighting(presetName = 'classroom') {
        console.log(`💡 Setting up ${presetName} lighting...`);
        
        const preset = this.presets[presetName] || this.presets.classroom;
        
        // 1. Ambient light - provides base illumination
        this.lights.ambient = new window.THREE.AmbientLight(
            preset.ambient.color, 
            preset.ambient.intensity
        );
        this.lights.ambient.name = 'VRM_AmbientLight';
        this.scene.add(this.lights.ambient);
        
        // 2. Main directional light - primary illumination (no shadows for VRM)
        this.lights.directional = new window.THREE.DirectionalLight(
            preset.directional.color, 
            preset.directional.intensity
        );
        this.lights.directional.position.set(...preset.directional.position);
        this.lights.directional.castShadow = false; // Critical: no shadows for VRM
        this.lights.directional.name = 'VRM_DirectionalLight';
        this.scene.add(this.lights.directional);
        
        // 3. Fill light - reduces harsh shadows
        this.lights.fill = new window.THREE.DirectionalLight(
            preset.fill.color, 
            preset.fill.intensity
        );
        this.lights.fill.position.set(...preset.fill.position);
        this.lights.fill.castShadow = false;
        this.lights.fill.name = 'VRM_FillLight';
        this.scene.add(this.lights.fill);
        
        // 4. Accent light - adds atmosphere
        this.lights.accent = new window.THREE.PointLight(
            preset.accent.color, 
            preset.accent.intensity, 
            50 // distance
        );
        this.lights.accent.position.set(...preset.accent.position);
        this.lights.accent.castShadow = false;
        this.lights.accent.name = 'VRM_AccentLight';
        this.scene.add(this.lights.accent);
        
        console.log(`✅ ${presetName} lighting setup complete`);
        this.logLightingInfo();
    }

    clearLights() {
        console.log('🧹 Clearing existing lights...');
        
        // Remove existing VRM lights
        Object.values(this.lights).forEach(light => {
            if (light && this.scene) {
                this.scene.remove(light);
                if (light.dispose) light.dispose();
            }
        });
        
        // Also remove any lights with VRM naming
        const lightsToRemove = [];
        this.scene.traverse((object) => {
            if (object.isLight && object.name && object.name.startsWith('VRM_')) {
                lightsToRemove.push(object);
            }
        });
        
        lightsToRemove.forEach(light => {
            this.scene.remove(light);
            if (light.dispose) light.dispose();
        });
        
        this.lights = {};
        console.log('✅ Lights cleared');
    }

    setPreset(presetName) {
        if (!this.presets[presetName]) {
            console.warn(`⚠️ Unknown lighting preset: ${presetName}`);
            return false;
        }
        
        console.log(`🔄 Switching to ${presetName} lighting preset...`);
        return this.setupLighting(presetName);
    }

    adjustBrightness(factor = 1.0) {
        console.log(`☀️ Adjusting lighting brightness by factor: ${factor}`);
        
        Object.values(this.lights).forEach(light => {
            if (light && light.intensity !== undefined) {
                const originalIntensity = light.userData.originalIntensity || light.intensity;
                light.userData.originalIntensity = originalIntensity;
                light.intensity = originalIntensity * factor;
            }
        });
        
        console.log(`✅ Brightness adjusted to ${(factor * 100).toFixed(0)}%`);
    }

    toggleLighting() {
        const isOn = this.lights.directional?.intensity > 0;
        
        if (isOn) {
            // Turn lights down
            this.adjustBrightness(0.2);
            console.log('🌙 Lighting dimmed');
            return false;
        } else {
            // Turn lights up
            this.adjustBrightness(1.0);
            console.log('☀️ Lighting brightened');
            return true;
        }
    }

    // Character-specific lighting adjustments
    optimizeForVRM(vrmCharacter) {
        if (!vrmCharacter || !vrmCharacter.scene) {
            console.warn('⚠️ No VRM character provided for lighting optimization');
            return;
        }
        
        console.log('🎭 Optimizing lighting for VRM character...');
        
        // Get character position for dynamic lighting
        const charPos = vrmCharacter.scene.position;
        
        // Adjust directional light to illuminate character
        if (this.lights.directional) {
            const lightOffset = new window.THREE.Vector3(5, 8, 3);
            this.lights.directional.position.copy(charPos).add(lightOffset);
            this.lights.directional.target.position.copy(charPos);
            this.lights.directional.target.updateMatrixWorld();
        }
        
        // Adjust fill light to reduce harsh shadows on character
        if (this.lights.fill) {
            const fillOffset = new window.THREE.Vector3(-3, 5, -2);
            this.lights.fill.position.copy(charPos).add(fillOffset);
        }
        
        // Position accent light above character
        if (this.lights.accent) {
            const accentOffset = new window.THREE.Vector3(0, 6, 1);
            this.lights.accent.position.copy(charPos).add(accentOffset);
        }
        
        console.log('✅ Lighting optimized for VRM character');
    }

    // Environment-specific lighting
    optimizeForClassroom(classroomModel) {
        if (!classroomModel) {
            console.log('ℹ️ No classroom model - using default lighting');
            return;
        }
        
        console.log('🏫 Optimizing lighting for classroom environment...');
        
        // Calculate classroom bounds
        const box = new window.THREE.Box3().setFromObject(classroomModel);
        const center = box.getCenter(new window.THREE.Vector3());
        const size = box.getSize(new window.THREE.Vector3());
        
        // Adjust ambient light for indoor environment
        if (this.lights.ambient) {
            this.lights.ambient.intensity = 0.5; // Updated ambient for classroom
        }
        
        // Position main light to illuminate the classroom
        if (this.lights.directional) {
            this.lights.directional.position.set(
                center.x + size.x * 0.5,
                center.y + size.y * 1.2,
                center.z + size.z * 0.3
            );
            this.lights.directional.intensity = 1.0; // Updated directional for classroom
        }
        
        // Add classroom-specific accent lighting
        if (this.lights.accent) {
            this.lights.accent.position.set(center.x, center.y + size.y * 0.8, center.z);
            this.lights.accent.intensity = 0.5; // Updated accent for classroom
            this.lights.accent.color.setHex(0x87ceeb); // Sky blue tint
        }
        
        console.log(`✅ Lighting optimized for classroom (bounds: ${size.x.toFixed(1)}x${size.y.toFixed(1)}x${size.z.toFixed(1)})`);
    }

    // Dynamic lighting effects
    animateLighting(deltaTime) {
        if (!this.initialized) return;
        
        // Subtle breathing effect on ambient light
        if (this.lights.ambient) {
            const time = Date.now() * 0.001;
            const breathe = Math.sin(time * 0.5) * 0.1 + 1.0;
            const baseIntensity = this.lights.ambient.userData.originalIntensity || 0.6;
            this.lights.ambient.intensity = baseIntensity * breathe;
        }
        
        // Gentle sway on accent light
        if (this.lights.accent) {
            const time = Date.now() * 0.0005;
            const sway = Math.sin(time) * 0.5;
            const baseY = this.lights.accent.userData.originalY || this.lights.accent.position.y;
            this.lights.accent.userData.originalY = baseY;
            this.lights.accent.position.y = baseY + sway;
        }
    }

    // Lighting analysis and debugging
    analyzeLighting() {
        console.log('🔍 Lighting Analysis:');
        console.log(`  Lights count: ${Object.keys(this.lights).length}`);
        
        Object.entries(this.lights).forEach(([name, light]) => {
            if (light) {
                console.log(`  ${name}: ${light.type} - Intensity: ${light.intensity.toFixed(2)}, Color: #${light.color.getHexString()}`);
                if (light.position) {
                    console.log(`    Position: (${light.position.x.toFixed(2)}, ${light.position.y.toFixed(2)}, ${light.position.z.toFixed(2)})`);
                }
            }
        });
    }

    logLightingInfo() {
        console.log('💡 Current Lighting Setup:');
        console.log(`  Ambient: ${this.lights.ambient?.intensity.toFixed(2)} intensity`);
        console.log(`  Directional: ${this.lights.directional?.intensity.toFixed(2)} intensity`);
        console.log(`  Fill: ${this.lights.fill?.intensity.toFixed(2)} intensity`);
        console.log(`  Accent: ${this.lights.accent?.intensity.toFixed(2)} intensity`);
        console.log(`  Shadows: ${this.renderer.shadowMap.enabled ? 'Enabled' : 'Disabled (VRM-safe)'}`);
    }

    // Get lighting info for external access
    getLightingInfo() {
        return {
            initialized: this.initialized,
            lightCount: Object.keys(this.lights).length,
            shadowsEnabled: this.renderer.shadowMap.enabled,
            toneMappingExposure: this.renderer.toneMappingExposure,
            lights: Object.entries(this.lights).map(([name, light]) => ({
                name,
                type: light?.type,
                intensity: light?.intensity,
                color: light?.color?.getHex(),
                position: light?.position ? {
                    x: light.position.x,
                    y: light.position.y,
                    z: light.position.z
                } : null
            }))
        };
    }

    // Cleanup
    dispose() {
        console.log('🧹 Disposing VRM lighting manager...');
        
        this.clearLights();
        this.initialized = false;
        
        console.log('✅ VRM lighting manager disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VRMLightingManager;
} else if (typeof window !== 'undefined') {
    window.VRMLightingManager = VRMLightingManager;
}
