/**
 * Core 3D Player for Motion Viewer
 * Extracted and adapted from @navi/player to work independently
 */

import * as THREE from 'three';

export class Player {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas,
            antialias: true,
            alpha: true 
        });
        
        this.clock = new THREE.Clock();
        this.mixers = [];
        this.eventListeners = new Map();
        
        this.setupLighting();
        this.setupCamera();
        this.startRenderLoop();
        
        console.log('Motion Viewer Player initialized');
    }
    
    setupLighting() {
        // Ambient light for overall illumination
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Main directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Fill light from opposite side
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 3, -5);
        this.scene.add(fillLight);
    }
    
    setupCamera() {
        this.camera.position.set(0, 1.6, 3);
        this.camera.lookAt(0, 1, 0);
        
        // Create camera driver for smooth movement
        this.cameraDriver = new FreeCameraDriver(this.camera);
    }
    
    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            
            const delta = this.clock.getDelta();
            
            // Update animation mixers
            this.mixers.forEach(mixer => mixer.update(delta));
            
            // Update camera driver
            this.cameraDriver.update(delta);
            
            // Emit tick event for external listeners
            this.emit('tick', delta);
            
            // Render the scene
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    async loadAvatar(avatarConfig) {
        const loader = new THREE.GLTFLoader();
        
        try {
            const gltf = await new Promise((resolve, reject) => {
                loader.load(avatarConfig.url, resolve, undefined, reject);
            });
            
            const avatar = gltf.scene;
            this.scene.add(avatar);
            
            // Create avatar instance with animation capabilities
            const instance = new AvatarInstance(avatar, gltf);
            
            // Set up initial position and scale
            if (avatarConfig.position) {
                avatar.position.copy(avatarConfig.position);
            }
            if (avatarConfig.scale) {
                avatar.scale.setScalar(avatarConfig.scale);
            }
            
            return instance;
            
        } catch (error) {
            console.error('Failed to load avatar:', error);
            throw error;
        }
    }
    
    loadMotion(motionData) {
        // Handle different motion formats (JSON, BVH, etc.)
        if (motionData.format === 'bvh') {
            return this.loadBVHMotion(motionData);
        } else if (motionData.format === 'json') {
            return this.loadJSONMotion(motionData);
        } else {
            throw new Error(`Unsupported motion format: ${motionData.format}`);
        }
    }
    
    loadBVHMotion(bvhData) {
        // BVH motion loading logic
        // This will be integrated with RSMT's BVH parser
        console.log('Loading BVH motion data...');
        // Implementation will be added in Phase 2
    }
    
    loadJSONMotion(jsonData) {
        // JSON motion loading (from chat interface)
        console.log('Loading JSON motion data...');
        
        const clip = new THREE.AnimationClip(jsonData.name, jsonData.duration, jsonData.tracks);
        return clip;
    }
    
    handleCanvasResize() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        const pixelRatio = window.devicePixelRatio;
        
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(pixelRatio);
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }
    
    // Event system
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.eventListeners.has(event)) {
            const listeners = this.eventListeners.get(event);
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }
    
    emit(event, ...args) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                callback(...args);
            });
        }
    }
    
    dispose() {
        // Clean up resources
        this.renderer.dispose();
        this.scene.clear();
        this.eventListeners.clear();
        console.log('Player disposed');
    }
}

export class FreeCameraDriver {
    constructor(camera) {
        this.camera = camera;
        this.target = new THREE.Vector3();
        this.position = new THREE.Vector3();
        this.smoothing = 0.1;
    }
    
    setView(viewConfig) {
        if (viewConfig.position) {
            this.position.copy(viewConfig.position);
        }
        if (viewConfig.target) {
            this.target.copy(viewConfig.target);
        }
    }
    
    update(delta) {
        // Smooth camera movement
        this.camera.position.lerp(this.position, this.smoothing);
        this.camera.lookAt(this.target);
    }
}

export class AvatarFacingCameraDriver extends FreeCameraDriver {
    constructor(camera) {
        super(camera);
        this.radius = 3;
        this.height = 1.6;
        this.angle = 0;
    }
    
    update(delta) {
        // Orbit around the avatar while facing it
        this.angle += delta * 0.1; // Slow rotation
        
        this.position.x = Math.cos(this.angle) * this.radius;
        this.position.y = this.height;
        this.position.z = Math.sin(this.angle) * this.radius;
        
        super.update(delta);
    }
}

export class AvatarInstance {
    constructor(avatar, gltf) {
        this.avatar = avatar;
        this.gltf = gltf;
        this.mixer = null;
        this.currentAnimation = null;
        this.idleAnimations = null;
        
        if (gltf.animations && gltf.animations.length > 0) {
            this.mixer = new THREE.AnimationMixer(avatar);
        }
    }
    
    setIdleAnimations(config) {
        this.idleAnimations = config;
        
        if (this.mixer && config.animations && config.animations.length > 0) {
            const animationClip = config.animations[0]; // Use first animation as idle
            const action = this.mixer.clipAction(animationClip);
            action.setLoop(THREE.LoopRepeat);
            action.play();
            this.currentAnimation = action;
        }
    }
    
    playAnimation(animationClip, loop = false) {
        if (!this.mixer) return;
        
        // Stop current animation
        if (this.currentAnimation) {
            this.currentAnimation.stop();
        }
        
        // Play new animation
        const action = this.mixer.clipAction(animationClip);
        action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
        action.reset();
        action.play();
        
        this.currentAnimation = action;
    }
    
    lookAtCameraAsIfHuman(camera) {
        // Orient avatar to face camera naturally
        const direction = new THREE.Vector3();
        direction.subVectors(camera.position, this.avatar.position);
        direction.y = 0; // Keep upright
        direction.normalize();
        
        this.avatar.lookAt(
            this.avatar.position.x + direction.x,
            this.avatar.position.y,
            this.avatar.position.z + direction.z
        );
    }
    
    tick(delta) {
        // Update animation mixer if present
        if (this.mixer) {
            this.mixer.update(delta);
        }
    }
    
    dispose() {
        if (this.mixer) {
            this.mixer.stopAllAction();
        }
        if (this.avatar.parent) {
            this.avatar.parent.remove(this.avatar);
        }
        console.log('Avatar instance disposed');
    }
}
