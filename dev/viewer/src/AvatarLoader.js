/**
 * Avatar Loader - Handles VRM and GLTF avatar loading
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRM, VRMLoaderPlugin } from '@pixiv/three-vrm';

export class AvatarLoader {
    constructor() {
        this.loader = new GLTFLoader();
        this.loader.register((parser) => new VRMLoaderPlugin(parser));
        
        this.cache = new Map();
    }
    
    async loadAvatar(avatarConfig) {
        // Check cache first
        const cacheKey = avatarConfig.url || avatarConfig.id;
        if (this.cache.has(cacheKey)) {
            return this.cloneAvatar(this.cache.get(cacheKey));
        }
        
        try {
            console.log(`Loading avatar: ${avatarConfig.name || avatarConfig.url}`);
            
            const gltf = await this.loadGLTF(avatarConfig.url);
            let avatar;
            
            // Check if it's a VRM file
            if (gltf.userData.vrm) {
                avatar = await this.processVRM(gltf);
            } else {
                avatar = this.processGLTF(gltf);
            }
            
            // Cache the loaded avatar
            this.cache.set(cacheKey, avatar);
            
            // Return a clone for use
            return this.cloneAvatar(avatar);
            
        } catch (error) {
            console.error('Failed to load avatar:', error);
            throw new Error(`Avatar loading failed: ${error.message}`);
        }
    }
    
    async loadGLTF(url) {
        return new Promise((resolve, reject) => {
            this.loader.load(
                url,
                (gltf) => resolve(gltf),
                (progress) => {
                    console.log(`Loading progress: ${(progress.loaded / progress.total * 100)}%`);
                },
                (error) => reject(error)
            );
        });
    }
    
    async processVRM(gltf) {
        const vrm = gltf.userData.vrm;
        
        // VRM-specific processing
        await vrm.ready;
        
        // Set up VRM for animation
        if (vrm.humanoid) {
            this.setupVRMHumanoid(vrm);
        }
        
        // Set up VRM materials
        if (vrm.materials) {
            this.setupVRMMaterials(vrm);
        }
        
        return {
            type: 'vrm',
            scene: gltf.scene,
            vrm: vrm,
            animations: gltf.animations || [],
            mixer: new THREE.AnimationMixer(gltf.scene)
        };
    }
    
    processGLTF(gltf) {
        // Standard GLTF processing
        return {
            type: 'gltf',
            scene: gltf.scene,
            animations: gltf.animations || [],
            mixer: new THREE.AnimationMixer(gltf.scene)
        };
    }
    
    setupVRMHumanoid(vrm) {
        // Configure VRM humanoid for better animation
        const humanoid = vrm.humanoid;
        
        // Enable auto update for humanoid bones
        if (humanoid.autoUpdateHumanBones) {
            humanoid.autoUpdateHumanBones = true;
        }
        
        // Set up bone constraints if available
        if (vrm.springBoneManager) {
            vrm.springBoneManager.reset();
        }
    }
    
    setupVRMMaterials(vrm) {
        // Optimize VRM materials for real-time rendering
        vrm.scene.traverse((child) => {
            if (child.isMesh && child.material) {
                // Enable shadows
                child.castShadow = true;
                child.receiveShadow = true;
                
                // Optimize material settings
                if (child.material.map) {
                    child.material.map.flipY = false;
                }
            }
        });
    }
    
    cloneAvatar(avatarData) {
        // Create a deep clone of the avatar for independent use
        const clonedScene = avatarData.scene.clone();
        
        const clonedData = {
            ...avatarData,
            scene: clonedScene,
            mixer: new THREE.AnimationMixer(clonedScene)
        };
        
        // Clone VRM-specific data if present
        if (avatarData.vrm) {
            clonedData.vrm = this.cloneVRM(avatarData.vrm, clonedScene);
        }
        
        // Clone animations
        if (avatarData.animations && avatarData.animations.length > 0) {
            clonedData.animations = avatarData.animations.map(clip => clip.clone());
        }
        
        return clonedData;
    }
    
    cloneVRM(originalVRM, clonedScene) {
        // Create a new VRM instance for the cloned scene
        // This is a simplified clone - in practice, VRM cloning is complex
        return {
            humanoid: originalVRM.humanoid,
            meta: originalVRM.meta,
            materials: originalVRM.materials,
            springBoneManager: originalVRM.springBoneManager
        };
    }
    
    // Avatar configuration helpers
    static createAvatarConfig(name, url, options = {}) {
        return {
            name: name,
            url: url,
            scale: options.scale || 1,
            position: options.position || new THREE.Vector3(0, 0, 0),
            previewCamera: options.previewCamera || {
                position: new THREE.Vector3(0, 1.6, 3),
                target: new THREE.Vector3(0, 1, 0)
            },
            previewAnimation: options.previewAnimation || null,
            ...options
        };
    }
    
    // Helper to get available avatars from assets
    async getAvailableAvatars(assetsPath = '/assets/avatars/') {
        try {
            const response = await fetch(`${assetsPath}index.json`);
            const avatarList = await response.json();
            
            return avatarList.map(avatar => 
                AvatarLoader.createAvatarConfig(
                    avatar.name,
                    `${assetsPath}${avatar.file}`,
                    avatar.config
                )
            );
        } catch (error) {
            console.warn('Could not load avatar list, using defaults');
            return this.getDefaultAvatars();
        }
    }
    
    getDefaultAvatars() {
        // Fallback avatars if no index is available
        return [
            AvatarLoader.createAvatarConfig(
                'Default Avatar',
                '/assets/avatars/default.vrm'
            )
        ];
    }
    
    dispose() {
        this.cache.clear();
        console.log('Avatar loader disposed');
    }
}
