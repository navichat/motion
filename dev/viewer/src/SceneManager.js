/**
 * Scene Manager - Handles 3D environment and classroom setup
 */

import * as THREE from 'three';

export class SceneManager {
    constructor(scene) {
        this.scene = scene;
        this.environments = new Map();
        this.currentEnvironment = null;
        this.avatars = new Map();
        
        // Environment presets
        this.environmentPresets = {
            classroom: this.createClassroomEnvironment.bind(this),
            stage: this.createStageEnvironment.bind(this),
            studio: this.createStudioEnvironment.bind(this),
            outdoor: this.createOutdoorEnvironment.bind(this)
        };
        
        console.log('Scene Manager initialized');
    }
    
    // Load and switch to environment
    async loadEnvironment(environmentType, options = {}) {
        console.log(`Loading environment: ${environmentType}`);
        
        if (this.environments.has(environmentType)) {
            this.switchToEnvironment(environmentType);
            return;
        }
        
        let environment;
        
        if (this.environmentPresets[environmentType]) {
            environment = await this.environmentPresets[environmentType](options);
        } else {
            environment = await this.createCustomEnvironment(environmentType, options);
        }
        
        this.environments.set(environmentType, environment);
        this.switchToEnvironment(environmentType);
    }
    
    switchToEnvironment(environmentType) {
        // Remove current environment
        if (this.currentEnvironment) {
            this.removeEnvironmentFromScene(this.currentEnvironment);
        }
        
        // Add new environment
        const environment = this.environments.get(environmentType);
        if (environment) {
            this.addEnvironmentToScene(environment);
            this.currentEnvironment = environment;
            console.log(`Switched to ${environmentType} environment`);
        }
    }
    
    // Create classroom environment for educational demonstrations
    createClassroomEnvironment(options = {}) {
        const environment = {
            type: 'classroom',
            objects: [],
            lights: [],
            camera: {
                position: new THREE.Vector3(0, 2, 5),
                target: new THREE.Vector3(0, 1, 0)
            }
        };
        
        // Create floor
        const floorGeometry = new THREE.PlaneGeometry(20, 20);
        const floorMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x8B4513,
            transparent: true,
            opacity: 0.8
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        environment.objects.push(floor);
        
        // Create walls
        this.createClassroomWalls(environment);
        
        // Create whiteboard
        this.createWhiteboard(environment);
        
        // Create desks and chairs
        this.createClassroomFurniture(environment, options);
        
        // Create lighting specific to classroom
        this.createClassroomLighting(environment);
        
        return environment;
    }
    
    createClassroomWalls(environment) {
        const wallMaterial = new THREE.MeshLambertMaterial({ color: 0xF5F5DC });
        
        // Back wall
        const backWallGeometry = new THREE.PlaneGeometry(20, 8);
        const backWall = new THREE.Mesh(backWallGeometry, wallMaterial);
        backWall.position.set(0, 4, -10);
        environment.objects.push(backWall);
        
        // Side walls
        const sideWallGeometry = new THREE.PlaneGeometry(20, 8);
        
        const leftWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
        leftWall.rotation.y = Math.PI / 2;
        leftWall.position.set(-10, 4, 0);
        environment.objects.push(leftWall);
        
        const rightWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
        rightWall.rotation.y = -Math.PI / 2;
        rightWall.position.set(10, 4, 0);
        environment.objects.push(rightWall);
    }
    
    createWhiteboard(environment) {
        // Whiteboard frame
        const frameGeometry = new THREE.BoxGeometry(6, 3, 0.1);
        const frameMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
        const frame = new THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.set(0, 3, -9.9);
        environment.objects.push(frame);
        
        // Whiteboard surface
        const boardGeometry = new THREE.PlaneGeometry(5.8, 2.8);
        const boardMaterial = new THREE.MeshLambertMaterial({ color: 0xFFFFFF });
        const board = new THREE.Mesh(boardGeometry, boardMaterial);
        board.position.set(0, 3, -9.8);
        environment.objects.push(board);
    }
    
    createClassroomFurniture(environment, options) {
        const deskCount = options.deskCount || 12;
        const rows = options.rows || 3;
        const desksPerRow = Math.ceil(deskCount / rows);
        
        for (let row = 0; row < rows; row++) {
            for (let desk = 0; desk < desksPerRow && (row * desksPerRow + desk) < deskCount; desk++) {
                this.createDeskAndChair(environment, {
                    x: (desk - desksPerRow / 2) * 3 + 1.5,
                    z: row * 3 - 2,
                    y: 0
                });
            }
        }
    }
    
    createDeskAndChair(environment, position) {
        // Simple desk
        const deskGeometry = new THREE.BoxGeometry(1.2, 0.1, 0.8);
        const deskMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
        const desk = new THREE.Mesh(deskGeometry, deskMaterial);
        desk.position.set(position.x, position.y + 0.75, position.z);
        desk.castShadow = true;
        environment.objects.push(desk);
        
        // Desk legs
        const legGeometry = new THREE.BoxGeometry(0.05, 0.7, 0.05);
        const legMaterial = new THREE.MeshLambertMaterial({ color: 0x666666 });
        
        const legPositions = [
            { x: -0.5, z: -0.35 },
            { x: 0.5, z: -0.35 },
            { x: -0.5, z: 0.35 },
            { x: 0.5, z: 0.35 }
        ];
        
        legPositions.forEach(legPos => {
            const leg = new THREE.Mesh(legGeometry, legMaterial);
            leg.position.set(
                position.x + legPos.x,
                position.y + 0.35,
                position.z + legPos.z
            );
            leg.castShadow = true;
            environment.objects.push(leg);
        });
        
        // Simple chair
        const chairSeatGeometry = new THREE.BoxGeometry(0.4, 0.05, 0.4);
        const chairMaterial = new THREE.MeshLambertMaterial({ color: 0x654321 });
        const chairSeat = new THREE.Mesh(chairSeatGeometry, chairMaterial);
        chairSeat.position.set(position.x, position.y + 0.45, position.z + 0.7);
        chairSeat.castShadow = true;
        environment.objects.push(chairSeat);
        
        // Chair back
        const chairBackGeometry = new THREE.BoxGeometry(0.4, 0.5, 0.05);
        const chairBack = new THREE.Mesh(chairBackGeometry, chairMaterial);
        chairBack.position.set(position.x, position.y + 0.7, position.z + 0.9);
        chairBack.castShadow = true;
        environment.objects.push(chairBack);
    }
    
    createClassroomLighting(environment) {
        // Overhead fluorescent-style lighting
        const fluorescent1 = new THREE.DirectionalLight(0xFFFFFF, 0.6);
        fluorescent1.position.set(-5, 7, 0);
        fluorescent1.target.position.set(-5, 0, 0);
        fluorescent1.castShadow = true;
        environment.lights.push(fluorescent1);
        
        const fluorescent2 = new THREE.DirectionalLight(0xFFFFFF, 0.6);
        fluorescent2.position.set(5, 7, 0);
        fluorescent2.target.position.set(5, 0, 0);
        fluorescent2.castShadow = true;
        environment.lights.push(fluorescent2);
        
        // Soft ambient light
        const ambient = new THREE.AmbientLight(0x404040, 0.3);
        environment.lights.push(ambient);
    }
    
    // Create stage environment for performances
    createStageEnvironment(options = {}) {
        const environment = {
            type: 'stage',
            objects: [],
            lights: [],
            camera: {
                position: new THREE.Vector3(0, 2, 8),
                target: new THREE.Vector3(0, 1, 0)
            }
        };
        
        // Stage platform
        const stageGeometry = new THREE.BoxGeometry(10, 0.5, 6);
        const stageMaterial = new THREE.MeshLambertMaterial({ color: 0x2F4F4F });
        const stage = new THREE.Mesh(stageGeometry, stageMaterial);
        stage.position.set(0, 0.25, -2);
        stage.castShadow = true;
        stage.receiveShadow = true;
        environment.objects.push(stage);
        
        // Curtains
        this.createStageCurtains(environment);
        
        // Stage lighting
        this.createStageLighting(environment);
        
        return environment;
    }
    
    createStageCurtains(environment) {
        const curtainMaterial = new THREE.MeshLambertMaterial({ color: 0x8B0000 });
        
        // Back curtain
        const backCurtainGeometry = new THREE.PlaneGeometry(12, 8);
        const backCurtain = new THREE.Mesh(backCurtainGeometry, curtainMaterial);
        backCurtain.position.set(0, 4, -5);
        environment.objects.push(backCurtain);
        
        // Side curtains
        const sideCurtainGeometry = new THREE.PlaneGeometry(6, 8);
        
        const leftCurtain = new THREE.Mesh(sideCurtainGeometry, curtainMaterial);
        leftCurtain.rotation.y = Math.PI / 2;
        leftCurtain.position.set(-6, 4, -2);
        environment.objects.push(leftCurtain);
        
        const rightCurtain = new THREE.Mesh(sideCurtainGeometry, curtainMaterial);
        rightCurtain.rotation.y = -Math.PI / 2;
        rightCurtain.position.set(6, 4, -2);
        environment.objects.push(rightCurtain);
    }
    
    createStageLighting(environment) {
        // Spotlight on center stage
        const spotlight = new THREE.SpotLight(0xFFFFFF, 1.0);
        spotlight.position.set(0, 10, 2);
        spotlight.target.position.set(0, 0.5, -2);
        spotlight.angle = Math.PI / 6;
        spotlight.penumbra = 0.1;
        spotlight.castShadow = true;
        environment.lights.push(spotlight);
        
        // Side lights
        const sideLight1 = new THREE.SpotLight(0xFF8C00, 0.5);
        sideLight1.position.set(-8, 6, 0);
        sideLight1.target.position.set(-2, 0.5, -2);
        environment.lights.push(sideLight1);
        
        const sideLight2 = new THREE.SpotLight(0x00CED1, 0.5);
        sideLight2.position.set(8, 6, 0);
        sideLight2.target.position.set(2, 0.5, -2);
        environment.lights.push(sideLight2);
        
        // Dim ambient
        const ambient = new THREE.AmbientLight(0x202020, 0.2);
        environment.lights.push(ambient);
    }
    
    // Create studio environment for motion capture style
    createStudioEnvironment(options = {}) {
        const environment = {
            type: 'studio',
            objects: [],
            lights: [],
            camera: {
                position: new THREE.Vector3(3, 2, 3),
                target: new THREE.Vector3(0, 1, 0)
            }
        };
        
        // Simple grid floor
        const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0x444444);
        environment.objects.push(gridHelper);
        
        // Clean lighting
        const mainLight = new THREE.DirectionalLight(0xFFFFFF, 0.8);
        mainLight.position.set(5, 10, 5);
        mainLight.castShadow = true;
        environment.lights.push(mainLight);
        
        const fillLight = new THREE.DirectionalLight(0xFFFFFF, 0.4);
        fillLight.position.set(-5, 5, -5);
        environment.lights.push(fillLight);
        
        const ambient = new THREE.AmbientLight(0x404040, 0.3);
        environment.lights.push(ambient);
        
        return environment;
    }
    
    // Create outdoor environment
    createOutdoorEnvironment(options = {}) {
        const environment = {
            type: 'outdoor',
            objects: [],
            lights: [],
            camera: {
                position: new THREE.Vector3(0, 2, 5),
                target: new THREE.Vector3(0, 1, 0)
            }
        };
        
        // Ground plane with grass texture
        const groundGeometry = new THREE.PlaneGeometry(50, 50);
        const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x228B22 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        environment.objects.push(ground);
        
        // Sky
        const skyGeometry = new THREE.SphereGeometry(100, 32, 32);
        const skyMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x87CEEB,
            side: THREE.BackSide
        });
        const sky = new THREE.Mesh(skyGeometry, skyMaterial);
        environment.objects.push(sky);
        
        // Sun lighting
        const sunLight = new THREE.DirectionalLight(0xFFFFAA, 1.0);
        sunLight.position.set(10, 20, 10);
        sunLight.castShadow = true;
        environment.lights.push(sunLight);
        
        const skyLight = new THREE.AmbientLight(0x404080, 0.4);
        environment.lights.push(skyLight);
        
        return environment;
    }
    
    // Load custom environment from file
    async createCustomEnvironment(environmentType, options) {
        try {
            const response = await fetch(`/assets/scenes/${environmentType}/scene.json`);
            const sceneData = await response.json();
            
            return this.buildEnvironmentFromData(sceneData);
        } catch (error) {
            console.warn(`Could not load custom environment ${environmentType}, using studio`);
            return this.createStudioEnvironment(options);
        }
    }
    
    buildEnvironmentFromData(sceneData) {
        // Build environment from JSON scene description
        const environment = {
            type: sceneData.type || 'custom',
            objects: [],
            lights: [],
            camera: sceneData.camera || {
                position: new THREE.Vector3(0, 2, 5),
                target: new THREE.Vector3(0, 1, 0)
            }
        };
        
        // Process objects
        if (sceneData.objects) {
            sceneData.objects.forEach(objData => {
                const object = this.createObjectFromData(objData);
                if (object) {
                    environment.objects.push(object);
                }
            });
        }
        
        // Process lights
        if (sceneData.lights) {
            sceneData.lights.forEach(lightData => {
                const light = this.createLightFromData(lightData);
                if (light) {
                    environment.lights.push(light);
                }
            });
        }
        
        return environment;
    }
    
    createObjectFromData(objData) {
        // Create Three.js objects from scene data
        // This would be expanded based on scene format requirements
        console.log('Creating object from data:', objData);
        return null; // Placeholder
    }
    
    createLightFromData(lightData) {
        // Create Three.js lights from scene data
        console.log('Creating light from data:', lightData);
        return null; // Placeholder
    }
    
    // Environment management
    addEnvironmentToScene(environment) {
        environment.objects.forEach(obj => this.scene.add(obj));
        environment.lights.forEach(light => this.scene.add(light));
    }
    
    removeEnvironmentFromScene(environment) {
        environment.objects.forEach(obj => this.scene.remove(obj));
        environment.lights.forEach(light => this.scene.remove(light));
    }
    
    // Avatar management in scene
    addAvatar(id, avatarInstance, position = new THREE.Vector3(0, 0, 0)) {
        if (this.avatars.has(id)) {
            this.removeAvatar(id);
        }
        
        avatarInstance.scene.position.copy(position);
        this.scene.add(avatarInstance.scene);
        this.avatars.set(id, avatarInstance);
        
        console.log(`Added avatar ${id} to scene`);
    }
    
    removeAvatar(id) {
        const avatarInstance = this.avatars.get(id);
        if (avatarInstance) {
            this.scene.remove(avatarInstance.scene);
            this.avatars.delete(id);
            console.log(`Removed avatar ${id} from scene`);
        }
    }
    
    getAvatar(id) {
        return this.avatars.get(id);
    }
    
    getAllAvatars() {
        return Array.from(this.avatars.values());
    }
    
    // Get available environments
    getAvailableEnvironments() {
        return Object.keys(this.environmentPresets);
    }
    
    getCurrentEnvironment() {
        return this.currentEnvironment;
    }
    
    dispose() {
        // Clean up all environments
        this.environments.forEach(env => {
            this.removeEnvironmentFromScene(env);
        });
        this.environments.clear();
        
        // Clean up avatars
        this.avatars.forEach((avatar, id) => {
            this.removeAvatar(id);
        });
        
        console.log('Scene Manager disposed');
    }
}
