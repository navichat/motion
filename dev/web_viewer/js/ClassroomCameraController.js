/**
 * Classroom Camera Controller - Advanced camera management for classroom environment
 * Provides cinematic views and smooth transitions
 */

class ClassroomCameraController {
    constructor(camera, characterSystem, controls) {
        this.camera = camera;
        this.characterSystem = characterSystem;
        this.controls = controls;
        this.cameraMode = 'free';
        
        // Get THREE from global window object
        this.THREE = window.THREE;
        if (!this.THREE) {
            console.warn('⚠️ THREE.js not available in ClassroomCameraController');
        }
        
        // Camera settings
        this.followDistance = 3;
        this.followHeight = 1.5;
        this.smoothness = 0.05;
        this.isTransitioning = false;
        this.transitionDuration = 1000;
        
        // Preset camera positions
        this.presetPositions = {
            'free': { position: [0, 2, 5], target: [0, 1, 0] },
            'follow': { position: [0, 1.5, 3], target: [0, 1, 0] },
            'front': { position: [0, 1.6, 2.5], target: [0, 1, 0] },
            'side': { position: [2.5, 1.6, 0], target: [0, 1, 0] },
            'classroom': { position: [-3, 2.5, 4], target: [0, 1, 0] },
            'overhead': { position: [0, 5, 0], target: [0, 0, 0] },
            'close': { position: [0, 1.8, 1.5], target: [0, 1.7, 0] },
            'dramatic': { position: [-2, 1.2, 2], target: [0, 1.5, 0] }
        };
        
        // Animation state
        this.currentTarget = new this.THREE.Vector3();
        this.currentPosition = new this.THREE.Vector3();
        this.targetPosition = new this.THREE.Vector3();
        this.targetLookAt = new this.THREE.Vector3();
        
        console.log('📹 ClassroomCameraController initialized');
    }

    setCameraMode(mode) {
        if (this.cameraMode === mode) return;
        
        console.log(`📹 Camera mode: ${this.cameraMode} → ${mode}`);
        this.cameraMode = mode;
        
        const preset = this.presetPositions[mode];
        if (preset) {
            this.transitionToPosition(
                new this.THREE.Vector3(...preset.position),
                new this.THREE.Vector3(...preset.target)
            );
        }
    }

    transitionToPosition(position, target, duration = this.transitionDuration) {
        this.isTransitioning = true;
        
        // Store current position
        this.currentPosition.copy(this.camera.position);
        this.currentTarget.copy(this.getCurrentLookAt());
        
        // Set target position
        this.targetPosition.copy(position);
        this.targetLookAt.copy(target);
        
        // Animate transition
        const startTime = Date.now();
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-in-out)
            const eased = progress < 0.5 
                ? 2 * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;
            
            // Interpolate position
            this.camera.position.lerpVectors(this.currentPosition, this.targetPosition, eased);
            
            // Interpolate look-at
            const lookAt = new this.THREE.Vector3();
            lookAt.lerpVectors(this.currentTarget, this.targetLookAt, eased);
            this.camera.lookAt(lookAt);
            
            // Update controls target if available
            if (this.controls && this.controls.target) {
                this.controls.target.copy(lookAt);
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.isTransitioning = false;
                console.log('📹 Camera transition complete');
            }
        };
        
        animate();
    }

    getCurrentLookAt() {
        // Calculate current look-at point
        const direction = new this.THREE.Vector3();
        this.camera.getWorldDirection(direction);
        const lookAt = new this.THREE.Vector3();
        lookAt.copy(this.camera.position).add(direction);
        return lookAt;
    }

    update() {
        if (this.isTransitioning) return;
        
        switch (this.cameraMode) {
            case 'follow':
                this.updateFollowMode();
                break;
            case 'orbit':
                this.updateOrbitMode();
                break;
            case 'cinematic':
                this.updateCinematicMode();
                break;
        }
    }

    updateFollowMode() {
        if (!this.characterSystem.vrmModel) return;
        
        const characterPos = this.characterSystem.vrmModel.scene.position;
        const targetPos = new this.THREE.Vector3(
            characterPos.x,
            characterPos.y + this.followHeight,
            characterPos.z + this.followDistance
        );
        
        // Smooth follow
        this.camera.position.lerp(targetPos, this.smoothness);
        
        // Look at character
        const lookAtPos = new this.THREE.Vector3(
            characterPos.x,
            characterPos.y + 1,
            characterPos.z
        );
        
        // Smooth look-at
        const currentLookAt = this.getCurrentLookAt();
        currentLookAt.lerp(lookAtPos, this.smoothness);
        this.camera.lookAt(currentLookAt);
    }

    updateOrbitMode() {
        if (!this.characterSystem.vrmModel) return;
        
        const time = Date.now() * 0.001;
        const characterPos = this.characterSystem.vrmModel.scene.position;
        
        // Orbit around character
        const radius = 3;
        const height = 1.5;
        
        const x = characterPos.x + Math.cos(time * 0.2) * radius;
        const z = characterPos.z + Math.sin(time * 0.2) * radius;
        const y = characterPos.y + height;
        
        this.camera.position.set(x, y, z);
        this.camera.lookAt(characterPos.x, characterPos.y + 1, characterPos.z);
    }

    updateCinematicMode() {
        // Cinematic camera movement
        const time = Date.now() * 0.001;
        
        // Subtle camera shake/movement
        const shake = {
            x: Math.sin(time * 0.5) * 0.02,
            y: Math.cos(time * 0.3) * 0.01,
            z: Math.sin(time * 0.7) * 0.01
        };
        
        this.camera.position.add(new this.THREE.Vector3(shake.x, shake.y, shake.z));
    }

    // Preset camera movements
    performDollyZoom(targetFOV, duration = 2000) {
        const startFOV = this.camera.fov;
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Interpolate FOV
            this.camera.fov = startFOV + (targetFOV - startFOV) * progress;
            this.camera.updateProjectionMatrix();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    performSlowPan(direction, speed = 0.5, duration = 3000) {
        const startPosition = this.camera.position.clone();
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Pan camera
            const offset = new this.THREE.Vector3(
                direction.x * speed * progress,
                direction.y * speed * progress,
                direction.z * speed * progress
            );
            
            this.camera.position.copy(startPosition).add(offset);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    // Focus on specific body parts
    focusOnFace() {
        if (!this.characterSystem.vrmModel) return;
        
        const headBone = this.characterSystem.vrmModel.humanoid?.getNormalizedBoneNode('head');
        if (headBone) {
            const headPos = new this.THREE.Vector3();
            headBone.getWorldPosition(headPos);
            
            const cameraPos = headPos.clone().add(new this.THREE.Vector3(0, 0.2, 0.5));
            this.transitionToPosition(cameraPos, headPos);
        }
    }

    focusOnHands() {
        if (!this.characterSystem.vrmModel) return;
        
        const leftHand = this.characterSystem.vrmModel.humanoid?.getRawBoneNode('leftHand');
        const rightHand = this.characterSystem.vrmModel.humanoid?.getRawBoneNode('rightHand');
        
        if (leftHand && rightHand) {
            const leftPos = new this.THREE.Vector3();
            const rightPos = new this.THREE.Vector3();
            leftHand.getWorldPosition(leftPos);
            rightHand.getWorldPosition(rightPos);
            
            const centerPos = leftPos.clone().add(rightPos).multiplyScalar(0.5);
            const cameraPos = centerPos.clone().add(new this.THREE.Vector3(0, 0.3, 0.8));
            
            this.transitionToPosition(cameraPos, centerPos);
        }
    }

    // Environmental focus
    focusOnClassroom() {
        const classroomCenter = new this.THREE.Vector3(0, 1, 0);
        const cameraPos = new this.THREE.Vector3(-3, 2.5, 4);
        this.transitionToPosition(cameraPos, classroomCenter);
    }

    focusOnWhiteboard() {
        const boardPos = new this.THREE.Vector3(0, 2, -4.9);
        const cameraPos = new this.THREE.Vector3(0, 2, -2);
        this.transitionToPosition(cameraPos, boardPos);
    }

    // Camera effects
    shake(intensity = 0.1, duration = 500) {
        const startTime = Date.now();
        const originalPos = this.camera.position.clone();
        
        const shakeLoop = () => {
            const elapsed = Date.now() - startTime;
            const progress = elapsed / duration;
            
            if (progress < 1) {
                const shakeX = (Math.random() - 0.5) * intensity;
                const shakeY = (Math.random() - 0.5) * intensity;
                const shakeZ = (Math.random() - 0.5) * intensity;
                
                this.camera.position.copy(originalPos).add(new this.THREE.Vector3(shakeX, shakeY, shakeZ));
                requestAnimationFrame(shakeLoop);
            } else {
                this.camera.position.copy(originalPos);
            }
        };
        
        shakeLoop();
    }

    // Utility methods
    saveCurrentView() {
        const currentView = {
            position: this.camera.position.clone(),
            target: this.getCurrentLookAt(),
            fov: this.camera.fov
        };
        
        localStorage.setItem('savedCameraView', JSON.stringify({
            position: currentView.position.toArray(),
            target: currentView.target.toArray(),
            fov: currentView.fov
        }));
        
        console.log('📸 Camera view saved');
    }

    loadSavedView() {
        const saved = localStorage.getItem('savedCameraView');
        if (saved) {
            const data = JSON.parse(saved);
            this.transitionToPosition(
                new this.THREE.Vector3(...data.position),
                new this.THREE.Vector3(...data.target)
            );
            this.camera.fov = data.fov;
            this.camera.updateProjectionMatrix();
            console.log('📸 Camera view loaded');
        }
    }

    getCameraInfo() {
        return {
            mode: this.cameraMode,
            position: this.camera.position.toArray(),
            target: this.getCurrentLookAt().toArray(),
            fov: this.camera.fov,
            isTransitioning: this.isTransitioning
        };
    }

    dispose() {
        this.camera = null;
        this.characterSystem = null;
        this.controls = null;
        
        console.log('🗑️ Camera controller disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClassroomCameraController;
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.ClassroomCameraController = ClassroomCameraController;
}
