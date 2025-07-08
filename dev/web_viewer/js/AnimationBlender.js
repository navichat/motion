/**
 * Animation Blender - Smoothly blends between BVH and character animations
 * Provides seamless transitions and idle pose blending
 */

class AnimationBlender {
    constructor(characterSystem) {
        this.characterSystem = characterSystem;
        this.blendFactor = 1.0; // 1.0 = full BVH, 0.0 = full idle
        this.targetBlendFactor = 1.0;
        this.blendSpeed = 0.02;
        this.idlePose = null;
        this.isBlending = false;
        this.blendingTimeout = null;
        
        console.log('🎨 AnimationBlender initialized');
        
        // Initialize idle pose
        this.initializeIdlePose();
    }

    initializeIdlePose() {
        // Create a neutral idle pose for the character
        this.idlePose = {
            // Root position and rotation
            rootPosition: [0, 0, 0],
            rootRotation: [0, 0, 0],
            
            // Joint rotations (in degrees)
            jointRotations: [
                // Hips
                0, 0, 0,
                // Spine
                0, 0, 0,
                // Chest
                0, 0, 0,
                // Neck
                0, 0, 0,
                // Head
                0, 0, 0,
                // Left shoulder
                0, 0, -10,
                // Left arm
                0, 0, -15,
                // Left forearm
                0, 0, -20,
                // Left hand
                0, 0, 0,
                // Right shoulder
                0, 0, 10,
                // Right arm
                0, 0, 15,
                // Right forearm
                0, 0, 20,
                // Right hand
                0, 0, 0,
                // Left hip
                0, 0, 0,
                // Left leg
                0, 0, 0,
                // Left foot
                0, 0, 0,
                // Right hip
                0, 0, 0,
                // Right leg
                0, 0, 0,
                // Right foot
                0, 0, 0
            ]
        };
        
        console.log('🧘 Idle pose initialized');
    }

    setBlendFactor(factor) {
        this.targetBlendFactor = Math.max(0, Math.min(1, factor));
        this.isBlending = true;
        
        // Clear any existing timeout
        if (this.blendingTimeout) {
            clearTimeout(this.blendingTimeout);
        }
        
        // Set timeout to finish blending
        this.blendingTimeout = setTimeout(() => {
            this.isBlending = false;
        }, 2000);
        
        console.log(`🎨 Blend factor target set to: ${this.targetBlendFactor.toFixed(2)}`);
    }

    blendToIdle(duration = 1000) {
        console.log('🧘 Blending to idle pose...');
        this.setBlendFactor(0.0);
        
        // Automatically return to BVH after duration
        setTimeout(() => {
            this.setBlendFactor(1.0);
        }, duration);
    }

    blendToBVH() {
        console.log('🎯 Blending to BVH animation...');
        this.setBlendFactor(1.0);
    }

    blendPoses(bvhFrame, idleFrame) {
        if (!bvhFrame || !idleFrame) return bvhFrame;
        
        const blended = new Float32Array(bvhFrame.length);
        
        // Blend root position
        for (let i = 0; i < 3; i++) {
            const bvhVal = bvhFrame[i] || 0;
            const idleVal = idleFrame.rootPosition[i] || 0;
            blended[i] = bvhVal * this.blendFactor + idleVal * (1 - this.blendFactor);
        }
        
        // Blend root rotation
        for (let i = 3; i < 6; i++) {
            const bvhVal = bvhFrame[i] || 0;
            const idleVal = idleFrame.rootRotation[i - 3] || 0;
            blended[i] = bvhVal * this.blendFactor + idleVal * (1 - this.blendFactor);
        }
        
        // Blend joint rotations
        for (let i = 6; i < bvhFrame.length; i++) {
            const bvhVal = bvhFrame[i] || 0;
            const idleIdx = i - 6;
            const idleVal = idleFrame.jointRotations[idleIdx] || 0;
            blended[i] = bvhVal * this.blendFactor + idleVal * (1 - this.blendFactor);
        }
        
        return blended;
    }

    update() {
        // Smooth blend factor transition
        if (this.isBlending) {
            const diff = this.targetBlendFactor - this.blendFactor;
            if (Math.abs(diff) > 0.001) {
                this.blendFactor += diff * this.blendSpeed;
            } else {
                this.blendFactor = this.targetBlendFactor;
                this.isBlending = false;
            }
        }
    }

    processFrame(frameData) {
        if (!frameData || this.blendFactor >= 0.99) {
            return frameData; // No blending needed
        }
        
        if (this.blendFactor <= 0.01) {
            // Pure idle pose
            return this.createIdleFrame(frameData.length);
        }
        
        // Blend between BVH and idle
        return this.blendPoses(frameData, this.idlePose);
    }

    createIdleFrame(length) {
        const frame = new Float32Array(length);
        
        // Root position
        for (let i = 0; i < 3; i++) {
            frame[i] = this.idlePose.rootPosition[i];
        }
        
        // Root rotation
        for (let i = 3; i < 6; i++) {
            frame[i] = this.idlePose.rootRotation[i - 3];
        }
        
        // Joint rotations
        for (let i = 6; i < length; i++) {
            const idleIdx = i - 6;
            frame[i] = this.idlePose.jointRotations[idleIdx] || 0;
        }
        
        return frame;
    }

    getBlendFactor() {
        return this.blendFactor;
    }

    isIdleMode() {
        return this.blendFactor < 0.1;
    }

    isBVHMode() {
        return this.blendFactor > 0.9;
    }

    // Preset blend animations
    performBreathing() {
        const originalBlend = this.blendFactor;
        
        // Subtle breathing animation
        const breathingCycle = () => {
            if (this.isIdleMode()) {
                // Modify idle pose for breathing
                this.idlePose.jointRotations[6] = Math.sin(Date.now() * 0.003) * 2; // Chest
                this.idlePose.jointRotations[9] = Math.sin(Date.now() * 0.003) * 1; // Neck
            }
        };
        
        return breathingCycle;
    }

    performSubtleMovement() {
        const originalBlend = this.blendFactor;
        
        // Subtle weight shifting
        const subtleMovement = () => {
            if (this.isIdleMode()) {
                const time = Date.now() * 0.001;
                this.idlePose.rootPosition[0] = Math.sin(time * 0.5) * 0.1;
                this.idlePose.rootRotation[1] = Math.sin(time * 0.3) * 2;
            }
        };
        
        return subtleMovement;
    }

    createCustomIdlePose(poseConfig) {
        if (!poseConfig) return;
        
        // Allow custom idle pose configuration
        if (poseConfig.rootPosition) {
            this.idlePose.rootPosition = [...poseConfig.rootPosition];
        }
        
        if (poseConfig.rootRotation) {
            this.idlePose.rootRotation = [...poseConfig.rootRotation];
        }
        
        if (poseConfig.jointRotations) {
            this.idlePose.jointRotations = [...poseConfig.jointRotations];
        }
        
        console.log('🎭 Custom idle pose created');
    }

    // Animation state management
    saveCurrentPoseAsIdle() {
        if (this.characterSystem.vrmModel) {
            // Extract current pose from VRM model
            const currentPose = this.extractCurrentPose();
            if (currentPose) {
                this.idlePose = currentPose;
                console.log('💾 Current pose saved as idle');
            }
        }
    }

    extractCurrentPose() {
        // Extract current pose from VRM model
        // This is a placeholder - would need actual VRM bone extraction
        return null;
    }

    dispose() {
        if (this.blendingTimeout) {
            clearTimeout(this.blendingTimeout);
            this.blendingTimeout = null;
        }
        
        this.characterSystem = null;
        this.idlePose = null;
        
        console.log('🗑️ Animation blender disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationBlender;
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.AnimationBlender = AnimationBlender;
}
