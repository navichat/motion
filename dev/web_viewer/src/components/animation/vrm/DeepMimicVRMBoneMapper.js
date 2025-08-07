/**
 * DeepMimic to VRM/BVH Bone Mapping System
 * 
 * Maps DeepMimic's simplified 15-joint hierarchy to our comprehensive 34+ bone VRM/BVH system.
 * This allows us to use DeepMimic physics-based animation while maintaining full VRM compatibility.
 */

class DeepMimicVRMBoneMapper {
    constructor(options = {}) {
        this.scaleFactor = options.scaleFactor || 1.0;
        this.interpolationSmoothing = options.interpolationSmoothing || 0.8;
        this.physicsInfluence = options.physicsInfluence || 1.0;
        
        // Initialize bone mapping tables
        this.deepMimicToVRM = this.createDeepMimicToVRMMapping();
        this.vrmToDeepMimic = this.createVRMToDeepMimicMapping();
        this.boneConstraints = this.createBoneConstraints();
        this.interpolationRules = this.createInterpolationRules();
        
        console.log('[DeepMimic VRM Mapper] Initialized with', Object.keys(this.deepMimicToVRM).length, 'mappings');
    }
    
    /**
     * Create mapping from DeepMimic joints to VRM bones
     */
    createDeepMimicToVRMMapping() {
        return {
            // Root mapping
            'root': {
                primary: 'hips',
                secondary: [],
                influence: 1.0,
                type: 'direct'
            },
            
            // Torso mapping
            'chest': {
                primary: 'spine2',
                secondary: ['spine', 'spine1'],
                influence: 1.0,
                type: 'hierarchical',
                distribution: [0.3, 0.4, 0.3] // spine, spine1, spine2
            },
            
            // Head/neck mapping
            'neck': {
                primary: 'neck',
                secondary: ['head'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.7, 0.3] // neck, head
            },
            
            // Right arm mapping
            'right_shoulder': {
                primary: 'rightShoulder',
                secondary: ['rightArm'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.6, 0.4]
            },
            'right_elbow': {
                primary: 'rightArm',
                secondary: ['rightForeArm'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.3, 0.7]
            },
            'right_wrist': {
                primary: 'rightForeArm',
                secondary: ['rightHand'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.2, 0.8]
            },
            
            // Left arm mapping (mirrored)
            'left_shoulder': {
                primary: 'leftShoulder',
                secondary: ['leftArm'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.6, 0.4]
            },
            'left_elbow': {
                primary: 'leftArm',
                secondary: ['leftForeArm'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.3, 0.7]
            },
            'left_wrist': {
                primary: 'leftForeArm',
                secondary: ['leftHand'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.2, 0.8]
            },
            
            // Right leg mapping
            'right_hip': {
                primary: 'rightUpLeg',
                secondary: [],
                influence: 1.0,
                type: 'direct'
            },
            'right_knee': {
                primary: 'rightLeg',
                secondary: [],
                influence: 1.0,
                type: 'direct'
            },
            'right_ankle': {
                primary: 'rightFoot',
                secondary: ['rightToe'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.8, 0.2]
            },
            
            // Left leg mapping (mirrored)
            'left_hip': {
                primary: 'leftUpLeg',
                secondary: [],
                influence: 1.0,
                type: 'direct'
            },
            'left_knee': {
                primary: 'leftLeg',
                secondary: [],
                influence: 1.0,
                type: 'direct'
            },
            'left_ankle': {
                primary: 'leftFoot',
                secondary: ['leftToe'],
                influence: 1.0,
                type: 'chain',
                distribution: [0.8, 0.2]
            }
        };
    }
    
    /**
     * Create reverse mapping for VRM bones that don't have DeepMimic equivalents
     */
    createVRMToDeepMimicMapping() {
        return {
            // Fingers - derive from hand position
            'leftThumb1': { source: 'left_wrist', method: 'derive', offset: { x: 0.1, y: 0, z: 0.05 } },
            'leftThumb2': { source: 'left_wrist', method: 'derive', offset: { x: 0.15, y: 0, z: 0.08 } },
            'leftIndex1': { source: 'left_wrist', method: 'derive', offset: { x: 0.12, y: 0, z: -0.02 } },
            'leftIndex2': { source: 'left_wrist', method: 'derive', offset: { x: 0.18, y: 0, z: -0.02 } },
            'leftMiddle1': { source: 'left_wrist', method: 'derive', offset: { x: 0.12, y: 0, z: 0 } },
            'leftMiddle2': { source: 'left_wrist', method: 'derive', offset: { x: 0.18, y: 0, z: 0 } },
            
            'rightThumb1': { source: 'right_wrist', method: 'derive', offset: { x: 0.1, y: 0, z: -0.05 } },
            'rightThumb2': { source: 'right_wrist', method: 'derive', offset: { x: 0.15, y: 0, z: -0.08 } },
            'rightIndex1': { source: 'right_wrist', method: 'derive', offset: { x: 0.12, y: 0, z: 0.02 } },
            'rightIndex2': { source: 'right_wrist', method: 'derive', offset: { x: 0.18, y: 0, z: 0.02 } },
            'rightMiddle1': { source: 'right_wrist', method: 'derive', offset: { x: 0.12, y: 0, z: 0 } },
            'rightMiddle2': { source: 'right_wrist', method: 'derive', offset: { x: 0.18, y: 0, z: 0 } }
        };
    }
    
    /**
     * Define bone constraints based on DeepMimic joint limits
     */
    createBoneConstraints() {
        return {
            'chest': { // DeepMimic joint limits: -1.2 to 1.2 radians
                minRotation: { x: -1.2, y: -1.2, z: -1.2 },
                maxRotation: { x: 1.2, y: 1.2, z: 1.2 }
            },
            'neck': {
                minRotation: { x: -1.0, y: -1.0, z: -1.0 },
                maxRotation: { x: 1.0, y: 1.0, z: 1.0 }
            },
            'right_hip': {
                minRotation: { x: -1.2, y: -1.0, z: -1.57 },
                maxRotation: { x: 1.2, y: 1.0, z: 2.57 }
            },
            'right_knee': { // Revolute joint: -3.14 to 0
                minRotation: { x: -3.14, y: 0, z: 0 },
                maxRotation: { x: 0, y: 0, z: 0 }
            },
            'right_ankle': {
                minRotation: { x: -1.0, y: -1.0, z: -1.57 },
                maxRotation: { x: 1.0, y: 1.0, z: 1.57 }
            },
            'right_shoulder': {
                minRotation: { x: -3.14, y: -1.5, z: -0.7 },
                maxRotation: { x: 0.5, y: 1.5, z: 3.14 }
            },
            'right_elbow': { // Revolute joint: 0 to 3.14
                minRotation: { x: 0, y: 0, z: 0 },
                maxRotation: { x: 3.14, y: 0, z: 0 }
            },
            // Mirror constraints for left side
            'left_hip': {
                minRotation: { x: -1.2, y: -1.0, z: -1.57 },
                maxRotation: { x: 1.2, y: 1.0, z: 2.57 }
            },
            'left_knee': {
                minRotation: { x: -3.14, y: 0, z: 0 },
                maxRotation: { x: 0, y: 0, z: 0 }
            },
            'left_ankle': {
                minRotation: { x: -1.0, y: -1.0, z: -1.57 },
                maxRotation: { x: 1.0, y: 1.0, z: 1.57 }
            },
            'left_shoulder': {
                minRotation: { x: -0.5, y: -1.5, z: -0.7 },
                maxRotation: { x: 3.14, y: 1.5, z: 3.14 }
            },
            'left_elbow': {
                minRotation: { x: 0, y: 0, z: 0 },
                maxRotation: { x: 3.14, y: 0, z: 0 }
            }
        };
    }
    
    /**
     * Define interpolation rules for unmapped bones
     */
    createInterpolationRules() {
        return {
            // Finger movement based on hand position and velocity
            fingerMovement: {
                relaxedCurl: 0.1,    // Base finger curl amount
                velocityResponse: 0.3, // How much fingers react to hand velocity
                randomVariation: 0.05  // Small random variation for naturalness
            },
            
            // Spine distribution from chest movement
            spineDistribution: {
                spineRatio: 0.3,    // Lower spine gets 30% of chest movement
                spine1Ratio: 0.4,   // Middle spine gets 40%
                spine2Ratio: 0.3    // Upper spine gets 30%
            },
            
            // Head follow-through from neck
            headFollowThrough: {
                followRatio: 0.3,   // Head follows 30% of neck movement
                delayFrames: 2      // Small delay for natural follow-through
            }
        };
    }
    
    /**
     * Convert DeepMimic frame to VRM BVH frame
     */
    convertDeepMimicToVRM(deepMimicFrame) {
        const vrmFrame = {
            frameNumber: deepMimicFrame.frameNumber || 0,
            time: deepMimicFrame.time || 0,
            bones: {},
            metadata: {
                source: 'deepmimic_mapped',
                originalSource: 'deepmimic',
                mappingVersion: '1.0'
            }
        };
        
        // Process each DeepMimic joint
        for (const [deepMimicJoint, mapping] of Object.entries(this.deepMimicToVRM)) {
            const deepMimicData = deepMimicFrame.joints?.[deepMimicJoint] || 
                                  this.getDeepMimicJointData(deepMimicFrame, deepMimicJoint);
            
            if (deepMimicData) {
                this.applyJointMapping(vrmFrame, deepMimicJoint, deepMimicData, mapping);
            }
        }
        
        // Fill in unmapped VRM bones
        this.interpolateUnmappedBones(vrmFrame, deepMimicFrame);
        
        // Apply constraints
        this.applyBoneConstraints(vrmFrame);
        
        return vrmFrame;
    }
    
    /**
     * Extract joint data from DeepMimic frame format
     */
    getDeepMimicJointData(frame, jointName) {
        // Handle different DeepMimic output formats
        if (frame.joints && frame.joints[jointName]) {
            return frame.joints[jointName];
        }
        
        if (frame.pose && Array.isArray(frame.pose)) {
            // Extract from pose array based on joint index
            const jointIndex = this.getDeepMimicJointIndex(jointName);
            if (jointIndex !== -1 && jointIndex < frame.pose.length) {
                return this.parsePoseData(frame.pose[jointIndex], jointName);
            }
        }
        
        return null;
    }
    
    /**
     * Get DeepMimic joint index from humanoid3d.txt structure
     */
    getDeepMimicJointIndex(jointName) {
        const jointIndices = {
            'root': 0, 'chest': 1, 'neck': 2, 'right_hip': 3, 'right_knee': 4,
            'right_ankle': 5, 'right_shoulder': 6, 'right_elbow': 7, 'right_wrist': 8,
            'left_hip': 9, 'left_knee': 10, 'left_ankle': 11, 'left_shoulder': 12,
            'left_elbow': 13, 'left_wrist': 14
        };
        return jointIndices[jointName] || -1;
    }
    
    /**
     * Parse pose data based on joint type
     */
    parsePoseData(poseData, jointName) {
        const jointTypes = {
            'root': 'none',
            'chest': 'spherical', 'neck': 'spherical',
            'right_hip': 'spherical', 'left_hip': 'spherical',
            'right_ankle': 'spherical', 'left_ankle': 'spherical',
            'right_shoulder': 'spherical', 'left_shoulder': 'spherical',
            'right_knee': 'revolute', 'left_knee': 'revolute',
            'right_elbow': 'revolute', 'left_elbow': 'revolute',
            'right_wrist': 'fixed', 'left_wrist': 'fixed'
        };
        
        const jointType = jointTypes[jointName];
        
        switch (jointType) {
            case 'spherical':
                // 3 DOF: x, y, z rotations
                return {
                    rotation: {
                        x: poseData[0] || 0,
                        y: poseData[1] || 0,
                        z: poseData[2] || 0,
                        w: 1
                    }
                };
            case 'revolute':
                // 1 DOF: single axis rotation
                return {
                    rotation: {
                        x: poseData[0] || 0,
                        y: 0,
                        z: 0,
                        w: 1
                    }
                };
            case 'none':
                // Root joint with position
                return {
                    position: {
                        x: poseData[0] || 0,
                        y: poseData[1] || 0,
                        z: poseData[2] || 0
                    },
                    rotation: {
                        x: poseData[3] || 0,
                        y: poseData[4] || 0,
                        z: poseData[5] || 0,
                        w: 1
                    }
                };
            default:
                return {
                    rotation: { x: 0, y: 0, z: 0, w: 1 }
                };
        }
    }
    
    /**
     * Apply joint mapping to VRM frame
     */
    applyJointMapping(vrmFrame, deepMimicJoint, deepMimicData, mapping) {
        const { primary, secondary, distribution, type } = mapping;
        
        switch (type) {
            case 'direct':
                vrmFrame.bones[primary] = this.convertBoneData(deepMimicData);
                break;
                
            case 'chain':
                this.applyChainMapping(vrmFrame, deepMimicData, primary, secondary, distribution);
                break;
                
            case 'hierarchical':
                this.applyHierarchicalMapping(vrmFrame, deepMimicData, primary, secondary, distribution);
                break;
        }
    }
    
    /**
     * Apply chain mapping (distribute rotation along bone chain)
     */
    applyChainMapping(vrmFrame, deepMimicData, primary, secondary, distribution) {
        const bones = [primary, ...secondary];
        
        for (let i = 0; i < bones.length; i++) {
            const boneName = bones[i];
            const factor = distribution[i] || (1.0 / bones.length);
            
            vrmFrame.bones[boneName] = this.convertBoneData(deepMimicData, factor);
        }
    }
    
    /**
     * Apply hierarchical mapping (distribute to parent-child chain)
     */
    applyHierarchicalMapping(vrmFrame, deepMimicData, primary, secondary, distribution) {
        const bones = [...secondary, primary]; // Secondary first, then primary
        
        for (let i = 0; i < bones.length; i++) {
            const boneName = bones[i];
            const factor = distribution[i] || (1.0 / bones.length);
            
            vrmFrame.bones[boneName] = this.convertBoneData(deepMimicData, factor);
        }
    }
    
    /**
     * Convert DeepMimic bone data to VRM format
     */
    convertBoneData(deepMimicData, factor = 1.0) {
        const result = {
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
        };
        
        if (deepMimicData.position) {
            result.position = {
                x: deepMimicData.position.x * factor * this.scaleFactor,
                y: deepMimicData.position.y * factor * this.scaleFactor,
                z: deepMimicData.position.z * factor * this.scaleFactor
            };
        }
        
        if (deepMimicData.rotation) {
            // Scale rotation by factor for distribution
            result.rotation = {
                x: deepMimicData.rotation.x * factor,
                y: deepMimicData.rotation.y * factor,
                z: deepMimicData.rotation.z * factor,
                w: deepMimicData.rotation.w || 1
            };
        }
        
        return result;
    }
    
    /**
     * Interpolate bones that don't have DeepMimic equivalents
     */
    interpolateUnmappedBones(vrmFrame, deepMimicFrame) {
        for (const [vrmBone, mapping] of Object.entries(this.vrmToDeepMimic)) {
            if (!vrmFrame.bones[vrmBone]) {
                const sourceData = this.getDeepMimicJointData(deepMimicFrame, mapping.source);
                
                if (sourceData && mapping.method === 'derive') {
                    vrmFrame.bones[vrmBone] = this.deriveBoneFromSource(sourceData, mapping);
                }
            }
        }
        
        // Add procedural finger movement
        this.addProceduralFingerMovement(vrmFrame, deepMimicFrame);
    }
    
    /**
     * Derive bone data from source with offset
     */
    deriveBoneFromSource(sourceData, mapping) {
        const derived = {
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
        };
        
        if (sourceData.position && mapping.offset) {
            derived.position = {
                x: sourceData.position.x + mapping.offset.x,
                y: sourceData.position.y + mapping.offset.y,
                z: sourceData.position.z + mapping.offset.z
            };
        }
        
        // Derive subtle rotation from source
        if (sourceData.rotation) {
            derived.rotation = {
                x: sourceData.rotation.x * 0.3, // Reduced influence
                y: sourceData.rotation.y * 0.3,
                z: sourceData.rotation.z * 0.3,
                w: 1
            };
        }
        
        return derived;
    }
    
    /**
     * Add procedural finger movement based on hand motion
     */
    addProceduralFingerMovement(vrmFrame, deepMimicFrame) {
        const rules = this.interpolationRules.fingerMovement;
        
        // Generate subtle finger movement based on hand velocity and position
        const fingerBones = [
            'leftThumb1', 'leftThumb2', 'leftIndex1', 'leftIndex2', 'leftMiddle1', 'leftMiddle2',
            'rightThumb1', 'rightThumb2', 'rightIndex1', 'rightIndex2', 'rightMiddle1', 'rightMiddle2'
        ];
        
        for (const fingerBone of fingerBones) {
            if (!vrmFrame.bones[fingerBone]) {
                const handSide = fingerBone.includes('left') ? 'left' : 'right';
                const handData = this.getDeepMimicJointData(deepMimicFrame, `${handSide}_wrist`);
                
                if (handData) {
                    const curl = rules.relaxedCurl + Math.random() * rules.randomVariation;
                    
                    vrmFrame.bones[fingerBone] = {
                        position: { x: 0, y: 0, z: 0 },
                        rotation: {
                            x: curl * (fingerBone.includes('Thumb') ? 0.5 : 1.0),
                            y: 0,
                            z: curl * 0.3,
                            w: 1
                        }
                    };
                }
            }
        }
    }
    
    /**
     * Apply anatomical constraints to prevent impossible poses
     */
    applyBoneConstraints(vrmFrame) {
        for (const boneName in vrmFrame.bones) {
            const bone = vrmFrame.bones[boneName];
            const constraints = this.boneConstraints[boneName];
            
            if (constraints && bone.rotation) {
                bone.rotation.x = Math.max(constraints.minRotation.x, 
                                         Math.min(constraints.maxRotation.x, bone.rotation.x));
                bone.rotation.y = Math.max(constraints.minRotation.y, 
                                         Math.min(constraints.maxRotation.y, bone.rotation.y));
                bone.rotation.z = Math.max(constraints.minRotation.z, 
                                         Math.min(constraints.maxRotation.z, bone.rotation.z));
            }
        }
    }
    
    /**
     * Batch convert multiple DeepMimic frames
     */
    convertDeepMimicSequence(deepMimicFrames) {
        console.log(`[DeepMimic VRM Mapper] Converting ${deepMimicFrames.length} frames`);
        
        const vrmFrames = [];
        
        for (let i = 0; i < deepMimicFrames.length; i++) {
            const vrmFrame = this.convertDeepMimicToVRM(deepMimicFrames[i]);
            vrmFrame.frameNumber = i;
            vrmFrame.time = i / 30; // Assuming 30 FPS
            
            vrmFrames.push(vrmFrame);
        }
        
        // Apply smoothing across frames
        this.applyCrossFrameSmoothing(vrmFrames);
        
        return vrmFrames;
    }
    
    /**
     * Apply smoothing across frame sequence
     */
    applyCrossFrameSmoothing(frames) {
        if (frames.length < 2) return;
        
        for (let i = 1; i < frames.length; i++) {
            const currentFrame = frames[i];
            const previousFrame = frames[i - 1];
            
            for (const boneName in currentFrame.bones) {
                if (previousFrame.bones[boneName]) {
                    const current = currentFrame.bones[boneName];
                    const previous = previousFrame.bones[boneName];
                    
                    // Smooth rotation
                    current.rotation.x = this.lerp(previous.rotation.x, current.rotation.x, this.interpolationSmoothing);
                    current.rotation.y = this.lerp(previous.rotation.y, current.rotation.y, this.interpolationSmoothing);
                    current.rotation.z = this.lerp(previous.rotation.z, current.rotation.z, this.interpolationSmoothing);
                    
                    // Smooth position if available
                    if (current.position && previous.position) {
                        current.position.x = this.lerp(previous.position.x, current.position.x, this.interpolationSmoothing);
                        current.position.y = this.lerp(previous.position.y, current.position.y, this.interpolationSmoothing);
                        current.position.z = this.lerp(previous.position.z, current.position.z, this.interpolationSmoothing);
                    }
                }
            }
        }
    }
    
    /**
     * Linear interpolation helper
     */
    lerp(a, b, t) {
        return a + (b - a) * t;
    }
    
    /**
     * Get mapping statistics
     */
    getMappingStats() {
        return {
            deepMimicJoints: Object.keys(this.deepMimicToVRM).length,
            vrmBones: Object.keys(this.vrmToDeepMimic).length + Object.keys(this.deepMimicToVRM).length,
            directMappings: Object.values(this.deepMimicToVRM).filter(m => m.type === 'direct').length,
            chainMappings: Object.values(this.deepMimicToVRM).filter(m => m.type === 'chain').length,
            hierarchicalMappings: Object.values(this.deepMimicToVRM).filter(m => m.type === 'hierarchical').length,
            derivedBones: Object.keys(this.vrmToDeepMimic).length
        };
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DeepMimicVRMBoneMapper };
} else {
    // Browser global
    window.DeepMimicVRMBoneMapper = DeepMimicVRMBoneMapper;
}
