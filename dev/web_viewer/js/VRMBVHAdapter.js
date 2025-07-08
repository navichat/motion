/**
 * VRM BVH Adapter - Maps BVH skeleton data to VRM character bones
 * Integrates anime characters with existing BVH motion capture system
 * Enhanced with idle animations and facial expressions
 */

class VRMBVHAdapter {
    constructor(vrmModelObject, bvhSkeleton) {
        // vrmModelObject should contain both .vrm and .scene properties
        this.vrmModelObject = vrmModelObject;
        this.vrmModel = vrmModelObject.vrm; // The actual VRM instance
        this.vrmScene = vrmModelObject.scene; // The THREE.js scene object
        this.bvhSkeleton = bvhSkeleton;
        this.boneMapping = this.createBoneMapping();
        this.availableBones = {}; // Bones that are available in the VRM model
        this.initialized = false;
        this.debugMode = true; // Set to true for debugging
        
        // Coordinate system conversion settings - centralized in VRMBVHAdapter
        this.coordinateConversion = {
            positionScale: 0.01,     // Scale BVH units to display size
            flipX: false,            // X: as-is from BVH
            flipY: false,            // Y: as-is from BVH (let it control height)
            flipZ: true,             // Z: flip to match VRM coordinate system
            rotationOrder: 'YXZ',    // RSMT uses YXZ rotation order
            flipRotX: true,          // X: Forward/back lean (pitch) - flip for VRM
            flipRotY: true,          // Y: Left/right turning (yaw) - flip to match Z flip
            flipRotZ: true           // Z: Roll - flip for VRM
        };
        
        // Idle animation system
        this.idleAnimations = {
            breathing: { enabled: true, amplitude: 0.01, frequency: 0.3, phase: 0 },
            blinking: { enabled: true, interval: 3000, lastBlink: 0, duration: 150 },
            headMovement: { enabled: true, amplitude: 0.05, frequency: 0.1, phase: 0 },
            facialExpressions: { enabled: true, currentExpression: 'neutral', duration: 0 }
        };
        
        // Camera interaction
        this.lookAtCamera = false;
        this.currentTime = 0;
        
        console.log('🎭 Enhanced VRMBVHAdapter initialized with idle animations');
        console.log('VRM Model Object:', vrmModelObject);
        console.log('VRM Instance:', this.vrmModel);
        console.log('VRM Scene:', this.vrmScene);
        console.log('BVH Skeleton:', bvhSkeleton);
        if (this.debugMode) {
            console.log('Initial bone mapping:', this.boneMapping);
        }
        
        // Initialize immediately to capture true rest pose
        if (this.vrmModel) {
            try {
                this.initialize();
            } catch (error) {
                console.warn('⚠️ Initial VRMBVHAdapter initialization failed, will retry on first frame:', error.message);
                this.initialized = false;
            }
        }
    }

    // Enable idle animations like the chat implementation
    setIdleAnimations(config) {
        console.log('🎭 Setting idle animations:', config);
        
        Object.assign(this.idleAnimations, config);
        
        // Start breathing animation
        if (this.idleAnimations.breathing.enabled) {
            this.startBreathingAnimation();
        }
        
        // Start blinking animation
        if (this.idleAnimations.blinking.enabled) {
            this.startBlinkingAnimation();
        }
        
        console.log('✅ Idle animations configured');
    }

    // Look at camera like a human (from chat implementation)
    lookAtCameraAsIfHuman(camera) {
        this.lookAtCamera = true;
        this.camera = camera;
        console.log('👁️ VRM character will look at camera');
    }

    // Tick function for animations (like chat implementation)
    tick(deltaTime) {
        this.currentTime += deltaTime;
        
        if (this.vrmModel) {
            // Update idle animations
            this.updateIdleAnimations(deltaTime);
            
            // Update camera looking
            if (this.lookAtCamera && this.camera) {
                this.updateCameraLook();
            }
            
            // Update VRM internal systems
            this.vrmModel.update(deltaTime);
        }
    }

    updateIdleAnimations(deltaTime) {
        if (!this.vrmModel || !this.vrmModel.humanoid) return;
        
        const time = this.currentTime;
        
        // Breathing animation
        if (this.idleAnimations.breathing.enabled) {
            const breathingOffset = Math.sin(time * this.idleAnimations.breathing.frequency) * this.idleAnimations.breathing.amplitude;
            const chestBone = this.vrmModel.humanoid.getNormalizedBoneNode('chest');
            if (chestBone) {
                chestBone.rotation.x += breathingOffset * 0.1;
            }
        }
        
        // Blinking animation
        if (this.idleAnimations.blinking.enabled) {
            const timeSinceLastBlink = time - this.idleAnimations.blinking.lastBlink;
            if (timeSinceLastBlink > this.idleAnimations.blinking.interval) {
                this.triggerBlink();
                this.idleAnimations.blinking.lastBlink = time;
            }
        }
        
        // Subtle head movement
        if (this.idleAnimations.headMovement.enabled) {
            const headOffset = Math.sin(time * this.idleAnimations.headMovement.frequency) * this.idleAnimations.headMovement.amplitude;
            const headBone = this.vrmModel.humanoid.getRawBoneNode('head');
            if (headBone) {
                headBone.rotation.y += headOffset * 0.1;
            }
        }
    }

    triggerBlink() {
        if (this.vrmModel && this.vrmModel.expressionManager) {
            // Trigger blink expression
            const expressions = this.vrmModel.expressionManager;
            if (expressions.getExpressionValue) {
                // Quick blink animation
                setTimeout(() => {
                    if (expressions.setValue) {
                        expressions.setValue('blink', 1.0);
                        setTimeout(() => {
                            expressions.setValue('blink', 0.0);
                        }, this.idleAnimations.blinking.duration);
                    }
                }, 10);
            }
        }
    }

    updateCameraLook() {
        if (!this.camera || !this.vrmModel) return;
        
        const headBone = this.vrmModel.humanoid.getRawBoneNode('head');
        if (headBone) {
            // Calculate direction to camera - use safe THREE.js access
            const Vector3 = window.THREE?.Vector3 || (typeof THREE !== 'undefined' ? THREE.Vector3 : null);
            if (!Vector3) return; // THREE.js not available yet
            
            const cameraPos = this.camera.position.clone();
            const headPos = headBone.getWorldPosition(new Vector3());
            const direction = cameraPos.sub(headPos).normalize();
            
            // Apply subtle head rotation toward camera
            const lookIntensity = 0.3; // Subtle looking
            headBone.lookAt(headPos.add(direction.multiplyScalar(lookIntensity)));
        }
    }

    startBreathingAnimation() {
        console.log('🫁 Starting breathing animation');
        // Breathing is handled in tick function
    }

    startBlinkingAnimation() {
        console.log('👁️ Starting blinking animation');
        // Blinking is handled in tick function
    }

    createBoneMapping() {
        // Map BVH joint names to VRM humanoid bone names
        // Fixed mapping to match VRM 1.0 humanoid specification
        return {
            // Root
            'Hips': 'hips',

            // Spine chain - corrected VRM bone names
            'Chest': 'spine',
            'Chest2': 'chest', 
            'Chest3': 'upperChest',
            'Chest4': 'upperChest', // Map Chest4 to upperChest instead of neck to avoid conflict
            'Neck': 'neck',         // This is the actual neck bone
            'Head': 'head',

            // Right arm chain - corrected VRM bone names
            'RightCollar': 'rightShoulder',     // Clavicle/shoulder
            'RightShoulder': 'rightUpperArm',   // Upper arm
            'RightElbow': 'rightLowerArm',      // Forearm
            'RightWrist': 'rightHand',          // Hand

            // Left arm chain - corrected VRM bone names
            'LeftCollar': 'leftShoulder',       // Clavicle/shoulder
            'LeftShoulder': 'leftUpperArm',     // Upper arm
            'LeftElbow': 'leftLowerArm',        // Forearm
            'LeftWrist': 'leftHand',            // Hand

            // Right leg chain - corrected VRM bone names
            'RightHip': 'rightUpperLeg',        // Thigh
            'RightKnee': 'rightLowerLeg',       // Shin
            'RightAnkle': 'rightFoot',          // Foot
            'RightToe': 'rightToes',            // Toes

            // Left leg chain - corrected VRM bone names
            'LeftHip': 'leftUpperLeg',          // Thigh
            'LeftKnee': 'leftLowerLeg',         // Shin
            'LeftAnkle': 'leftFoot',            // Foot
            'LeftToe': 'leftToes'               // Toes
        };
    }

    initialize() {
        if (!this.vrmModel || !this.vrmModel.humanoid) {
            console.error('❌ VRM model or humanoid not available');
            return false;
        }
        
        // Check available bones
        const availableBones = {};
        for (const [bvhJoint, vrmBone] of Object.entries(this.boneMapping)) {
            // Use getNormalizedBoneNode for better compatibility and to avoid deprecation warning
            const bone = this.vrmModel.humanoid.getNormalizedBoneNode(vrmBone);
            if (bone) {
                availableBones[bvhJoint] = { vrmBone, bone };
            }
        }
        
        this.availableBones = availableBones;
        
        // CRITICAL: Store the original rest pose of the VRM skeleton
        this.originalRestPose = {};
        this.originalWorldTransforms = {};
        
        for (const [bvhJoint, boneData] of Object.entries(availableBones)) {
            const bone = boneData.bone;
            
            // Store original local transform
            this.originalRestPose[bvhJoint] = {
                position: bone.position.clone(),
                rotation: bone.rotation.clone(),
                quaternion: bone.quaternion.clone(),
                scale: bone.scale.clone()
            };
            
            // Store original world transform for analysis
            const worldPos = new (window.THREE?.Vector3 || (typeof THREE !== 'undefined' ? THREE.Vector3 : Object))();
            const worldRot = new (window.THREE?.Quaternion || (typeof THREE !== 'undefined' ? THREE.Quaternion : Object))();
            const worldScale = new (window.THREE?.Vector3 || (typeof THREE !== 'undefined' ? THREE.Vector3 : Object))();
            
            if (worldPos.constructor === Object) {
                // THREE.js not available yet, skip world transform capture for now
                this.originalWorldTransforms[bvhJoint] = {
                    position: { x: 0, y: 0, z: 0 },
                    rotation: { x: 0, y: 0, z: 0, w: 1 },
                    scale: { x: 1, y: 1, z: 1 }
                };
                continue;
            }
            
            bone.getWorldPosition(worldPos);
            bone.getWorldQuaternion(worldRot);
            bone.getWorldScale(worldScale);
            
            this.originalWorldTransforms[bvhJoint] = {
                position: worldPos.clone(),
                rotation: worldRot.clone(),
                scale: worldScale.clone()
            };
            
            if (this.debugMode) {
                console.log(`📸 Stored rest pose for ${bvhJoint} (${boneData.vrmBone}):`, {
                    localRot: bone.rotation.clone(),
                    worldPos: worldPos.clone(),
                    worldRot: worldRot.clone()
                });
            }
        }
        
        this.initialized = true;

        // Ensure autoUpdateHumanBones and lookAt.autoUpdate are true for proper VRM animation
        this.vrmModel.humanoid.autoUpdateHumanBones = true;
        if (this.vrmModel.lookAt) {
            this.vrmModel.lookAt.autoUpdate = true;
        }
        
        console.log('✅ VRMBVHAdapter initialized with', Object.keys(availableBones).length, 'mapped bones');
        console.log('📸 Captured original rest pose for all bones');
        
        if (this.debugMode) {
            console.log('Available bones:', Object.keys(availableBones));
            this.analyzeRestPose();
        }
        
        return true;
    }

    // New method to analyze the VRM rest pose
    analyzeRestPose() {
        console.log('=== VRM REST POSE ANALYSIS ===');
        
        // Check for common rest pose patterns
        const leftShoulder = this.originalRestPose['LeftShoulder'];
        const rightShoulder = this.originalRestPose['RightShoulder'];
        const leftUpperArm = this.originalRestPose['LeftCollar'];
        const rightUpperArm = this.originalRestPose['RightCollar'];
        
        if (leftShoulder && rightShoulder) {
            console.log('Shoulder rotations:');
            console.log(`  Left Shoulder: x=${leftShoulder.rotation.x.toFixed(3)}, y=${leftShoulder.rotation.y.toFixed(3)}, z=${leftShoulder.rotation.z.toFixed(3)}`);
            console.log(`  Right Shoulder: x=${rightShoulder.rotation.x.toFixed(3)}, y=${rightShoulder.rotation.y.toFixed(3)}, z=${rightShoulder.rotation.z.toFixed(3)}`);
            
            // Check if arms are already raised in rest pose
            const leftArmRaise = Math.abs(leftShoulder.rotation.z);
            const rightArmRaise = Math.abs(rightShoulder.rotation.z);
            
            if (leftArmRaise > 0.1 || rightArmRaise > 0.1) {
                console.warn('⚠️ VRM appears to have arms raised in rest pose!');
                console.log(`  Left arm raise: ${(leftArmRaise * 180 / Math.PI).toFixed(1)}°`);
                console.log(`  Right arm raise: ${(rightArmRaise * 180 / Math.PI).toFixed(1)}°`);
            }
        }
        
        // Analyze world positions to understand skeleton structure
        console.log('=== WORLD POSITION ANALYSIS ===');
        Object.keys(this.originalWorldTransforms).forEach(bvhJoint => {
            const worldTransform = this.originalWorldTransforms[bvhJoint];
            const vrmBone = this.boneMapping[bvhJoint];
            console.log(`${bvhJoint} (${vrmBone}): world pos (${worldTransform.position.x.toFixed(2)}, ${worldTransform.position.y.toFixed(2)}, ${worldTransform.position.z.toFixed(2)})`);
        });
    }

    // Diagnose the difference between VRM rest pose and BVH first frame
    diagnoseRestPoseMismatch(bvhFirstFrame) {
        if (!bvhFirstFrame || !this.originalRestPose) {
            console.warn('⚠️ Cannot diagnose rest pose mismatch - missing data');
            return;
        }
        
        console.log('=== BVH vs VRM REST POSE MISMATCH DIAGNOSIS ===');
        
        // BVH joint order from the actual file structure
        const bvhJointOrder = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ];
        
        let channelIndex = 6; // Start after root position/rotation
        
        console.log('Checking for significant rest pose differences:');
        
        for (const bvhJoint of bvhJointOrder) {
            if (bvhJoint === 'Hips') continue; // Skip root
            
            const vrmBone = this.boneMapping[bvhJoint];
            const vrmRestPose = this.originalRestPose[bvhJoint];
            
            if (vrmRestPose && channelIndex + 2 < bvhFirstFrame.length) {
                // Get BVH rotations from first frame
                const bvhRotY = (bvhFirstFrame[channelIndex] || 0);     // Degrees
                const bvhRotX = (bvhFirstFrame[channelIndex + 1] || 0); // Degrees  
                const bvhRotZ = (bvhFirstFrame[channelIndex + 2] || 0); // Degrees
                
                // Get VRM rest pose rotations (in degrees)
                const vrmRotX = vrmRestPose.rotation.x * 180 / Math.PI;
                const vrmRotY = vrmRestPose.rotation.y * 180 / Math.PI;
                const vrmRotZ = vrmRestPose.rotation.z * 180 / Math.PI;
                
                // Calculate differences
                const diffX = Math.abs(bvhRotX - vrmRotX);
                const diffY = Math.abs(bvhRotY - vrmRotY);
                const diffZ = Math.abs(bvhRotZ - vrmRotZ);
                const totalDiff = diffX + diffY + diffZ;
                
                // Flag significant differences (likely causing pose issues)
                if (totalDiff > 15) { // More than 15 degrees total difference
                    console.warn(`🚨 ${bvhJoint} (${vrmBone}) has significant rest pose mismatch:`);
                    console.log(`  BVH expects: Y=${bvhRotY.toFixed(1)}°, X=${bvhRotX.toFixed(1)}°, Z=${bvhRotZ.toFixed(1)}°`);
                    console.log(`  VRM has: Y=${vrmRotY.toFixed(1)}°, X=${vrmRotX.toFixed(1)}°, Z=${vrmRotZ.toFixed(1)}°`);
                    console.log(`  Difference: Y=${Math.abs(bvhRotY - vrmRotY).toFixed(1)}°, X=${diffX.toFixed(1)}°, Z=${diffZ.toFixed(1)}° (total: ${totalDiff.toFixed(1)}°)`);
                    
                    // Special check for arm raising issues
                    if (bvhJoint.includes('Shoulder') || bvhJoint.includes('Collar')) {
                        if (Math.abs(bvhRotZ) > 10) {
                            console.log(`  ⚠️ This appears to be causing arm raising! BVH Z-rotation: ${bvhRotZ.toFixed(1)}°`);
                        }
                    }
                }
            }
            
            channelIndex += 3;
        }
    }

    // Restore VRM to original rest pose
    restoreRestPose() {
        if (!this.originalRestPose) {
            console.warn('⚠️ No original rest pose stored');
            return false;
        }
        
        console.log('🔄 Restoring VRM to original rest pose...');
        
        for (const [bvhJoint, restPose] of Object.entries(this.originalRestPose)) {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                
                // Restore original transform
                bone.position.copy(restPose.position);
                bone.rotation.copy(restPose.rotation);
                bone.quaternion.copy(restPose.quaternion);
                bone.scale.copy(restPose.scale);
            }
        }
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ VRM restored to original rest pose');
        return true;
    }

    // Apply rest pose compensation to handle T-pose differences
    applyRestPoseCompensation() {
        if (!this.originalRestPose) {
            console.warn('⚠️ No original rest pose available for compensation');
            return;
        }
        
        console.log('🔧 Applying rest pose compensation...');
        
        // Common VRM to BVH rest pose adjustments
        const compensations = {
            'LeftShoulder': { x: 0, y: 0, z: -0.2 },   // Lower left arm
            'RightShoulder': { x: 0, y: 0, z: 0.2 },   // Lower right arm
            'LeftCollar': { x: 0, y: 0, z: -0.1 },     // Adjust left collar
            'RightCollar': { x: 0, y: 0, z: 0.1 },     // Adjust right collar
            // Leg compensations
            'LeftHip': { x: 0, y: 0, z: 0.1 },        // Slight outward for left hip
            'RightHip': { x: 0, y: 0, z: -0.1 },      // Slight outward for right hip
            'LeftKnee': { x: 0.1, y: 0, z: 0 },       // Slight bend for left knee
            'RightKnee': { x: 0.1, y: 0, z: 0 }       // Slight bend for right knee
        };
        
        for (const [bvhJoint, compensation] of Object.entries(compensations)) {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                const originalRot = this.originalRestPose[bvhJoint].rotation;
                
                // Apply compensation on top of original rotation
                bone.rotation.set(
                    originalRot.x + compensation.x,
                    originalRot.y + compensation.y,
                    originalRot.z + compensation.z
                );
                
                console.log(`Applied compensation to ${bvhJoint}: ${JSON.stringify(compensation)}`);
            }
        }
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
    }

    // Debug method to compare current pose with rest pose
    debugPoseComparison() {
        if (!this.originalRestPose) {
            console.warn('⚠️ No original rest pose to compare with');
            return;
        }
        
        console.log('=== CURRENT VS REST POSE COMPARISON ===');
        
        for (const [bvhJoint, restPose] of Object.entries(this.originalRestPose)) {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                const currentRot = bone.rotation;
                const restRot = restPose.rotation;
                
                const diffX = currentRot.x - restRot.x;
                const diffY = currentRot.y - restRot.y;
                const diffZ = currentRot.z - restRot.z;
                
                const totalDiff = Math.sqrt(diffX*diffX + diffY*diffY + diffZ*diffZ);
                
                if (totalDiff > 0.01) { // Only show significant differences
                    console.log(`${bvhJoint}:`);
                    console.log(`  Rest: (${restRot.x.toFixed(3)}, ${restRot.y.toFixed(3)}, ${restRot.z.toFixed(3)})`);
                    console.log(`  Current: (${currentRot.x.toFixed(3)}, ${currentRot.y.toFixed(3)}, ${currentRot.z.toFixed(3)})`);
                    console.log(`  Diff: (${diffX.toFixed(3)}, ${diffY.toFixed(3)}, ${diffZ.toFixed(3)}) [${totalDiff.toFixed(3)}]`);
                }
            }
        }
    }

    // Analyze BVH first frame to understand expected rest pose
    analyzeBVHRestPose(bvhFrameData) {
        if (!bvhFrameData || bvhFrameData.length < 6) {
            console.warn('⚠️ Invalid BVH frame data for rest pose analysis');
            return;
        }
        
        console.log('=== BVH REST POSE ANALYSIS ===');
        console.log('Analyzing first frame as BVH expected rest pose...');
        
        // BVH joint order from the actual file structure
        const bvhJointOrder = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ];
        
        let channelIndex = 6; // Start after root position/rotation
        
        const bvhRestPose = {};
        
        for (const bvhJoint of bvhJointOrder) {
            if (bvhJoint === 'Hips') continue; // Skip root
            
            if (channelIndex + 2 < bvhFrameData.length) {
                const rotY = bvhFrameData[channelIndex] || 0;     // Yaw
                const rotX = bvhFrameData[channelIndex + 1] || 0; // Pitch  
                const rotZ = bvhFrameData[channelIndex + 2] || 0; // Roll
                
                bvhRestPose[bvhJoint] = { rotX, rotY, rotZ };
                
                // Flag significant rotations that might indicate non-neutral pose
                const totalRot = Math.abs(rotX) + Math.abs(rotY) + Math.abs(rotZ);
                if (totalRot > 5) { // More than 5 degrees total
                    console.log(`📢 ${bvhJoint} has significant rest rotation: Y=${rotY.toFixed(1)}°, X=${rotX.toFixed(1)}°, Z=${rotZ.toFixed(1)}°`);
                }
                
                channelIndex += 3;
            }
        }
        
        // Store BVH rest pose for comparison
        this.bvhRestPose = bvhRestPose;
        
        // Compare with VRM rest pose
        if (this.originalRestPose) {
            console.log('=== BVH vs VRM REST POSE COMPARISON ===');
            
            Object.keys(bvhRestPose).forEach(bvhJoint => {
                const bvhRest = bvhRestPose[bvhJoint];
                const vrmRestData = this.originalRestPose[bvhJoint];
                
                if (vrmRestData) {
                    const vrmRest = {
                        rotX: vrmRestData.rotation.x * 180 / Math.PI,
                        rotY: vrmRestData.rotation.y * 180 / Math.PI,
                        rotZ: vrmRestData.rotation.z * 180 / Math.PI
                    };
                    
                    const diffX = Math.abs(bvhRest.rotX - vrmRest.rotX);
                    const diffY = Math.abs(bvhRest.rotY - vrmRest.rotY);
                    const diffZ = Math.abs(bvhRest.rotZ - vrmRest.rotZ);
                    const totalDiff = diffX + diffY + diffZ;
                    
                    if (totalDiff > 10) { // Significant difference
                        console.log(`🔍 ${bvhJoint} rest pose mismatch (${totalDiff.toFixed(1)}° total):`);
                        console.log(`  BVH: Y=${bvhRest.rotY.toFixed(1)}°, X=${bvhRest.rotX.toFixed(1)}°, Z=${bvhRest.rotZ.toFixed(1)}°`);
                        console.log(`  VRM: Y=${vrmRest.rotY.toFixed(1)}°, X=${vrmRest.rotX.toFixed(1)}°, Z=${vrmRest.rotZ.toFixed(1)}°`);
                        console.log(`  Diff: Y=${Math.abs(bvhRest.rotY - vrmRest.rotY).toFixed(1)}°, X=${diffX.toFixed(1)}°, Z=${diffZ.toFixed(1)}°`);
                    }
                }
            });
        }
        
        return bvhRestPose;
    }

    // Apply rest pose offset compensation based on BVH vs VRM differences
    applyBVHRestPoseCompensation() {
        if (!this.bvhRestPose || !this.originalRestPose) {
            console.warn('⚠️ BVH rest pose analysis required first');
            return false;
        }
        
        console.log('🔧 Applying BVH-based rest pose compensation...');
        
        // Apply the difference between BVH expected rest pose and VRM rest pose
        Object.keys(this.bvhRestPose).forEach(bvhJoint => {
            const boneData = this.availableBones[bvhJoint];
            const bvhRest = this.bvhRestPose[bvhJoint];
            const vrmRestData = this.originalRestPose[bvhJoint];
            
            if (boneData && boneData.bone && vrmRestData) {
                const bone = boneData.bone;
                
                // Calculate the offset needed to match BVH expected rest pose
                const offsetX = (bvhRest.rotX * Math.PI / 180) - vrmRestData.rotation.x;
                const offsetY = (bvhRest.rotY * Math.PI / 180) - vrmRestData.rotation.y;
                const offsetZ = (bvhRest.rotZ * Math.PI / 180) - vrmRestData.rotation.z;
                
                // Apply the offset
                bone.rotation.set(
                    vrmRestData.rotation.x + offsetX,
                    vrmRestData.rotation.y + offsetY,
                    vrmRestData.rotation.z + offsetZ
                );
                
                console.log(`Applied BVH rest compensation to ${bvhJoint}: offset (${(offsetX*180/Math.PI).toFixed(1)}°, ${(offsetY*180/Math.PI).toFixed(1)}°, ${(offsetZ*180/Math.PI).toFixed(1)}°)`);
            }
        });
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ BVH rest pose compensation applied');
        return true;
    }

    applyBVHFrameToVRM(frameData) {
        if (!this.initialized) {
            if (!this.initialize()) {
                return false;
            }
        }
        
        if (!frameData || frameData.length < 6) {
            if (this.debugMode) console.warn('⚠️ Invalid frame data for VRMBVHAdapter');
            return false;
        }

        try {
            // Apply root position ONLY (first 3 values) to the entire VRM scene
            // This moves the whole character, preserving internal skeleton proportions
            
            // BVH uses centimeters, VRM uses meters; coordinate system conversion
            const rootX = frameData[0] * this.coordinateConversion.positionScale * (this.coordinateConversion.flipX ? -1 : 1);
            const rootY = frameData[1] * this.coordinateConversion.positionScale * (this.coordinateConversion.flipY ? -1 : 1);
            const rootZ = frameData[2] * this.coordinateConversion.positionScale * (this.coordinateConversion.flipZ ? -1 : 1);
            
            // Apply position to VRM scene (moves entire character)
            // Keep character on ground level by setting Y to 0 instead of using BVH Y data
            this.vrmModel.scene.position.set(rootX, 0, rootZ);

            // Apply root rotation (next 3 values) to VRM scene
            const rootRotY = (frameData[3] || 0) * Math.PI / 180; // Yaw first in BVH
            const rootRotX = (frameData[4] || 0) * Math.PI / 180; // Pitch second
            const rootRotZ = (frameData[5] || 0) * Math.PI / 180; // Roll third
            
            // Apply coordinate system corrections for VRM scene rotation
            this.vrmModel.scene.rotation.order = this.coordinateConversion.rotationOrder;
            this.vrmModel.scene.rotation.set(
                rootRotX * (this.coordinateConversion.flipRotX ? -1 : 1),
                rootRotY * (this.coordinateConversion.flipRotY ? -1 : 1),
                rootRotZ * (this.coordinateConversion.flipRotZ ? -1 : 1)
            );
            
            // Debug circular movement issue
            if (this.debugMode && (Math.abs(rootRotY) > 0.01 || Math.abs(frameData[0]) > 1 || Math.abs(frameData[2]) > 1)) {
                console.log(`🔄 Root transform: pos(${rootX.toFixed(3)}, ${rootY.toFixed(3)}, ${rootZ.toFixed(3)}) rot(${(rootRotX * (this.coordinateConversion.flipRotX ? -1 : 1) * 180/Math.PI).toFixed(1)}°, ${(rootRotY * (this.coordinateConversion.flipRotY ? -1 : 1) * 180/Math.PI).toFixed(1)}°, ${(rootRotZ * (this.coordinateConversion.flipRotZ ? -1 : 1) * 180/Math.PI).toFixed(1)}°)`);
            }

            // BVH joint order from the actual file structure (matches rsmt_showcase_modern.html)
            const bvhJointOrder = [
                'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
                'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
                'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
                'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
                'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
            ];

            // Start from index 6 for joint rotations (after root pos + rot)
            let channelIndex = 6;
            
            // Process joints in BVH file order - APPLY ROTATIONS ONLY to preserve VRM proportions
            for (const bvhJoint of bvhJointOrder) {
                // Skip root joint as it's already processed
                if (bvhJoint === 'Hips') continue;
                
                // Check if this BVH joint maps to a VRM bone
                const vrmBoneName = this.boneMapping[bvhJoint];
                if (!vrmBoneName) {
                    // Skip unmapped joints but advance channel index
                    channelIndex += 3;
                    continue;
                }
                
                const boneData = this.availableBones[bvhJoint];
                if (!boneData || !boneData.bone) {
                    // Skip unavailable bones but advance channel index
                    channelIndex += 3;
                    continue;
                }
                
                const boneNode = boneData.bone;
                
                // Check if we have enough data for this joint
                if (channelIndex + 2 >= frameData.length) {
                    console.warn(`⚠️ Not enough data for ${bvhJoint} (${vrmBoneName}). Frame data length: ${frameData.length}, needed index: ${channelIndex + 2}`);
                    break;
                }
                
                // BVH rotation order is typically Y, X, Z (yaw, pitch, roll)
                const rotY = (frameData[channelIndex] || 0) * Math.PI / 180;     // Yaw
                const rotX = (frameData[channelIndex + 1] || 0) * Math.PI / 180; // Pitch  
                const rotZ = (frameData[channelIndex + 2] || 0) * Math.PI / 180; // Roll
                
                // CRITICAL: Apply ONLY rotation to preserve VRM skeleton proportions
                // Do NOT modify bone position - this maintains VRM character's bone lengths
                boneNode.rotation.order = this.coordinateConversion.rotationOrder;
                
                // Special handling for arm joints to fix backwards bending while preserving natural joint motion
                let finalRotX = rotX;
                let finalRotY = rotY;
                let finalRotZ = rotZ;
                
                // Fix arm joint coordinate conversion for proper bending direction
                if (bvhJoint.includes('Shoulder')) {
                    // Shoulder joints: Apply targeted fixes without full axis remapping
                    if (bvhJoint.includes('Right')) {
                        // Right shoulder: Keep primary axes but fix direction
                        finalRotX = rotX * -1;    // Invert X to fix forward/back swing
                        finalRotY = rotY * 1;     // Keep Y normal for up/down movement
                        finalRotZ = rotZ * -1;    // Invert Z to fix arm raising direction
                    } else if (bvhJoint.includes('Left')) {
                        // Left shoulder: Mirror right but with opposite Z behavior
                        finalRotX = rotX * -1;    // Invert X to fix forward/back swing  
                        finalRotY = rotY * 1;     // Keep Y normal for up/down movement
                        finalRotZ = rotZ * -1;    // Also invert Z for left side (same as right)
                    }
                } else if (bvhJoint.includes('Elbow')) {
                    // Elbow joints: Focus on fixing the primary bending axis only
                    if (bvhJoint.includes('Right')) {
                        // Right elbow: Fix the primary flexion/extension axis
                        finalRotX = rotX * -1;    // Invert X to reverse bend direction
                        finalRotY = rotY * 1;     // Keep Y normal for twist
                        finalRotZ = rotZ * 1;     // Keep Z normal for side bend
                    } else if (bvhJoint.includes('Left')) {
                        // Left elbow: Mirror right elbow behavior
                        finalRotX = rotX * -1;    // Invert X to reverse bend direction
                        finalRotY = rotY * 1;     // Keep Y normal for twist
                        finalRotZ = rotZ * 1;     // Keep Z normal for side bend
                    }
                } else if (bvhJoint.includes('Collar')) {
                    // Collar bones: Reduce influence instead of completely zeroing
                    finalRotX = rotX * 0.5;  // Reduce collar bone X influence
                    finalRotY = rotY * 0.5;  // Reduce collar bone Y influence
                    finalRotZ = rotZ * 0.5;  // Reduce collar bone Z influence
                } else if (bvhJoint === 'Head') {
                    // Head bone: Special handling to prevent looking down
                    finalRotX = rotX * -0.5; // Reduce and flip head pitch to prevent looking down
                    finalRotY = rotY * -1;   // Flip head yaw for proper left/right turning
                    finalRotZ = rotZ * -1;   // Flip head roll for proper tilting
                    
                    // Clamp head rotation to prevent extreme downward looking
                    const maxHeadPitch = Math.PI / 6; // 30 degrees max pitch
                    if (Math.abs(finalRotX) > maxHeadPitch) {
                        finalRotX = Math.sign(finalRotX) * maxHeadPitch;
                    }
                } else if (bvhJoint === 'Neck') {
                    // Neck bone: Special handling to support head properly
                    finalRotX = rotX * -0.3; // Reduce and flip neck pitch
                    finalRotY = rotY * -1;   // Flip neck yaw for proper turning
                    finalRotZ = rotZ * -1;   // Flip neck roll for proper tilting
                    
                    // Limit neck rotation to prevent unnatural bending
                    const maxNeckPitch = Math.PI / 8; // 22.5 degrees max pitch
                    if (Math.abs(finalRotX) > maxNeckPitch) {
                        finalRotX = Math.sign(finalRotX) * maxNeckPitch;
                    }
                } else {
                    // For non-arm joints, use standard coordinate conversion
                    finalRotX = rotX * (this.coordinateConversion.flipRotX ? -1 : 1);
                    finalRotY = rotY * (this.coordinateConversion.flipRotY ? -1 : 1);
                    finalRotZ = rotZ * (this.coordinateConversion.flipRotZ ? -1 : 1);
                }
                
                // Apply the corrected rotations
                boneNode.rotation.set(finalRotX, finalRotY, finalRotZ);
                
                if (this.debugMode && (channelIndex < 15 || bvhJoint.includes('Shoulder') || bvhJoint.includes('Elbow') || bvhJoint.includes('Collar') || bvhJoint === 'Head' || bvhJoint === 'Neck')) {
                    const isArmJoint = bvhJoint.includes('Shoulder') || bvhJoint.includes('Elbow') || bvhJoint.includes('Collar');
                    const isHeadNeckJoint = bvhJoint === 'Head' || bvhJoint === 'Neck';
                    let conversionType = 'STANDARD';
                    if (isArmJoint) conversionType = 'ARM-SPECIFIC';
                    if (isHeadNeckJoint) conversionType = 'HEAD/NECK-SPECIFIC';
                    
                    console.log(`${bvhJoint} -> ${vrmBoneName} (${conversionType}): Y=${(rotY*180/Math.PI).toFixed(1)}, X=${(rotX*180/Math.PI).toFixed(1)}, Z=${(rotZ*180/Math.PI).toFixed(1)} -> Final: Y=${(finalRotY*180/Math.PI).toFixed(1)}, X=${(finalRotX*180/Math.PI).toFixed(1)}, Z=${(finalRotZ*180/Math.PI).toFixed(1)}`);
                }
                
                channelIndex += 3;
            }
            
            // Update VRM using the correct method - this is critical!
            // VRM models need to be updated with deltaTime, not just humanoid.update()
            if (this.vrmModel && this.vrmModel.update) {
                // Use a fixed deltaTime if not provided - this is essential for VRM animation
                const deltaTime = 1/60; // 60fps
                this.vrmModel.update(deltaTime);
            } else if (this.vrmModel.humanoid?.update) {
                // Fallback for older VRM versions
                this.vrmModel.humanoid.update();
            }
            
            return true;
            
        } catch (error) {
            console.error('❌ Error applying BVH frame to VRM:', error);
            return false;
        }
    }

    // Restore VRM to original rest pose
    restoreRestPose() {
        if (!this.originalRestPose) {
            console.warn('⚠️ No original rest pose stored');
            return false;
        }
        
        console.log('🔄 Restoring VRM to original rest pose...');
        
        for (const [bvhJoint, restPose] of Object.entries(this.originalRestPose)) {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                
                // Restore original transform
                bone.position.copy(restPose.position);
                bone.rotation.copy(restPose.rotation);
                bone.quaternion.copy(restPose.quaternion);
                bone.scale.copy(restPose.scale);
            }
        }
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ VRM restored to original rest pose');
        return true;
    }

    // Apply rest pose compensation to handle T-pose differences
    applyRestPoseCompensation() {
        if (!this.originalRestPose) {
            console.warn('⚠️ No original rest pose available for compensation');
            return;
        }
        
        console.log('🔧 Applying rest pose compensation...');
        
        // Common VRM to BVH rest pose adjustments
        const compensations = {
            'LeftShoulder': { x: 0, y: 0, z: -0.2 },  // Lower left arm
            'RightShoulder': { x: 0, y: 0, z: 0.2 },  // Lower right arm
            'LeftCollar': { x: 0, y: 0, z: -0.1 },    // Adjust left collar
            'RightCollar': { x: 0, y: 0, z: 0.1 }     // Adjust right collar
        };
        
        for (const [bvhJoint, compensation] of Object.entries(compensations)) {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                const originalRot = this.originalRestPose[bvhJoint].rotation;
                
                // Apply compensation on top of original rotation
                bone.rotation.set(
                    originalRot.x + compensation.x,
                    originalRot.y + compensation.y,
                    originalRot.z + compensation.z
                );
                
                console.log(`Applied compensation to ${bvhJoint}: ${JSON.stringify(compensation)}`);
            }
        }
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
    }

    // Debug method to compare current pose with rest pose
    debugPoseComparison() {
        if (!this.originalRestPose) {
            console.warn('⚠️ No original rest pose to compare with');
            return;
        }
        
        console.log('=== CURRENT VS REST POSE COMPARISON ===');
        
        for (const [bvhJoint, restPose] of Object.entries(this.originalRestPose)) {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                const currentRot = bone.rotation;
                const restRot = restPose.rotation;
                
                const diffX = currentRot.x - restRot.x;
                const diffY = currentRot.y - restRot.y;
                const diffZ = currentRot.z - restRot.z;
                
                const totalDiff = Math.sqrt(diffX*diffX + diffY*diffY + diffZ*diffZ);
                
                if (totalDiff > 0.01) { // Only show significant differences
                    console.log(`${bvhJoint}:`);
                    console.log(`  Rest: (${restRot.x.toFixed(3)}, ${restRot.y.toFixed(3)}, ${restRot.z.toFixed(3)})`);
                    console.log(`  Current: (${currentRot.x.toFixed(3)}, ${currentRot.y.toFixed(3)}, ${currentRot.z.toFixed(3)})`);
                    console.log(`  Diff: (${diffX.toFixed(3)}, ${diffY.toFixed(3)}, ${diffZ.toFixed(3)}) [${totalDiff.toFixed(3)}]`);
                }
            }
        }
    }

    // Analyze BVH first frame to understand expected rest pose
    analyzeBVHRestPose(bvhFrameData) {
        if (!bvhFrameData || bvhFrameData.length < 6) {
            console.warn('⚠️ Invalid BVH frame data for rest pose analysis');
            return;
        }
        
        console.log('=== BVH REST POSE ANALYSIS ===');
        console.log('Analyzing first frame as BVH expected rest pose...');
        
        // BVH joint order from the actual file structure
        const bvhJointOrder = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ];
        
        let channelIndex = 6; // Start after root position/rotation
        
        const bvhRestPose = {};
        
        for (const bvhJoint of bvhJointOrder) {
            if (bvhJoint === 'Hips') continue; // Skip root
            
            if (channelIndex + 2 < bvhFrameData.length) {
                const rotY = bvhFrameData[channelIndex] || 0;     // Yaw
                const rotX = bvhFrameData[channelIndex + 1] || 0; // Pitch  
                const rotZ = bvhFrameData[channelIndex + 2] || 0; // Roll
                
                bvhRestPose[bvhJoint] = { rotX, rotY, rotZ };
                
                // Flag significant rotations that might indicate non-neutral pose
                const totalRot = Math.abs(rotX) + Math.abs(rotY) + Math.abs(rotZ);
                if (totalRot > 5) { // More than 5 degrees total
                    console.log(`📢 ${bvhJoint} has significant rest rotation: Y=${rotY.toFixed(1)}°, X=${rotX.toFixed(1)}°, Z=${rotZ.toFixed(1)}°`);
                }
                
                channelIndex += 3;
            }
        }
        
        // Store BVH rest pose for comparison
        this.bvhRestPose = bvhRestPose;
        
        // Compare with VRM rest pose
        if (this.originalRestPose) {
            console.log('=== BVH vs VRM REST POSE COMPARISON ===');
            
            Object.keys(bvhRestPose).forEach(bvhJoint => {
                const bvhRest = bvhRestPose[bvhJoint];
                const vrmRestData = this.originalRestPose[bvhJoint];
                
                if (vrmRestData) {
                    const vrmRest = {
                        rotX: vrmRestData.rotation.x * 180 / Math.PI,
                        rotY: vrmRestData.rotation.y * 180 / Math.PI,
                        rotZ: vrmRestData.rotation.z * 180 / Math.PI
                    };
                    
                    const diffX = Math.abs(bvhRest.rotX - vrmRest.rotX);
                    const diffY = Math.abs(bvhRest.rotY - vrmRest.rotY);
                    const diffZ = Math.abs(bvhRest.rotZ - vrmRest.rotZ);
                    const totalDiff = diffX + diffY + diffZ;
                    
                    if (totalDiff > 10) { // Significant difference
                        console.log(`🔍 ${bvhJoint} rest pose mismatch (${totalDiff.toFixed(1)}° total):`);
                        console.log(`  BVH: Y=${bvhRest.rotY.toFixed(1)}°, X=${bvhRest.rotX.toFixed(1)}°, Z=${bvhRest.rotZ.toFixed(1)}°`);
                        console.log(`  VRM: Y=${vrmRest.rotY.toFixed(1)}°, X=${vrmRest.rotX.toFixed(1)}°, Z=${vrmRest.rotZ.toFixed(1)}°`);
                        console.log(`  Diff: Y=${Math.abs(bvhRest.rotY - vrmRest.rotY).toFixed(1)}°, X=${diffX.toFixed(1)}°, Z=${diffZ.toFixed(1)}°`);
                    }
                }
            });
        }
        
        return bvhRestPose;
    }

    // Apply rest pose offset compensation based on BVH vs VRM differences
    applyBVHRestPoseCompensation() {
        if (!this.bvhRestPose || !this.originalRestPose) {
            console.warn('⚠️ BVH rest pose analysis required first');
            return false;
        }
        
        console.log('🔧 Applying BVH-based rest pose compensation...');
        
        // Apply the difference between BVH expected rest pose and VRM rest pose
        Object.keys(this.bvhRestPose).forEach(bvhJoint => {
            const boneData = this.availableBones[bvhJoint];
            const bvhRest = this.bvhRestPose[bvhJoint];
            const vrmRestData = this.originalRestPose[bvhJoint];
            
            if (boneData && boneData.bone && vrmRestData) {
                const bone = boneData.bone;
                
                // Calculate the offset needed to match BVH expected rest pose
                const offsetX = (bvhRest.rotX * Math.PI / 180) - vrmRestData.rotation.x;
                const offsetY = (bvhRest.rotY * Math.PI / 180) - vrmRestData.rotation.y;
                const offsetZ = (bvhRest.rotZ * Math.PI / 180) - vrmRestData.rotation.z;
                
                // Apply the offset
                bone.rotation.set(
                    vrmRestData.rotation.x + offsetX,
                    vrmRestData.rotation.y + offsetY,
                    vrmRestData.rotation.z + offsetZ
                );
                
                console.log(`Applied BVH rest compensation to ${bvhJoint}: offset (${(offsetX*180/Math.PI).toFixed(1)}°, ${(offsetY*180/Math.PI).toFixed(1)}°, ${(offsetZ*180/Math.PI).toFixed(1)}°)`);
            }
        });
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ BVH rest pose compensation applied');
        return true;
    }

    // Set initial VRM position and rotation (called once on load)
    setInitialTransform(position = { x: 0, y: 0, z: 0 }, rotation = { x: 0, y: 0, z: 0 }) {
        if (!this.vrmModel || !this.vrmModel.scene) {
            console.warn('⚠️ VRM model not available for initial transform');
            return false;
        }
        
        console.log('🔧 Setting initial VRM transform...');
        
        // Set initial position
        this.vrmModel.scene.position.set(position.x, position.y, position.z);
        
        // Set initial rotation
        this.vrmModel.scene.rotation.set(rotation.x, rotation.y, rotation.z);
        
        console.log(`✅ Initial VRM transform set: pos(${position.x}, ${position.y}, ${position.z}) rot(${rotation.x}, ${rotation.y}, ${rotation.z})`);
        return true;
    }

    // Reset all bones to neutral position (T-pose equivalent)
    resetToNeutralPose() {
        if (!this.availableBones) {
            console.warn('⚠️ No available bones for neutral pose reset');
            return false;
        }
        
        console.log('🔄 Resetting VRM to neutral pose...');
        
        // Reset all bones to neutral rotation
        Object.keys(this.availableBones).forEach(bvhJoint => {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                
                // Reset rotation to neutral
                bone.rotation.set(0, 0, 0);
                bone.rotation.order = 'XYZ';
                
                // Don't modify position - preserve VRM skeleton proportions
                // Don't modify scale - preserve VRM bone sizes
            }
        });
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ VRM reset to neutral pose');
        return true;
    }

    // Set global VRM position and rotation (for repositioning the entire character)
    setGlobalTransform(position, rotation) {
        if (!this.vrmModel || !this.vrmModel.scene) {
            console.warn('⚠️ VRM model not available for global transform');
            return false;
        }
        
        if (position) {
            this.vrmModel.scene.position.set(position.x || 0, position.y || 0, position.z || 0);
        }
        
        if (rotation) {
            this.vrmModel.scene.rotation.set(rotation.x || 0, rotation.y || 0, rotation.z || 0);
        }
        
        if (this.debugMode) {
            console.log(`🔧 Global VRM transform updated: pos(${this.vrmModel.scene.position.x.toFixed(3)}, ${this.vrmModel.scene.position.y.toFixed(3)}, ${this.vrmModel.scene.position.z.toFixed(3)}) rot(${this.vrmModel.scene.rotation.x.toFixed(3)}, ${this.vrmModel.scene.rotation.y.toFixed(3)}, ${this.vrmModel.scene.rotation.z.toFixed(3)})`);
        }
        
        return true;
    }

    // Get current global VRM position and rotation
    getGlobalTransform() {
        if (!this.vrmModel || !this.vrmModel.scene) {
            return null;
        }
        
        return {
            position: {
                x: this.vrmModel.scene.position.x,
                y: this.vrmModel.scene.position.y,
                z: this.vrmModel.scene.position.z
            },
            rotation: {
                x: this.vrmModel.scene.rotation.x,
                y: this.vrmModel.scene.rotation.y,
                z: this.vrmModel.scene.rotation.z
            }
        };
    }

    // Get bone mapping for external access (read-only)
    getBoneMapping() {
        return { ...this.boneMapping }; // Return a copy to prevent external modification
    }

    // Get available bones for external access (read-only)
    getAvailableBones() {
        const result = {};
        Object.keys(this.availableBones).forEach(bvhJoint => {
            const boneData = this.availableBones[bvhJoint];
            if (boneData) {
                result[bvhJoint] = {
                    vrmBone: boneData.vrmBone,
                    bone: boneData.bone // Direct reference for reading only
                };
            }
        });
        return result;
    }

    // Validate bone hierarchy consistency
    validateBoneHierarchy() {
        if (!this.availableBones) {
            console.warn('⚠️ No available bones for hierarchy validation');
            return false;
        }
        
        console.log('🔍 Validating VRM bone hierarchy...');
        
        let valid = true;
        Object.keys(this.availableBones).forEach(bvhJoint => {
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                const bone = boneData.bone;
                
                // Check if bone has valid parent-child relationships
                if (!bone.parent && bvhJoint !== 'Hips') {
                    console.warn(`⚠️ Bone ${bvhJoint} (${boneData.vrmBone}) has no parent but is not root`);
                    valid = false;
                }
                
                // Check for position consistency (should not be modified during animation)
                if (this.originalRestPose && this.originalRestPose[bvhJoint]) {
                    const originalPos = this.originalRestPose[bvhJoint].position;
                    const currentPos = bone.position;
                    const posDiff = Math.sqrt(
                        Math.pow(currentPos.x - originalPos.x, 2) +
                        Math.pow(currentPos.y - originalPos.y, 2) +
                        Math.pow(currentPos.z - originalPos.z, 2)
                    );
                    
                    if (posDiff > 0.001) { // More than 1mm difference
                        console.warn(`⚠️ Bone ${bvhJoint} position has drifted from original: ${posDiff.toFixed(6)}m`);
                    }
                }
            }
        });
        
        if (valid) {
            console.log('✅ VRM bone hierarchy is valid');
        } else {
            console.warn('⚠️ VRM bone hierarchy has issues');
        }
        
        return valid;
    }

    // Coordinate conversion configuration methods
    setCoordinateConfig(config) {
        if (config.flipX !== undefined) this.coordinateConversion.flipX = config.flipX;
        if (config.flipY !== undefined) this.coordinateConversion.flipY = config.flipY;
        if (config.flipZ !== undefined) this.coordinateConversion.flipZ = config.flipZ;
        if (config.flipRotX !== undefined) this.coordinateConversion.flipRotX = config.flipRotX;
        if (config.flipRotY !== undefined) this.coordinateConversion.flipRotY = config.flipRotY;
        if (config.flipRotZ !== undefined) this.coordinateConversion.flipRotZ = config.flipRotZ;
        if (config.rotationOrder !== undefined) this.coordinateConversion.rotationOrder = config.rotationOrder;
        if (config.positionScale !== undefined) this.coordinateConversion.positionScale = config.positionScale;
        
        console.log('🔧 VRMBVHAdapter coordinate config updated:', this.coordinateConversion);
        return true;
    }

    getCoordinateConfig() {
        return { ...this.coordinateConversion };
    }

    // Manual head orientation fix
    fixHeadOrientation() {
        if (!this.availableBones || !this.initialized) {
            console.warn('⚠️ VRMBVHAdapter not initialized - cannot fix head orientation');
            return false;
        }
        
        console.log('🔧 Manually fixing head orientation...');
        
        // Fix head bone if available
        const headBone = this.availableBones['Head'];
        if (headBone && headBone.bone) {
            // Apply manual head lift to counteract downward looking
            const currentRotX = headBone.bone.rotation.x;
            headBone.bone.rotation.x = Math.max(currentRotX - 0.3, -Math.PI/6); // Lift head up, max 30 degrees
            console.log(`✅ Head rotation X adjusted: ${currentRotX.toFixed(3)} -> ${headBone.bone.rotation.x.toFixed(3)}`);
        }
        
        // Fix neck bone if available
        const neckBone = this.availableBones['Neck'];
        if (neckBone && neckBone.bone) {
            // Apply manual neck adjustment
            const currentRotX = neckBone.bone.rotation.x;
            neckBone.bone.rotation.x = Math.max(currentRotX - 0.2, -Math.PI/8); // Lift neck, max 22.5 degrees
            console.log(`✅ Neck rotation X adjusted: ${currentRotX.toFixed(3)} -> ${neckBone.bone.rotation.x.toFixed(3)}`);
        }
        
        // Fix upper chest to help with overall posture
        const upperChestBone = this.availableBones['Chest3'] || this.availableBones['Chest4'];
        if (upperChestBone && upperChestBone.bone) {
            // Slight backward tilt of upper chest to improve posture
            upperChestBone.bone.rotation.x -= 0.1; // Tilt back slightly
            console.log('✅ Upper chest posture adjusted');
        }
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ Head orientation manually fixed');
        return true;
    }

    // Reset head to neutral looking forward position
    resetHeadToNeutral() {
        if (!this.availableBones || !this.initialized) {
            console.warn('⚠️ VRMBVHAdapter not initialized - cannot reset head');
            return false;
        }
        
        console.log('🔄 Resetting head to neutral position...');
        
        // Reset head bone
        const headBone = this.availableBones['Head'];
        if (headBone && headBone.bone) {
            headBone.bone.rotation.set(0, 0, 0); // Neutral head position
            console.log('✅ Head reset to neutral');
        }
        
        // Reset neck bone
        const neckBone = this.availableBones['Neck'];
        if (neckBone && neckBone.bone) {
            neckBone.bone.rotation.set(0, 0, 0); // Neutral neck position
            console.log('✅ Neck reset to neutral');
        }
        
        // Update VRM
        if (this.vrmModel && this.vrmModel.update) {
            this.vrmModel.update(0.016);
        }
        
        console.log('✅ Head orientation reset to neutral');
        return true;
    }

    // Quick fix methods for common issues
    toggleFlipRotX() {
        this.coordinateConversion.flipRotX = !this.coordinateConversion.flipRotX;
        console.log('🔧 Toggled flipRotX in VRMBVHAdapter:', this.coordinateConversion.flipRotX);
        return this.coordinateConversion.flipRotX;
    }

    toggleFlipRotY() {
        this.coordinateConversion.flipRotY = !this.coordinateConversion.flipRotY;
        console.log('🔧 Toggled flipRotY in VRMBVHAdapter:', this.coordinateConversion.flipRotY);
        return this.coordinateConversion.flipRotY;
    }

    toggleFlipRotZ() {
        this.coordinateConversion.flipRotZ = !this.coordinateConversion.flipRotZ;
        console.log('🔧 Toggled flipRotZ in VRMBVHAdapter:', this.coordinateConversion.flipRotZ);
        return this.coordinateConversion.flipRotZ;
    }

    toggleFlipZ() {
        this.coordinateConversion.flipZ = !this.coordinateConversion.flipZ;
        console.log('🔧 Toggled flipZ in VRMBVHAdapter:', this.coordinateConversion.flipZ);
        return this.coordinateConversion.flipZ;
    }

    setRotationOrder(order) {
        const validOrders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'];
        if (validOrders.includes(order)) {
            this.coordinateConversion.rotationOrder = order;
            console.log('🔄 VRMBVHAdapter rotation order:', order);
            return true;
        } else {
            console.log('❌ Invalid rotation order. Valid: XYZ, XZY, YXZ, YZX, ZXY, ZYX');
            return false;
        }
    }
}

    // Export for use in other modules
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = VRMBVHAdapter;
    }

// Global export for browser
if (typeof window !== 'undefined') {
    window.VRMBVHAdapter = VRMBVHAdapter;
}
