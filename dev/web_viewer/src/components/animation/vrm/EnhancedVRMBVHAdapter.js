/**
 * Enhanced VRM BVH Adapter - Based on working chat implementation
 * Uses correct VRM 3.0+ API patterns for proper animation
 */

class EnhancedVRMBVHAdapter {
    constructor(vrmObject, scene) {
        try {
            this.vrm = vrmObject; // The VRM object itself, not wrapped
            this.scene = scene;
            this.initialized = false;
            this.debugMode = true;
            
            // Validate inputs
            if (!vrmObject) {
                throw new Error('VRM object is required');
            }
            if (!scene) {
                throw new Error('Scene is required');
            }
            
            // Animation state
            this.clock = new THREE.Clock();
            this.breathingPhase = 0;
            this.blinkTimer = 0;
            this.nextBlinkTime = 3;
            this.isAnimating = true;
            
            // BVH to VRM bone mapping - ULTRA-COMPREHENSIVE mapping for maximum compatibility
            this.boneMapping = {
                // Root and spine hierarchy - Multiple naming conventions
                'Hips': 'hips',
                'Root': 'hips',
                'Hip': 'hips',
                'Pelvis': 'hips',
                'mixamorig:Hips': 'hips',  // Mixamo naming convention
                'Bip01 Pelvis': 'hips',    // 3DS Max Biped naming
                'Bip001 Pelvis': 'hips',   // Alternative Biped naming
                
                // Spine chain - Comprehensive BVH hierarchy mapping
                'Chest': 'spine',
                'Chest2': 'chest', 
                'Chest3': 'upperChest',
                'Chest4': 'neck',
                'Neck': 'neck',
                'Head': 'head',
                'Spine': 'spine',
                'Spine1': 'chest',
                'Spine2': 'upperChest',
                'Spine3': 'neck',
                'UpperChest': 'upperChest',
                'Lower Spine': 'spine',
                'Middle Spine': 'chest',
                'Upper Spine': 'upperChest',
                'Torso': 'chest',
                'mixamorig:Spine': 'spine',
                'mixamorig:Spine1': 'chest',
                'mixamorig:Spine2': 'upperChest',
                'mixamorig:Neck': 'neck',
                'mixamorig:Head': 'head',
                'Bip01 Spine': 'spine',
                'Bip01 Spine1': 'chest',
                'Bip01 Spine2': 'upperChest',
                'Bip01 Neck': 'neck',
                'Bip01 Head': 'head',
                
                // Eye tracking - Multiple naming conventions
                'LeftEye': 'leftEye',
                'RightEye': 'rightEye',
                'Left Eye': 'leftEye',
                'Right Eye': 'rightEye',
                'L_Eye': 'leftEye',
                'R_Eye': 'rightEye',
                'eye.L': 'leftEye',
                'eye.R': 'rightEye',
                'mixamorig:LeftEye': 'leftEye',
                'mixamorig:RightEye': 'rightEye',
                
                // Right arm - Ultra-comprehensive hierarchical chain
                'RightCollar': 'rightShoulder',
                'RightShoulder': 'rightUpperArm',
                'RightUpperArm': 'rightUpperArm',
                'RightElbow': 'rightLowerArm',
                'RightLowerArm': 'rightLowerArm',
                'RightWrist': 'rightHand',
                'RightHand': 'rightHand',
                'Right Shoulder': 'rightShoulder',
                'Right Upper Arm': 'rightUpperArm',
                'Right Lower Arm': 'rightLowerArm',
                'Right Hand': 'rightHand',
                'R_Shoulder': 'rightShoulder',
                'R_UpperArm': 'rightUpperArm',
                'R_LowerArm': 'rightLowerArm',
                'R_Hand': 'rightHand',
                'RightArm': 'rightUpperArm',
                'RightForeArm': 'rightLowerArm',
                'shoulder.R': 'rightShoulder',
                'upper_arm.R': 'rightUpperArm',
                'forearm.R': 'rightLowerArm',
                'hand.R': 'rightHand',
                'mixamorig:RightShoulder': 'rightShoulder',
                'mixamorig:RightArm': 'rightUpperArm',
                'mixamorig:RightForeArm': 'rightLowerArm',
                'mixamorig:RightHand': 'rightHand',
                'Bip01 R Clavicle': 'rightShoulder',
                'Bip01 R UpperArm': 'rightUpperArm',
                'Bip01 R Forearm': 'rightLowerArm',
                'Bip01 R Hand': 'rightHand',
                
                // Left arm - Ultra-comprehensive hierarchical chain
                'LeftCollar': 'leftShoulder',
                'LeftShoulder': 'leftUpperArm',
                'LeftUpperArm': 'leftUpperArm',
                'LeftElbow': 'leftLowerArm',
                'LeftLowerArm': 'leftLowerArm',
                'LeftWrist': 'leftHand',
                'LeftHand': 'leftHand',
                'Left Shoulder': 'leftShoulder',
                'Left Upper Arm': 'leftUpperArm',
                'Left Lower Arm': 'leftLowerArm',
                'Left Hand': 'leftHand',
                'L_Shoulder': 'leftShoulder',
                'L_UpperArm': 'leftUpperArm',
                'L_LowerArm': 'leftLowerArm',
                'L_Hand': 'leftHand',
                'LeftArm': 'leftUpperArm',
                'LeftForeArm': 'leftLowerArm',
                'shoulder.L': 'leftShoulder',
                'upper_arm.L': 'leftUpperArm',
                'forearm.L': 'leftLowerArm',
                'hand.L': 'leftHand',
                'mixamorig:LeftShoulder': 'leftShoulder',
                'mixamorig:LeftArm': 'leftUpperArm',
                'mixamorig:LeftForeArm': 'leftLowerArm',
                'mixamorig:LeftHand': 'leftHand',
                'Bip01 L Clavicle': 'leftShoulder',
                'Bip01 L UpperArm': 'leftUpperArm',
                'Bip01 L Forearm': 'leftLowerArm',
                'Bip01 L Hand': 'leftHand',
                
                // Right leg - Ultra-comprehensive hierarchical chain
                'RightHip': 'rightUpperLeg',
                'RightUpperLeg': 'rightUpperLeg',
                'RightKnee': 'rightLowerLeg',
                'RightLowerLeg': 'rightLowerLeg',
                'RightAnkle': 'rightFoot',
                'RightFoot': 'rightFoot',
                'RightToe': 'rightToes',
                'RightToes': 'rightToes',
                'Right Hip': 'rightUpperLeg',
                'Right Upper Leg': 'rightUpperLeg',
                'Right Lower Leg': 'rightLowerLeg',
                'Right Foot': 'rightFoot',
                'Right Toe': 'rightToes',
                'R_Hip': 'rightUpperLeg',
                'R_UpperLeg': 'rightUpperLeg',
                'R_LowerLeg': 'rightLowerLeg',
                'R_Foot': 'rightFoot',
                'R_Toe': 'rightToes',
                'RightLeg': 'rightUpperLeg',
                'RightShin': 'rightLowerLeg',
                'RightThigh': 'rightUpperLeg',
                'thigh.R': 'rightUpperLeg',
                'shin.R': 'rightLowerLeg',
                'foot.R': 'rightFoot',
                'toe.R': 'rightToes',
                'mixamorig:RightUpLeg': 'rightUpperLeg',
                'mixamorig:RightLeg': 'rightLowerLeg',
                'mixamorig:RightFoot': 'rightFoot',
                'mixamorig:RightToeBase': 'rightToes',
                'Bip01 R Thigh': 'rightUpperLeg',
                'Bip01 R Calf': 'rightLowerLeg',
                'Bip01 R Foot': 'rightFoot',
                'Bip01 R Toe0': 'rightToes',
                
                // Left leg - Ultra-comprehensive hierarchical chain
                'LeftHip': 'leftUpperLeg',
                'LeftUpperLeg': 'leftUpperLeg',
                'LeftKnee': 'leftLowerLeg',
                'LeftLowerLeg': 'leftLowerLeg',
                'LeftAnkle': 'leftFoot',
                'LeftFoot': 'leftFoot',
                'LeftToe': 'leftToes',
                'LeftToes': 'leftToes',
                'Left Hip': 'leftUpperLeg',
                'Left Upper Leg': 'leftUpperLeg',
                'Left Lower Leg': 'leftLowerLeg',
                'Left Foot': 'leftFoot',
                'Left Toe': 'leftToes',
                'L_Hip': 'leftUpperLeg',
                'L_UpperLeg': 'leftUpperLeg',
                'L_LowerLeg': 'leftLowerLeg',
                'L_Foot': 'leftFoot',
                'L_Toe': 'leftToes',
                'LeftLeg': 'leftUpperLeg',
                'LeftShin': 'leftLowerLeg',
                'LeftThigh': 'leftUpperLeg',
                'thigh.L': 'leftUpperLeg',
                'shin.L': 'leftLowerLeg',
                'foot.L': 'leftFoot',
                'toe.L': 'leftToes',
                'mixamorig:LeftUpLeg': 'leftUpperLeg',
                'mixamorig:LeftLeg': 'leftLowerLeg',
                'mixamorig:LeftFoot': 'leftFoot',
                'mixamorig:LeftToeBase': 'leftToes',
                'Bip01 L Thigh': 'leftUpperLeg',
                'Bip01 L Calf': 'leftLowerLeg',
                'Bip01 L Foot': 'leftFoot',
                'Bip01 L Toe0': 'leftToes',
                
                // Ultra-comprehensive finger bones - Right hand
                'RightThumb1': 'rightThumbMetacarpal',
                'RightThumb2': 'rightThumbProximal',
                'RightThumb3': 'rightThumbDistal',
                'RightThumbMetacarpal': 'rightThumbMetacarpal',
                'RightThumbProximal': 'rightThumbProximal',
                'RightThumbDistal': 'rightThumbDistal',
                'RightIndex1': 'rightIndexProximal',
                'RightIndex2': 'rightIndexIntermediate',
                'RightIndex3': 'rightIndexDistal',
                'RightIndexProximal': 'rightIndexProximal',
                'RightIndexIntermediate': 'rightIndexIntermediate',
                'RightIndexDistal': 'rightIndexDistal',
                'RightMiddle1': 'rightMiddleProximal',
                'RightMiddle2': 'rightMiddleIntermediate',
                'RightMiddle3': 'rightMiddleDistal',
                'RightMiddleProximal': 'rightMiddleProximal',
                'RightMiddleIntermediate': 'rightMiddleIntermediate',
                'RightMiddleDistal': 'rightMiddleDistal',
                'RightRing1': 'rightRingProximal',
                'RightRing2': 'rightRingIntermediate',
                'RightRing3': 'rightRingDistal',
                'RightRingProximal': 'rightRingProximal',
                'RightRingIntermediate': 'rightRingIntermediate',
                'RightRingDistal': 'rightRingDistal',
                'RightPinky1': 'rightLittleProximal',
                'RightPinky2': 'rightLittleIntermediate',
                'RightPinky3': 'rightLittleDistal',
                'RightLittle1': 'rightLittleProximal',
                'RightLittle2': 'rightLittleIntermediate',
                'RightLittle3': 'rightLittleDistal',
                'RightLittleProximal': 'rightLittleProximal',
                'RightLittleIntermediate': 'rightLittleIntermediate',
                'RightLittleDistal': 'rightLittleDistal',
                // Mixamo finger naming
                'mixamorig:RightHandThumb1': 'rightThumbMetacarpal',
                'mixamorig:RightHandThumb2': 'rightThumbProximal',
                'mixamorig:RightHandThumb3': 'rightThumbDistal',
                'mixamorig:RightHandIndex1': 'rightIndexProximal',
                'mixamorig:RightHandIndex2': 'rightIndexIntermediate',
                'mixamorig:RightHandIndex3': 'rightIndexDistal',
                'mixamorig:RightHandMiddle1': 'rightMiddleProximal',
                'mixamorig:RightHandMiddle2': 'rightMiddleIntermediate',
                'mixamorig:RightHandMiddle3': 'rightMiddleDistal',
                'mixamorig:RightHandRing1': 'rightRingProximal',
                'mixamorig:RightHandRing2': 'rightRingIntermediate',
                'mixamorig:RightHandRing3': 'rightRingDistal',
                'mixamorig:RightHandPinky1': 'rightLittleProximal',
                'mixamorig:RightHandPinky2': 'rightLittleIntermediate',
                'mixamorig:RightHandPinky3': 'rightLittleDistal',
                // Biped finger naming
                'Bip01 R Finger0': 'rightThumbMetacarpal',
                'Bip01 R Finger01': 'rightThumbProximal',
                'Bip01 R Finger02': 'rightThumbDistal',
                'Bip01 R Finger1': 'rightIndexProximal',
                'Bip01 R Finger11': 'rightIndexIntermediate',
                'Bip01 R Finger12': 'rightIndexDistal',
                'Bip01 R Finger2': 'rightMiddleProximal',
                'Bip01 R Finger21': 'rightMiddleIntermediate',
                'Bip01 R Finger22': 'rightMiddleDistal',
                'Bip01 R Finger3': 'rightRingProximal',
                'Bip01 R Finger31': 'rightRingIntermediate',
                'Bip01 R Finger32': 'rightRingDistal',
                'Bip01 R Finger4': 'rightLittleProximal',
                'Bip01 R Finger41': 'rightLittleIntermediate',
                'Bip01 R Finger42': 'rightLittleDistal',
                
                // Ultra-comprehensive finger bones - Left hand
                'LeftThumb1': 'leftThumbMetacarpal',
                'LeftThumb2': 'leftThumbProximal',
                'LeftThumb3': 'leftThumbDistal',
                'LeftThumbMetacarpal': 'leftThumbMetacarpal',
                'LeftThumbProximal': 'leftThumbProximal',
                'LeftThumbDistal': 'leftThumbDistal',
                'LeftIndex1': 'leftIndexProximal',
                'LeftIndex2': 'leftIndexIntermediate',
                'LeftIndex3': 'leftIndexDistal',
                'LeftIndexProximal': 'leftIndexProximal',
                'LeftIndexIntermediate': 'leftIndexIntermediate',
                'LeftIndexDistal': 'leftIndexDistal',
                'LeftMiddle1': 'leftMiddleProximal',
                'LeftMiddle2': 'leftMiddleIntermediate',
                'LeftMiddle3': 'leftMiddleDistal',
                'LeftMiddleProximal': 'leftMiddleProximal',
                'LeftMiddleIntermediate': 'leftMiddleIntermediate',
                'LeftMiddleDistal': 'leftMiddleDistal',
                'LeftRing1': 'leftRingProximal',
                'LeftRing2': 'leftRingIntermediate',
                'LeftRing3': 'leftRingDistal',
                'LeftRingProximal': 'leftRingProximal',
                'LeftRingIntermediate': 'leftRingIntermediate',
                'LeftRingDistal': 'leftRingDistal',
                'LeftPinky1': 'leftLittleProximal',
                'LeftPinky2': 'leftLittleIntermediate',
                'LeftPinky3': 'leftLittleDistal',
                'LeftLittle1': 'leftLittleProximal',
                'LeftLittle2': 'leftLittleIntermediate',
                'LeftLittle3': 'leftLittleDistal',
                'LeftLittleProximal': 'leftLittleProximal',
                'LeftLittleIntermediate': 'leftLittleIntermediate',
                'LeftLittleDistal': 'leftLittleDistal',
                // Mixamo finger naming
                'mixamorig:LeftHandThumb1': 'leftThumbMetacarpal',
                'mixamorig:LeftHandThumb2': 'leftThumbProximal',
                'mixamorig:LeftHandThumb3': 'leftThumbDistal',
                'mixamorig:LeftHandIndex1': 'leftIndexProximal',
                'mixamorig:LeftHandIndex2': 'leftIndexIntermediate',
                'mixamorig:LeftHandIndex3': 'leftIndexDistal',
                'mixamorig:LeftHandMiddle1': 'leftMiddleProximal',
                'mixamorig:LeftHandMiddle2': 'leftMiddleIntermediate',
                'mixamorig:LeftHandMiddle3': 'leftMiddleDistal',
                'mixamorig:LeftHandRing1': 'leftRingProximal',
                'mixamorig:LeftHandRing2': 'leftRingIntermediate',
                'mixamorig:LeftHandRing3': 'leftRingDistal',
                'mixamorig:LeftHandPinky1': 'leftLittleProximal',
                'mixamorig:LeftHandPinky2': 'leftLittleIntermediate',
                'mixamorig:LeftHandPinky3': 'leftLittleDistal',
                // Biped finger naming
                'Bip01 L Finger0': 'leftThumbMetacarpal',
                'Bip01 L Finger01': 'leftThumbProximal',
                'Bip01 L Finger02': 'leftThumbDistal',
                'Bip01 L Finger1': 'leftIndexProximal',
                'Bip01 L Finger11': 'leftIndexIntermediate',
                'Bip01 L Finger12': 'leftIndexDistal',
                'Bip01 L Finger2': 'leftMiddleProximal',
                'Bip01 L Finger21': 'leftMiddleIntermediate',
                'Bip01 L Finger22': 'leftMiddleDistal',
                'Bip01 L Finger3': 'leftRingProximal',
                'Bip01 L Finger31': 'leftRingIntermediate',
                'Bip01 L Finger32': 'leftRingDistal',
                'Bip01 L Finger4': 'leftLittleProximal',
                'Bip01 L Finger41': 'leftLittleIntermediate',
                'Bip01 L Finger42': 'leftLittleDistal'
            };
            
            this.availableBones = {};
            
            console.log('🎭 Enhanced VRM BVH Adapter created successfully');
            console.log('VRM object type:', typeof vrmObject);
            console.log('VRM humanoid available:', !!vrmObject?.humanoid);
            console.log('🚀 ULTRA-COMPREHENSIVE bone mapping defined:', Object.keys(this.boneMapping).length, 'BVH->VRM mappings');
            console.log('📊 Supports: Standard, Mixamo, 3DS Max Biped, and custom naming conventions');
            
            // Auto-initialize if possible
            if (vrmObject?.humanoid) {
                try {
                    this.initialize();
                    
                    // Automatically validate bone mapping in debug mode
                    if (this.debugMode) {
                        setTimeout(() => {
                            this.validateBoneMapping();
                        }, 100);
                    }
                } catch (initError) {
                    console.warn('⚠️ Auto-initialization failed:', initError.message);
                    console.log('🔧 Manual initialization will be available');
                }
            }
            
        } catch (error) {
            console.error('❌ Enhanced VRM Adapter construction failed:', error.message);
            throw error;
        }
    }
    
    initialize() {
        if (!this.vrm || !this.vrm.humanoid) {
            console.error('❌ VRM or humanoid not available for Enhanced Adapter');
            return false;
        }
        
        if (!this.boneMapping) {
            console.error('❌ Bone mapping not initialized');
            return false;
        }
        
        console.log('🔧 Initializing Enhanced VRM-BVH adapter...');
        console.log('Available bone mappings:', Object.keys(this.boneMapping).length);
        
        // Use getRawBoneNode API like the working implementation
        const humanoid = this.vrm.humanoid;
        let mappedCount = 0;
        
        // Configure VRM for animation
        if (humanoid.autoUpdateHumanBones !== undefined) {
            humanoid.autoUpdateHumanBones = true;
        }
        
        if (this.vrm.lookAt && this.vrm.lookAt.autoUpdate !== undefined) {
            this.vrm.lookAt.autoUpdate = true;
        }
        
        // Initialize available bones storage
        this.availableBones = {};
        
        // First, get ALL available VRM bones for comprehensive mapping validation
        const allVRMBones = [];
        const vrmBoneNames = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 
                             'leftEye', 'rightEye', 'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
                             'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
                             'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
                             'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes'];
        
        for (const vrmBoneName of vrmBoneNames) {
            try {
                const bone = humanoid.getRawBoneNode(vrmBoneName);
                if (bone) {
                    allVRMBones.push(vrmBoneName);
                }
            } catch (e) {
                // Bone not available
            }
        }
        
        console.log(`🦴 Found ${allVRMBones.length} available VRM bones:`, allVRMBones);
        
        // Map bones using correct API with detailed validation
        for (const [bvhJoint, vrmBone] of Object.entries(this.boneMapping)) {
            try {
                const boneNode = humanoid.getRawBoneNode(vrmBone);
                
                if (boneNode) {
                    this.availableBones[bvhJoint] = {
                        bone: boneNode,
                        vrmBone: vrmBone,
                        originalName: boneNode.name || 'unnamed'
                    };
                    mappedCount++;
                    
                    if (this.debugMode) {
                        console.log(`✅ Mapped: ${bvhJoint} -> ${vrmBone} (${boneNode.name})`);
                    }
                } else {
                    if (this.debugMode) {
                        console.warn(`❌ BVH bone '${bvhJoint}' -> VRM bone '${vrmBone}' returned null`);
                    }
                }
            } catch (error) {
                if (this.debugMode) {
                    console.warn(`⚠️ Could not map ${bvhJoint} -> ${vrmBone}:`, error.message);
                }
            }
        }
        
        // Check for missing critical bones
        const criticalBones = ['hips', 'spine', 'chest', 'neck', 'head', 
                              'leftUpperArm', 'leftLowerArm', 'rightUpperArm', 'rightLowerArm',
                              'leftUpperLeg', 'leftLowerLeg', 'rightUpperLeg', 'rightLowerLeg'];
        
        const missingCritical = criticalBones.filter(bone => 
            !Object.values(this.availableBones).some(mapped => mapped.vrmBone === bone)
        );
        
        if (missingCritical.length > 0) {
            console.warn(`⚠️ Missing critical bones: ${missingCritical.join(', ')}`);
        }
        
        console.log(`🎭 Enhanced VRM-BVH adapter: ${mappedCount} bones mapped successfully`);
        
        if (mappedCount === 0) {
            console.error('❌ No bones mapped! VRM might not be compatible.');
            return false;
        }
        
        // Configure VRM position and orientation based on chat implementation
        if (this.vrm.scene) {
            // Position the VRM properly
            this.vrm.scene.position.set(0, 0, 0);
            
            // Set initial pose to match VRM standard pose
            this.setInitialVRMPose();
            
            console.log('🎭 VRM positioned for direct BVH bone control like legacy rsmt_showcase');
        }
        
        this.initialized = true;
        return true;
    }
    
    // Comprehensive bone mapping validation method
    validateBoneMapping() {
        if (!this.vrm || !this.vrm.humanoid) {
            console.error('❌ VRM or humanoid not available for bone validation');
            return false;
        }
        
        console.log('🔍 === COMPREHENSIVE BONE MAPPING VALIDATION ===');
        
        // Get all available VRM bones
        const allVRMBones = [];
        const standardVRMBones = [
            'hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 
            'leftEye', 'rightEye', 'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
            'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
            'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
            'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
            // Finger bones
            'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal',
            'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
            'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
            'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
            'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
            'rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal',
            'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
            'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
            'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
            'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal'
        ];
        
        for (const vrmBoneName of standardVRMBones) {
            try {
                const bone = this.vrm.humanoid.getRawBoneNode(vrmBoneName);
                if (bone) {
                    allVRMBones.push({
                        name: vrmBoneName,
                        bone: bone,
                        originalName: bone.name || 'unnamed',
                        mapped: Object.values(this.availableBones).some(mapped => mapped.vrmBone === vrmBoneName)
                    });
                }
            } catch (e) {
                // Bone not available in this VRM
            }
        }
        
        console.log(`🦴 Found ${allVRMBones.length} total VRM bones`);
        console.log(`✅ Mapped ${Object.keys(this.availableBones).length} BVH joints to VRM bones`);
        
        // Show mapped bones
        console.log('📋 MAPPED BONES:');
        for (const [bvhJoint, boneData] of Object.entries(this.availableBones)) {
            console.log(`  ${bvhJoint} -> ${boneData.vrmBone} (${boneData.originalName})`);
        }
        
        // Show unmapped VRM bones
        const unmappedBones = allVRMBones.filter(bone => !bone.mapped);
        if (unmappedBones.length > 0) {
            console.log('⚠️ UNMAPPED VRM BONES:');
            unmappedBones.forEach(bone => {
                console.log(`  ${bone.name} (${bone.originalName})`);
            });
        }
        
        // Check for critical bone coverage
        const criticalBones = ['hips', 'spine', 'chest', 'neck', 'head', 
                              'leftUpperArm', 'leftLowerArm', 'rightUpperArm', 'rightLowerArm',
                              'leftUpperLeg', 'leftLowerLeg', 'rightUpperLeg', 'rightLowerLeg'];
        
        const mappedCritical = criticalBones.filter(bone => 
            Object.values(this.availableBones).some(mapped => mapped.vrmBone === bone)
        );
        
        console.log(`✅ Critical bone coverage: ${mappedCritical.length}/${criticalBones.length}`);
        
        const missingCritical = criticalBones.filter(bone => !mappedCritical.includes(bone));
        if (missingCritical.length > 0) {
            console.warn(`❌ Missing critical bones: ${missingCritical.join(', ')}`);
        }
        
        console.log('🔍 === BONE VALIDATION COMPLETE ===');
        
        return {
            totalVRMBones: allVRMBones.length,
            mappedBones: Object.keys(this.availableBones).length,
            criticalCoverage: `${mappedCritical.length}/${criticalBones.length}`,
            unmappedBones: unmappedBones.map(b => b.name),
            missingCritical: missingCritical
        };
    }
    
    applyBVHFrameToVRM(frameData) {
        if (!this.initialized && !this.initialize()) {
            return false;
        }
        
        if (!frameData || frameData.length < 6) {
            return false;
        }
        
        try {
            // === PROPER VRM BVH TRANSFORMATION ===
            // Deep analysis of coordinate system differences and T-pose alignment
            let channelIndex = 0;
            
            // Root position (first 3 channels) - BVH typically in centimeters
            const rootX = frameData[channelIndex++] || 0;
            const rootY = frameData[channelIndex++] || 0;
            const rootZ = frameData[channelIndex++] || 0;
            
            // Root rotation (next 3 channels) - BVH standard YXZ order
            const rootRotY = (frameData[channelIndex++] || 0) * Math.PI / 180; // Y: Yaw (turning)
            const rootRotX = (frameData[channelIndex++] || 0) * Math.PI / 180; // X: Pitch (lean)
            const rootRotZ = (frameData[channelIndex++] || 0) * Math.PI / 180; // Z: Roll
            
            // === COORDINATE SYSTEM TRANSFORMATION ===
            // Transform BVH coordinate system to match VRM expectations
            if (this.vrm.scene) {
                // Scale conversion: BVH (cm) -> VRM (meters)
                const BVH_TO_METERS = 0.01;
                const TYPICAL_HIP_HEIGHT = 0.98; // Standard VRM character hip height in meters
                
                // VRM coordinate system: Y-up, forward is -Z, right is +X
                // BVH coordinate system: Y-up, forward varies (often +Z), right is +X
                
                // Position transformation with proper coordinate mapping
                this.vrm.scene.position.set(
                    rootX * BVH_TO_METERS,                         // X: lateral movement (same)
                    (rootY * BVH_TO_METERS) - TYPICAL_HIP_HEIGHT,  // Y: vertical with ground adjustment
                    -rootZ * BVH_TO_METERS                         // Z: forward/back (flip for VRM)
                );
                
                // Rotation transformation with VRM orientation correction
                this.vrm.scene.rotation.order = 'YXZ'; // Match BVH rotation order
                
                // Apply rotations with coordinate system correction
                this.vrm.scene.rotation.set(
                    rootRotX,                    // X: pitch (lean forward/back)
                    rootRotY + Math.PI,          // Y: yaw (turn) + 180° for VRM forward direction
                    rootRotZ                     // Z: roll (lean left/right)
                );
            }
            
            // === BONE ROTATION TRANSFORMATION WITH T-POSE ALIGNMENT ===
            // Apply joint rotations with proper VRM T-pose considerations
            const bvhJointOrder = [
                // Spine chain - all possible spine segments
                'Chest', 'Chest2', 'Chest3', 'Chest4', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
                
                // Right arm chain
                'RightCollar', 'RightShoulder', 'RightArm', 'RightElbow', 'RightWrist',
                
                // Left arm chain  
                'LeftCollar', 'LeftShoulder', 'LeftArm', 'LeftElbow', 'LeftWrist',
                
                // Right leg chain
                'RightHip', 'RightLeg', 'RightKnee', 'RightAnkle', 'RightToe',
                
                // Left leg chain
                'LeftHip', 'LeftLeg', 'LeftKnee', 'LeftAnkle', 'LeftToe'
            ];
            
            bvhJointOrder.forEach(bvhJoint => {
                if (channelIndex + 2 >= frameData.length) return;
                
                const boneData = this.availableBones[bvhJoint];
                if (boneData && boneData.bone) {
                    // Extract BVH rotation data (YXZ order)
                    const rotY = (frameData[channelIndex++] || 0) * Math.PI / 180; // Y: yaw
                    const rotX = (frameData[channelIndex++] || 0) * Math.PI / 180; // X: pitch  
                    const rotZ = (frameData[channelIndex++] || 0) * Math.PI / 180; // Z: roll
                    
                    // Apply T-pose alignment transformations based on bone type
                    let finalRotX = rotX;
                    let finalRotY = rotY;
                    let finalRotZ = rotZ;
                    
                    // === T-POSE SPECIFIC TRANSFORMATIONS ===
                    // VRM characters start in T-pose (arms horizontal), BVH often assumes A-pose
                    
                    if (bvhJoint.includes('Shoulder') || bvhJoint.includes('Collar')) {
                        // Shoulder bones: adjust for T-pose arm positioning
                        // VRM T-pose has arms horizontal, may need shoulder adjustment
                        if (bvhJoint.includes('Right')) {
                            finalRotZ += Math.PI / 6;  // Slight adjustment for right shoulder
                        } else if (bvhJoint.includes('Left')) {
                            finalRotZ -= Math.PI / 6;  // Slight adjustment for left shoulder
                        }
                    }
                    
                    if (bvhJoint.includes('Arm') || bvhJoint.includes('UpperArm')) {
                        // Upper arm bones: major T-pose correction needed
                        // VRM T-pose: arms horizontal along X-axis
                        if (bvhJoint.includes('Right')) {
                            finalRotZ -= Math.PI / 2;  // Rotate right arm from vertical to horizontal
                        } else if (bvhJoint.includes('Left')) {
                            finalRotZ += Math.PI / 2;  // Rotate left arm from vertical to horizontal
                        }
                    }
                    
                    if (bvhJoint.includes('Hip') && !bvhJoint.includes('Right') && !bvhJoint.includes('Left')) {
                        // Root hip bone: coordinate system alignment
                        finalRotY = rotY + Math.PI; // Align hip forward direction with VRM
                    }
                    
                    // Set rotation order for consistent bone transformation
                    boneData.bone.rotation.order = 'YXZ';
                    
                    // Apply final transformed rotations
                    boneData.bone.rotation.set(
                        finalRotX,  // X: pitch (up/down)
                        finalRotY,  // Y: yaw (left/right turn)
                        finalRotZ   // Z: roll (lean)
                    );
                } else {
                    // Skip these 3 channels if bone not available
                    channelIndex += 3;
                }
            });
            
            // Update VRM using correct method like chat system
            const deltaTime = this.clock.getDelta();
            if (this.vrm.update) {
                this.vrm.update(deltaTime);
            }
            
            // Debug logging for coordinate transformations (every 60 frames)
            if (this.debugMode && Math.floor(this.clock.getElapsedTime() * 60) % 60 === 0) {
                console.log('🔄 VRM Coordinate Transform Applied:');
                console.log(`  Root: pos(${this.vrm.scene.position.x.toFixed(2)}, ${this.vrm.scene.position.y.toFixed(2)}, ${this.vrm.scene.position.z.toFixed(2)}) rot(${(this.vrm.scene.rotation.x * 180/Math.PI).toFixed(1)}°, ${(this.vrm.scene.rotation.y * 180/Math.PI).toFixed(1)}°, ${(this.vrm.scene.rotation.z * 180/Math.PI).toFixed(1)}°)`);
                console.log(`  T-pose corrections applied for ${Object.keys(this.availableBones).length} bones`);
            }
            
            return true;
            
        } catch (error) {
            console.error('❌ Enhanced VRM adapter error:', error);
            return false;
        }
    }
    
    setupAnimationMixer() {
        if (!this.vrm || !this.vrm.scene) return;
        
        // Create animation mixer like chat avatar system
        this.animationMixer = new THREE.AnimationMixer(this.vrm.scene);
        this.currentClip = null;
        this.currentAction = null;
        
        console.log('🎬 Animation mixer setup for VRM like chat system');
    }
    
    applyBVHToAnimationMixer(frameData) {
        if (!this.animationMixer || !frameData) return;
        
        // Convert BVH frame data to animation tracks that work with VRM system
        const bvhJointOrder = [
            'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
            'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
            'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
            'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
        ];
        
        let channelIndex = 6;
        
        // Apply BVH rotations to available bones using natural VRM bone system
        for (const bvhJoint of bvhJointOrder) {
            if (channelIndex + 2 >= frameData.length) break;
            
            const boneData = this.availableBones[bvhJoint];
            if (boneData && boneData.bone) {
                // Extract BVH rotations (YXZ order) - natural application
                const rotY = frameData[channelIndex] * Math.PI / 180;     // Y rotation
                const rotX = frameData[channelIndex + 1] * Math.PI / 180; // X rotation  
                const rotZ = frameData[channelIndex + 2] * Math.PI / 180; // Z rotation
                
                // Apply to VRM bone naturally without fighting the system
                boneData.bone.rotation.set(rotX, rotY, rotZ);
                boneData.bone.rotation.order = 'YXZ'; // Match BVH order
            }
            
            channelIndex += 3;
        }
    }
    
    setInitialVRMPose() {
        if (!this.vrm || !this.vrm.humanoid) return;
        
        // Simply reset all bones to neutral pose - let BVH data handle positioning
        for (const [bvhJoint, boneData] of Object.entries(this.availableBones)) {
            if (boneData && boneData.bone) {
                boneData.bone.rotation.set(0, 0, 0);
                boneData.bone.rotation.order = 'YXZ';
            }
        }
        
        console.log('🎭 VRM initial pose set - neutral position for BVH control');
    }

    // Alias for backward compatibility
    applyBVHFrame(frameData) {
        return this.applyBVHFrameToVRM(frameData);
    }
    
    // Method expected by the animation loop - simplified like chat avatar system
    tick(deltaTime) {
        if (!this.vrm) return;
        
        // Update VRM using correct method like chat system
        if (this.vrm.update) {
            this.vrm.update(deltaTime);
        }
        
        // Update animation mixer like chat avatar system
        if (this.animationMixer) {
            this.animationMixer.update(deltaTime);
        }
        
        // Only apply very subtle facial animations (no bone conflicts)
        this.updateSubtleIdleAnimations(deltaTime);
    }
    
    updateSubtleIdleAnimations(deltaTime) {
        if (!this.isAnimating || !this.vrm.humanoid) return;
        
        // Only apply very subtle facial animations that don't conflict with BVH bone data
        // Blinking only - no bone rotations that compete with BVH
        this.blinkTimer += deltaTime;
        if (this.blinkTimer >= this.nextBlinkTime) {
            this.triggerBlink();
            this.blinkTimer = 0;
            this.nextBlinkTime = Math.random() * 4 + 2;
        }
    }
    
    updateIdleAnimations(deltaTime) {
        // Full idle animations when not using BVH data
        if (!this.isAnimating || !this.vrm.humanoid) return;
        
        // Breathing animation
        this.breathingPhase += deltaTime * 2;
        const breathingIntensity = Math.sin(this.breathingPhase) * 0.015;
        
        const chest = this.vrm.humanoid.getRawBoneNode('chest');
        const spine = this.vrm.humanoid.getRawBoneNode('spine');
        
        if (chest) {
            chest.rotation.x += breathingIntensity;
        }
        if (spine) {
            spine.rotation.x += breathingIntensity * 0.5;
        }
        
        // Blinking
        this.blinkTimer += deltaTime;
        if (this.blinkTimer >= this.nextBlinkTime) {
            this.triggerBlink();
            this.blinkTimer = 0;
            this.nextBlinkTime = Math.random() * 4 + 2;
        }
        
        // Subtle head movement
        const headSway = Math.sin(this.breathingPhase * 0.3) * 0.008;
        const head = this.vrm.humanoid.getRawBoneNode('head');
        if (head) {
            head.rotation.y += headSway;
        }
    }
    
    triggerBlink() {
        if (this.vrm.expressionManager) {
            this.vrm.expressionManager.setValue('blink', 1.0);
            
            setTimeout(() => {
                if (this.vrm.expressionManager) {
                    this.vrm.expressionManager.setValue('blink', 0.0);
                }
            }, 120);
        }
    }
    
    // Compatibility method for old character system
    setIdleAnimations(config) {
        console.log('🎭 Setting idle animations config:', config);
        // The enhanced adapter handles idle animations automatically in updateIdleAnimations()
        // This method is for compatibility with the old character system
        if (config.breathing !== undefined) {
            this.isAnimating = config.breathing.enabled;
        }
        return true;
    }
    
    // Compatibility method for camera interaction - simplified like chat implementation
    lookAtCameraAsIfHuman(camera) {
        if (!this.vrm || !this.vrm.lookAt) return;
        
        try {
            // Simple look-at behavior like the working chat implementation
            const cameraPosition = camera.position.clone();
            
            // Use VRM's lookAt system naturally without complex offset calculations
            if (this.vrm.lookAt.target) {
                this.vrm.lookAt.target.copy(cameraPosition);
                
                // Enable auto look-at like in chat implementation
                if (this.vrm.lookAt.autoUpdate !== undefined) {
                    this.vrm.lookAt.autoUpdate = true;
                }
            }
            
            console.log('👀 VRM looking at camera - simple mode');
        } catch (error) {
            console.warn('⚠️ Could not make VRM look at camera:', error.message);
        }
    }
    
    // Simple root transform method like chat implementation
    setRootTransform(transform) {
        if (!this.vrm || !this.vrm.scene) return;
        
        try {
            if (transform.position) {
                this.vrm.scene.position.copy(transform.position);
            }
            if (transform.rotation) {
                this.vrm.scene.rotation.copy(transform.rotation);
            }
            if (transform.scale) {
                this.vrm.scene.scale.copy(transform.scale);
            }
            
            console.log('🎭 VRM root transform set');
        } catch (error) {
            console.warn('⚠️ Could not set VRM root transform:', error.message);
        }
    }
    
    getAvailableBones() {
        return Object.keys(this.availableBones);
    }
    
    setDebugMode(enabled) {
        this.debugMode = enabled;
    }
    
    setAnimationEnabled(enabled) {
        this.isAnimating = enabled;
    }
    
    dispose() {
        this.availableBones = {};
        this.initialized = false;
    }
}

// Make globally available
window.EnhancedVRMBVHAdapter = EnhancedVRMBVHAdapter;

// Global bone mapping validation function
window.validateVRMBoneMapping = function() {
    // Check multiple possible locations for the VRM adapter
    let vrmAdapter = null;
    
    if (window.currentVRMAdapter) {
        vrmAdapter = window.currentVRMAdapter;
        console.log('🔍 Found VRM adapter at window.currentVRMAdapter');
    } else if (window.characterSystem && window.characterSystem.vrmAdapter) {
        vrmAdapter = window.characterSystem.vrmAdapter;
        console.log('🔍 Found VRM adapter at window.characterSystem.vrmAdapter');
    } else if (window.vrmCharacter && window.vrmCharacter.adapter) {
        vrmAdapter = window.vrmCharacter.adapter;
        console.log('🔍 Found VRM adapter at window.vrmCharacter.adapter');
    }
    
    if (vrmAdapter && typeof vrmAdapter.validateBoneMapping === 'function') {
        console.log('✅ Running comprehensive bone mapping validation...');
        return vrmAdapter.validateBoneMapping();
    } else {
        console.warn('⚠️ No VRM adapter found with validateBoneMapping method.');
        console.log('💡 Available objects:', {
            currentVRMAdapter: !!window.currentVRMAdapter,
            characterSystem: !!window.characterSystem,
            vrmCharacter: !!window.vrmCharacter
        });
        return null;
    }
};

console.log('✅ Enhanced VRM BVH Adapter loaded - based on working chat implementation');
console.log('🔧 Use validateVRMBoneMapping() to check bone mapping completeness');
