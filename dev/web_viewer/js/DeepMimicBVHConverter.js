/**
 * DeepMimic BVH Converter
 * 
 * Converts DeepMimic neural network output to BVH animation format
 * for integration with the BVH Timeline compositor system.
 * 
 * DeepMimic generates physics-based character animation using deep reinforcement learning.
 * This converter handles the output from DeepMimic models and converts it to BVH format
 * for timeline-based animation composition.
 */

class DeepMimicBVHConverter {
    constructor(options = {}) {
        // Core conversion options
        this.scaleFactor = options.scaleFactor || 1.0;
        this.frameRate = options.frameRate || 30;
        this.smoothing = options.smoothing !== false;
        this.smoothingFactor = options.smoothingFactor || 0.8;
        
        // DeepMimic specific options
        this.physicsIntegration = options.physicsIntegration !== false;
        this.contactConstraints = options.contactConstraints !== false;
        this.rootMotionEnabled = options.rootMotionEnabled !== false;
        this.characterScale = options.characterScale || 1.0;
        
        // Animation style options
        this.motionStyle = options.motionStyle || 'natural'; // 'natural', 'athletic', 'dramatic', 'precise'
        this.energyLevel = options.energyLevel || 1.0; // 0.1 to 2.0
        this.stabilityFactor = options.stabilityFactor || 1.0; // Balance vs dynamics
        
        // DeepMimic bone structure (humanoid character)
        this.boneStructure = this.initializeBoneStructure();
        
        // Physics simulation parameters
        this.physicsParams = {
            gravity: options.gravity || -9.81,
            groundHeight: options.groundHeight || 0.0,
            friction: options.friction || 0.8,
            damping: options.damping || 0.95,
            springStiffness: options.springStiffness || 1000.0,
            springDamping: options.springDamping || 50.0
        };
        
        // Contact detection and constraints
        this.contactPoints = new Map(); // Track foot/hand contacts
        this.constraintSolver = new DeepMimicConstraintSolver(this.physicsParams);
        
        // Motion quality metrics
        this.qualityMetrics = {
            stability: 0,
            naturalness: 0,
            energyEfficiency: 0,
            goalAchievement: 0
        };
        
        // Frame processing cache
        this.frameCache = new Map();
        this.maxCacheSize = options.maxCacheSize || 1000;
        
        // Statistics tracking
        this.stats = {
            framesProcessed: 0,
            physicsViolations: 0,
            contactEvents: 0,
            averageProcessingTime: 0,
            totalProcessingTime: 0
        };
        
        console.log('[DeepMimic BVH Converter] Initialized with options:', {
            scaleFactor: this.scaleFactor,
            frameRate: this.frameRate,
            motionStyle: this.motionStyle,
            physicsIntegration: this.physicsIntegration,
            rootMotionEnabled: this.rootMotionEnabled
        });
    }
    
    /**
     * Initialize DeepMimic bone structure mapping
     */
    initializeBoneStructure() {
        return {
            // Root and pelvis
            root: { index: 0, parent: null, children: ['pelvis'], type: 'root' },
            pelvis: { index: 1, parent: 'root', children: ['spine', 'leftHip', 'rightHip'], type: 'core' },
            
            // Spine chain
            spine: { index: 2, parent: 'pelvis', children: ['spine1'], type: 'spine' },
            spine1: { index: 3, parent: 'spine', children: ['spine2'], type: 'spine' },
            spine2: { index: 4, parent: 'spine1', children: ['neck', 'leftShoulder', 'rightShoulder'], type: 'spine' },
            neck: { index: 5, parent: 'spine2', children: ['head'], type: 'neck' },
            head: { index: 6, parent: 'neck', children: [], type: 'head' },
            
            // Left arm chain
            leftShoulder: { index: 7, parent: 'spine2', children: ['leftArm'], type: 'shoulder' },
            leftArm: { index: 8, parent: 'leftShoulder', children: ['leftForeArm'], type: 'arm' },
            leftForeArm: { index: 9, parent: 'leftArm', children: ['leftHand'], type: 'forearm' },
            leftHand: { index: 10, parent: 'leftForeArm', children: [], type: 'hand' },
            
            // Right arm chain
            rightShoulder: { index: 11, parent: 'spine2', children: ['rightArm'], type: 'shoulder' },
            rightArm: { index: 12, parent: 'rightShoulder', children: ['rightForeArm'], type: 'arm' },
            rightForeArm: { index: 13, parent: 'rightArm', children: ['rightHand'], type: 'forearm' },
            rightHand: { index: 14, parent: 'rightForeArm', children: [], type: 'hand' },
            
            // Left leg chain
            leftHip: { index: 15, parent: 'pelvis', children: ['leftThigh'], type: 'hip' },
            leftThigh: { index: 16, parent: 'leftHip', children: ['leftShin'], type: 'thigh' },
            leftShin: { index: 17, parent: 'leftThigh', children: ['leftFoot'], type: 'shin' },
            leftFoot: { index: 18, parent: 'leftShin', children: ['leftToe'], type: 'foot' },
            leftToe: { index: 19, parent: 'leftFoot', children: [], type: 'toe' },
            
            // Right leg chain
            rightHip: { index: 20, parent: 'pelvis', children: ['rightThigh'], type: 'hip' },
            rightThigh: { index: 21, parent: 'rightHip', children: ['rightShin'], type: 'thigh' },
            rightShin: { index: 22, parent: 'rightThigh', children: ['rightFoot'], type: 'shin' },
            rightFoot: { index: 23, parent: 'rightShin', children: ['rightToe'], type: 'foot' },
            rightToe: { index: 24, parent: 'rightFoot', children: [], type: 'toe' }
        };
    }
    
    /**
     * Convert DeepMimic output to BVH timeline clip
     */
    createTimelineClip(deepMimicOutput, referenceData, startTime = 0) {
        const startProcessing = performance.now();
        
        try {
            console.log('[DeepMimic BVH] Creating timeline clip from DeepMimic output');
            
            // Parse DeepMimic output format
            const parsedData = this.parseDeepMimicOutput(deepMimicOutput);
            
            // Generate BVH frames from DeepMimic data
            const bvhFrames = this.generateBVHFrames(parsedData, referenceData);
            
            // Apply physics constraints and quality improvements
            const refinedFrames = this.applyPhysicsConstraints(bvhFrames);
            
            // Calculate motion metrics
            const motionMetrics = this.calculateMotionMetrics(refinedFrames);
            
            // Create timeline clip structure
            const clip = {
                id: `deepmimic_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                startTime: startTime,
                duration: refinedFrames.length / this.frameRate,
                frameRate: this.frameRate,
                frames: refinedFrames,
                
                // DeepMimic specific metadata
                metadata: {
                    source: 'deepmimic',
                    converter: 'DeepMimicBVHConverter',
                    version: '1.0.0',
                    motionStyle: this.motionStyle,
                    energyLevel: this.energyLevel,
                    physicsEnabled: this.physicsIntegration,
                    rootMotionEnabled: this.rootMotionEnabled,
                    qualityMetrics: motionMetrics,
                    characterScale: this.characterScale,
                    originalFrameCount: parsedData.frames?.length || 0,
                    processedFrameCount: refinedFrames.length,
                    processingTime: performance.now() - startProcessing
                },
                
                // Animation properties
                loop: false,
                blendMode: 'override', // DeepMimic usually provides complete poses
                priority: 100, // High priority for physics-based animation
                channels: ['root', 'body', 'arms', 'legs', 'spine', 'head'],
                
                // Physics properties
                physicsConstraints: this.extractPhysicsConstraints(refinedFrames),
                contactEvents: this.extractContactEvents(refinedFrames),
                rootMotion: this.extractRootMotion(refinedFrames)
            };
            
            // Update statistics
            this.updateStats(performance.now() - startProcessing, refinedFrames.length);
            
            console.log('[DeepMimic BVH] Timeline clip created successfully:', {
                id: clip.id,
                duration: clip.duration,
                frameCount: clip.frames.length,
                motionStyle: clip.metadata.motionStyle,
                qualityScore: motionMetrics.overall
            });
            
            return clip;
            
        } catch (error) {
            console.error('[DeepMimic BVH] Error creating timeline clip:', error);
            throw new Error(`DeepMimic BVH conversion failed: ${error.message}`);
        }
    }
    
    /**
     * Parse DeepMimic output format
     */
    parseDeepMimicOutput(deepMimicOutput) {
        console.log('[DeepMimic BVH] Parsing DeepMimic output format');
        
        // Handle different DeepMimic output formats
        if (deepMimicOutput.poses) {
            // Standard pose format
            return this.parsePoseFormat(deepMimicOutput.poses);
        } else if (deepMimicOutput.states) {
            // State-action format from RL training
            return this.parseStateActionFormat(deepMimicOutput.states, deepMimicOutput.actions);
        } else if (deepMimicOutput.motion_data) {
            // Motion capture format
            return this.parseMotionDataFormat(deepMimicOutput.motion_data);
        } else if (Array.isArray(deepMimicOutput)) {
            // Direct frame array
            return this.parseFrameArray(deepMimicOutput);
        } else {
            // Try to auto-detect format
            return this.autoDetectFormat(deepMimicOutput);
        }
    }
    
    /**
     * Parse pose format (joint positions/rotations)
     */
    parsePoseFormat(poses) {
        const frames = [];
        
        poses.forEach((pose, frameIndex) => {
            const frame = {
                frameNumber: frameIndex,
                timestamp: frameIndex / this.frameRate * 1000,
                bones: {},
                physics: {
                    rootPosition: pose.root_pos || [0, 0, 0],
                    rootRotation: pose.root_rot || [0, 0, 0, 1],
                    rootVelocity: pose.root_vel || [0, 0, 0],
                    rootAngularVelocity: pose.root_ang_vel || [0, 0, 0]
                },
                contacts: pose.contacts || []
            };
            
            // Process joint data
            if (pose.joint_pos && pose.joint_rot) {
                this.processPoseJoints(frame, pose.joint_pos, pose.joint_rot, pose.joint_vel);
            } else if (pose.dof_pos) {
                this.processDOFData(frame, pose.dof_pos, pose.dof_vel);
            }
            
            frames.push(frame);
        });
        
        return { frames, format: 'pose' };
    }
    
    /**
     * Parse state-action format from reinforcement learning
     */
    parseStateActionFormat(states, actions) {
        const frames = [];
        
        states.forEach((state, frameIndex) => {
            const action = actions[frameIndex] || {};
            
            const frame = {
                frameNumber: frameIndex,
                timestamp: frameIndex / this.frameRate * 1000,
                bones: {},
                physics: {
                    rootPosition: state.slice(0, 3) || [0, 0, 0],
                    rootRotation: state.slice(3, 7) || [0, 0, 0, 1],
                    rootVelocity: state.slice(7, 10) || [0, 0, 0],
                    rootAngularVelocity: state.slice(10, 13) || [0, 0, 0]
                },
                action: action,
                reward: action.reward || 0,
                done: action.done || false
            };
            
            // Extract joint information from state vector
            this.extractJointsFromState(frame, state);
            
            frames.push(frame);
        });
        
        return { frames, format: 'state_action' };
    }
    
    /**
     * Process joint positions and rotations
     */
    processPoseJoints(frame, jointPos, jointRot, jointVel = null) {
        Object.keys(this.boneStructure).forEach(boneName => {
            const boneInfo = this.boneStructure[boneName];
            const index = boneInfo.index;
            
            if (index < jointPos.length / 3 && index < jointRot.length / 4) {
                const posIndex = index * 3;
                const rotIndex = index * 4;
                const velIndex = jointVel ? index * 3 : null;
                
                frame.bones[boneName] = {
                    position: {
                        x: jointPos[posIndex] * this.scaleFactor,
                        y: jointPos[posIndex + 1] * this.scaleFactor,
                        z: jointPos[posIndex + 2] * this.scaleFactor
                    },
                    rotation: this.quaternionToEuler(
                        jointRot[rotIndex],     // x
                        jointRot[rotIndex + 1], // y
                        jointRot[rotIndex + 2], // z
                        jointRot[rotIndex + 3]  // w
                    ),
                    velocity: jointVel ? {
                        x: jointVel[velIndex],
                        y: jointVel[velIndex + 1],
                        z: jointVel[velIndex + 2]
                    } : { x: 0, y: 0, z: 0 }
                };
            }
        });
    }
    
    /**
     * Generate BVH frames from parsed DeepMimic data
     */
    generateBVHFrames(parsedData, referenceData) {
        console.log('[DeepMimic BVH] Generating BVH frames');
        
        const bvhFrames = [];
        
        parsedData.frames.forEach((frame, index) => {
            const bvhFrame = {
                frameNumber: frame.frameNumber,
                timestamp: frame.timestamp,
                motionData: [],
                bones: {},
                physics: frame.physics,
                metadata: {
                    source: 'deepmimic',
                    motionStyle: this.motionStyle,
                    energyLevel: this.energyLevel,
                    contacts: frame.contacts || []
                }
            };
            
            // Convert bone data to BVH format
            Object.keys(this.boneStructure).forEach(boneName => {
                const boneData = frame.bones[boneName];
                if (boneData) {
                    // Apply motion style modifications
                    const styledBone = this.applyMotionStyle(boneData, boneName);
                    
                    // Store in BVH format
                    bvhFrame.bones[boneName] = styledBone;
                    
                    // Create motion data array (position + rotation)
                    const motionValues = [
                        styledBone.position.x,
                        styledBone.position.y,
                        styledBone.position.z,
                        styledBone.rotation.x,
                        styledBone.rotation.y,
                        styledBone.rotation.z
                    ];
                    
                    bvhFrame.motionData.push(...motionValues);
                }
            });
            
            bvhFrames.push(bvhFrame);
        });
        
        // Apply temporal smoothing if enabled
        if (this.smoothing) {
            return this.applySmoothingFilter(bvhFrames);
        }
        
        return bvhFrames;
    }
    
    /**
     * Apply motion style modifications
     */
    applyMotionStyle(boneData, boneName) {
        const styledBone = JSON.parse(JSON.stringify(boneData)); // Deep copy
        
        switch (this.motionStyle) {
            case 'athletic':
                return this.applyAthleticStyle(styledBone, boneName);
            case 'dramatic':
                return this.applyDramaticStyle(styledBone, boneName);
            case 'precise':
                return this.applyPreciseStyle(styledBone, boneName);
            case 'natural':
            default:
                return this.applyNaturalStyle(styledBone, boneName);
        }
    }
    
    /**
     * Apply athletic motion style
     */
    applyAthleticStyle(boneData, boneName) {
        const athleticMultiplier = 1.0 + (this.energyLevel - 1.0) * 0.5;
        
        // Enhance power movements
        if (boneName.includes('Thigh') || boneName.includes('Shin')) {
            boneData.rotation.x *= athleticMultiplier;
            boneData.rotation.z *= athleticMultiplier;
        }
        
        // Add dynamic arm movements
        if (boneName.includes('Arm') || boneName.includes('ForeArm')) {
            boneData.rotation.y *= athleticMultiplier;
        }
        
        // Enhance spine engagement
        if (boneName.includes('spine')) {
            boneData.rotation.x *= 1.1;
        }
        
        return boneData;
    }
    
    /**
     * Apply dramatic motion style
     */
    applyDramaticStyle(boneData, boneName) {
        const dramaticMultiplier = 1.2 + (this.energyLevel - 1.0) * 0.3;
        
        // Exaggerate all movements
        boneData.rotation.x *= dramaticMultiplier;
        boneData.rotation.y *= dramaticMultiplier;
        boneData.rotation.z *= dramaticMultiplier;
        
        // Special emphasis on expressive bones
        if (boneName === 'head' || boneName.includes('Shoulder')) {
            const emphasisMultiplier = 1.5;
            boneData.rotation.x *= emphasisMultiplier;
            boneData.rotation.y *= emphasisMultiplier;
        }
        
        return boneData;
    }
    
    /**
     * Apply precise motion style
     */
    applyPreciseStyle(boneData, boneName) {
        const precisionFactor = 0.8 + (this.energyLevel - 1.0) * 0.1;
        
        // Reduce movement magnitude for precision
        boneData.rotation.x *= precisionFactor;
        boneData.rotation.y *= precisionFactor;
        boneData.rotation.z *= precisionFactor;
        
        // Enhance stability
        if (boneName === 'pelvis' || boneName === 'spine') {
            boneData.rotation.x *= 0.9;
            boneData.rotation.z *= 0.9;
        }
        
        return boneData;
    }
    
    /**
     * Apply natural motion style (default)
     */
    applyNaturalStyle(boneData, boneName) {
        // Apply subtle energy level adjustments
        const naturalMultiplier = 0.95 + (this.energyLevel - 1.0) * 0.1;
        
        boneData.rotation.x *= naturalMultiplier;
        boneData.rotation.y *= naturalMultiplier;
        boneData.rotation.z *= naturalMultiplier;
        
        return boneData;
    }
    
    /**
     * Apply physics constraints and quality improvements
     */
    applyPhysicsConstraints(frames) {
        if (!this.physicsIntegration) {
            return frames;
        }
        
        console.log('[DeepMimic BVH] Applying physics constraints');
        
        const constrainedFrames = [];
        
        frames.forEach((frame, index) => {
            const constrainedFrame = JSON.parse(JSON.stringify(frame));
            
            // Apply joint limits
            this.applyJointLimits(constrainedFrame);
            
            // Resolve contact constraints
            if (this.contactConstraints) {
                this.resolveContactConstraints(constrainedFrame, index, frames);
            }
            
            // Apply stability corrections
            this.applyStabilityCorrections(constrainedFrame, index, frames);
            
            // Update physics properties
            this.updatePhysicsProperties(constrainedFrame, index, frames);
            
            constrainedFrames.push(constrainedFrame);
        });
        
        return constrainedFrames;
    }
    
    /**
     * Apply joint angle limits
     */
    applyJointLimits(frame) {
        Object.keys(frame.bones).forEach(boneName => {
            const bone = frame.bones[boneName];
            const limits = this.getJointLimits(boneName);
            
            if (limits) {
                bone.rotation.x = Math.max(limits.x.min, Math.min(limits.x.max, bone.rotation.x));
                bone.rotation.y = Math.max(limits.y.min, Math.min(limits.y.max, bone.rotation.y));
                bone.rotation.z = Math.max(limits.z.min, Math.min(limits.z.max, bone.rotation.z));
            }
        });
    }
    
    /**
     * Get anatomical joint limits for a bone
     */
    getJointLimits(boneName) {
        const limits = {
            // Spine limits (in radians)
            spine: { x: { min: -0.5, max: 0.8 }, y: { min: -0.3, max: 0.3 }, z: { min: -0.4, max: 0.4 } },
            spine1: { x: { min: -0.4, max: 0.6 }, y: { min: -0.2, max: 0.2 }, z: { min: -0.3, max: 0.3 } },
            spine2: { x: { min: -0.3, max: 0.5 }, y: { min: -0.2, max: 0.2 }, z: { min: -0.3, max: 0.3 } },
            
            // Neck and head limits
            neck: { x: { min: -0.6, max: 0.6 }, y: { min: -0.8, max: 0.8 }, z: { min: -0.4, max: 0.4 } },
            head: { x: { min: -0.3, max: 0.3 }, y: { min: -0.5, max: 0.5 }, z: { min: -0.2, max: 0.2 } },
            
            // Arm limits
            leftShoulder: { x: { min: -1.5, max: 3.0 }, y: { min: -1.0, max: 2.0 }, z: { min: -0.5, max: 1.5 } },
            rightShoulder: { x: { min: -1.5, max: 3.0 }, y: { min: -2.0, max: 1.0 }, z: { min: -1.5, max: 0.5 } },
            leftArm: { x: { min: -0.5, max: 2.5 }, y: { min: -1.0, max: 1.0 }, z: { min: -2.5, max: 0.5 } },
            rightArm: { x: { min: -0.5, max: 2.5 }, y: { min: -1.0, max: 1.0 }, z: { min: -0.5, max: 2.5 } },
            leftForeArm: { x: { min: 0, max: 2.5 }, y: { min: -0.1, max: 0.1 }, z: { min: -0.1, max: 0.1 } },
            rightForeArm: { x: { min: 0, max: 2.5 }, y: { min: -0.1, max: 0.1 }, z: { min: -0.1, max: 0.1 } },
            
            // Leg limits
            leftThigh: { x: { min: -1.5, max: 0.5 }, y: { min: -0.3, max: 0.3 }, z: { min: -0.8, max: 0.8 } },
            rightThigh: { x: { min: -1.5, max: 0.5 }, y: { min: -0.3, max: 0.3 }, z: { min: -0.8, max: 0.8 } },
            leftShin: { x: { min: 0, max: 2.3 }, y: { min: -0.1, max: 0.1 }, z: { min: -0.1, max: 0.1 } },
            rightShin: { x: { min: 0, max: 2.3 }, y: { min: -0.1, max: 0.1 }, z: { min: -0.1, max: 0.1 } },
            leftFoot: { x: { min: -0.8, max: 0.6 }, y: { min: -0.3, max: 0.3 }, z: { min: -0.2, max: 0.2 } },
            rightFoot: { x: { min: -0.8, max: 0.6 }, y: { min: -0.3, max: 0.3 }, z: { min: -0.2, max: 0.2 } }
        };
        
        return limits[boneName] || null;
    }
    
    /**
     * Resolve contact constraints (foot-ground contact)
     */
    resolveContactConstraints(frame, frameIndex, allFrames) {
        if (!frame.metadata.contacts) return;
        
        frame.metadata.contacts.forEach(contact => {
            if (contact.type === 'foot_ground') {
                const footBone = contact.bone;
                if (frame.bones[footBone]) {
                    // Ensure foot stays on ground during contact
                    if (contact.active) {
                        frame.bones[footBone].position.y = Math.max(
                            this.physicsParams.groundHeight,
                            frame.bones[footBone].position.y
                        );
                    }
                }
            }
        });
    }
    
    /**
     * Apply stability corrections
     */
    applyStabilityCorrections(frame, frameIndex, allFrames) {
        if (frameIndex === 0) return;
        
        const prevFrame = allFrames[frameIndex - 1];
        const stabilityThreshold = 1.0 / this.stabilityFactor;
        
        // Check for excessive movement
        Object.keys(frame.bones).forEach(boneName => {
            const currentBone = frame.bones[boneName];
            const prevBone = prevFrame.bones[boneName];
            
            if (currentBone && prevBone) {
                // Calculate rotation change
                const rotChange = Math.abs(currentBone.rotation.x - prevBone.rotation.x) +
                               Math.abs(currentBone.rotation.y - prevBone.rotation.y) +
                               Math.abs(currentBone.rotation.z - prevBone.rotation.z);
                
                // Apply stability correction if change is too large
                if (rotChange > stabilityThreshold) {
                    const blendFactor = stabilityThreshold / rotChange;
                    currentBone.rotation.x = this.lerp(prevBone.rotation.x, currentBone.rotation.x, blendFactor);
                    currentBone.rotation.y = this.lerp(prevBone.rotation.y, currentBone.rotation.y, blendFactor);
                    currentBone.rotation.z = this.lerp(prevBone.rotation.z, currentBone.rotation.z, blendFactor);
                }
            }
        });
    }
    
    /**
     * Calculate motion quality metrics
     */
    calculateMotionMetrics(frames) {
        const metrics = {
            stability: 0,
            naturalness: 0,
            energyEfficiency: 0,
            goalAchievement: 0,
            overall: 0
        };
        
        if (frames.length < 2) return metrics;
        
        let totalMovement = 0;
        let stabilityScore = 0;
        let energyScore = 0;
        
        // Analyze frame-to-frame changes
        for (let i = 1; i < frames.length; i++) {
            const currentFrame = frames[i];
            const prevFrame = frames[i - 1];
            
            let frameMovement = 0;
            let frameStability = 0;
            
            Object.keys(currentFrame.bones).forEach(boneName => {
                const currentBone = currentFrame.bones[boneName];
                const prevBone = prevFrame.bones[boneName];
                
                if (currentBone && prevBone) {
                    // Calculate movement magnitude
                    const movement = Math.abs(currentBone.rotation.x - prevBone.rotation.x) +
                                   Math.abs(currentBone.rotation.y - prevBone.rotation.y) +
                                   Math.abs(currentBone.rotation.z - prevBone.rotation.z);
                    
                    frameMovement += movement;
                    
                    // Calculate stability (consistency of movement)
                    if (i > 1) {
                        const prevPrevBone = frames[i - 2].bones[boneName];
                        if (prevPrevBone) {
                            const prevMovement = Math.abs(prevBone.rotation.x - prevPrevBone.rotation.x) +
                                               Math.abs(prevBone.rotation.y - prevPrevBone.rotation.y) +
                                               Math.abs(prevBone.rotation.z - prevPrevBone.rotation.z);
                            
                            frameStability += 1.0 - Math.abs(movement - prevMovement);
                        }
                    }
                }
            });
            
            totalMovement += frameMovement;
            stabilityScore += frameStability;
        }
        
        // Calculate metrics
        const avgMovement = totalMovement / (frames.length - 1);
        metrics.stability = Math.max(0, Math.min(1, stabilityScore / (frames.length - 2)));
        metrics.naturalness = Math.max(0, Math.min(1, 1.0 - Math.abs(avgMovement - 0.5) * 2));
        metrics.energyEfficiency = Math.max(0, Math.min(1, 1.0 - avgMovement * 0.5));
        metrics.goalAchievement = 0.8; // Placeholder - would need goal information
        
        // Overall score
        metrics.overall = (metrics.stability * 0.3 + 
                          metrics.naturalness * 0.3 + 
                          metrics.energyEfficiency * 0.2 + 
                          metrics.goalAchievement * 0.2);
        
        return metrics;
    }
    
    /**
     * Extract physics constraints from frames
     */
    extractPhysicsConstraints(frames) {
        const constraints = [];
        
        frames.forEach((frame, index) => {
            if (frame.metadata.contacts) {
                frame.metadata.contacts.forEach(contact => {
                    if (contact.active) {
                        constraints.push({
                            type: contact.type,
                            frame: index,
                            bone: contact.bone,
                            position: contact.position,
                            normal: contact.normal,
                            force: contact.force || 0
                        });
                    }
                });
            }
        });
        
        return constraints;
    }
    
    /**
     * Extract contact events (foot strikes, hand contacts, etc.)
     */
    extractContactEvents(frames) {
        const events = [];
        let inContact = new Set();
        
        frames.forEach((frame, index) => {
            const currentContacts = new Set();
            
            if (frame.metadata.contacts) {
                frame.metadata.contacts.forEach(contact => {
                    if (contact.active) {
                        currentContacts.add(contact.bone);
                        
                        // Check for new contact
                        if (!inContact.has(contact.bone)) {
                            events.push({
                                type: 'contact_start',
                                frame: index,
                                time: frame.timestamp / 1000,
                                bone: contact.bone,
                                contactType: contact.type,
                                position: contact.position
                            });
                        }
                    }
                });
            }
            
            // Check for ended contacts
            inContact.forEach(bone => {
                if (!currentContacts.has(bone)) {
                    events.push({
                        type: 'contact_end',
                        frame: index,
                        time: frame.timestamp / 1000,
                        bone: bone
                    });
                }
            });
            
            inContact = currentContacts;
        });
        
        return events;
    }
    
    /**
     * Extract root motion from frames
     */
    extractRootMotion(frames) {
        if (!this.rootMotionEnabled || frames.length === 0) {
            return { enabled: false };
        }
        
        const rootMotion = {
            enabled: true,
            positions: [],
            rotations: [],
            velocities: [],
            totalDistance: 0,
            averageSpeed: 0
        };
        
        let totalDistance = 0;
        
        frames.forEach((frame, index) => {
            const physics = frame.physics || {};
            
            rootMotion.positions.push(physics.rootPosition || [0, 0, 0]);
            rootMotion.rotations.push(physics.rootRotation || [0, 0, 0, 1]);
            rootMotion.velocities.push(physics.rootVelocity || [0, 0, 0]);
            
            // Calculate distance traveled
            if (index > 0) {
                const prevPos = rootMotion.positions[index - 1];
                const currPos = rootMotion.positions[index];
                const distance = Math.sqrt(
                    Math.pow(currPos[0] - prevPos[0], 2) +
                    Math.pow(currPos[1] - prevPos[1], 2) +
                    Math.pow(currPos[2] - prevPos[2], 2)
                );
                totalDistance += distance;
            }
        });
        
        rootMotion.totalDistance = totalDistance;
        rootMotion.averageSpeed = totalDistance / (frames.length / this.frameRate);
        
        return rootMotion;
    }
    
    /**
     * Apply smoothing filter to frames
     */
    applySmoothingFilter(frames) {
        if (frames.length < 3) return frames;
        
        const smoothedFrames = [frames[0]]; // Keep first frame as-is
        
        for (let i = 1; i < frames.length - 1; i++) {
            const prevFrame = frames[i - 1];
            const currentFrame = frames[i];
            const nextFrame = frames[i + 1];
            
            const smoothedFrame = JSON.parse(JSON.stringify(currentFrame));
            
            // Apply smoothing to bone rotations
            Object.keys(smoothedFrame.bones).forEach(boneName => {
                const prevBone = prevFrame.bones[boneName];
                const currentBone = smoothedFrame.bones[boneName];
                const nextBone = nextFrame.bones[boneName];
                
                if (prevBone && currentBone && nextBone) {
                    // Smooth rotations using weighted average
                    currentBone.rotation.x = this.smoothValue(
                        prevBone.rotation.x,
                        currentBone.rotation.x,
                        nextBone.rotation.x,
                        this.smoothingFactor
                    );
                    currentBone.rotation.y = this.smoothValue(
                        prevBone.rotation.y,
                        currentBone.rotation.y,
                        nextBone.rotation.y,
                        this.smoothingFactor
                    );
                    currentBone.rotation.z = this.smoothValue(
                        prevBone.rotation.z,
                        currentBone.rotation.z,
                        nextBone.rotation.z,
                        this.smoothingFactor
                    );
                }
            });
            
            smoothedFrames.push(smoothedFrame);
        }
        
        smoothedFrames.push(frames[frames.length - 1]); // Keep last frame as-is
        
        return smoothedFrames;
    }
    
    /**
     * Smooth a single value using neighboring values
     */
    smoothValue(prev, current, next, factor) {
        const avg = (prev + current + next) / 3;
        return current + (avg - current) * factor;
    }
    
    /**
     * Utility methods
     */
    
    // Convert quaternion to Euler angles
    quaternionToEuler(x, y, z, w) {
        // Roll (x-axis rotation)
        const sinr_cosp = 2 * (w * x + y * z);
        const cosr_cosp = 1 - 2 * (x * x + y * y);
        const roll = Math.atan2(sinr_cosp, cosr_cosp);
        
        // Pitch (y-axis rotation)
        const sinp = 2 * (w * y - z * x);
        const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
        
        // Yaw (z-axis rotation)
        const siny_cosp = 2 * (w * z + x * y);
        const cosy_cosp = 1 - 2 * (y * y + z * z);
        const yaw = Math.atan2(siny_cosp, cosy_cosp);
        
        return { x: roll, y: pitch, z: yaw };
    }
    
    // Linear interpolation
    lerp(a, b, t) {
        return a + (b - a) * t;
    }
    
    // Auto-detect DeepMimic output format
    autoDetectFormat(output) {
        console.log('[DeepMimic BVH] Auto-detecting output format');
        
        // Check for common DeepMimic properties
        if (output.joint_pos || output.dof_pos) {
            return this.parsePoseFormat([output]);
        }
        
        if (Array.isArray(output) && output.length > 0) {
            if (typeof output[0] === 'number') {
                // Flat array - likely state vector
                return this.parseStateVector(output);
            } else if (output[0].joint_pos || output[0].dof_pos) {
                return this.parsePoseFormat(output);
            }
        }
        
        // Default fallback
        console.warn('[DeepMimic BVH] Unknown format, creating default frame');
        return { frames: [this.createDefaultFrame()], format: 'unknown' };
    }
    
    // Create a default frame
    createDefaultFrame() {
        const frame = {
            frameNumber: 0,
            timestamp: 0,
            bones: {},
            physics: {
                rootPosition: [0, 0, 0],
                rootRotation: [0, 0, 0, 1],
                rootVelocity: [0, 0, 0],
                rootAngularVelocity: [0, 0, 0]
            }
        };
        
        // Initialize all bones to neutral pose
        Object.keys(this.boneStructure).forEach(boneName => {
            frame.bones[boneName] = {
                position: { x: 0, y: 0, z: 0 },
                rotation: { x: 0, y: 0, z: 0 },
                velocity: { x: 0, y: 0, z: 0 }
            };
        });
        
        return frame;
    }
    
    /**
     * Update processing statistics
     */
    updateStats(processingTime, frameCount) {
        this.stats.framesProcessed += frameCount;
        this.stats.totalProcessingTime += processingTime;
        this.stats.averageProcessingTime = 
            this.stats.totalProcessingTime / this.stats.framesProcessed;
    }
    
    /**
     * Get performance statistics
     */
    getStats() {
        return {
            ...this.stats,
            cacheSize: this.frameCache.size,
            options: {
                scaleFactor: this.scaleFactor,
                frameRate: this.frameRate,
                motionStyle: this.motionStyle,
                energyLevel: this.energyLevel,
                physicsIntegration: this.physicsIntegration,
                rootMotionEnabled: this.rootMotionEnabled
            }
        };
    }
    
    /**
     * Configuration methods
     */
    setMotionStyle(style) {
        this.motionStyle = style;
        console.log('[DeepMimic BVH] Motion style updated:', style);
    }
    
    setEnergyLevel(level) {
        this.energyLevel = Math.max(0.1, Math.min(2.0, level));
        console.log('[DeepMimic BVH] Energy level updated:', this.energyLevel);
    }
    
    setStabilityFactor(factor) {
        this.stabilityFactor = Math.max(0.1, Math.min(2.0, factor));
        console.log('[DeepMimic BVH] Stability factor updated:', this.stabilityFactor);
    }
    
    /**
     * Cleanup and disposal
     */
    dispose() {
        this.frameCache.clear();
        this.contactPoints.clear();
        
        if (this.constraintSolver) {
            this.constraintSolver.dispose();
        }
        
        console.log('[DeepMimic BVH Converter] Disposed');
    }
}

/**
 * DeepMimic Constraint Solver
 * 
 * Handles physics constraints and corrections for DeepMimic animations
 */
class DeepMimicConstraintSolver {
    constructor(physicsParams) {
        this.physicsParams = physicsParams;
        this.constraints = [];
        this.solverIterations = 10;
    }
    
    addConstraint(constraint) {
        this.constraints.push(constraint);
    }
    
    solve(frame) {
        // Implement constraint solving logic
        // This would be a full physics constraint solver
        // For now, just apply basic corrections
        
        for (let i = 0; i < this.solverIterations; i++) {
            this.constraints.forEach(constraint => {
                this.applyConstraint(frame, constraint);
            });
        }
    }
    
    applyConstraint(frame, constraint) {
        // Apply individual constraint corrections
        switch (constraint.type) {
            case 'position':
                this.applyPositionConstraint(frame, constraint);
                break;
            case 'distance':
                this.applyDistanceConstraint(frame, constraint);
                break;
            case 'angle':
                this.applyAngleConstraint(frame, constraint);
                break;
        }
    }
    
    applyPositionConstraint(frame, constraint) {
        // Keep a bone at a specific position
        if (frame.bones[constraint.bone]) {
            frame.bones[constraint.bone].position = { ...constraint.targetPosition };
        }
    }
    
    applyDistanceConstraint(frame, constraint) {
        // Maintain distance between two bones
        const bone1 = frame.bones[constraint.bone1];
        const bone2 = frame.bones[constraint.bone2];
        
        if (bone1 && bone2) {
            // Calculate current distance and adjust if needed
            // Implementation would go here
        }
    }
    
    applyAngleConstraint(frame, constraint) {
        // Maintain angle relationships between bones
        if (frame.bones[constraint.bone]) {
            // Implementation would go here
        }
    }
    
    dispose() {
        this.constraints = [];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DeepMimicBVHConverter, DeepMimicConstraintSolver };
} else {
    window.DeepMimicBVHConverter = DeepMimicBVHConverter;
    window.DeepMimicConstraintSolver = DeepMimicConstraintSolver;
}
