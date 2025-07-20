/**
 * Audio2Gesture to BVH Converter
 * 
 * This module converts Audio2Gesture neural network outputs (body gesture animation data)
 * to BVH format for integration with the BVH Timeline compositor system.
 * 
 * Audio2Gesture typically outputs joint positions, rotations, or pose parameters
 * which we convert to full body bone rotations compatible with BVH format.
 */

class Audio2GestureBVHConverter {
    constructor(options = {}) {
        this.options = {
            // Conversion settings
            scaleFactor: options.scaleFactor || 1.0,
            smoothing: options.smoothing !== false, // Default enabled
            smoothingFactor: options.smoothingFactor || 0.7,
            
            // Body part toggles
            enableUpperBody: options.enableUpperBody !== false,
            enableLowerBody: options.enableLowerBody !== false,
            enableFingers: options.enableFingers !== false,
            enableSpine: options.enableSpine !== false,
            
            // Gesture characteristics
            gestureIntensity: options.gestureIntensity || 1.0,
            emotionalModulation: options.emotionalModulation !== false,
            rhythmSensitivity: options.rhythmSensitivity || 0.8,
            
            // Frame rate and timing
            targetFrameRate: options.targetFrameRate || 30,
            maxLookAhead: options.maxLookAhead || 15, // frames for predictive smoothing
            
            // Audio analysis
            enableAudioFeatures: options.enableAudioFeatures !== false,
            pitchSensitivity: options.pitchSensitivity || 0.6,
            amplitudeSensitivity: options.amplitudeSensitivity || 0.8,
            
            // Debug options
            verbose: options.verbose || false,
            logConversions: options.logConversions || false
        };
        
        // Internal state
        this.frameHistory = [];
        this.maxHistorySize = 50; // Keep last 50 frames for smoothing
        this.audioFeatures = [];
        this.isInitialized = false;
        
        // Body bone structure for BVH (full body hierarchy)
        this.bodyBones = this.initializeBodyBoneStructure();
        
        // Gesture patterns and mappings
        this.gesturePatterns = this.initializeGesturePatterns();
        
        // Statistics
        this.stats = {
            framesConverted: 0,
            averageConversionTime: 0,
            lastConversionTime: 0,
            errorCount: 0,
            audioFeaturesExtracted: 0
        };
        
        console.log('[Audio2Gesture BVH Converter] Initialized with options:', this.options);
    }
    
    /**
     * Initialize the full body bone structure for BVH conversion
     */
    initializeBodyBoneStructure() {
        return {
            // Root and spine
            hips: { index: 0, parent: null, channels: ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation'] },
            spine: { index: 1, parent: 'hips', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            spine1: { index: 2, parent: 'spine', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            spine2: { index: 3, parent: 'spine1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            neck: { index: 4, parent: 'spine2', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            head: { index: 5, parent: 'neck', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left arm
            leftShoulder: { index: 6, parent: 'spine2', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftArm: { index: 7, parent: 'leftShoulder', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftForeArm: { index: 8, parent: 'leftArm', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftHand: { index: 9, parent: 'leftForeArm', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left fingers
            leftThumb1: { index: 10, parent: 'leftHand', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftThumb2: { index: 11, parent: 'leftThumb1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftIndex1: { index: 12, parent: 'leftHand', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftIndex2: { index: 13, parent: 'leftIndex1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftMiddle1: { index: 14, parent: 'leftHand', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftMiddle2: { index: 15, parent: 'leftMiddle1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right arm
            rightShoulder: { index: 16, parent: 'spine2', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightArm: { index: 17, parent: 'rightShoulder', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightForeArm: { index: 18, parent: 'rightArm', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightHand: { index: 19, parent: 'rightForeArm', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right fingers
            rightThumb1: { index: 20, parent: 'rightHand', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightThumb2: { index: 21, parent: 'rightThumb1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightIndex1: { index: 22, parent: 'rightHand', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightIndex2: { index: 23, parent: 'rightIndex1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightMiddle1: { index: 24, parent: 'rightHand', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightMiddle2: { index: 25, parent: 'rightMiddle1', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Left leg
            leftUpLeg: { index: 26, parent: 'hips', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftLeg: { index: 27, parent: 'leftUpLeg', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftFoot: { index: 28, parent: 'leftLeg', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftToe: { index: 29, parent: 'leftFoot', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Right leg
            rightUpLeg: { index: 30, parent: 'hips', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightLeg: { index: 31, parent: 'rightUpLeg', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightFoot: { index: 32, parent: 'rightLeg', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightToe: { index: 33, parent: 'rightFoot', channels: ['Xrotation', 'Yrotation', 'Zrotation'] }
        };
    }
    
    /**
     * Initialize gesture patterns for different emotional/contextual states
     */
    initializeGesturePatterns() {
        return {
            neutral: {
                armSwing: 0.3,
                spineMovement: 0.2,
                headNod: 0.1,
                handGestures: 0.2
            },
            excited: {
                armSwing: 0.8,
                spineMovement: 0.6,
                headNod: 0.4,
                handGestures: 0.7
            },
            calm: {
                armSwing: 0.1,
                spineMovement: 0.1,
                headNod: 0.05,
                handGestures: 0.1
            },
            emphatic: {
                armSwing: 0.9,
                spineMovement: 0.4,
                headNod: 0.6,
                handGestures: 0.8
            },
            questioning: {
                armSwing: 0.2,
                spineMovement: 0.1,
                headNod: 0.3,
                handGestures: 0.4
            }
        };
    }
    
    /**
     * Convert Audio2Gesture output to BVH frame format
     * @param {Object} audio2GestureOutput - Raw output from Audio2Gesture neural network
     * @param {Float32Array} audioFeatures - Audio features for this frame
     * @param {number} timestamp - Frame timestamp
     * @param {Object} options - Conversion options for this frame
     * @returns {Object} BVH frame compatible with timeline system
     */
    async convertToBVH(audio2GestureOutput, audioFeatures, timestamp, options = {}) {
        const startTime = performance.now();
        
        try {
            // Validate input
            if (!audio2GestureOutput) {
                throw new Error('Audio2Gesture output is null or undefined');
            }
            
            // Extract audio features if provided
            if (audioFeatures && this.options.enableAudioFeatures) {
                this.extractAudioFeatures(audioFeatures, timestamp);
            }
            
            // Detect Audio2Gesture output format and convert accordingly
            const normalizedData = this.normalizeAudio2GestureOutput(audio2GestureOutput);
            
            // Convert to body bone rotations
            const bodyRotations = this.calculateBodyRotations(normalizedData, audioFeatures, options);
            
            // Apply smoothing if enabled
            const smoothedRotations = this.options.smoothing ? 
                this.applySmoothingToRotations(bodyRotations, timestamp) : 
                bodyRotations;
            
            // Generate BVH frame
            const bvhFrame = this.generateBVHFrame(smoothedRotations, timestamp, normalizedData, audioFeatures);
            
            // Store frame in history for smoothing
            this.updateFrameHistory(bvhFrame, normalizedData, timestamp);
            
            // Update statistics
            this.updateConversionStats(startTime);
            
            if (this.options.logConversions) {
                console.log('[Audio2Gesture BVH] Converted frame:', {
                    timestamp,
                    inputType: this.detectAudio2GestureFormat(audio2GestureOutput),
                    outputBones: Object.keys(smoothedRotations).length,
                    conversionTime: `${(performance.now() - startTime).toFixed(2)}ms`
                });
            }
            
            return bvhFrame;
            
        } catch (error) {
            this.stats.errorCount++;
            console.error('[Audio2Gesture BVH] Conversion error:', error);
            
            // Return neutral frame on error
            return this.generateNeutralBVHFrame(timestamp);
        }
    }
    
    /**
     * Normalize different Audio2Gesture output formats to a common structure
     */
    normalizeAudio2GestureOutput(output) {
        const format = this.detectAudio2GestureFormat(output);
        
        switch (format) {
            case 'poses':
                return this.normalizePoses(output);
            case 'joints':
                return this.normalizeJoints(output);
            case 'rotations':
                return this.normalizeRotations(output);
            case 'coordinates':
                return this.normalizeCoordinates(output);
            case 'smplx':
                return this.normalizeSMPLX(output);
            default:
                console.warn('[Audio2Gesture BVH] Unknown format, using raw data:', format);
                return this.normalizeRawData(output);
        }
    }
    
    /**
     * Detect the format of Audio2Gesture output
     */
    detectAudio2GestureFormat(output) {
        if (output.poses || (Array.isArray(output) && output.length > 0 && output[0].pose)) {
            return 'poses'; // Pose sequence
        } else if (output.joints || (Array.isArray(output) && output.length > 0 && output[0].joints)) {
            return 'joints'; // Joint positions/rotations
        } else if (output.rotations || output.joint_rotations) {
            return 'rotations'; // Direct rotation matrices/quaternions
        } else if (output.coordinates || output.keypoints) {
            return 'coordinates'; // 2D/3D coordinates
        } else if (output.smplx || output.smpl || output.body_pose) {
            return 'smplx'; // SMPL/SMPL-X format
        } else {
            return 'unknown';
        }
    }
    
    /**
     * Normalize pose sequence data
     */
    normalizePoses(output) {
        const poses = output.poses || output;
        
        if (!Array.isArray(poses) || poses.length === 0) {
            throw new Error('Invalid poses data: expected non-empty array');
        }
        
        // Take the most recent pose or average if multiple
        const currentPose = poses[poses.length - 1];
        
        return {
            format: 'poses',
            bodyPose: currentPose.body_pose || currentPose.pose || currentPose,
            handPoseL: currentPose.left_hand || currentPose.handL || null,
            handPoseR: currentPose.right_hand || currentPose.handR || null,
            facePose: currentPose.face_pose || null,
            confidence: currentPose.confidence || output.confidence || 1.0,
            timestamp: currentPose.timestamp || output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize joint data
     */
    normalizeJoints(output) {
        const joints = output.joints || output;
        
        return {
            format: 'joints',
            joints: this.parseJointData(joints),
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize rotation data
     */
    normalizeRotations(output) {
        const rotations = output.rotations || output.joint_rotations || output;
        
        return {
            format: 'rotations',
            rotations: this.parseRotationData(rotations),
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize coordinate data (2D/3D keypoints)
     */
    normalizeCoordinates(output) {
        const coordinates = output.coordinates || output.keypoints || output;
        
        return {
            format: 'coordinates',
            keypoints: this.parseCoordinateData(coordinates),
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize SMPL/SMPL-X format data
     */
    normalizeSMPLX(output) {
        const smplData = output.smplx || output.smpl || output;
        
        return {
            format: 'smplx',
            bodyPose: smplData.body_pose || smplData.pose,
            handPoseL: smplData.left_hand_pose,
            handPoseR: smplData.right_hand_pose,
            shape: smplData.betas || smplData.shape,
            globalOrient: smplData.global_orient || smplData.root_orient,
            translation: smplData.transl || smplData.translation,
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Fallback normalization for unknown formats
     */
    normalizeRawData(output) {
        return {
            format: 'raw',
            data: output,
            confidence: output.confidence || 0.5,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Extract audio features for gesture modulation
     */
    extractAudioFeatures(audioData, timestamp) {
        if (!audioData || audioData.length === 0) return;
        
        // Calculate basic audio features
        const features = {
            timestamp: timestamp,
            
            // Amplitude/Energy
            rms: this.calculateRMS(audioData),
            amplitude: this.calculateAmplitude(audioData),
            
            // Spectral features (simplified)
            spectralCentroid: this.calculateSpectralCentroid(audioData),
            spectralRolloff: this.calculateSpectralRolloff(audioData),
            
            // Rhythm/Tempo (basic beat detection)
            beatStrength: this.calculateBeatStrength(audioData),
            
            // Pitch estimation (simplified)
            fundamentalFreq: this.estimateFundamentalFrequency(audioData)
        };
        
        // Store in feature history
        this.audioFeatures.push(features);
        if (this.audioFeatures.length > 100) { // Keep last 100 features
            this.audioFeatures.shift();
        }
        
        this.stats.audioFeaturesExtracted++;
        
        return features;
    }
    
    /**
     * Calculate body bone rotations from normalized data
     */
    calculateBodyRotations(normalizedData, audioFeatures, options = {}) {
        const rotations = {};
        
        // Get gesture pattern based on audio characteristics
        const gesturePattern = this.selectGesturePattern(audioFeatures);
        
        switch (normalizedData.format) {
            case 'poses':
                Object.assign(rotations, this.posesToRotations(normalizedData, gesturePattern));
                break;
            case 'joints':
                Object.assign(rotations, this.jointsToRotations(normalizedData, gesturePattern));
                break;
            case 'rotations':
                Object.assign(rotations, this.rotationsToRotations(normalizedData, gesturePattern));
                break;
            case 'coordinates':
                Object.assign(rotations, this.coordinatesToRotations(normalizedData, gesturePattern));
                break;
            case 'smplx':
                Object.assign(rotations, this.smplxToRotations(normalizedData, gesturePattern));
                break;
            default:
                Object.assign(rotations, this.rawDataToRotations(normalizedData, gesturePattern));
                break;
        }
        
        // Apply audio-driven modulation
        if (audioFeatures && this.options.enableAudioFeatures) {
            this.applyAudioModulation(rotations, audioFeatures, gesturePattern);
        }
        
        // Apply scaling and constraints
        return this.applyRotationConstraints(rotations, options);
    }
    
    /**
     * Select appropriate gesture pattern based on audio characteristics
     */
    selectGesturePattern(audioFeatures) {
        if (!audioFeatures || !this.options.emotionalModulation) {
            return this.gesturePatterns.neutral;
        }
        
        const recentFeatures = this.audioFeatures.slice(-10); // Last 10 frames
        if (recentFeatures.length === 0) {
            return this.gesturePatterns.neutral;
        }
        
        // Calculate average features
        const avgAmplitude = recentFeatures.reduce((sum, f) => sum + f.amplitude, 0) / recentFeatures.length;
        const avgBeatStrength = recentFeatures.reduce((sum, f) => sum + f.beatStrength, 0) / recentFeatures.length;
        const avgPitch = recentFeatures.reduce((sum, f) => sum + f.fundamentalFreq, 0) / recentFeatures.length;
        
        // Select pattern based on audio characteristics
        if (avgAmplitude > 0.7 && avgBeatStrength > 0.6) {
            return this.gesturePatterns.excited;
        } else if (avgAmplitude < 0.2 && avgBeatStrength < 0.3) {
            return this.gesturePatterns.calm;
        } else if (avgBeatStrength > 0.8) {
            return this.gesturePatterns.emphatic;
        } else if (avgPitch > 200 && avgAmplitude > 0.4) { // Higher pitch might indicate questioning
            return this.gesturePatterns.questioning;
        } else {
            return this.gesturePatterns.neutral;
        }
    }
    
    /**
     * Convert pose data to bone rotations
     */
    posesToRotations(data, gesturePattern) {
        const rotations = {};
        const pose = data.bodyPose;
        
        if (!pose || pose.length === 0) {
            return this.generateBasicGestures(gesturePattern);
        }
        
        // Map pose parameters to bone rotations
        // Assuming pose is in axis-angle or quaternion format
        if (pose.length >= 72) { // SMPL format (24 joints * 3 rotation params)
            rotations.hips = this.poseToRotation(pose.slice(0, 3), 'hips');
            rotations.spine = this.poseToRotation(pose.slice(3, 6), 'spine');
            rotations.spine1 = this.poseToRotation(pose.slice(6, 9), 'spine1');
            rotations.neck = this.poseToRotation(pose.slice(9, 12), 'neck');
            rotations.head = this.poseToRotation(pose.slice(12, 15), 'head');
            
            // Arms
            rotations.leftShoulder = this.poseToRotation(pose.slice(15, 18), 'leftShoulder');
            rotations.leftArm = this.poseToRotation(pose.slice(18, 21), 'leftArm');
            rotations.leftForeArm = this.poseToRotation(pose.slice(21, 24), 'leftForeArm');
            rotations.rightShoulder = this.poseToRotation(pose.slice(24, 27), 'rightShoulder');
            rotations.rightArm = this.poseToRotation(pose.slice(27, 30), 'rightArm');
            rotations.rightForeArm = this.poseToRotation(pose.slice(30, 33), 'rightForeArm');
            
            // Legs (if enabled)
            if (this.options.enableLowerBody && pose.length >= 48) {
                rotations.leftUpLeg = this.poseToRotation(pose.slice(33, 36), 'leftUpLeg');
                rotations.leftLeg = this.poseToRotation(pose.slice(36, 39), 'leftLeg');
                rotations.leftFoot = this.poseToRotation(pose.slice(39, 42), 'leftFoot');
                rotations.rightUpLeg = this.poseToRotation(pose.slice(42, 45), 'rightUpLeg');
                rotations.rightLeg = this.poseToRotation(pose.slice(45, 48), 'rightLeg');
                rotations.rightFoot = this.poseToRotation(pose.slice(48, 51), 'rightFoot');
            }
        }
        
        // Add hand gestures if available
        if (data.handPoseL && this.options.enableFingers) {
            Object.assign(rotations, this.handPoseToRotations(data.handPoseL, 'left'));
        }
        if (data.handPoseR && this.options.enableFingers) {
            Object.assign(rotations, this.handPoseToRotations(data.handPoseR, 'right'));
        }
        
        return rotations;
    }
    
    /**
     * Convert pose parameters to rotation
     */
    poseToRotation(poseParams, boneName) {
        if (!poseParams || poseParams.length < 3) {
            return { x: 0, y: 0, z: 0 };
        }
        
        // Convert axis-angle to Euler angles (simplified)
        const angle = Math.sqrt(poseParams[0] * poseParams[0] + 
                               poseParams[1] * poseParams[1] + 
                               poseParams[2] * poseParams[2]);
        
        if (angle < 1e-6) {
            return { x: 0, y: 0, z: 0 };
        }
        
        const axis = [
            poseParams[0] / angle,
            poseParams[1] / angle,
            poseParams[2] / angle
        ];
        
        // Convert to Euler angles (simplified conversion)
        const radToDeg = 180 / Math.PI;
        
        return {
            x: axis[0] * angle * radToDeg,
            y: axis[1] * angle * radToDeg,
            z: axis[2] * angle * radToDeg
        };
    }
    
    /**
     * Generate basic gestures when no input data is available
     */
    generateBasicGestures(gesturePattern) {
        const time = Date.now() / 1000;
        const rotations = {};
        
        // Basic arm swinging based on gesture pattern
        const armSwingIntensity = gesturePattern.armSwing * this.options.gestureIntensity;
        const spineIntensity = gesturePattern.spineMovement * this.options.gestureIntensity;
        
        // Arm swinging
        rotations.leftArm = {
            x: Math.sin(time * 2) * armSwingIntensity * 20,
            y: 0,
            z: Math.cos(time * 2) * armSwingIntensity * 10
        };
        
        rotations.rightArm = {
            x: -Math.sin(time * 2) * armSwingIntensity * 20,
            y: 0,
            z: -Math.cos(time * 2) * armSwingIntensity * 10
        };
        
        // Subtle spine movement
        rotations.spine = {
            x: Math.sin(time * 1.5) * spineIntensity * 5,
            y: Math.cos(time * 1.3) * spineIntensity * 3,
            z: 0
        };
        
        // Head nodding
        if (gesturePattern.headNod > 0.1) {
            rotations.head = {
                x: Math.sin(time * 3) * gesturePattern.headNod * 8,
                y: Math.cos(time * 2.5) * gesturePattern.headNod * 4,
                z: 0
            };
        }
        
        return rotations;
    }
    
    /**
     * Apply audio-driven modulation to rotations
     */
    applyAudioModulation(rotations, audioFeatures, gesturePattern) {
        if (!audioFeatures) return;
        
        const amplitudeScale = audioFeatures.amplitude * this.options.amplitudeSensitivity;
        const beatScale = audioFeatures.beatStrength * this.options.rhythmSensitivity;
        const pitchScale = Math.min(1.0, audioFeatures.fundamentalFreq / 200) * this.options.pitchSensitivity;
        
        // Modulate arm movements based on amplitude
        if (rotations.leftArm) {
            rotations.leftArm.x *= (1 + amplitudeScale * 0.5);
            rotations.leftArm.z *= (1 + amplitudeScale * 0.3);
        }
        
        if (rotations.rightArm) {
            rotations.rightArm.x *= (1 + amplitudeScale * 0.5);
            rotations.rightArm.z *= (1 + amplitudeScale * 0.3);
        }
        
        // Modulate spine movement based on beat strength
        if (rotations.spine) {
            rotations.spine.x *= (1 + beatScale * 0.4);
            rotations.spine.y *= (1 + beatScale * 0.2);
        }
        
        // Modulate head movement based on pitch
        if (rotations.head) {
            rotations.head.x *= (1 + pitchScale * 0.3);
            rotations.head.y *= (1 + pitchScale * 0.2);
        }
        
        // Add rhythmic hand gestures based on beat
        if (beatScale > 0.5) {
            const time = Date.now() / 1000;
            const handGestureIntensity = gesturePattern.handGestures * beatScale;
            
            if (!rotations.leftHand) rotations.leftHand = { x: 0, y: 0, z: 0 };
            if (!rotations.rightHand) rotations.rightHand = { x: 0, y: 0, z: 0 };
            
            rotations.leftHand.z += Math.sin(time * 4) * handGestureIntensity * 15;
            rotations.rightHand.z += -Math.sin(time * 4) * handGestureIntensity * 15;
        }
    }
    
    /**
     * Apply smoothing to rotations using frame history
     */
    applySmoothingToRotations(currentRotations, timestamp) {
        if (this.frameHistory.length === 0) {
            return currentRotations;
        }
        
        const smoothedRotations = {};
        const factor = this.options.smoothingFactor;
        
        // Get recent frames for multi-frame smoothing
        const recentFrames = this.frameHistory.slice(-3); // Last 3 frames
        
        for (const boneName in currentRotations) {
            const current = currentRotations[boneName];
            let smoothed = { ...current };
            
            // Average with recent frames
            let validFrameCount = 1;
            
            for (const frame of recentFrames) {
                if (frame.rotations[boneName]) {
                    const prev = frame.rotations[boneName];
                    smoothed.x = this.lerpAngle(smoothed.x, prev.x, factor / validFrameCount);
                    smoothed.y = this.lerpAngle(smoothed.y, prev.y, factor / validFrameCount);
                    smoothed.z = this.lerpAngle(smoothed.z, prev.z, factor / validFrameCount);
                    validFrameCount++;
                }
            }
            
            smoothedRotations[boneName] = smoothed;
        }
        
        return smoothedRotations;
    }
    
    /**
     * Generate BVH frame from body rotations
     */
    generateBVHFrame(bodyRotations, timestamp, originalData, audioFeatures) {
        const totalBones = Object.keys(this.bodyBones).length;
        const motionData = new Array(totalBones * 6).fill(0);
        
        // Convert rotations to BVH format (position + rotation for each bone)
        for (const boneName in bodyRotations) {
            const boneInfo = this.bodyBones[boneName];
            if (boneInfo) {
                const rotation = bodyRotations[boneName];
                const channelCount = boneInfo.channels.length;
                const baseIndex = this.calculateBoneBaseIndex(boneName);
                
                // Handle different channel configurations
                if (channelCount === 6) { // Position + rotation
                    motionData[baseIndex] = rotation.px || 0;     // X position
                    motionData[baseIndex + 1] = rotation.py || 0; // Y position
                    motionData[baseIndex + 2] = rotation.pz || 0; // Z position
                    motionData[baseIndex + 3] = rotation.x || 0;  // X rotation
                    motionData[baseIndex + 4] = rotation.y || 0;  // Y rotation
                    motionData[baseIndex + 5] = rotation.z || 0;  // Z rotation
                } else if (channelCount === 3) { // Rotation only
                    motionData[baseIndex] = rotation.x || 0;      // X rotation
                    motionData[baseIndex + 1] = rotation.y || 0;  // Y rotation
                    motionData[baseIndex + 2] = rotation.z || 0;  // Z rotation
                }
            }
        }
        
        return {
            frameNumber: this.stats.framesConverted,
            timestamp: timestamp,
            motionData: motionData,
            metadata: {
                type: 'body',
                source: 'audio2gesture',
                format: originalData.format,
                confidence: originalData.confidence,
                boneCount: Object.keys(bodyRotations).length,
                channels: motionData.length,
                audioFeatures: audioFeatures ? {
                    amplitude: audioFeatures.amplitude,
                    beatStrength: audioFeatures.beatStrength,
                    fundamentalFreq: audioFeatures.fundamentalFreq
                } : null,
                converter: 'Audio2GestureBVHConverter',
                version: '1.0.0'
            }
        };
    }
    
    /**
     * Calculate base index for bone in motion data array
     */
    calculateBoneBaseIndex(boneName) {
        let index = 0;
        
        for (const [currentBoneName, boneInfo] of Object.entries(this.bodyBones)) {
            if (currentBoneName === boneName) {
                return index;
            }
            index += boneInfo.channels.length;
        }
        
        return 0;
    }
    
    /**
     * Generate neutral BVH frame (rest pose) for error cases
     */
    generateNeutralBVHFrame(timestamp) {
        const totalBones = Object.keys(this.bodyBones).length;
        const motionData = new Array(totalBones * 6).fill(0);
        
        return {
            frameNumber: this.stats.framesConverted,
            timestamp: timestamp,
            motionData: motionData,
            metadata: {
                type: 'body',
                source: 'audio2gesture',
                format: 'neutral',
                confidence: 0.0,
                boneCount: 0,
                channels: motionData.length,
                converter: 'Audio2GestureBVHConverter',
                version: '1.0.0',
                isNeutral: true
            }
        };
    }
    
    /**
     * Audio analysis helper methods
     */
    calculateRMS(audioData) {
        const sum = audioData.reduce((acc, sample) => acc + sample * sample, 0);
        return Math.sqrt(sum / audioData.length);
    }
    
    calculateAmplitude(audioData) {
        return audioData.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0);
    }
    
    calculateSpectralCentroid(audioData) {
        // Simplified spectral centroid calculation
        // In a real implementation, you'd use FFT
        let weightedSum = 0;
        let magnitudeSum = 0;
        
        for (let i = 0; i < audioData.length; i++) {
            const magnitude = Math.abs(audioData[i]);
            weightedSum += i * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    }
    
    calculateSpectralRolloff(audioData) {
        // Simplified spectral rolloff (85% of spectral energy)
        let totalEnergy = 0;
        const energies = [];
        
        for (let i = 0; i < audioData.length; i++) {
            const energy = audioData[i] * audioData[i];
            energies.push(energy);
            totalEnergy += energy;
        }
        
        const threshold = totalEnergy * 0.85;
        let cumulativeEnergy = 0;
        
        for (let i = 0; i < energies.length; i++) {
            cumulativeEnergy += energies[i];
            if (cumulativeEnergy >= threshold) {
                return i / energies.length;
            }
        }
        
        return 1.0;
    }
    
    calculateBeatStrength(audioData) {
        // Simple beat detection based on energy variations
        const windowSize = Math.floor(audioData.length / 4);
        const energies = [];
        
        for (let i = 0; i < audioData.length - windowSize; i += windowSize) {
            const window = audioData.slice(i, i + windowSize);
            const energy = window.reduce((sum, sample) => sum + sample * sample, 0) / windowSize;
            energies.push(energy);
        }
        
        if (energies.length < 2) return 0;
        
        // Calculate energy variance as beat strength indicator
        const mean = energies.reduce((sum, energy) => sum + energy, 0) / energies.length;
        const variance = energies.reduce((sum, energy) => sum + Math.pow(energy - mean, 2), 0) / energies.length;
        
        return Math.min(1.0, Math.sqrt(variance) / (mean + 1e-6));
    }
    
    estimateFundamentalFrequency(audioData) {
        // Simplified pitch estimation using autocorrelation
        const sampleRate = 16000; // Assume 16kHz
        const minPeriod = Math.floor(sampleRate / 500); // 500 Hz max
        const maxPeriod = Math.floor(sampleRate / 50);  // 50 Hz min
        
        let maxCorrelation = 0;
        let bestPeriod = minPeriod;
        
        for (let period = minPeriod; period <= maxPeriod && period < audioData.length / 2; period++) {
            let correlation = 0;
            const samples = audioData.length - period;
            
            for (let i = 0; i < samples; i++) {
                correlation += audioData[i] * audioData[i + period];
            }
            
            correlation /= samples;
            
            if (correlation > maxCorrelation) {
                maxCorrelation = correlation;
                bestPeriod = period;
            }
        }
        
        return sampleRate / bestPeriod;
    }
    
    /**
     * Utility methods
     */
    lerpAngle(from, to, t) {
        // Handle angle wrapping for smooth interpolation
        let diff = to - from;
        if (diff > 180) diff -= 360;
        if (diff < -180) diff += 360;
        return from + diff * t;
    }
    
    applyRotationConstraints(rotations, options = {}) {
        const constrained = {};
        
        for (const boneName in rotations) {
            const rotation = rotations[boneName];
            const constraints = this.getBoneConstraints(boneName);
            
            constrained[boneName] = {
                x: this.clampAngle(rotation.x * this.options.scaleFactor, constraints.x.min, constraints.x.max),
                y: this.clampAngle(rotation.y * this.options.scaleFactor, constraints.y.min, constraints.y.max),
                z: this.clampAngle(rotation.z * this.options.scaleFactor, constraints.z.min, constraints.z.max)
            };
        }
        
        return constrained;
    }
    
    getBoneConstraints(boneName) {
        // Define realistic rotation limits for different bones
        const defaultConstraints = { x: { min: -180, max: 180 }, y: { min: -180, max: 180 }, z: { min: -180, max: 180 } };
        
        const constraints = {
            // Spine constraints
            spine: { x: { min: -30, max: 30 }, y: { min: -45, max: 45 }, z: { min: -30, max: 30 } },
            spine1: { x: { min: -20, max: 20 }, y: { min: -30, max: 30 }, z: { min: -20, max: 20 } },
            spine2: { x: { min: -15, max: 15 }, y: { min: -20, max: 20 }, z: { min: -15, max: 15 } },
            
            // Arm constraints
            leftArm: { x: { min: -180, max: 180 }, y: { min: -90, max: 180 }, z: { min: -90, max: 90 } },
            rightArm: { x: { min: -180, max: 180 }, y: { min: -180, max: 90 }, z: { min: -90, max: 90 } },
            leftForeArm: { x: { min: 0, max: 150 }, y: { min: -90, max: 90 }, z: { min: -90, max: 90 } },
            rightForeArm: { x: { min: 0, max: 150 }, y: { min: -90, max: 90 }, z: { min: -90, max: 90 } },
            
            // Head/neck constraints
            neck: { x: { min: -45, max: 45 }, y: { min: -60, max: 60 }, z: { min: -30, max: 30 } },
            head: { x: { min: -30, max: 30 }, y: { min: -45, max: 45 }, z: { min: -20, max: 20 } }
        };
        
        return constraints[boneName] || defaultConstraints;
    }
    
    clampAngle(angle, min, max) {
        return Math.max(min, Math.min(max, angle));
    }
    
    /**
     * Frame history management
     */
    updateFrameHistory(bvhFrame, originalData, timestamp) {
        this.frameHistory.push({
            timestamp: timestamp,
            rotations: this.extractRotationsFromBVHFrame(bvhFrame),
            originalData: originalData,
            confidence: originalData.confidence
        });
        
        // Keep history size manageable
        if (this.frameHistory.length > this.maxHistorySize) {
            this.frameHistory.shift();
        }
    }
    
    extractRotationsFromBVHFrame(bvhFrame) {
        const rotations = {};
        
        for (const boneName in this.bodyBones) {
            const baseIndex = this.calculateBoneBaseIndex(boneName);
            const channelCount = this.bodyBones[boneName].channels.length;
            
            if (channelCount === 6 && baseIndex + 5 < bvhFrame.motionData.length) {
                rotations[boneName] = {
                    px: bvhFrame.motionData[baseIndex],
                    py: bvhFrame.motionData[baseIndex + 1],
                    pz: bvhFrame.motionData[baseIndex + 2],
                    x: bvhFrame.motionData[baseIndex + 3],
                    y: bvhFrame.motionData[baseIndex + 4],
                    z: bvhFrame.motionData[baseIndex + 5]
                };
            } else if (channelCount === 3 && baseIndex + 2 < bvhFrame.motionData.length) {
                rotations[boneName] = {
                    x: bvhFrame.motionData[baseIndex],
                    y: bvhFrame.motionData[baseIndex + 1],
                    z: bvhFrame.motionData[baseIndex + 2]
                };
            }
        }
        
        return rotations;
    }
    
    /**
     * Statistics and monitoring
     */
    updateConversionStats(startTime) {
        const conversionTime = performance.now() - startTime;
        
        this.stats.framesConverted++;
        this.stats.lastConversionTime = conversionTime;
        
        // Update running average
        const count = this.stats.framesConverted;
        this.stats.averageConversionTime = 
            (this.stats.averageConversionTime * (count - 1) + conversionTime) / count;
    }
    
    getStats() {
        return {
            ...this.stats,
            historySize: this.frameHistory.length,
            audioFeaturesSize: this.audioFeatures.length,
            isInitialized: this.isInitialized,
            bodyBonesCount: Object.keys(this.bodyBones).length
        };
    }
    
    /**
     * Timeline integration methods
     */
    createTimelineClip(audio2GestureFrames, audioFrames, startTime = 0, duration = null) {
        if (!Array.isArray(audio2GestureFrames)) {
            throw new Error('audio2GestureFrames must be an array');
        }
        
        const clipDuration = duration || (audio2GestureFrames.length / this.options.targetFrameRate);
        const bvhFrames = [];
        
        for (let i = 0; i < audio2GestureFrames.length; i++) {
            const timestamp = startTime + (i / this.options.targetFrameRate);
            const audioFeatures = audioFrames && audioFrames[i] ? audioFrames[i] : null;
            const bvhFrame = this.convertToBVH(audio2GestureFrames[i], audioFeatures, timestamp);
            bvhFrames.push(bvhFrame);
        }
        
        return {
            type: 'audio2gesture',
            startTime: startTime,
            duration: clipDuration,
            frames: bvhFrames,
            metadata: {
                sourceFrameCount: audio2GestureFrames.length,
                targetFrameRate: this.options.targetFrameRate,
                converter: 'Audio2GestureBVHConverter',
                hasAudioFeatures: !!audioFrames
            }
        };
    }
    
    /**
     * Real-time processing for live Audio2Gesture input
     */
    async processLiveFrame(audio2GestureOutput, audioFeatures, timestamp) {
        try {
            const bvhFrame = await this.convertToBVH(audio2GestureOutput, audioFeatures, timestamp);
            
            // Emit frame update event if connected to timeline
            if (this.onFrameReady) {
                this.onFrameReady(bvhFrame, timestamp);
            }
            
            return bvhFrame;
            
        } catch (error) {
            console.error('[Audio2Gesture BVH] Live frame processing error:', error);
            return this.generateNeutralBVHFrame(timestamp);
        }
    }
    
    /**
     * Cleanup and disposal
     */
    dispose() {
        this.frameHistory = [];
        this.audioFeatures = [];
        this.stats = {
            framesConverted: 0,
            averageConversionTime: 0,
            lastConversionTime: 0,
            errorCount: 0,
            audioFeaturesExtracted: 0
        };
        this.onFrameReady = null;
        
        console.log('[Audio2Gesture BVH Converter] Disposed');
    }
}

// Placeholder methods for different input formats
// These would need specific implementations based on Audio2Gesture model outputs

Audio2GestureBVHConverter.prototype.parseJointData = function(joints) {
    // Parse joint position/rotation data
    return Array.isArray(joints) ? joints : [];
};

Audio2GestureBVHConverter.prototype.parseRotationData = function(rotations) {
    // Parse rotation matrices or quaternions
    return Array.isArray(rotations) ? rotations : [];
};

Audio2GestureBVHConverter.prototype.parseCoordinateData = function(coordinates) {
    // Parse 2D/3D keypoint coordinates
    return Array.isArray(coordinates) ? coordinates : [];
};

Audio2GestureBVHConverter.prototype.jointsToRotations = function(data, gesturePattern) {
    // Convert joint data to bone rotations
    const rotations = this.generateBasicGestures(gesturePattern);
    
    // Apply joint data if available
    if (data.joints && data.joints.length > 0) {
        // Implementation would depend on specific joint format
        console.log('[Audio2Gesture BVH] Joint data conversion not fully implemented');
    }
    
    return rotations;
};

Audio2GestureBVHConverter.prototype.rotationsToRotations = function(data, gesturePattern) {
    // Convert rotation data to bone rotations
    const rotations = {};
    
    if (data.rotations && data.rotations.length > 0) {
        // Map rotation data to bone structure
        const boneNames = Object.keys(this.bodyBones);
        
        for (let i = 0; i < Math.min(data.rotations.length, boneNames.length); i++) {
            const rotation = data.rotations[i];
            if (rotation && typeof rotation === 'object') {
                rotations[boneNames[i]] = {
                    x: rotation.x || rotation[0] || 0,
                    y: rotation.y || rotation[1] || 0,
                    z: rotation.z || rotation[2] || 0
                };
            }
        }
    }
    
    return Object.keys(rotations).length > 0 ? rotations : this.generateBasicGestures(gesturePattern);
};

Audio2GestureBVHConverter.prototype.coordinatesToRotations = function(data, gesturePattern) {
    // Convert 2D/3D coordinates to bone rotations
    const rotations = this.generateBasicGestures(gesturePattern);
    
    if (data.keypoints && data.keypoints.length > 0) {
        // Implementation would depend on keypoint format and skeleton structure
        console.log('[Audio2Gesture BVH] Coordinate data conversion not fully implemented');
    }
    
    return rotations;
};

Audio2GestureBVHConverter.prototype.smplxToRotations = function(data, gesturePattern) {
    // Convert SMPL/SMPL-X format to bone rotations
    if (data.bodyPose) {
        return this.posesToRotations(data, gesturePattern);
    }
    
    return this.generateBasicGestures(gesturePattern);
};

Audio2GestureBVHConverter.prototype.rawDataToRotations = function(data, gesturePattern) {
    // Fallback for unknown data formats
    return this.generateBasicGestures(gesturePattern);
};

Audio2GestureBVHConverter.prototype.handPoseToRotations = function(handPose, side) {
    // Convert hand pose data to finger rotations
    const rotations = {};
    const prefix = side === 'left' ? 'left' : 'right';
    
    if (handPose && handPose.length >= 6) {
        rotations[`${prefix}Thumb1`] = { x: handPose[0] * 30, y: handPose[1] * 20, z: 0 };
        rotations[`${prefix}Thumb2`] = { x: handPose[1] * 20, y: 0, z: 0 };
        rotations[`${prefix}Index1`] = { x: handPose[2] * 40, y: 0, z: 0 };
        rotations[`${prefix}Index2`] = { x: handPose[3] * 30, y: 0, z: 0 };
        rotations[`${prefix}Middle1`] = { x: handPose[4] * 40, y: 0, z: 0 };
        rotations[`${prefix}Middle2`] = { x: handPose[5] * 30, y: 0, z: 0 };
    }
    
    return rotations;
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Audio2GestureBVHConverter;
} else {
    window.Audio2GestureBVHConverter = Audio2GestureBVHConverter;
}
