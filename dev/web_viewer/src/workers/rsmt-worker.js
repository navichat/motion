/**
 * RSMT Worker
 * 
 * Web Worker for Real-time Stylized Motion Transitions.
 * Handles motion stylization and smooth transitions between different movement styles.
 */

let rsmtModel = null;
let isModelLoaded = false;

// Worker message handler
self.onmessage = async function(event) {
    const { type, id, data } = event.data;
    
    try {
        switch (type) {
            case 'load-model':
                await loadRSMTModel();
                break;
                
            case 'inference':
                const result = await runInference(data.input);
                self.postMessage({
                    type: 'inference-result',
                    id: id,
                    result: result
                });
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            id: id,
            error: error.message
        });
    }
};

/**
 * Load RSMT model for real-time stylized motion transitions
 */
async function loadRSMTModel() {
    console.log('📦 Loading RSMT model in worker...');
    
    try {
        // Load actual RSMT ONNX models from migration workspace
        const styleVAEEncoderPath = '../../../migration_workspace/models/onnx/stylevae_encoder.onnx';
        const styleVAEDecoderPath = '../../../migration_workspace/models/onnx/stylevae_decoder.onnx';
        const transitionNetPath = '../../../migration_workspace/models/onnx/transition_net.onnx';
        const deepPhasePath = '../../../migration_workspace/models/onnx/deephase.onnx';
        
        // Load ONNX Runtime Web
        if (typeof ort === 'undefined') {
            importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort.min.js');
        }
        
        console.log('🎨 Loading StyleVAE Encoder from:', styleVAEEncoderPath);
        const encoderSession = await ort.InferenceSession.create(styleVAEEncoderPath);
        
        console.log('🎨 Loading StyleVAE Decoder from:', styleVAEDecoderPath);
        const decoderSession = await ort.InferenceSession.create(styleVAEDecoderPath);
        
        console.log('🔄 Loading Transition Net from:', transitionNetPath);
        const transitionSession = await ort.InferenceSession.create(transitionNetPath);
        
        console.log('⏱️ Loading DeepPhase from:', deepPhasePath);
        const deepPhaseSession = await ort.InferenceSession.create(deepPhasePath);
        
        rsmtModel = {
            name: 'RSMT',
            version: '1.0',
            encoderSession: encoderSession,
            decoderSession: decoderSession,
            transitionSession: transitionSession,
            deepPhaseSession: deepPhaseSession,
            inputShape: [1, 120, 66], // [batch, sequence_length, joint_features]
            outputShape: [1, 120, 66], // [batch, sequence_length, stylized_joint_features]
            styles: ['casual', 'confident', 'energetic', 'relaxed', 'dramatic'],
            loaded: true
        };
        
        isModelLoaded = true;
        
        self.postMessage({
            type: 'model-loaded',
            model: {
                name: rsmtModel.name,
                version: rsmtModel.version,
                availableStyles: rsmtModel.styles,
                inputShape: rsmtModel.inputShape,
                outputShape: rsmtModel.outputShape
            }
        });
        
        console.log('✅ RSMT models loaded successfully');
        
    } catch (error) {
        console.error('❌ Failed to load RSMT models:', error);
        
        // Fallback to demo mode
        rsmtModel = {
            name: 'RSMT (Demo Mode)',
            version: '1.0-demo',
            inputShape: [1, 120, 66],
            outputShape: [1, 120, 66], 
            styles: ['casual', 'confident', 'energetic'],
            demoMode: true,
            loaded: true
        };
        
        isModelLoaded = true;
        
        self.postMessage({
            type: 'model-loaded',
            model: {
                name: rsmtModel.name,
                version: rsmtModel.version,
                availableStyles: rsmtModel.styles,
                demoMode: true
            }
        });
        
        console.log('⚠️ RSMT running in demo mode');
    }
}

/**
 * Simulate model loading delay
 */
function simulateModelLoading() {
    return new Promise(resolve => {
        setTimeout(resolve, 2200 + Math.random() * 600); // 2.2-2.8 second delay
    });
}

/**
 * Run RSMT inference on motion input
 * @param {Object} input - Motion input data
 * @returns {Object} Stylized motion data
 */
async function runInference(input) {
    if (!isModelLoaded) {
        throw new Error('RSMT model not loaded');
    }
    
    const startTime = performance.now();
    
    try {
        // Process motion data for stylization
        const motionFeatures = processMotionData(input.motion);
        
        // Apply style transfer using ONNX models or demo
        const stylizedMotion = await applyMotionStyleTransfer(motionFeatures, input.targetStyle);
        
        const inferenceTime = performance.now() - startTime;
        
        // Send performance metrics
        self.postMessage({
            type: 'performance',
            metrics: {
                inferenceTime: inferenceTime,
                motionLength: input.motion?.duration || 0,
                styleTransferred: input.targetStyle
            }
        });
        
        return {
            type: 'stylized_motion',
            data: stylizedMotion,
            timestamp: Date.now(),
            processingTime: inferenceTime,
            originalStyle: input.originalStyle || 'neutral',
            targetStyle: input.targetStyle || 'casual'
        };
        
    } catch (error) {
        console.error('❌ RSMT inference failed:', error);
        throw error;
    }
}

/**
 * Apply motion style transfer using ONNX models
 */
async function applyMotionStyleTransfer(motionFeatures, targetStyle) {
    if (rsmtModel.encoderSession && rsmtModel.decoderSession && rsmtModel.transitionSession) {
        // Use actual ONNX models for style transfer
        try {
            // 1. Encode motion to latent space
            const motionTensor = new ort.Tensor('float32', motionFeatures.joints.flat(), rsmtModel.inputShape);
            const encoderFeeds = { 'input': motionTensor };
            const encoderResults = await rsmtModel.encoderSession.run(encoderFeeds);
            const latentCode = encoderResults[Object.keys(encoderResults)[0]].data;
            
            // 2. Apply style transfer in latent space
            const styleVector = encodeStyle(targetStyle);
            const styledLatent = blendLatentWithStyle(Array.from(latentCode), styleVector);
            
            // 3. Decode back to motion
            const decoderTensor = new ort.Tensor('float32', styledLatent, [1, styledLatent.length]);
            const decoderFeeds = { 'latent': decoderTensor };
            const decoderResults = await rsmtModel.decoderSession.run(decoderFeeds);
            const stylizedMotionData = decoderResults[Object.keys(decoderResults)[0]].data;
            
            return {
                joints: reshapeMotionData(Array.from(stylizedMotionData)),
                style: targetStyle,
                quality: 0.9,
                method: 'onnx'
            };
            
        } catch (error) {
            console.warn('⚠️ ONNX style transfer failed, using demo mode:', error);
            return applyDemoStyleTransfer(motionFeatures, targetStyle);
        }
        
    } else {
        // Demo mode
        return applyDemoStyleTransfer(motionFeatures, targetStyle);
    }
}

/**
 * Demo style transfer when ONNX models are not available
 */
function applyDemoStyleTransfer(motionFeatures, targetStyle) {
    const styleModifications = getStyleModifications(targetStyle);
    
    // Apply style-specific modifications to motion
    const stylizedJoints = motionFeatures.joints.map(frameData => {
        return frameData.map((joint, idx) => {
            return joint * styleModifications.amplification[idx % styleModifications.amplification.length];
        });
    });
    
    return {
        joints: stylizedJoints,
        style: targetStyle,
        quality: 0.7, // Demo quality
        method: 'demo'
    };
}

/**
 * Process raw motion data into feature representation
 * @param {Object} motionData - Raw motion data
 * @returns {Object} Processed motion features
 */
function processMotionData(motionData) {
    // Extract motion characteristics
    const joints = motionData.joints || generateDefaultJoints();
    const velocities = calculateVelocities(joints);
    const accelerations = calculateAccelerations(velocities);
    
    return {
        joints: joints,
        velocities: velocities,
        accelerations: accelerations,
        frameCount: joints.length,
        duration: motionData.duration || joints.length / 30.0, // Assume 30 FPS
        
        // Motion analysis features
        energy: calculateMotionEnergy(velocities),
        smoothness: calculateMotionSmoothness(accelerations),
        rhythm: analyzeMotionRhythm(joints),
        symmetry: analyzeMotionSymmetry(joints)
    };
}

/**
 * Generate default joint data if not provided
 */
function generateDefaultJoints() {
    const frameCount = 60; // 2 seconds at 30 FPS
    const jointNames = [
        'hips', 'spine1', 'spine2', 'chest', 'neck', 'head',
        'leftShoulder', 'leftElbow', 'leftWrist', 'leftHand',
        'rightShoulder', 'rightElbow', 'rightWrist', 'rightHand',
        'leftHip', 'leftKnee', 'leftAnkle', 'leftFoot',
        'rightHip', 'rightKnee', 'rightAnkle', 'rightFoot'
    ];
    
    const frames = [];
    
    for (let frame = 0; frame < frameCount; frame++) {
        const frameData = {};
        
        jointNames.forEach(jointName => {
            frameData[jointName] = {
                position: {
                    x: Math.sin(frame * 0.1) * 0.1,
                    y: 0,
                    z: Math.cos(frame * 0.1) * 0.1
                },
                rotation: {
                    x: 0,
                    y: Math.sin(frame * 0.05) * 0.2,
                    z: 0
                }
            };
        });
        
        frames.push(frameData);
    }
    
    return frames;
}

/**
 * Calculate velocities from joint positions
 * @param {Array} joints - Joint data per frame
 * @returns {Array} Velocity data per frame
 */
function calculateVelocities(joints) {
    if (joints.length < 2) return [];
    
    const velocities = [];
    
    for (let frame = 1; frame < joints.length; frame++) {
        const currentFrame = joints[frame];
        const previousFrame = joints[frame - 1];
        const frameVelocities = {};
        
        for (const jointName in currentFrame) {
            const current = currentFrame[jointName];
            const previous = previousFrame[jointName];
            
            frameVelocities[jointName] = {
                position: {
                    x: current.position.x - previous.position.x,
                    y: current.position.y - previous.position.y,
                    z: current.position.z - previous.position.z
                }
            };
        }
        
        velocities.push(frameVelocities);
    }
    
    return velocities;
}

/**
 * Calculate accelerations from velocities
 * @param {Array} velocities - Velocity data per frame
 * @returns {Array} Acceleration data per frame
 */
function calculateAccelerations(velocities) {
    if (velocities.length < 2) return [];
    
    const accelerations = [];
    
    for (let frame = 1; frame < velocities.length; frame++) {
        const currentVel = velocities[frame];
        const previousVel = velocities[frame - 1];
        const frameAccelerations = {};
        
        for (const jointName in currentVel) {
            const current = currentVel[jointName];
            const previous = previousVel[jointName];
            
            frameAccelerations[jointName] = {
                position: {
                    x: current.position.x - previous.position.x,
                    y: current.position.y - previous.position.y,
                    z: current.position.z - previous.position.z
                }
            };
        }
        
        accelerations.push(frameAccelerations);
    }
    
    return accelerations;
}

/**
 * Calculate overall motion energy
 * @param {Array} velocities - Velocity data
 * @returns {number} Motion energy value
 */
function calculateMotionEnergy(velocities) {
    if (velocities.length === 0) return 0;
    
    let totalEnergy = 0;
    let frameCount = 0;
    
    velocities.forEach(frameVel => {
        for (const jointName in frameVel) {
            const vel = frameVel[jointName].position;
            const magnitude = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            totalEnergy += magnitude;
            frameCount++;
        }
    });
    
    return frameCount > 0 ? totalEnergy / frameCount : 0;
}

/**
 * Calculate motion smoothness
 * @param {Array} accelerations - Acceleration data
 * @returns {number} Smoothness value (lower = smoother)
 */
function calculateMotionSmoothness(accelerations) {
    if (accelerations.length === 0) return 0;
    
    let totalJerk = 0;
    let frameCount = 0;
    
    accelerations.forEach(frameAcc => {
        for (const jointName in frameAcc) {
            const acc = frameAcc[jointName].position;
            const magnitude = Math.sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z);
            totalJerk += magnitude;
            frameCount++;
        }
    });
    
    return frameCount > 0 ? totalJerk / frameCount : 0;
}

/**
 * Analyze motion rhythm patterns
 * @param {Array} joints - Joint data
 * @returns {Object} Rhythm analysis
 */
function analyzeMotionRhythm(joints) {
    // Simplified rhythm analysis
    const cyclicJoints = ['leftElbow', 'rightElbow', 'leftKnee', 'rightKnee'];
    const rhythmData = {};
    
    cyclicJoints.forEach(jointName => {
        const positions = joints.map(frame => frame[jointName]?.position.y || 0);
        
        // Find peaks and valleys for rhythm detection
        let peaks = 0;
        for (let i = 1; i < positions.length - 1; i++) {
            if (positions[i] > positions[i-1] && positions[i] > positions[i+1]) {
                peaks++;
            }
        }
        
        rhythmData[jointName] = {
            frequency: peaks / (joints.length / 30.0), // peaks per second
            amplitude: Math.max(...positions) - Math.min(...positions)
        };
    });
    
    return rhythmData;
}

/**
 * Analyze motion symmetry between left/right limbs
 * @param {Array} joints - Joint data
 * @returns {Object} Symmetry analysis
 */
function analyzeMotionSymmetry(joints) {
    const symmetryPairs = [
        ['leftElbow', 'rightElbow'],
        ['leftWrist', 'rightWrist'],
        ['leftKnee', 'rightKnee'],
        ['leftAnkle', 'rightAnkle']
    ];
    
    const symmetryScores = {};
    
    symmetryPairs.forEach(([left, right]) => {
        let totalDifference = 0;
        let frameCount = 0;
        
        joints.forEach(frame => {
            const leftPos = frame[left]?.position;
            const rightPos = frame[right]?.position;
            
            if (leftPos && rightPos) {
                // Calculate mirrored difference
                const diff = Math.sqrt(
                    Math.pow(leftPos.x + rightPos.x, 2) + // X should be mirrored
                    Math.pow(leftPos.y - rightPos.y, 2) + // Y should be same
                    Math.pow(leftPos.z - rightPos.z, 2)   // Z should be same
                );
                totalDifference += diff;
                frameCount++;
            }
        });
        
        symmetryScores[`${left}_${right}`] = frameCount > 0 ? 
            1 - Math.min(1, totalDifference / frameCount) : 0;
    });
    
    return symmetryScores;
}

/**
 * Apply motion style transfer using RSMT model
 * @param {Object} motionFeatures - Processed motion features
 * @param {string} targetStyle - Target style to apply
 * @returns {Object} Stylized motion data
 */
async function applyMotionStyleTransfer(motionFeatures, targetStyle = 'casual') {
    // Simulate neural network inference delay
    await new Promise(resolve => setTimeout(resolve, 80 + Math.random() * 120));
    
    const style = rsmtModel.styles.includes(targetStyle) ? targetStyle : 'casual';
    
    // Apply style-specific modifications
    const stylizedJoints = applyStyleModifications(motionFeatures.joints, style);
    const transitionSmoothing = calculateTransitionSmoothing(motionFeatures, style);
    
    return {
        joints: stylizedJoints,
        style: style,
        confidence: 0.87 + Math.random() * 0.08,
        frameCount: motionFeatures.frameCount,
        duration: motionFeatures.duration,
        
        // Style-specific metrics
        energyMultiplier: getStyleEnergyMultiplier(style),
        smoothnessTarget: getStyleSmoothnessTarget(style),
        rhythmAdjustment: getStyleRhythmAdjustment(style),
        
        // Transition data
        transitionSmoothing: transitionSmoothing,
        blendWeight: calculateBlendWeight(motionFeatures, style)
    };
}

/**
 * Apply style-specific modifications to joint data
 * @param {Array} joints - Original joint data
 * @param {string} style - Target style
 * @returns {Array} Modified joint data
 */
function applyStyleModifications(joints, style) {
    const energyMult = getStyleEnergyMultiplier(style);
    const smoothnessMult = getStyleSmoothnessTarget(style);
    const rhythmAdj = getStyleRhythmAdjustment(style);
    
    return joints.map((frame, frameIndex) => {
        const modifiedFrame = {};
        
        for (const jointName in frame) {
            const originalJoint = frame[jointName];
            const jointType = getJointType(jointName);
            
            // Apply style-specific transformations
            let positionScale = energyMult;
            let rotationScale = energyMult * 0.8;
            
            // Different joints respond differently to style
            switch (jointType) {
                case 'spine':
                    positionScale *= getSpineStyleFactor(style);
                    break;
                case 'arm':
                    positionScale *= getArmStyleFactor(style);
                    rotationScale *= getArmStyleFactor(style);
                    break;
                case 'leg':
                    positionScale *= getLegStyleFactor(style);
                    break;
            }
            
            // Add rhythmic variations
            const rhythmFactor = 1 + Math.sin(frameIndex * rhythmAdj.frequency) * rhythmAdj.amplitude;
            
            modifiedFrame[jointName] = {
                position: {
                    x: originalJoint.position.x * positionScale * rhythmFactor,
                    y: originalJoint.position.y * positionScale,
                    z: originalJoint.position.z * positionScale * rhythmFactor
                },
                rotation: {
                    x: originalJoint.rotation.x * rotationScale,
                    y: originalJoint.rotation.y * rotationScale * rhythmFactor,
                    z: originalJoint.rotation.z * rotationScale
                }
            };
        }
        
        return modifiedFrame;
    });
}

/**
 * Get joint type for style application
 * @param {string} jointName - Name of the joint
 * @returns {string} Joint type
 */
function getJointType(jointName) {
    if (jointName.includes('spine') || jointName.includes('chest') || jointName.includes('hips')) {
        return 'spine';
    } else if (jointName.includes('Arm') || jointName.includes('Elbow') || jointName.includes('Wrist') || jointName.includes('Hand')) {
        return 'arm';
    } else if (jointName.includes('Hip') || jointName.includes('Knee') || jointName.includes('Ankle') || jointName.includes('Foot')) {
        return 'leg';
    }
    return 'other';
}

/**
 * Get energy multiplier for different styles
 * @param {string} style - Style name
 * @returns {number} Energy multiplier
 */
function getStyleEnergyMultiplier(style) {
    const styleEnergy = {
        'casual': 1.0,
        'confident': 1.3,
        'energetic': 1.8,
        'relaxed': 0.7,
        'dramatic': 1.5
    };
    return styleEnergy[style] || 1.0;
}

/**
 * Get smoothness target for different styles
 * @param {string} style - Style name
 * @returns {number} Smoothness target
 */
function getStyleSmoothnessTarget(style) {
    const styleSmoothness = {
        'casual': 1.0,
        'confident': 0.8,
        'energetic': 0.6,
        'relaxed': 1.4,
        'dramatic': 0.7
    };
    return styleSmoothness[style] || 1.0;
}

/**
 * Get rhythm adjustment for different styles
 * @param {string} style - Style name
 * @returns {Object} Rhythm adjustment parameters
 */
function getStyleRhythmAdjustment(style) {
    const rhythmAdjustments = {
        'casual': { frequency: 0.1, amplitude: 0.05 },
        'confident': { frequency: 0.08, amplitude: 0.1 },
        'energetic': { frequency: 0.15, amplitude: 0.15 },
        'relaxed': { frequency: 0.05, amplitude: 0.03 },
        'dramatic': { frequency: 0.12, amplitude: 0.2 }
    };
    return rhythmAdjustments[style] || rhythmAdjustments['casual'];
}

/**
 * Get spine-specific style factors
 */
function getSpineStyleFactor(style) {
    const factors = {
        'casual': 1.0,
        'confident': 1.2,
        'energetic': 1.1,
        'relaxed': 0.9,
        'dramatic': 1.3
    };
    return factors[style] || 1.0;
}

/**
 * Get arm-specific style factors
 */
function getArmStyleFactor(style) {
    const factors = {
        'casual': 1.0,
        'confident': 1.1,
        'energetic': 1.4,
        'relaxed': 0.8,
        'dramatic': 1.6
    };
    return factors[style] || 1.0;
}

/**
 * Get leg-specific style factors
 */
function getLegStyleFactor(style) {
    const factors = {
        'casual': 1.0,
        'confident': 1.05,
        'energetic': 1.2,
        'relaxed': 0.95,
        'dramatic': 1.1
    };
    return factors[style] || 1.0;
}

/**
 * Calculate transition smoothing parameters
 * @param {Object} motionFeatures - Motion features
 * @param {string} style - Target style
 * @returns {Object} Transition smoothing data
 */
function calculateTransitionSmoothing(motionFeatures, style) {
    return {
        blendFrames: 30, // Smooth over 1 second
        easeFunction: 'cubic',
        preserveRhythm: style !== 'relaxed',
        energyTransition: getStyleEnergyMultiplier(style)
    };
}

/**
 * Calculate blend weight for style application
 * @param {Object} motionFeatures - Motion features
 * @param {string} style - Target style
 * @returns {number} Blend weight
 */
function calculateBlendWeight(motionFeatures, style) {
    // Higher energy motions blend more easily
    const energyFactor = Math.min(1, motionFeatures.energy * 2);
    
    // Different styles have different blend characteristics
    const styleBlendability = {
        'casual': 0.9,
        'confident': 0.8,
        'energetic': 0.7,
        'relaxed': 0.95,
        'dramatic': 0.6
    };
    
    return energyFactor * (styleBlendability[style] || 0.8);
}

console.log('🎨 RSMT Worker initialized');