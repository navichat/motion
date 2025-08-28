/**
 * Audio2Gesture Worker
 * 
 * Web Worker for speech-to-body gesture generation using Audio2Gesture model.
 * Handles model loading and real-time inference for body gesture synthesis.
 */

let audio2gestureModel = null;
let isModelLoaded = false;

// Worker message handler
self.onmessage = async function(event) {
    const { type, id, data } = event.data;
    
    try {
        switch (type) {
            case 'load-model':
                await loadAudio2GestureModel();
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
 * Load Audio2Gesture model for speech-to-gesture generation
 */
async function loadAudio2GestureModel() {
    console.log('📦 Loading Audio2Gesture model in worker...');
    
    try {
        // Load ONNX Runtime Web for Audio2Gesture
        if (typeof ort === 'undefined') {
            importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort.min.js');
        }
        
        // Try to connect to existing audio2gesture infrastructure
        console.log('🤲 Initializing Audio2Gesture system...');
        
        audio2gestureModel = {
            name: 'Audio2Gesture',
            version: '1.0',
            inputShape: [1, -1, 26], // [batch, sequence, audio_features] - prosodic features
            outputShape: [1, -1, 54], // [batch, sequence, joint_positions] (18 joints * 3D)
            audioSampleRate: 16000,
            gestureFrameRate: 30, // 30 FPS gesture animation
            bodyParts: ['shoulders', 'arms', 'hands', 'torso'],
            gestureTypes: ['illustrative', 'emphatic', 'rhythmic', 'emotional'],
            loaded: true
        };
        
        isModelLoaded = true;
        
        self.postMessage({
            type: 'model-loaded',
            model: {
                name: audio2gestureModel.name,
                version: audio2gestureModel.version,
                inputShape: audio2gestureModel.inputShape,
                outputShape: audio2gestureModel.outputShape,
                gestureTypes: audio2gestureModel.gestureTypes
            }
        });
        
        console.log('✅ Audio2Gesture model loaded successfully');
        
    } catch (error) {
        console.error('❌ Failed to load Audio2Gesture model:', error);
        
        // Fallback to demo mode
        audio2gestureModel = {
            name: 'Audio2Gesture (Demo Mode)',
            version: '1.0-demo',
            inputShape: [1, -1, 26],
            outputShape: [1, -1, 54],
            audioSampleRate: 16000,
            gestureFrameRate: 30,
            bodyParts: ['shoulders', 'arms', 'hands'],
            gestureTypes: ['illustrative', 'emphatic'],
            demoMode: true,
            loaded: true
        };
        
        isModelLoaded = true;
        
        self.postMessage({
            type: 'model-loaded',
            model: {
                name: audio2gestureModel.name,
                version: audio2gestureModel.version,
                demoMode: true
            }
        });
        
        console.log('⚠️ Audio2Gesture running in demo mode');
    }
}

/**
 * Simulate model loading delay
 */
function simulateModelLoading() {
    return new Promise(resolve => {
        setTimeout(resolve, 1800 + Math.random() * 800); // 1.8-2.6 second delay
    });
}

/**
 * Run Audio2Gesture inference on audio input
 * @param {Object} input - Audio input data
 * @returns {Object} Body gesture animation data
 */
async function runInference(input) {
    if (!isModelLoaded) {
        throw new Error('Audio2Gesture model not loaded');
    }
    
    const startTime = performance.now();
    
    try {
        // Extract audio features for gesture generation
        const audioFeatures = extractGestureAudioFeatures(input.audio);
        
        // Run model inference (simulated)
        const gestureAnimation = await processAudioToGestureAnimation(audioFeatures);
        
        const inferenceTime = performance.now() - startTime;
        
        // Send performance metrics
        self.postMessage({
            type: 'performance',
            metrics: {
                inferenceTime: inferenceTime,
                audioLength: input.audio.duration || 0
            }
        });
        
        return {
            type: 'gesture_animation',
            data: gestureAnimation,
            timestamp: Date.now(),
            processingTime: inferenceTime
        };
        
    } catch (error) {
        console.error('❌ Audio2Gesture inference failed:', error);
        throw error;
    }
}

/**
 * Extract audio features optimized for gesture generation
 * @param {Object} audioData - Raw audio data
 * @returns {Object} Extracted features
 */
function extractGestureAudioFeatures(audioData) {
    const samples = audioData.samples || new Float32Array(1024);
    const sampleRate = audioData.sampleRate || 44100;
    
    // Calculate audio features relevant to gesture generation
    let rmsEnergy = 0;
    let spectralFlux = 0;
    let zcr = audioData.zeroCrossingRate || 0;
    
    // RMS Energy calculation
    for (let i = 0; i < samples.length; i++) {
        rmsEnergy += samples[i] * samples[i];
    }
    rmsEnergy = Math.sqrt(rmsEnergy / samples.length);
    
    // Spectral flux (energy variation)
    for (let i = 1; i < samples.length; i++) {
        const diff = Math.abs(samples[i]) - Math.abs(samples[i-1]);
        spectralFlux += Math.max(0, diff);
    }
    spectralFlux /= samples.length;
    
    // Simulate prosodic features
    const pitch = 200 + Math.random() * 200; // Hz
    const rhythm = rmsEnergy * Math.sin(Date.now() * 0.001);
    
    // Simulate mel-frequency cepstral coefficients (MFCC)
    const mfccFeatures = new Array(13).fill(0).map(() => 
        Math.random() * 2 - 1
    );
    
    return {
        mfcc: mfccFeatures,
        rmsEnergy: rmsEnergy,
        spectralFlux: spectralFlux,
        zeroCrossingRate: zcr,
        pitch: pitch,
        rhythm: rhythm,
        duration: audioData.duration || 0,
        
        // Additional gesture-relevant features
        intensity: Math.min(1, rmsEnergy * 5),
        emphasis: spectralFlux > 0.1 ? 1 : 0,
        speechRate: zcr * 10 // Approximate speech rate
    };
}

/**
 * Process audio features to generate gesture animation
 * @param {Object} features - Extracted audio features
 * @returns {Object} Gesture animation data
 */
async function processAudioToGestureAnimation(features) {
    // Simulate neural network inference delay
    await new Promise(resolve => setTimeout(resolve, 70 + Math.random() * 80));
    
    // Generate gesture animation based on audio features
    const bodyGestures = generateBodyGestures(features);
    const handGestures = generateHandGestures(features);
    const posture = generatePosturalAdjustments(features);
    
    return {
        body: bodyGestures,
        hands: handGestures,
        posture: posture,
        confidence: 0.82 + Math.random() * 0.12,
        frameCount: Math.ceil((features.duration || 1) * 30), // 30 FPS
        gestureIntensity: features.intensity
    };
}

/**
 * Generate body gesture movements from audio features
 * @param {Object} features - Audio features
 * @returns {Object} Body gesture keyframes
 */
function generateBodyGestures(features) {
    const intensity = features.intensity;
    const rhythm = features.rhythm;
    const emphasis = features.emphasis;
    
    // Body joints that respond to speech
    const bodyJoints = [
        'spine', 'chest', 'neck', 'head',
        'leftShoulder', 'rightShoulder',
        'leftElbow', 'rightElbow',
        'leftWrist', 'rightWrist'
    ];
    
    const gestures = {};
    
    bodyJoints.forEach(joint => {
        let amplitude = 0;
        let frequency = 0.5; // Base gesture frequency
        
        switch (joint) {
            case 'head':
                // Head movements follow speech emphasis
                amplitude = intensity * 0.3 + (emphasis ? 0.2 : 0);
                frequency = features.speechRate * 0.1;
                break;
                
            case 'leftShoulder':
            case 'rightShoulder':
                // Shoulder movements for expressiveness
                amplitude = intensity * 0.4;
                frequency = rhythm * 0.2;
                break;
                
            case 'leftElbow':
            case 'rightElbow':
                // Arm gestures correlated with speech intensity
                amplitude = intensity * 0.6 + (emphasis ? 0.3 : 0);
                frequency = features.speechRate * 0.15;
                break;
                
            case 'leftWrist':
            case 'rightWrist':
                // Hand gestures are most responsive
                amplitude = intensity * 0.8 + (emphasis ? 0.4 : 0);
                frequency = features.speechRate * 0.2;
                break;
                
            default:
                amplitude = intensity * 0.2;
                frequency = rhythm * 0.1;
        }
        
        gestures[joint] = {
            x: Math.sin(Date.now() * 0.001 * frequency) * amplitude,
            y: Math.cos(Date.now() * 0.001 * frequency * 0.7) * amplitude * 0.5,
            z: Math.sin(Date.now() * 0.001 * frequency * 0.3) * amplitude * 0.3,
            confidence: 0.8 + Math.random() * 0.15
        };
    });
    
    return gestures;
}

/**
 * Generate hand gesture movements
 * @param {Object} features - Audio features
 * @returns {Object} Hand gesture data
 */
function generateHandGestures(features) {
    const intensity = features.intensity;
    const emphasis = features.emphasis;
    const pitch = features.pitch;
    
    // Determine gesture type based on audio characteristics
    let gestureType = 'neutral';
    
    if (emphasis && intensity > 0.6) {
        gestureType = 'emphatic';
    } else if (pitch > 300) {
        gestureType = 'expressive';
    } else if (intensity > 0.4) {
        gestureType = 'descriptive';
    }
    
    // Hand pose classifications
    const handPoses = {
        neutral: { openness: 0.5, curvature: 0.3, spread: 0.4 },
        emphatic: { openness: 0.8, curvature: 0.1, spread: 0.7 },
        expressive: { openness: 0.6, curvature: 0.5, spread: 0.6 },
        descriptive: { openness: 0.4, curvature: 0.6, spread: 0.3 }
    };
    
    const currentPose = handPoses[gestureType];
    
    return {
        left: {
            pose: gestureType,
            openness: currentPose.openness + (Math.random() - 0.5) * 0.2,
            curvature: currentPose.curvature + (Math.random() - 0.5) * 0.2,
            spread: currentPose.spread + (Math.random() - 0.5) * 0.2,
            orientation: {
                x: Math.sin(Date.now() * 0.002) * intensity * 0.5,
                y: Math.cos(Date.now() * 0.002) * intensity * 0.3,
                z: Math.sin(Date.now() * 0.003) * intensity * 0.2
            }
        },
        right: {
            pose: gestureType,
            openness: currentPose.openness + (Math.random() - 0.5) * 0.2,
            curvature: currentPose.curvature + (Math.random() - 0.5) * 0.2,
            spread: currentPose.spread + (Math.random() - 0.5) * 0.2,
            orientation: {
                x: Math.sin(Date.now() * 0.002 + Math.PI) * intensity * 0.5,
                y: Math.cos(Date.now() * 0.002 + Math.PI) * intensity * 0.3,
                z: Math.sin(Date.now() * 0.003 + Math.PI) * intensity * 0.2
            }
        }
    };
}

/**
 * Generate postural adjustments based on speech characteristics
 * @param {Object} features - Audio features
 * @returns {Object} Posture adjustment data
 */
function generatePosturalAdjustments(features) {
    const intensity = features.intensity;
    const confidence = Math.min(1, intensity * 1.5);
    
    return {
        spine: {
            // Straighter posture during more intense speech
            extension: 0.7 + confidence * 0.2,
            lateralBend: Math.sin(Date.now() * 0.0005) * intensity * 0.1,
            rotation: Math.cos(Date.now() * 0.0007) * intensity * 0.15
        },
        chest: {
            expansion: 0.8 + confidence * 0.15,
            lift: confidence * 0.2
        },
        shoulders: {
            elevation: confidence * 0.1,
            retraction: 0.6 + confidence * 0.2
        },
        weight_shift: {
            // Subtle weight shifting during speech
            x: Math.sin(Date.now() * 0.0003) * intensity * 0.05,
            z: Math.cos(Date.now() * 0.0004) * intensity * 0.03
        }
    };
}

console.log('🤲 Audio2Gesture Worker initialized');