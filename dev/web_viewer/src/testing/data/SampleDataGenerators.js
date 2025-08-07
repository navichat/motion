/**
 * Test Data Generators
 * 
 * Provides consistent test data generation for all animation backends
 * to ensure reliable and repeatable testing scenarios.
 */

/**
 * Sample Data Generators
 */
class SampleDataGenerators {
    constructor() {
        this.defaultBoneHierarchy = [
            'hips', 'spine', 'spine1', 'spine2', 'neck', 'head',
            'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
            'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
            'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToe',
            'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToe'
        ];
        
        this.facialBones = [
            'head', 'jaw', 'leftEye', 'rightEye', 'leftEyebrow', 'rightEyebrow',
            'nose', 'mouth', 'leftCheek', 'rightCheek'
        ];
    }
    
    /**
     * Generate Sample BVH Frame
     */
    generateBVHFrame(options = {}) {
        const frameNumber = options.frameNumber || 0;
        const time = options.time || frameNumber / 30;
        const bones = {};
        
        // Generate bone positions and rotations
        const boneList = options.bones || this.defaultBoneHierarchy;
        
        for (const boneName of boneList) {
            bones[boneName] = this.generateBoneData(boneName, time, options);
        }
        
        return {
            frameNumber: frameNumber,
            time: time,
            bones: bones,
            metadata: {
                source: options.source || 'test_generator',
                type: options.type || 'sample',
                ...options.metadata
            }
        };
    }
    
    /**
     * Generate bone data with realistic constraints
     */
    generateBoneData(boneName, time, options = {}) {
        const basePositions = this.getBoneBasePosition(boneName);
        const animationOffset = options.animationOffset || 0;
        
        // Add time-based animation
        const animationPhase = time * 2 * Math.PI + animationOffset;
        const intensity = options.intensity || 0.1;
        
        const position = {
            x: basePositions.x + Math.sin(animationPhase) * intensity * 0.1,
            y: basePositions.y + Math.cos(animationPhase * 0.5) * intensity * 0.05,
            z: basePositions.z + Math.sin(animationPhase * 1.5) * intensity * 0.08
        };
        
        // Generate rotation with anatomical limits
        const rotation = this.generateRealisticRotation(boneName, animationPhase, intensity);
        
        return {
            position: position,
            rotation: rotation
        };
    }
    
    /**
     * Get base position for bone
     */
    getBoneBasePosition(boneName) {
        const positions = {
            hips: { x: 0, y: 1.0, z: 0 },
            spine: { x: 0, y: 1.2, z: 0 },
            spine1: { x: 0, y: 1.35, z: 0 },
            spine2: { x: 0, y: 1.45, z: 0 },
            neck: { x: 0, y: 1.55, z: 0 },
            head: { x: 0, y: 1.65, z: 0 },
            
            leftShoulder: { x: -0.2, y: 1.45, z: 0 },
            leftUpperArm: { x: -0.3, y: 1.4, z: 0 },
            leftLowerArm: { x: -0.3, y: 1.1, z: 0 },
            leftHand: { x: -0.3, y: 0.85, z: 0 },
            
            rightShoulder: { x: 0.2, y: 1.45, z: 0 },
            rightUpperArm: { x: 0.3, y: 1.4, z: 0 },
            rightLowerArm: { x: 0.3, y: 1.1, z: 0 },
            rightHand: { x: 0.3, y: 0.85, z: 0 },
            
            leftUpperLeg: { x: -0.1, y: 0.9, z: 0 },
            leftLowerLeg: { x: -0.1, y: 0.5, z: 0 },
            leftFoot: { x: -0.1, y: 0.1, z: 0 },
            leftToe: { x: -0.1, y: 0.05, z: 0.1 },
            
            rightUpperLeg: { x: 0.1, y: 0.9, z: 0 },
            rightLowerLeg: { x: 0.1, y: 0.5, z: 0 },
            rightFoot: { x: 0.1, y: 0.1, z: 0 },
            rightToe: { x: 0.1, y: 0.05, z: 0.1 },
            
            // Facial bones
            jaw: { x: 0, y: 1.58, z: 0.05 },
            leftEye: { x: -0.03, y: 1.67, z: 0.08 },
            rightEye: { x: 0.03, y: 1.67, z: 0.08 },
            leftEyebrow: { x: -0.03, y: 1.69, z: 0.07 },
            rightEyebrow: { x: 0.03, y: 1.69, z: 0.07 },
            nose: { x: 0, y: 1.65, z: 0.09 },
            mouth: { x: 0, y: 1.61, z: 0.08 },
            leftCheek: { x: -0.04, y: 1.63, z: 0.06 },
            rightCheek: { x: 0.04, y: 1.63, z: 0.06 }
        };
        
        return positions[boneName] || { x: 0, y: 0, z: 0 };
    }
    
    /**
     * Generate realistic rotation with anatomical constraints
     */
    generateRealisticRotation(boneName, phase, intensity) {
        // Define rotation limits for different bones
        const limits = {
            hips: { x: 0.2, y: 0.3, z: 0.1 },
            spine: { x: 0.3, y: 0.2, z: 0.2 },
            neck: { x: 0.4, y: 0.6, z: 0.3 },
            head: { x: 0.3, y: 0.5, z: 0.2 },
            
            leftUpperArm: { x: 1.5, y: 0.8, z: 1.2 },
            rightUpperArm: { x: 1.5, y: 0.8, z: 1.2 },
            leftLowerArm: { x: 0.1, y: 0.1, z: 1.4 },
            rightLowerArm: { x: 0.1, y: 0.1, z: 1.4 },
            
            leftUpperLeg: { x: 1.0, y: 0.3, z: 0.4 },
            rightUpperLeg: { x: 1.0, y: 0.3, z: 0.4 },
            leftLowerLeg: { x: 1.2, y: 0.1, z: 0.1 },
            rightLowerLeg: { x: 1.2, y: 0.1, z: 0.1 }
        };
        
        const limit = limits[boneName] || { x: 0.1, y: 0.1, z: 0.1 };
        
        const x = Math.sin(phase) * intensity * limit.x;
        const y = Math.sin(phase * 1.3) * intensity * limit.y;
        const z = Math.sin(phase * 0.7) * intensity * limit.z;
        
        // Convert to quaternion
        const w = Math.sqrt(1 - (x*x + y*y + z*z) * 0.5);
        
        return { x, y, z, w };
    }
    
    /**
     * Generate Audio Features
     */
    generateAudioFeatures(options = {}) {
        const duration = options.duration || 1.0;
        const sampleRate = options.sampleRate || 16000;
        const energy = options.energy || Math.random() * 0.5 + 0.3;
        
        return {
            duration: duration,
            sampleRate: sampleRate,
            energy: energy,
            pitch: 120 + Math.random() * 80,
            spectralCentroid: 2000 + Math.random() * 1000,
            zeroCrossingRate: 0.1 + Math.random() * 0.05,
            mfcc: Array.from({ length: 13 }, () => Math.random() - 0.5),
            rms: energy,
            confidence: 0.8 + Math.random() * 0.2,
            features: {
                tempo: 60 + Math.random() * 80,
                loudness: energy * 100,
                brightness: Math.random(),
                roughness: Math.random() * 0.3
            }
        };
    }
    
    /**
     * Generate Facial Landmarks
     */
    generateFacialLandmarks(options = {}) {
        const expression = options.expression || 'neutral';
        const confidence = options.confidence || 0.9;
        
        // Generate 68 facial landmarks (standard face detection format)
        const landmarks = [];
        
        // Face outline (0-16)
        for (let i = 0; i < 17; i++) {
            const angle = (i / 16) * Math.PI - Math.PI * 0.5;
            landmarks.push({
                x: 0.4 + Math.cos(angle) * 0.35,
                y: 0.6 + Math.sin(angle) * 0.4,
                confidence: confidence
            });
        }
        
        // Eyebrows (17-26)
        for (let i = 0; i < 10; i++) {
            const isLeft = i < 5;
            const localIndex = i % 5;
            landmarks.push({
                x: isLeft ? 0.25 + localIndex * 0.05 : 0.55 + localIndex * 0.05,
                y: 0.35 + Math.sin(localIndex * 0.5) * 0.02,
                confidence: confidence
            });
        }
        
        // Nose (27-35)
        for (let i = 0; i < 9; i++) {
            landmarks.push({
                x: 0.48 + (Math.random() - 0.5) * 0.08,
                y: 0.45 + i * 0.02,
                confidence: confidence
            });
        }
        
        // Eyes (36-47)
        for (let i = 0; i < 12; i++) {
            const isLeft = i < 6;
            const localIndex = i % 6;
            const angle = (localIndex / 6) * Math.PI * 2;
            landmarks.push({
                x: (isLeft ? 0.35 : 0.65) + Math.cos(angle) * 0.03,
                y: 0.4 + Math.sin(angle) * 0.02,
                confidence: confidence
            });
        }
        
        // Mouth (48-67)
        for (let i = 0; i < 20; i++) {
            const angle = (i / 20) * Math.PI * 2;
            const radius = i < 12 ? 0.06 : 0.03; // Outer vs inner lip
            landmarks.push({
                x: 0.5 + Math.cos(angle) * radius,
                y: 0.65 + Math.sin(angle) * radius * 0.5,
                confidence: confidence
            });
        }
        
        return {
            landmarks: landmarks,
            expression: expression,
            confidence: confidence,
            boundingBox: {
                x: 0.1, y: 0.1, width: 0.8, height: 0.8
            },
            quality: confidence
        };
    }
    
    /**
     * Generate Physics Constraints
     */
    generatePhysicsConstraints(options = {}) {
        const constraintType = options.type || 'joint';
        
        return {
            type: constraintType,
            bodyA: options.bodyA || 'upperArm',
            bodyB: options.bodyB || 'lowerArm',
            anchor: options.anchor || { x: 0, y: 0, z: 0 },
            limits: {
                minAngle: -1.5,
                maxAngle: 1.5,
                stiffness: 0.8,
                damping: 0.1
            },
            force: options.force || { x: 0, y: -9.81, z: 0 },
            active: true,
            priority: options.priority || 1.0
        };
    }
    
    /**
     * Generate Animation Clip
     */
    generateAnimationClip(options = {}) {
        const duration = options.duration || 2.0;
        const frameRate = options.frameRate || 30;
        const frameCount = Math.ceil(duration * frameRate);
        const frames = [];
        
        for (let i = 0; i < frameCount; i++) {
            const time = i / frameRate;
            frames.push(this.generateBVHFrame({
                frameNumber: i,
                time: time,
                intensity: options.intensity || 0.2,
                animationOffset: options.animationOffset || 0,
                source: options.source || 'clip_generator'
            }));
        }
        
        return {
            id: options.id || `clip_${Date.now()}`,
            name: options.name || 'Generated Clip',
            frames: frames,
            duration: duration,
            frameRate: frameRate,
            frameCount: frameCount,
            metadata: {
                generated: true,
                generator: 'SampleDataGenerators',
                ...options.metadata
            }
        };
    }
    
    /**
     * Generate Pathfinding Scenario
     */
    generatePathfindingScenario(options = {}) {
        const start = options.start || { x: 0, y: 0, z: 0 };
        const end = options.end || { x: 5, y: 0, z: 5 };
        const obstacles = options.obstacles || [];
        
        // Add some default obstacles if none provided
        if (obstacles.length === 0) {
            obstacles.push(
                { x: 2, y: 0, z: 2, radius: 0.5 },
                { x: 3, y: 0, z: 1, radius: 0.3 },
                { x: 1, y: 0, z: 4, radius: 0.4 }
            );
        }
        
        return {
            start: start,
            destination: end,
            obstacles: obstacles,
            gridSize: options.gridSize || { width: 10, height: 10 },
            movementCost: options.movementCost || 1.0,
            heuristic: options.heuristic || 'euclidean'
        };
    }
    
    /**
     * Generate Multi-Modal Test Data
     */
    generateMultiModalData(options = {}) {
        const duration = options.duration || 2.0;
        
        return {
            audio: this.generateAudioFeatures({ duration }),
            facial: this.generateFacialLandmarks({ expression: options.expression }),
            physics: this.generatePhysicsConstraints({ type: options.constraintType }),
            pathfinding: this.generatePathfindingScenario({ 
                end: options.destination || { x: 3, y: 0, z: 3 }
            }),
            timeline: {
                duration: duration,
                frameRate: 30,
                layers: [
                    { id: 'facial', priority: 3, blend: 'additive' },
                    { id: 'gesture', priority: 2, blend: 'override' },
                    { id: 'physics', priority: 1, blend: 'base' }
                ]
            }
        };
    }
    
    /**
     * Generate Performance Test Data
     */
    generatePerformanceTestData(options = {}) {
        const frameCount = options.frameCount || 1000;
        const boneCount = options.boneCount || 20;
        const clipCount = options.clipCount || 10;
        
        const clips = [];
        for (let i = 0; i < clipCount; i++) {
            clips.push(this.generateAnimationClip({
                id: `perf_clip_${i}`,
                duration: frameCount / 30,
                frameRate: 30,
                intensity: 0.1 + (i / clipCount) * 0.4
            }));
        }
        
        return {
            clips: clips,
            totalFrames: frameCount * clipCount,
            totalBones: boneCount,
            estimatedMemoryMB: (frameCount * clipCount * boneCount * 8 * 7) / (1024 * 1024), // Rough estimate
            testType: 'performance'
        };
    }
    
    /**
     * Generate Error Scenarios
     */
    generateErrorScenarios() {
        return [
            {
                name: 'invalid_frame_format',
                data: { frameNumber: 'invalid', bones: null },
                expectedError: 'Invalid frame format'
            },
            {
                name: 'missing_bone_data',
                data: { frameNumber: 0, time: 0, bones: {} },
                expectedError: 'No bone data'
            },
            {
                name: 'invalid_audio_features',
                data: { duration: -1, energy: 'invalid' },
                expectedError: 'Invalid audio features'
            },
            {
                name: 'malformed_landmarks',
                data: { landmarks: 'not_array' },
                expectedError: 'Invalid landmarks format'
            },
            {
                name: 'physics_constraint_error',
                data: { type: 'unknown', bodyA: null },
                expectedError: 'Invalid physics constraint'
            }
        ];
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SampleDataGenerators };
} else {
    // Browser global
    window.SampleDataGenerators = SampleDataGenerators;
}
