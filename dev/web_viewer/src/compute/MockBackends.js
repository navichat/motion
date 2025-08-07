/**
 * Mock Backend Implementations for Testing
 * 
 * Provides consistent mock implementations of all animation backends
 * for reliable testing without external dependencies.
 */

/**
 * Mock BVH Timeline Implementation
 */
class MockBVHTimeline {
    constructor() {
        this.frameRate = 30;
        this.currentTime = 0;
        this.isPlaying = false;
        this.layers = [];
        this.clips = new Map(); // trackName -> clips array
        this.initialized = false;
    }
    
    async initialize(options = {}) {
        this.frameRate = options.frameRate || 30;
        this.initialized = true;
        return true;
    }
    
    isInitialized() {
        return this.initialized;
    }
    
    async addFrame(trackName, frame) {
        if (!this.clips.has(trackName)) {
            this.clips.set(trackName, []);
        }
        
        // Add frame as a single-frame clip
        this.clips.get(trackName).push({
            id: `frame_${Date.now()}`,
            frames: [frame],
            startTime: frame.time * 1000,
            duration: 1000 / this.frameRate
        });
    }
    
    async getFrameAtTime(time) {
        // Find the most recent frame at or before the given time
        let latestFrame = null;
        let latestTime = -1;
        
        for (const clipArray of this.clips.values()) {
            for (const clip of clipArray) {
                for (const frame of clip.frames) {
                    if (frame.time <= time && frame.time > latestTime) {
                        latestFrame = frame;
                        latestTime = frame.time;
                    }
                }
            }
        }
        
        return latestFrame || {
            frameNumber: 0,
            time: time,
            bones: {},
            metadata: { source: 'mock' }
        };
    }
    
    async addLayer(layer) {
        this.layers.push(layer);
        if (!this.clips.has(layer.id)) {
            this.clips.set(layer.id, []);
        }
    }
    
    getLayers() {
        return this.layers;
    }
    
    async addClip(trackName, clip) {
        if (!this.clips.has(trackName)) {
            this.clips.set(trackName, []);
        }
        this.clips.get(trackName).push(clip);
    }
    
    getClips(trackName) {
        return this.clips.get(trackName) || [];
    }
    
    play() {
        this.isPlaying = true;
        this.startPlayback();
    }
    
    pause() {
        this.isPlaying = false;
    }
    
    reset() {
        this.currentTime = 0;
        this.isPlaying = false;
    }
    
    getCurrentTime() {
        return this.currentTime;
    }
    
    isPlaying() {
        return this.isPlaying;
    }
    
    startPlayback() {
        if (!this.isPlaying) return;
        
        const tick = () => {
            if (this.isPlaying) {
                this.currentTime += 1000 / this.frameRate; // Advance by one frame
                setTimeout(tick, 1000 / this.frameRate);
            }
        };
        
        tick();
    }
}

/**
 * Mock Audio2Gesture Converter Implementation
 */
class MockAudio2GestureConverter {
    constructor() {
        this.initialized = false;
        this.processingTime = 50; // Mock processing delay
    }
    
    async initialize() {
        await this.mockDelay(100);
        this.initialized = true;
        return true;
    }
    
    isInitialized() {
        return this.initialized;
    }
    
    async processAudioFeatures(audioFeatures) {
        await this.mockDelay(this.processingTime);
        
        return {
            gestureData: [
                { joint: 'leftUpperArm', rotation: { x: 0.1, y: 0, z: 0 }, confidence: 0.8 },
                { joint: 'rightUpperArm', rotation: { x: -0.1, y: 0, z: 0 }, confidence: 0.8 },
                { joint: 'leftLowerArm', rotation: { x: 0.2, y: 0, z: 0 }, confidence: 0.7 },
                { joint: 'rightLowerArm', rotation: { x: -0.2, y: 0, z: 0 }, confidence: 0.7 }
            ],
            intensity: audioFeatures.energy || 0.5,
            confidence: 0.85
        };
    }
    
    async generateBVHFromAudio(audioFeatures) {
        await this.mockDelay(this.processingTime);
        
        const frames = [];
        const duration = audioFeatures.duration || 1.0;
        const frameCount = Math.ceil(duration * 30); // 30 fps
        
        for (let i = 0; i < frameCount; i++) {
            const time = i / 30;
            const progress = i / (frameCount - 1);
            
            // Generate gesture animation based on audio energy
            const energy = audioFeatures.energy || 0.5;
            const armRotation = Math.sin(time * 2 * Math.PI) * energy * 0.2;
            
            frames.push({
                frameNumber: i,
                time: time,
                bones: {
                    leftUpperArm: {
                        position: { x: -0.3, y: 1.4, z: 0 },
                        rotation: { x: armRotation, y: 0, z: 0, w: Math.cos(armRotation / 2) }
                    },
                    rightUpperArm: {
                        position: { x: 0.3, y: 1.4, z: 0 },
                        rotation: { x: -armRotation, y: 0, z: 0, w: Math.cos(armRotation / 2) }
                    },
                    leftLowerArm: {
                        position: { x: -0.3, y: 1.1, z: 0 },
                        rotation: { x: armRotation * 1.5, y: 0, z: 0, w: Math.cos(armRotation * 1.5 / 2) }
                    },
                    rightLowerArm: {
                        position: { x: 0.3, y: 1.1, z: 0 },
                        rotation: { x: -armRotation * 1.5, y: 0, z: 0, w: Math.cos(armRotation * 1.5 / 2) }
                    }
                },
                metadata: { source: 'mock_audio2gesture', energy: energy }
            });
        }
        
        return frames;
    }
    
    async mockDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Mock FaceFormer Converter Implementation
 */
class MockFaceFormerConverter {
    constructor() {
        this.initialized = false;
        this.processingTime = 30;
    }
    
    async initialize() {
        await this.mockDelay(100);
        this.initialized = true;
        return true;
    }
    
    isInitialized() {
        return this.initialized;
    }
    
    async processLandmarks(facialLandmarks) {
        await this.mockDelay(this.processingTime);
        
        // Extract key facial features from landmarks
        const eyeDistance = this.calculateEyeDistance(facialLandmarks.landmarks);
        const mouthWidth = this.calculateMouthWidth(facialLandmarks.landmarks);
        
        return {
            blendShapes: {
                eyeBlinkLeft: Math.random() * 0.2,
                eyeBlinkRight: Math.random() * 0.2,
                mouthSmile: Math.random() * 0.5,
                browInnerUp: Math.random() * 0.3
            },
            expressions: {
                happiness: Math.random() * 0.6,
                surprise: Math.random() * 0.3,
                neutral: 0.7
            },
            confidence: facialLandmarks.confidence || 0.9
        };
    }
    
    async generateFacialBVH(facialLandmarks) {
        await this.mockDelay(this.processingTime);
        
        const processedLandmarks = await this.processLandmarks(facialLandmarks);
        const frames = [];
        const frameCount = 30; // 1 second of facial animation
        
        for (let i = 0; i < frameCount; i++) {
            const time = i / 30;
            const blinkPhase = Math.sin(time * 3 * Math.PI) * 0.5 + 0.5; // Blinking pattern
            
            frames.push({
                frameNumber: i,
                time: time,
                bones: {
                    head: {
                        position: { x: 0, y: 1.6, z: 0 },
                        rotation: { 
                            x: Math.sin(time * 0.5) * 0.05, // Subtle head movement
                            y: 0, 
                            z: 0, 
                            w: 1 
                        }
                    },
                    leftEye: {
                        position: { x: -0.03, y: 1.65, z: 0.08 },
                        rotation: { x: 0, y: 0, z: 0, w: 1 }
                    },
                    rightEye: {
                        position: { x: 0.03, y: 1.65, z: 0.08 },
                        rotation: { x: 0, y: 0, z: 0, w: 1 }
                    },
                    jaw: {
                        position: { x: 0, y: 1.55, z: 0.05 },
                        rotation: { 
                            x: processedLandmarks.blendShapes.mouthSmile * 0.1, 
                            y: 0, 
                            z: 0, 
                            w: 1 
                        }
                    }
                },
                metadata: { 
                    source: 'mock_faceformer',
                    blendShapes: processedLandmarks.blendShapes,
                    expressions: processedLandmarks.expressions
                }
            });
        }
        
        return frames;
    }
    
    calculateEyeDistance(landmarks) {
        // Mock calculation
        return 0.065; // Average eye distance in meters
    }
    
    calculateMouthWidth(landmarks) {
        // Mock calculation
        return 0.05; // Average mouth width in meters
    }
    
    async mockDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Mock DeepMimic Converter Implementation
 */
class MockDeepMimicConverter {
    constructor() {
        this.initialized = false;
        this.processingTime = 200; // Physics simulation takes longer
    }
    
    async initialize() {
        await this.mockDelay(200);
        this.initialized = true;
        return true;
    }
    
    isInitialized() {
        return this.initialized;
    }
    
    async runPhysicsSimulation(constraints) {
        await this.mockDelay(this.processingTime);
        
        // Mock physics simulation result
        return {
            stable: true,
            convergenceTime: 150,
            energy: 0.05,
            violations: [],
            iterations: 50,
            constraints: constraints
        };
    }
    
    async generateMotion(targetPose, options = {}) {
        await this.mockDelay(this.processingTime);
        
        const frames = [];
        const duration = options.duration || 1.0;
        const frameCount = Math.ceil(duration * 30);
        
        for (let i = 0; i < frameCount; i++) {
            const time = i / 30;
            const progress = i / (frameCount - 1);
            
            // Generate physics-based motion with natural dynamics
            const hipsBob = Math.sin(time * 4 * Math.PI) * 0.02; // Walking bob
            const armSwing = Math.sin(time * 4 * Math.PI + Math.PI) * 0.1; // Arm swing
            
            frames.push({
                frameNumber: i,
                time: time,
                bones: {
                    hips: {
                        position: { 
                            x: targetPose.bones.hips?.position.x || 0, 
                            y: (targetPose.bones.hips?.position.y || 1) + hipsBob, 
                            z: targetPose.bones.hips?.position.z || 0 
                        },
                        rotation: { x: 0, y: 0, z: 0, w: 1 }
                    },
                    spine: {
                        position: { x: 0, y: 1.2 + hipsBob * 0.5, z: 0 },
                        rotation: { x: hipsBob * 0.1, y: 0, z: 0, w: 1 }
                    },
                    leftUpperArm: {
                        position: { x: -0.3, y: 1.4, z: 0 },
                        rotation: { x: armSwing, y: 0, z: 0, w: Math.cos(armSwing / 2) }
                    },
                    rightUpperArm: {
                        position: { x: 0.3, y: 1.4, z: 0 },
                        rotation: { x: -armSwing, y: 0, z: 0, w: Math.cos(armSwing / 2) }
                    },
                    leftUpperLeg: {
                        position: { x: -0.1, y: 0.9, z: 0 },
                        rotation: { x: armSwing * 0.5, y: 0, z: 0, w: Math.cos(armSwing * 0.25) }
                    },
                    rightUpperLeg: {
                        position: { x: 0.1, y: 0.9, z: 0 },
                        rotation: { x: -armSwing * 0.5, y: 0, z: 0, w: Math.cos(armSwing * 0.25) }
                    }
                },
                metadata: { 
                    source: 'mock_deepmimic',
                    physics: {
                        energy: 0.05 + Math.random() * 0.02,
                        stability: 0.95 + Math.random() * 0.05
                    }
                }
            });
        }
        
        return frames;
    }
    
    async refinePhysics(data) {
        await this.mockDelay(this.processingTime);
        
        // Mock physics refinement
        return {
            ...data,
            refined: true,
            physicsQuality: 0.92,
            refinementTime: this.processingTime
        };
    }
    
    async mockDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Mock RSMT Converter Implementation
 */
class MockRSMTConverter {
    constructor() {
        this.initialized = false;
        this.processingTime = 75;
        this.animationLibrary = new Map();
        this.poseVectors = new Map();
    }
    
    async initialize() {
        await this.mockDelay(100);
        this.initialized = true;
        return true;
    }
    
    isInitialized() {
        return this.initialized;
    }
    
    extractPoseVector(frame) {
        // Mock pose vector extraction
        const vector = [];
        
        // Add position and rotation data from bones
        for (const boneName in frame.bones) {
            const bone = frame.bones[boneName];
            if (bone.position) {
                vector.push(bone.position.x, bone.position.y, bone.position.z);
            }
            if (bone.rotation) {
                vector.push(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w);
            }
        }
        
        // Pad to 128 dimensions
        while (vector.length < 128) {
            vector.push(0);
        }
        
        return vector.slice(0, 128);
    }
    
    async loadAnimation(animationName, animationData) {
        await this.mockDelay(50);
        
        // Process animation data
        const processedData = {
            name: animationName,
            frames: animationData.frames || [],
            fps: animationData.fps || 30,
            duration: (animationData.frames?.length || 0) / (animationData.fps || 30)
        };
        
        this.animationLibrary.set(animationName, processedData);
        
        // Generate pose vectors
        const vectors = [];
        for (let i = 0; i < processedData.frames.length; i++) {
            const vector = this.extractPoseVector(processedData.frames[i]);
            vectors.push({
                frameIndex: i,
                time: i / processedData.fps,
                vector: vector,
                magnitude: Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0))
            });
        }
        
        this.poseVectors.set(animationName, vectors);
        
        return processedData;
    }
    
    findBestMatch(currentPose, targetAnimationName) {
        const targetVectors = this.poseVectors.get(targetAnimationName);
        if (!targetVectors || targetVectors.length === 0) {
            return null;
        }
        
        const currentVector = this.extractPoseVector(currentPose);
        let bestMatch = null;
        let bestSimilarity = -1;
        
        // Mock similarity calculation
        for (const targetFrame of targetVectors) {
            const similarity = this.mockCosineSimilarity(currentVector, targetFrame.vector);
            
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = {
                    frameIndex: targetFrame.frameIndex,
                    time: targetFrame.time,
                    similarity: similarity,
                    animationName: targetAnimationName
                };
            }
        }
        
        return bestMatch;
    }
    
    async generateTransition(fromPose, targetAnimationName, options = {}) {
        await this.mockDelay(this.processingTime);
        
        const match = this.findBestMatch(fromPose, targetAnimationName);
        if (!match) {
            throw new Error(`No suitable transition point found in ${targetAnimationName}`);
        }
        
        const targetAnimation = this.animationLibrary.get(targetAnimationName);
        const targetFrame = targetAnimation.frames[match.frameIndex];
        
        // Generate transition frames
        const duration = options.duration || 1.0;
        const frameCount = Math.ceil(duration * 30);
        const frames = [];
        
        for (let i = 0; i < frameCount; i++) {
            const progress = i / (frameCount - 1);
            const interpolatedFrame = this.interpolatePoses(fromPose, targetFrame, progress);
            
            interpolatedFrame.frameNumber = i;
            interpolatedFrame.time = i / 30;
            interpolatedFrame.metadata = {
                ...interpolatedFrame.metadata,
                transitionProgress: progress,
                generatedBy: 'mock_rsmt'
            };
            
            frames.push(interpolatedFrame);
        }
        
        return {
            frames: frames,
            duration: duration,
            fromPose: fromPose,
            toPose: targetFrame,
            frameCount: frameCount,
            method: 'mock_interpolation',
            quality: 0.85
        };
    }
    
    interpolatePoses(fromPose, toPose, progress) {
        const result = {
            frameNumber: 0,
            time: 0,
            bones: {},
            metadata: { source: 'mock_interpolation' }
        };
        
        // Interpolate each bone
        const allBones = new Set([
            ...Object.keys(fromPose.bones || {}),
            ...Object.keys(toPose.bones || {})
        ]);
        
        for (const boneName of allBones) {
            const fromBone = fromPose.bones[boneName] || this.getDefaultBone();
            const toBone = toPose.bones[boneName] || this.getDefaultBone();
            
            result.bones[boneName] = {
                position: {
                    x: fromBone.position.x + (toBone.position.x - fromBone.position.x) * progress,
                    y: fromBone.position.y + (toBone.position.y - fromBone.position.y) * progress,
                    z: fromBone.position.z + (toBone.position.z - fromBone.position.z) * progress
                },
                rotation: {
                    x: fromBone.rotation.x + (toBone.rotation.x - fromBone.rotation.x) * progress,
                    y: fromBone.rotation.y + (toBone.rotation.y - fromBone.rotation.y) * progress,
                    z: fromBone.rotation.z + (toBone.rotation.z - fromBone.rotation.z) * progress,
                    w: fromBone.rotation.w + (toBone.rotation.w - fromBone.rotation.w) * progress
                }
            };
        }
        
        return result;
    }
    
    getDefaultBone() {
        return {
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
        };
    }
    
    mockCosineSimilarity(vector1, vector2) {
        // Simplified cosine similarity calculation
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        for (let i = 0; i < Math.min(vector1.length, vector2.length); i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }
        
        if (norm1 === 0 || norm2 === 0) return 0;
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    async mockDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Mock Pathfinding Planner Implementation
 */
class MockPathfindingPlanner {
    constructor() {
        this.initialized = false;
        this.processingTime = 100;
        this.obstacles = new Map();
    }
    
    async initialize() {
        await this.mockDelay(100);
        this.initialized = true;
        return true;
    }
    
    isInitialized() {
        return this.initialized;
    }
    
    async planPath(destination) {
        await this.mockDelay(this.processingTime);
        
        // Generate mock path waypoints
        const waypoints = [
            { x: 0, y: 0, z: 0, time: 0 },
            { x: destination.x * 0.3, y: 0, z: destination.z * 0.3, time: 0.3 },
            { x: destination.x * 0.6, y: 0, z: destination.z * 0.6, time: 0.6 },
            { x: destination.x, y: 0, z: destination.z, time: 1.0 }
        ];
        
        return {
            waypoints: waypoints,
            distance: Math.sqrt(destination.x * destination.x + destination.z * destination.z),
            duration: 1.0,
            success: true
        };
    }
    
    addObstacle(id, obstacle) {
        this.obstacles.set(id, obstacle);
    }
    
    removeObstacle(id) {
        this.obstacles.delete(id);
    }
    
    async planPathToDestination(destination, options = {}) {
        await this.mockDelay(this.processingTime);
        
        const path = await this.planPath(destination);
        
        // Generate keyframes
        const keyframes = [];
        for (let i = 0; i < path.waypoints.length; i++) {
            const waypoint = path.waypoints[i];
            keyframes.push({
                id: `keyframe_${i}`,
                time: waypoint.time,
                position: { x: waypoint.x, y: waypoint.y, z: waypoint.z },
                velocity: { x: 0, y: 0, z: 0 },
                orientation: { x: 0, y: 0, z: 0, w: 1 },
                movementType: 'walk',
                metadata: { pathIndex: i }
            });
        }
        
        // Generate BVH keyframes
        const bvhKeyframes = [];
        for (const keyframe of keyframes) {
            bvhKeyframes.push({
                frameNumber: bvhKeyframes.length,
                time: keyframe.time,
                bones: {
                    hips: {
                        position: keyframe.position,
                        rotation: keyframe.orientation
                    }
                },
                metadata: { source: 'mock_pathfinding' }
            });
        }
        
        return {
            path: path.waypoints,
            keyframes: keyframes,
            bvhKeyframes: bvhKeyframes,
            totalDistance: path.distance,
            estimatedDuration: path.duration,
            planningTime: this.processingTime
        };
    }
    
    async mockDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MockBVHTimeline,
        MockAudio2GestureConverter,
        MockFaceFormerConverter,
        MockDeepMimicConverter,
        MockRSMTConverter,
        MockPathfindingPlanner
    };
} else {
    // Browser global
    window.MockBVHTimeline = MockBVHTimeline;
    window.MockAudio2GestureConverter = MockAudio2GestureConverter;
    window.MockFaceFormerConverter = MockFaceFormerConverter;
    window.MockDeepMimicConverter = MockDeepMimicConverter;
    window.MockRSMTConverter = MockRSMTConverter;
    window.MockPathfindingPlanner = MockPathfindingPlanner;
}
