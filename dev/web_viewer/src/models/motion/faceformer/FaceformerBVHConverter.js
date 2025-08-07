/**
 * FaceFormer to BVH Converter
 * 
 * Integrates FaceFormer neural network (audio-to-facial animation) with the BVH Timeline system.
 * Converts FaceFormer vertex outputs to BVH bone rotations for facial animation.
 */

class FaceformerBVHConverter {
    constructor(options = {}) {
        this.faceformerPath = options.faceformerPath || '../faceformer/faceformer_simple_step.onnx';
        this.session = null;
        this.initialized = false;
        
        // Configuration
        this.maxSeqLen = options.maxSeqLen || 20;
        this.featureDim = options.featureDim || 64;
        this.vertexCount = options.vertexCount || 5023; // Standard FaceFormer vertex count
        this.framerate = options.framerate || 30;
        
        // BVH mapping configuration
        this.bvhBoneMapping = this.createBVHBoneMapping();
        this.templateMesh = null;
        this.blendshapeWeights = null;
        
        // Audio processing
        this.audioContext = null;
        this.sampleRate = 16000; // FaceFormer expects 16kHz
        
        // Performance monitoring
        this.stats = {
            audioProcessingTime: 0,
            inferenceTime: 0,
            bvhConversionTime: 0,
            totalFramesGenerated: 0
        };
        
        console.log('[FaceFormer BVH] Converter initialized');
    }
    
    /**
     * Initialize the FaceFormer model and audio processing
     */
    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('[FaceFormer BVH] Loading model...');
            
            // Initialize ONNX Runtime session
            if (typeof ort !== 'undefined') {
                this.session = await ort.InferenceSession.create(this.faceformerPath, {
                    executionProviders: ['webgl', 'wasm'] // Prefer WebGL for better performance
                });
            } else {
                throw new Error('ONNX Runtime not available. Please include onnxruntime-web.');
            }
            
            // Initialize Web Audio API
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            // Load template mesh and blendshape data
            await this.loadTemplateData();
            
            this.initialized = true;
            console.log('[FaceFormer BVH] Initialization complete');
            
        } catch (error) {
            console.error('[FaceFormer BVH] Initialization failed:', error);
            throw error;
        }
    }
    
    /**
     * Generate BVH facial animation from audio buffer
     */
    async generateBVHFromAudio(audioBuffer, options = {}) {
        if (!this.initialized) {
            throw new Error('FaceFormer not initialized. Call initialize() first.');
        }
        
        const startTime = performance.now();
        
        try {
            // Process audio to features
            const audioFeatures = await this.processAudioToFeatures(audioBuffer);
            
            // Generate facial mesh sequence
            const meshSequence = await this.generateFacialMeshSequence(audioFeatures, options);
            
            // Convert mesh sequence to BVH bone data
            const bvhFrames = this.convertMeshSequenceToBVH(meshSequence, options);
            
            // Update statistics
            this.stats.totalFramesGenerated += bvhFrames.length;
            
            console.log(`[FaceFormer BVH] Generated ${bvhFrames.length} BVH frames from ${audioBuffer.duration.toFixed(2)}s audio`);
            
            return {
                frames: bvhFrames,
                duration: audioBuffer.duration,
                framerate: this.framerate,
                metadata: {
                    audioLength: audioBuffer.duration,
                    frameCount: bvhFrames.length,
                    processingTime: performance.now() - startTime,
                    stats: { ...this.stats }
                }
            };
            
        } catch (error) {
            console.error('[FaceFormer BVH] Generation failed:', error);
            throw error;
        }
    }
    
    /**
     * Process audio buffer to FaceFormer-compatible features
     */
    async processAudioToFeatures(audioBuffer) {
        const startTime = performance.now();
        
        try {
            // Resample audio to 16kHz if needed
            let audioData;
            if (audioBuffer.sampleRate !== this.sampleRate) {
                audioData = await this.resampleAudio(audioBuffer, this.sampleRate);
            } else {
                audioData = audioBuffer.getChannelData(0);
            }
            
            // Extract audio features (simplified - in production you'd use Wav2Vec2)
            const features = this.extractAudioFeatures(audioData);
            
            this.stats.audioProcessingTime = performance.now() - startTime;
            return features;
            
        } catch (error) {
            console.error('[FaceFormer BVH] Audio processing failed:', error);
            throw error;
        }
    }
    
    /**
     * Generate facial mesh sequence using FaceFormer
     */
    async generateFacialMeshSequence(audioFeatures, options = {}) {
        const startTime = performance.now();
        
        try {
            const maxFrames = options.maxFrames || Math.ceil(audioFeatures.length / (this.sampleRate / this.framerate));
            const batchSize = 1;
            
            // Initialize sequence buffer
            let sequenceBuffer = new Array(batchSize * this.maxSeqLen * this.featureDim).fill(0);
            let currentLength = 1; // Start with template
            
            // Prepare input tensors
            const audioTensor = this.createAudioTensor(audioFeatures);
            const templateTensor = this.createTemplateTensor();
            const oneHotTensor = this.createOneHotTensor(options.subjectId || 0);
            
            const generatedMeshes = [];
            
            // Autoregressive generation
            for (let step = 0; step < maxFrames && currentLength < this.maxSeqLen; step++) {
                const sequenceTensor = new ort.Tensor('float32', 
                    new Float32Array(sequenceBuffer), 
                    [batchSize, this.maxSeqLen, this.featureDim]);
                
                const lengthTensor = new ort.Tensor('int64', 
                    new BigInt64Array([BigInt(currentLength)]), 
                    [batchSize, 1]);
                
                // Run FaceFormer inference
                const feeds = {
                    audio_features: audioTensor,
                    vertice_sequence: sequenceTensor,
                    current_length: lengthTensor,
                    one_hot: oneHotTensor,
                    template: templateTensor
                };
                
                const results = await this.session.run(feeds);
                
                // Extract results
                const newVertices = Array.from(results.new_vertice_out.data);
                const updatedSequence = Array.from(results.updated_sequence.data);
                const newLength = Number(results.new_length.data[0]);
                
                generatedMeshes.push(newVertices);
                
                // Update for next iteration
                sequenceBuffer = updatedSequence;
                currentLength = newLength;
            }
            
            this.stats.inferenceTime = performance.now() - startTime;
            return generatedMeshes;
            
        } catch (error) {
            console.error('[FaceFormer BVH] Mesh generation failed:', error);
            throw error;
        }
    }
    
    /**
     * Convert mesh sequence to BVH bone data
     */
    convertMeshSequenceToBVH(meshSequence, options = {}) {
        const startTime = performance.now();
        
        try {
            const bvhFrames = [];
            
            for (let frameIndex = 0; frameIndex < meshSequence.length; frameIndex++) {
                const vertices = meshSequence[frameIndex];
                const bvhFrame = this.convertSingleMeshToBVH(vertices, frameIndex, options);
                bvhFrames.push(bvhFrame);
            }
            
            this.stats.bvhConversionTime = performance.now() - startTime;
            return bvhFrames;
            
        } catch (error) {
            console.error('[FaceFormer BVH] BVH conversion failed:', error);
            throw error;
        }
    }
    
    /**
     * Convert single mesh frame to BVH bone data
     */
    convertSingleMeshToBVH(vertices, frameIndex, options = {}) {
        // This is where we map facial mesh vertices to BVH bone rotations
        const bvhFrame = {
            time: frameIndex / this.framerate,
            motionData: [],
            metadata: {
                type: 'faceformer_facial',
                frameIndex: frameIndex,
                vertexCount: vertices.length / 3 // Assuming 3D vertices
            }
        };
        
        // Extract facial landmark positions from mesh vertices
        const landmarks = this.extractFacialLandmarks(vertices);
        
        // Convert landmarks to bone rotations
        const boneRotations = this.landmarksToBoneRotations(landmarks);
        
        // Map to BVH motion data format
        bvhFrame.motionData = this.mapToBVHMotionData(boneRotations);
        
        return bvhFrame;
    }
    
    /**
     * Extract key facial landmarks from vertex data
     */
    extractFacialLandmarks(vertices) {
        // Map specific vertices to facial landmarks
        const landmarks = {};
        
        // Eye region landmarks
        landmarks.leftEye = this.getVertexPosition(vertices, this.bvhBoneMapping.leftEye.vertices);
        landmarks.rightEye = this.getVertexPosition(vertices, this.bvhBoneMapping.rightEye.vertices);
        
        // Eyebrow landmarks
        landmarks.leftEyebrow = this.getVertexPosition(vertices, this.bvhBoneMapping.leftEyebrow.vertices);
        landmarks.rightEyebrow = this.getVertexPosition(vertices, this.bvhBoneMapping.rightEyebrow.vertices);
        
        // Mouth landmarks
        landmarks.mouthCornerLeft = this.getVertexPosition(vertices, this.bvhBoneMapping.mouthCornerLeft.vertices);
        landmarks.mouthCornerRight = this.getVertexPosition(vertices, this.bvhBoneMapping.mouthCornerRight.vertices);
        landmarks.mouthTop = this.getVertexPosition(vertices, this.bvhBoneMapping.mouthTop.vertices);
        landmarks.mouthBottom = this.getVertexPosition(vertices, this.bvhBoneMapping.mouthBottom.vertices);
        
        // Jaw landmarks
        landmarks.jaw = this.getVertexPosition(vertices, this.bvhBoneMapping.jaw.vertices);
        
        // Cheek landmarks
        landmarks.leftCheek = this.getVertexPosition(vertices, this.bvhBoneMapping.leftCheek.vertices);
        landmarks.rightCheek = this.getVertexPosition(vertices, this.bvhBoneMapping.rightCheek.vertices);
        
        return landmarks;
    }
    
    /**
     * Convert landmarks to bone rotations
     */
    landmarksToBoneRotations(landmarks) {
        const rotations = {};
        
        // Calculate head rotation from overall facial movement
        rotations.head = this.calculateHeadRotation(landmarks);
        
        // Calculate jaw rotation from mouth landmarks
        rotations.jaw = this.calculateJawRotation(landmarks.mouthTop, landmarks.mouthBottom, landmarks.jaw);
        
        // Calculate eye rotations
        rotations.leftEye = this.calculateEyeRotation(landmarks.leftEye, 'left');
        rotations.rightEye = this.calculateEyeRotation(landmarks.rightEye, 'right');
        
        // Calculate eyebrow rotations (mapped to deformation bones)
        rotations.leftEyebrow = this.calculateEyebrowRotation(landmarks.leftEyebrow, 'left');
        rotations.rightEyebrow = this.calculateEyebrowRotation(landmarks.rightEyebrow, 'right');
        
        // Calculate mouth deformation (can be mapped to multiple bones)
        const mouthRotations = this.calculateMouthRotations(landmarks);
        Object.assign(rotations, mouthRotations);
        
        return rotations;
    }
    
    /**
     * Map bone rotations to BVH motion data format
     */
    mapToBVHMotionData(boneRotations) {
        const motionData = [];
        
        // Map each bone rotation to BVH format [Xpos, Ypos, Zpos, Zrot, Xrot, Yrot]
        for (const [boneName, rotation] of Object.entries(boneRotations)) {
            const bvhBone = this.bvhBoneMapping[boneName];
            if (bvhBone) {
                motionData[bvhBone.index] = [
                    rotation.position?.x || 0,
                    rotation.position?.y || 0,
                    rotation.position?.z || 0,
                    this.radiansToDegrees(rotation.euler?.z || 0),
                    this.radiansToDegrees(rotation.euler?.x || 0),
                    this.radiansToDegrees(rotation.euler?.y || 0)
                ];
            }
        }
        
        return motionData;
    }
    
    /**
     * Create BVH bone mapping for facial bones
     */
    createBVHBoneMapping() {
        return {
            head: {
                index: 0,
                vertices: [1, 2, 3, 4, 5] // Head region vertices
            },
            jaw: {
                index: 1,
                vertices: [100, 101, 102, 103] // Jaw vertices
            },
            leftEye: {
                index: 2,
                vertices: [200, 201, 202, 203] // Left eye vertices
            },
            rightEye: {
                index: 3,
                vertices: [300, 301, 302, 303] // Right eye vertices
            },
            leftEyebrow: {
                index: 4,
                vertices: [400, 401, 402] // Left eyebrow vertices
            },
            rightEyebrow: {
                index: 5,
                vertices: [500, 501, 502] // Right eyebrow vertices
            },
            mouthCornerLeft: {
                index: 6,
                vertices: [600, 601] // Left mouth corner
            },
            mouthCornerRight: {
                index: 7,
                vertices: [700, 701] // Right mouth corner
            },
            mouthTop: {
                index: 8,
                vertices: [800, 801, 802] // Upper lip
            },
            mouthBottom: {
                index: 9,
                vertices: [900, 901, 902] // Lower lip
            },
            leftCheek: {
                index: 10,
                vertices: [1000, 1001] // Left cheek
            },
            rightCheek: {
                index: 11,
                vertices: [1100, 1101] // Right cheek
            }
        };
    }
    
    /**
     * Utility functions for audio processing
     */
    async resampleAudio(audioBuffer, targetSampleRate) {
        const offlineContext = new OfflineAudioContext(
            1, 
            audioBuffer.duration * targetSampleRate, 
            targetSampleRate
        );
        
        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        
        const resampledBuffer = await offlineContext.startRendering();
        return resampledBuffer.getChannelData(0);
    }
    
    extractAudioFeatures(audioData) {
        // Simplified feature extraction - in production use Wav2Vec2
        const windowSize = 512;
        const hopSize = 256;
        const features = [];
        
        for (let i = 0; i < audioData.length - windowSize; i += hopSize) {
            const window = audioData.slice(i, i + windowSize);
            const feature = this.computeMFCC(window); // Simplified MFCC
            features.push(feature);
        }
        
        return features;
    }
    
    computeMFCC(window) {
        // Simplified MFCC computation - returns 768-dim features like Wav2Vec2
        const feature = new Array(768);
        for (let i = 0; i < feature.length; i++) {
            feature[i] = Math.random() * 0.1 - 0.05; // Placeholder
        }
        return feature;
    }
    
    /**
     * Tensor creation utilities
     */
    createAudioTensor(audioFeatures) {
        const batchSize = 1;
        const seqLen = audioFeatures.length;
        const featureDim = audioFeatures[0].length;
        
        return new ort.Tensor('float32', 
            new Float32Array(audioFeatures.flat()), 
            [batchSize, seqLen, featureDim]);
    }
    
    createTemplateTensor() {
        // Create template mesh tensor (neutral face)
        const batchSize = 1;
        const templateSeqLen = 1;
        const vertexDim = this.vertexCount * 3; // 3D vertices
        
        const templateData = new Array(batchSize * templateSeqLen * vertexDim).fill(0);
        
        return new ort.Tensor('float32', 
            new Float32Array(templateData), 
            [batchSize, templateSeqLen, vertexDim]);
    }
    
    createOneHotTensor(subjectId = 0) {
        const batchSize = 1;
        const numSubjects = 3; // Adjust based on your model
        
        const oneHotData = new Array(batchSize * numSubjects).fill(0);
        oneHotData[subjectId] = 1;
        
        return new ort.Tensor('float32', 
            new Float32Array(oneHotData), 
            [batchSize, numSubjects]);
    }
    
    /**
     * Geometric calculation utilities
     */
    getVertexPosition(vertices, vertexIndices) {
        const positions = [];
        for (const index of vertexIndices) {
            if (index * 3 + 2 < vertices.length) {
                positions.push({
                    x: vertices[index * 3],
                    y: vertices[index * 3 + 1],
                    z: vertices[index * 3 + 2]
                });
            }
        }
        
        // Return average position
        const avg = positions.reduce((acc, pos) => ({
            x: acc.x + pos.x,
            y: acc.y + pos.y,
            z: acc.z + pos.z
        }), { x: 0, y: 0, z: 0 });
        
        return {
            x: avg.x / positions.length,
            y: avg.y / positions.length,
            z: avg.z / positions.length
        };
    }
    
    calculateHeadRotation(landmarks) {
        // Calculate head rotation from facial landmarks
        return {
            position: { x: 0, y: 0, z: 0 },
            euler: { x: 0, y: 0, z: 0 }
        };
    }
    
    calculateJawRotation(mouthTop, mouthBottom, jaw) {
        // Calculate jaw opening from mouth positions
        const jawOpen = this.distance3D(mouthTop, mouthBottom) / 10; // Normalize
        return {
            position: { x: 0, y: 0, z: 0 },
            euler: { x: jawOpen, y: 0, z: 0 }
        };
    }
    
    calculateEyeRotation(eyePosition, side) {
        // Calculate eye rotation (simplified)
        return {
            position: { x: 0, y: 0, z: 0 },
            euler: { x: 0, y: 0, z: 0 }
        };
    }
    
    calculateEyebrowRotation(eyebrowPosition, side) {
        // Calculate eyebrow movement
        return {
            position: { x: 0, y: 0, z: 0 },
            euler: { x: 0, y: 0, z: 0 }
        };
    }
    
    calculateMouthRotations(landmarks) {
        // Calculate mouth deformation rotations
        return {
            mouthCornerLeft: {
                position: { x: 0, y: 0, z: 0 },
                euler: { x: 0, y: 0, z: 0 }
            },
            mouthCornerRight: {
                position: { x: 0, y: 0, z: 0 },
                euler: { x: 0, y: 0, z: 0 }
            }
        };
    }
    
    distance3D(p1, p2) {
        return Math.sqrt(
            Math.pow(p2.x - p1.x, 2) + 
            Math.pow(p2.y - p1.y, 2) + 
            Math.pow(p2.z - p1.z, 2)
        );
    }
    
    radiansToDegrees(radians) {
        return radians * (180 / Math.PI);
    }
    
    degreesToRadians(degrees) {
        return degrees * (Math.PI / 180);
    }
    
    /**
     * Load template data (placeholder)
     */
    async loadTemplateData() {
        // Load neutral face template and blendshape data
        // This would typically load from your face model assets
        console.log('[FaceFormer BVH] Template data loaded');
    }
    
    /**
     * Integration with BVH Timeline (with buffering support)
     */
    createTimelineClip(audioBuffer, startTime, options = {}) {
        // Pre-generate frames for better performance with buffering
        let generatedFrames = null;
        let isGenerating = false;
        
        return {
            type: 'faceformer_audio',
            startTime: startTime,
            duration: audioBuffer.duration,
            weight: options.weight || 1.0,
            blendMode: options.blendMode || 'additive',
            generator: async (time, frameIndex) => {
                // Lazy generation - generate all frames on first request
                if (!generatedFrames && !isGenerating) {
                    isGenerating = true;
                    try {
                        console.log('[FaceFormer] Generating all frames for timeline clip...');
                        const animationData = await this.generateBVHFromAudio(audioBuffer, {
                            maxFrames: Math.ceil(audioBuffer.duration * this.framerate),
                            ...options
                        });
                        generatedFrames = animationData.frames;
                        console.log(`[FaceFormer] Generated ${generatedFrames.length} frames for buffering`);
                    } catch (error) {
                        console.error('[FaceFormer] Frame generation failed:', error);
                        generatedFrames = [];
                    } finally {
                        isGenerating = false;
                    }
                }
                
                // Wait for generation to complete if in progress
                while (isGenerating) {
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
                
                // Return frame if available
                if (generatedFrames && frameIndex < generatedFrames.length) {
                    return {
                        ...generatedFrames[frameIndex],
                        time: time // Update timestamp
                    };
                }
                
                return {
                    time: time,
                    motionData: [],
                    metadata: { type: 'faceformer_empty' }
                };
            },
            
            // Buffer-friendly methods
            prebufferFrames: async () => {
                if (!generatedFrames && !isGenerating) {
                    // Trigger generation
                    await this.generator(0, 0);
                }
                return generatedFrames || [];
            },
            
            getFrameCount: () => {
                return generatedFrames ? generatedFrames.length : Math.ceil(audioBuffer.duration * this.framerate);
            },
            
            isReady: () => {
                return generatedFrames !== null;
            },
            
            metadata: {
                audioBuffer: audioBuffer,
                converter: 'faceformer',
                frameCount: Math.ceil(audioBuffer.duration * this.framerate),
                canPrebuffer: true,
                ...options
            }
        };
    }
    
    /**
     * Create optimized timeline clip for real-time use
     */
    createRealTimeClip(audioBuffer, startTime, options = {}) {
        const audioData = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;
        
        return {
            type: 'faceformer_realtime',
            startTime: startTime,
            duration: audioBuffer.duration,
            weight: options.weight || 1.0,
            blendMode: options.blendMode || 'additive',
            generator: async (time, frameIndex) => {
                try {
                    // Extract audio features for this frame
                    const audioStart = Math.floor((time - startTime) * sampleRate);
                    const audioEnd = audioStart + 1024; // 1024 sample window
                    
                    if (audioStart >= 0 && audioEnd <= audioData.length) {
                        const audioSegment = audioData.slice(audioStart, audioEnd);
                        
                        // Generate frame directly (faster but may have quality trade-offs)
                        const frame = await this.generateSingleFrame(audioSegment, frameIndex, options);
                        return {
                            ...frame,
                            time: time
                        };
                    }
                } catch (error) {
                    console.warn(`[FaceFormer] Real-time frame generation failed at ${time}:`, error);
                }
                
                return {
                    time: time,
                    motionData: [],
                    metadata: { type: 'faceformer_realtime_empty' }
                };
            },
            metadata: {
                audioBuffer: audioBuffer,
                converter: 'faceformer_realtime',
                realTime: true,
                ...options
            }
        };
    }
    
    /**
     * Generate single frame for real-time processing
     */
    async generateSingleFrame(audioSegment, frameIndex, options = {}) {
        // Simplified single-frame generation for real-time use
        // This would integrate with your existing FaceFormer model
        
        try {
            // Extract simple audio features
            const features = this.extractSimpleFeatures(audioSegment);
            
            // Generate facial landmarks (placeholder)
            const landmarks = this.generateLandmarksFromFeatures(features, frameIndex);
            
            // Convert to BVH
            const boneRotations = this.landmarksToBoneRotations(landmarks);
            const motionData = this.mapToBVHMotionData(boneRotations);
            
            return {
                time: frameIndex / this.framerate,
                motionData: motionData,
                metadata: {
                    type: 'faceformer_realtime',
                    frameIndex: frameIndex,
                    audioFeatures: features.length
                }
            };
            
        } catch (error) {
            console.warn('[FaceFormer] Single frame generation failed:', error);
            return {
                time: frameIndex / this.framerate,
                motionData: [],
                metadata: { type: 'faceformer_error' }
            };
        }
    }
    
    /**
     * Extract simple audio features for real-time use
     */
    extractSimpleFeatures(audioSegment) {
        // Simplified feature extraction for real-time performance
        const features = [];
        const windowSize = 64;
        
        for (let i = 0; i < audioSegment.length - windowSize; i += windowSize) {
            const window = audioSegment.slice(i, i + windowSize);
            
            // RMS energy
            const rms = Math.sqrt(window.reduce((sum, x) => sum + x * x, 0) / window.length);
            
            // Zero crossing rate
            let zcr = 0;
            for (let j = 1; j < window.length; j++) {
                if ((window[j] >= 0) !== (window[j-1] >= 0)) zcr++;
            }
            zcr /= window.length;
            
            features.push(rms, zcr);
        }
        
        return features;
    }
    
    /**
     * Generate facial landmarks from audio features (placeholder)
     */
    generateLandmarksFromFeatures(features, frameIndex) {
        // This is a placeholder - in production you'd use the actual FaceFormer model
        const landmarks = {};
        const time = frameIndex / this.framerate;
        
        // Simple sine wave-based animation for demo
        const intensity = features.length > 0 ? features.reduce((a, b) => a + b, 0) / features.length : 0;
        
        landmarks.jaw = {
            x: 0,
            y: intensity * Math.sin(time * 10) * 0.1,
            z: 0
        };
        
        landmarks.mouthCornerLeft = {
            x: -0.05 + intensity * Math.sin(time * 15) * 0.02,
            y: intensity * Math.cos(time * 12) * 0.01,
            z: 0
        };
        
        landmarks.mouthCornerRight = {
            x: 0.05 + intensity * Math.sin(time * 15) * 0.02,
            y: intensity * Math.cos(time * 12) * 0.01,
            z: 0
        };
        
        return landmarks;
    }
    
    /**
     * Get performance statistics
     */
    getStats() {
        return {
            ...this.stats,
            initialized: this.initialized,
            modelPath: this.faceformerPath
        };
    }
    
    /**
     * Cleanup resources
     */
    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.initialized = false;
        console.log('[FaceFormer BVH] Resources disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceformerBVHConverter;
} else {
    window.FaceformerBVHConverter = FaceformerBVHConverter;
}
