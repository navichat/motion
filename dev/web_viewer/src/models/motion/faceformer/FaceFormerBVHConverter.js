/**
 * FaceFormer to BVH Converter
 * 
 * This module converts FaceFormer neural network outputs (facial animation data)
 * to BVH format for integration with the BVH Timeline compositor system.
 * 
 * FaceFormer typically outputs facial landmarks, blend shapes, or vertex positions
 * which we convert to facial bone rotations compatible with BVH format.
 */

class FaceFormerBVHConverter {
    constructor(options = {}) {
        this.options = {
            // Conversion settings
            scaleFactor: options.scaleFactor || 1.0,
            smoothing: options.smoothing !== false, // Default enabled
            smoothingFactor: options.smoothingFactor || 0.8,
            
            // Facial bone mapping
            enableEyeMovements: options.enableEyeMovements !== false,
            enableJawMovement: options.enableJawMovement !== false,
            enableEyebrowMovement: options.enableEyebrowMovement !== false,
            enableCheekMovement: options.enableCheekMovement !== false,
            
            // Frame rate and timing
            targetFrameRate: options.targetFrameRate || 30,
            maxLookAhead: options.maxLookAhead || 10, // frames to look ahead for smoothing
            
            // Debug options
            verbose: options.verbose || false,
            logConversions: options.logConversions || false
        };
        
        // Internal state
        this.frameHistory = [];
        this.maxHistorySize = 30; // Keep last 30 frames for smoothing
        this.isInitialized = false;
        
        // Facial bone structure for BVH
        this.facialBones = this.initializeFacialBoneStructure();
        
        // Statistics
        this.stats = {
            framesConverted: 0,
            averageConversionTime: 0,
            lastConversionTime: 0,
            errorCount: 0
        };
        
        console.log('[FaceFormer BVH Converter] Initialized with options:', this.options);
    }
    
    /**
     * Initialize the facial bone structure for BVH conversion
     */
    initializeFacialBoneStructure() {
        return {
            // Head and neck bones
            head: { index: 0, parent: null, channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            neck: { index: 1, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Eye bones
            leftEye: { index: 2, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightEye: { index: 3, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Eyebrow bones
            leftEyebrow: { index: 4, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightEyebrow: { index: 5, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Jaw and mouth bones
            jaw: { index: 6, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            leftMouth: { index: 7, parent: 'jaw', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightMouth: { index: 8, parent: 'jaw', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Cheek bones
            leftCheek: { index: 9, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            rightCheek: { index: 10, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] },
            
            // Nose bone
            nose: { index: 11, parent: 'head', channels: ['Xrotation', 'Yrotation', 'Zrotation'] }
        };
    }
    
    /**
     * Convert FaceFormer output to BVH frame format
     * @param {Object} faceFormerOutput - Raw output from FaceFormer neural network
     * @param {number} timestamp - Frame timestamp
     * @param {Object} options - Conversion options for this frame
     * @returns {Object} BVH frame compatible with timeline system
     */
    async convertToBVH(faceFormerOutput, timestamp, options = {}) {
        const startTime = performance.now();
        
        try {
            // Validate input
            if (!faceFormerOutput) {
                throw new Error('FaceFormer output is null or undefined');
            }
            
            // Detect FaceFormer output format and convert accordingly
            const normalizedData = this.normalizeFaceFormerOutput(faceFormerOutput);
            
            // Convert to facial bone rotations
            const facialRotations = this.calculateFacialRotations(normalizedData, options);
            
            // Apply smoothing if enabled
            const smoothedRotations = this.options.smoothing ? 
                this.applySmoothingToRotations(facialRotations, timestamp) : 
                facialRotations;
            
            // Generate BVH frame
            const bvhFrame = this.generateBVHFrame(smoothedRotations, timestamp, normalizedData);
            
            // Store frame in history for smoothing
            this.updateFrameHistory(bvhFrame, normalizedData, timestamp);
            
            // Update statistics
            this.updateConversionStats(startTime);
            
            if (this.options.logConversions) {
                console.log('[FaceFormer BVH] Converted frame:', {
                    timestamp,
                    inputType: this.detectFaceFormerFormat(faceFormerOutput),
                    outputBones: Object.keys(smoothedRotations).length,
                    conversionTime: `${(performance.now() - startTime).toFixed(2)}ms`
                });
            }
            
            return bvhFrame;
            
        } catch (error) {
            this.stats.errorCount++;
            console.error('[FaceFormer BVH] Conversion error:', error);
            
            // Return neutral frame on error
            return this.generateNeutralBVHFrame(timestamp);
        }
    }
    
    /**
     * Normalize different FaceFormer output formats to a common structure
     */
    normalizeFaceFormerOutput(output) {
        const format = this.detectFaceFormerFormat(output);
        
        switch (format) {
            case 'landmarks':
                return this.normalizeLandmarks(output);
            case 'blendshapes':
                return this.normalizeBlendShapes(output);
            case 'vertices':
                return this.normalizeVertices(output);
            case 'coefficients':
                return this.normalizeCoefficients(output);
            default:
                console.warn('[FaceFormer BVH] Unknown format, using raw data:', format);
                return this.normalizeRawData(output);
        }
    }
    
    /**
     * Detect the format of FaceFormer output
     */
    detectFaceFormerFormat(output) {
        if (output.landmarks || (Array.isArray(output) && output.length === 68)) {
            return 'landmarks'; // 68-point facial landmarks
        } else if (output.blendshapes || output.blend_shapes) {
            return 'blendshapes'; // ARKit-style blend shapes
        } else if (output.vertices && Array.isArray(output.vertices)) {
            return 'vertices'; // 3D mesh vertices
        } else if (output.coefficients || output.expression_coefficients) {
            return 'coefficients'; // Expression coefficients
        } else if (output.face_mesh || output.mesh) {
            return 'vertices'; // Mesh data
        } else {
            return 'unknown';
        }
    }
    
    /**
     * Normalize facial landmark data (68-point format)
     */
    normalizeLandmarks(output) {
        const landmarks = output.landmarks || output;
        
        if (!Array.isArray(landmarks) || landmarks.length < 68) {
            throw new Error('Invalid landmarks data: expected 68 points');
        }
        
        // Extract key facial features from landmarks
        return {
            format: 'landmarks',
            eyeL: this.extractEyeLandmarks(landmarks, 'left'),
            eyeR: this.extractEyeLandmarks(landmarks, 'right'),
            eyebrowL: this.extractEyebrowLandmarks(landmarks, 'left'),
            eyebrowR: this.extractEyebrowLandmarks(landmarks, 'right'),
            mouth: this.extractMouthLandmarks(landmarks),
            jaw: this.extractJawLandmarks(landmarks),
            nose: this.extractNoseLandmarks(landmarks),
            cheekL: this.extractCheekLandmarks(landmarks, 'left'),
            cheekR: this.extractCheekLandmarks(landmarks, 'right'),
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize blend shape data (ARKit-style)
     */
    normalizeBlendShapes(output) {
        const blendshapes = output.blendshapes || output.blend_shapes || output;
        
        return {
            format: 'blendshapes',
            eyeBlinkL: blendshapes.eyeBlinkLeft || 0,
            eyeBlinkR: blendshapes.eyeBlinkRight || 0,
            eyeLookInL: blendshapes.eyeLookInLeft || 0,
            eyeLookInR: blendshapes.eyeLookInRight || 0,
            eyeLookOutL: blendshapes.eyeLookOutLeft || 0,
            eyeLookOutR: blendshapes.eyeLookOutRight || 0,
            eyeLookUpL: blendshapes.eyeLookUpLeft || 0,
            eyeLookUpR: blendshapes.eyeLookUpRight || 0,
            eyeLookDownL: blendshapes.eyeLookDownLeft || 0,
            eyeLookDownR: blendshapes.eyeLookDownRight || 0,
            jawOpen: blendshapes.jawOpen || 0,
            jawLeft: blendshapes.jawLeft || 0,
            jawRight: blendshapes.jawRight || 0,
            mouthSmileL: blendshapes.mouthSmileLeft || 0,
            mouthSmileR: blendshapes.mouthSmileRight || 0,
            mouthFrownL: blendshapes.mouthFrownLeft || 0,
            mouthFrownR: blendshapes.mouthFrownRight || 0,
            browInnerUp: blendshapes.browInnerUp || 0,
            browOuterUpL: blendshapes.browOuterUpLeft || 0,
            browOuterUpR: blendshapes.browOuterUpRight || 0,
            cheekPuffL: blendshapes.cheekPuff || blendshapes.cheekPuffLeft || 0,
            cheekPuffR: blendshapes.cheekPuff || blendshapes.cheekPuffRight || 0,
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize vertex mesh data
     */
    normalizeVertices(output) {
        const vertices = output.vertices || output.face_mesh || output.mesh || output;
        
        // Extract facial features from vertex positions
        // This requires knowledge of the specific mesh topology
        return {
            format: 'vertices',
            vertices: vertices,
            // Extract key vertex indices for facial features
            eyeVerticesL: this.extractEyeVertices(vertices, 'left'),
            eyeVerticesR: this.extractEyeVertices(vertices, 'right'),
            mouthVertices: this.extractMouthVertices(vertices),
            jawVertices: this.extractJawVertices(vertices),
            confidence: output.confidence || 1.0,
            timestamp: output.timestamp || Date.now()
        };
    }
    
    /**
     * Normalize expression coefficients
     */
    normalizeCoefficients(output) {
        const coefficients = output.coefficients || output.expression_coefficients || output;
        
        return {
            format: 'coefficients',
            coefficients: Array.isArray(coefficients) ? coefficients : Object.values(coefficients),
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
     * Calculate facial bone rotations from normalized data
     */
    calculateFacialRotations(normalizedData, options = {}) {
        const rotations = {};
        
        switch (normalizedData.format) {
            case 'landmarks':
                Object.assign(rotations, this.landmarksToRotations(normalizedData));
                break;
            case 'blendshapes':
                Object.assign(rotations, this.blendshapesToRotations(normalizedData));
                break;
            case 'vertices':
                Object.assign(rotations, this.verticesToRotations(normalizedData));
                break;
            case 'coefficients':
                Object.assign(rotations, this.coefficientsToRotations(normalizedData));
                break;
            default:
                Object.assign(rotations, this.rawDataToRotations(normalizedData));
                break;
        }
        
        // Apply scaling and constraints
        return this.applyRotationConstraints(rotations, options);
    }
    
    /**
     * Convert landmarks to bone rotations
     */
    landmarksToRotations(data) {
        const rotations = {};
        
        if (this.options.enableEyeMovements) {
            // Calculate eye rotations from landmark positions
            rotations.leftEye = this.calculateEyeRotationFromLandmarks(data.eyeL, 'left');
            rotations.rightEye = this.calculateEyeRotationFromLandmarks(data.eyeR, 'right');
        }
        
        if (this.options.enableEyebrowMovement) {
            rotations.leftEyebrow = this.calculateEyebrowRotationFromLandmarks(data.eyebrowL, 'left');
            rotations.rightEyebrow = this.calculateEyebrowRotationFromLandmarks(data.eyebrowR, 'right');
        }
        
        if (this.options.enableJawMovement) {
            rotations.jaw = this.calculateJawRotationFromLandmarks(data.mouth, data.jaw);
        }
        
        if (this.options.enableCheekMovement) {
            rotations.leftCheek = this.calculateCheekRotationFromLandmarks(data.cheekL, 'left');
            rotations.rightCheek = this.calculateCheekRotationFromLandmarks(data.cheekR, 'right');
        }
        
        return rotations;
    }
    
    /**
     * Convert blend shapes to bone rotations
     */
    blendshapesToRotations(data) {
        const rotations = {};
        
        if (this.options.enableEyeMovements) {
            rotations.leftEye = {
                x: -(data.eyeLookUpL - data.eyeLookDownL) * 20, // Up/down
                y: -(data.eyeLookInL - data.eyeLookOutL) * 15,  // Left/right
                z: data.eyeBlinkL * -10 // Blink affects Z slightly
            };
            
            rotations.rightEye = {
                x: -(data.eyeLookUpR - data.eyeLookDownR) * 20,
                y: (data.eyeLookInR - data.eyeLookOutR) * 15, // Reversed for right eye
                z: data.eyeBlinkR * -10
            };
        }
        
        if (this.options.enableJawMovement) {
            rotations.jaw = {
                x: data.jawOpen * 25, // Open/close
                y: (data.jawLeft - data.jawRight) * 10, // Side to side
                z: 0
            };
        }
        
        if (this.options.enableEyebrowMovement) {
            rotations.leftEyebrow = {
                x: (data.browInnerUp + data.browOuterUpL) * 15,
                y: -data.browOuterUpL * 5,
                z: 0
            };
            
            rotations.rightEyebrow = {
                x: (data.browInnerUp + data.browOuterUpR) * 15,
                y: data.browOuterUpR * 5,
                z: 0
            };
        }
        
        rotations.leftMouth = {
            x: 0,
            y: -(data.mouthSmileL - data.mouthFrownL) * 10,
            z: data.mouthSmileL * 5
        };
        
        rotations.rightMouth = {
            x: 0,
            y: (data.mouthSmileR - data.mouthFrownR) * 10,
            z: data.mouthSmileR * 5
        };
        
        if (this.options.enableCheekMovement) {
            rotations.leftCheek = {
                x: 0,
                y: 0,
                z: data.cheekPuffL * 8
            };
            
            rotations.rightCheek = {
                x: 0,
                y: 0,
                z: data.cheekPuffR * 8
            };
        }
        
        return rotations;
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
        
        // Get the most recent frame from history
        const lastFrame = this.frameHistory[this.frameHistory.length - 1];
        
        for (const boneName in currentRotations) {
            if (lastFrame.rotations[boneName]) {
                smoothedRotations[boneName] = {
                    x: this.lerpAngle(lastFrame.rotations[boneName].x || 0, currentRotations[boneName].x || 0, factor),
                    y: this.lerpAngle(lastFrame.rotations[boneName].y || 0, currentRotations[boneName].y || 0, factor),
                    z: this.lerpAngle(lastFrame.rotations[boneName].z || 0, currentRotations[boneName].z || 0, factor)
                };
            } else {
                smoothedRotations[boneName] = currentRotations[boneName];
            }
        }
        
        return smoothedRotations;
    }
    
    /**
     * Generate BVH frame from facial rotations
     */
    generateBVHFrame(facialRotations, timestamp, originalData) {
        const motionData = new Array(Object.keys(this.facialBones).length * 6).fill(0);
        
        // Convert rotations to BVH format (position + rotation for each bone)
        for (const boneName in facialRotations) {
            const boneInfo = this.facialBones[boneName];
            if (boneInfo) {
                const baseIndex = boneInfo.index * 6;
                const rotation = facialRotations[boneName];
                
                // Position (typically 0 for facial bones, except head)
                motionData[baseIndex] = 0;     // X position
                motionData[baseIndex + 1] = 0; // Y position
                motionData[baseIndex + 2] = 0; // Z position
                
                // Rotation (in degrees)
                motionData[baseIndex + 3] = rotation.x || 0; // X rotation
                motionData[baseIndex + 4] = rotation.y || 0; // Y rotation
                motionData[baseIndex + 5] = rotation.z || 0; // Z rotation
            }
        }
        
        return {
            frameNumber: this.stats.framesConverted,
            timestamp: timestamp,
            motionData: motionData,
            metadata: {
                type: 'facial',
                source: 'faceformer',
                format: originalData.format,
                confidence: originalData.confidence,
                boneCount: Object.keys(facialRotations).length,
                channels: Object.keys(this.facialBones).length * 6,
                converter: 'FaceFormerBVHConverter',
                version: '1.0.0'
            }
        };
    }
    
    /**
     * Generate neutral BVH frame (all zeros) for error cases
     */
    generateNeutralBVHFrame(timestamp) {
        const motionData = new Array(Object.keys(this.facialBones).length * 6).fill(0);
        
        return {
            frameNumber: this.stats.framesConverted,
            timestamp: timestamp,
            motionData: motionData,
            metadata: {
                type: 'facial',
                source: 'faceformer',
                format: 'neutral',
                confidence: 0.0,
                boneCount: 0,
                channels: motionData.length,
                converter: 'FaceFormerBVHConverter',
                version: '1.0.0',
                isNeutral: true
            }
        };
    }
    
    /**
     * Helper methods for landmark processing
     */
    extractEyeLandmarks(landmarks, side) {
        // Extract eye landmarks based on 68-point standard
        const leftEyeIndices = [36, 37, 38, 39, 40, 41];
        const rightEyeIndices = [42, 43, 44, 45, 46, 47];
        const indices = side === 'left' ? leftEyeIndices : rightEyeIndices;
        
        return indices.map(i => landmarks[i]).filter(p => p);
    }
    
    extractEyebrowLandmarks(landmarks, side) {
        const leftBrowIndices = [17, 18, 19, 20, 21];
        const rightBrowIndices = [22, 23, 24, 25, 26];
        const indices = side === 'left' ? leftBrowIndices : rightBrowIndices;
        
        return indices.map(i => landmarks[i]).filter(p => p);
    }
    
    extractMouthLandmarks(landmarks) {
        const mouthIndices = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67];
        return mouthIndices.map(i => landmarks[i]).filter(p => p);
    }
    
    extractJawLandmarks(landmarks) {
        const jawIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        return jawIndices.map(i => landmarks[i]).filter(p => p);
    }
    
    extractNoseLandmarks(landmarks) {
        const noseIndices = [27, 28, 29, 30, 31, 32, 33, 34, 35];
        return noseIndices.map(i => landmarks[i]).filter(p => p);
    }
    
    extractCheekLandmarks(landmarks, side) {
        // Approximate cheek area from jaw and face outline
        const leftCheekIndices = [1, 2, 3, 31, 49];
        const rightCheekIndices = [13, 14, 15, 35, 53];
        const indices = side === 'left' ? leftCheekIndices : rightCheekIndices;
        
        return indices.map(i => landmarks[i]).filter(p => p);
    }
    
    /**
     * Calculate rotation from landmark positions
     */
    calculateEyeRotationFromLandmarks(eyeLandmarks, side) {
        if (!eyeLandmarks || eyeLandmarks.length < 6) {
            return { x: 0, y: 0, z: 0 };
        }
        
        // Calculate eye center and direction
        const center = this.calculateCenterPoint(eyeLandmarks);
        const corners = side === 'left' ? 
            { inner: eyeLandmarks[3], outer: eyeLandmarks[0] } :
            { inner: eyeLandmarks[0], outer: eyeLandmarks[3] };
        
        // Calculate rotation based on eye shape changes
        const horizontalVector = {
            x: corners.outer.x - corners.inner.x,
            y: corners.outer.y - corners.inner.y
        };
        
        const eyeAngle = Math.atan2(horizontalVector.y, horizontalVector.x) * (180 / Math.PI);
        
        return {
            x: 0, // Vertical look direction
            y: eyeAngle * 0.5, // Horizontal look direction
            z: 0
        };
    }
    
    calculateJawRotationFromLandmarks(mouthLandmarks, jawLandmarks) {
        if (!mouthLandmarks || mouthLandmarks.length < 4) {
            return { x: 0, y: 0, z: 0 };
        }
        
        // Calculate mouth opening
        const topLip = mouthLandmarks[3]; // Approximate top
        const bottomLip = mouthLandmarks[9]; // Approximate bottom
        const mouthHeight = Math.abs(topLip.y - bottomLip.y);
        
        // Calculate jaw lateral movement
        const leftCorner = mouthLandmarks[0];
        const rightCorner = mouthLandmarks[6];
        const mouthCenter = {
            x: (leftCorner.x + rightCorner.x) / 2,
            y: (leftCorner.y + rightCorner.y) / 2
        };
        
        return {
            x: Math.min(30, mouthHeight * 2), // Open/close (limited to 30 degrees)
            y: 0, // Side movement would need more complex calculation
            z: 0
        };
    }
    
    /**
     * Utility methods
     */
    calculateCenterPoint(points) {
        const sum = points.reduce((acc, point) => ({
            x: acc.x + point.x,
            y: acc.y + point.y,
            z: (acc.z || 0) + (point.z || 0)
        }), { x: 0, y: 0, z: 0 });
        
        return {
            x: sum.x / points.length,
            y: sum.y / points.length,
            z: sum.z / points.length
        };
    }
    
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
            constrained[boneName] = {
                x: this.clampAngle(rotation.x * this.options.scaleFactor, -45, 45),
                y: this.clampAngle(rotation.y * this.options.scaleFactor, -45, 45),
                z: this.clampAngle(rotation.z * this.options.scaleFactor, -45, 45)
            };
        }
        
        return constrained;
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
        
        for (const boneName in this.facialBones) {
            const boneInfo = this.facialBones[boneName];
            const baseIndex = boneInfo.index * 6;
            
            if (baseIndex + 5 < bvhFrame.motionData.length) {
                rotations[boneName] = {
                    x: bvhFrame.motionData[baseIndex + 3],
                    y: bvhFrame.motionData[baseIndex + 4],
                    z: bvhFrame.motionData[baseIndex + 5]
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
            isInitialized: this.isInitialized,
            facialBonesCount: Object.keys(this.facialBones).length
        };
    }
    
    /**
     * Timeline integration methods
     */
    createTimelineClip(faceFormerFrames, startTime = 0, duration = null) {
        if (!Array.isArray(faceFormerFrames)) {
            throw new Error('faceFormerFrames must be an array');
        }
        
        const clipDuration = duration || (faceFormerFrames.length / this.options.targetFrameRate);
        const bvhFrames = [];
        
        for (let i = 0; i < faceFormerFrames.length; i++) {
            const timestamp = startTime + (i / this.options.targetFrameRate);
            const bvhFrame = this.convertToBVH(faceFormerFrames[i], timestamp);
            bvhFrames.push(bvhFrame);
        }
        
        return {
            type: 'faceformer',
            startTime: startTime,
            duration: clipDuration,
            frames: bvhFrames,
            metadata: {
                sourceFrameCount: faceFormerFrames.length,
                targetFrameRate: this.options.targetFrameRate,
                converter: 'FaceFormerBVHConverter'
            }
        };
    }
    
    /**
     * Real-time processing for live FaceFormer input
     */
    async processLiveFrame(faceFormerOutput, timestamp) {
        try {
            const bvhFrame = await this.convertToBVH(faceFormerOutput, timestamp);
            
            // Emit frame update event if connected to timeline
            if (this.onFrameReady) {
                this.onFrameReady(bvhFrame, timestamp);
            }
            
            return bvhFrame;
            
        } catch (error) {
            console.error('[FaceFormer BVH] Live frame processing error:', error);
            return this.generateNeutralBVHFrame(timestamp);
        }
    }
    
    /**
     * Cleanup and disposal
     */
    dispose() {
        this.frameHistory = [];
        this.stats = {
            framesConverted: 0,
            averageConversionTime: 0,
            lastConversionTime: 0,
            errorCount: 0
        };
        this.onFrameReady = null;
        
        console.log('[FaceFormer BVH Converter] Disposed');
    }
}

// Placeholder methods for vertex and coefficient processing
// These would need to be implemented based on specific FaceFormer model outputs

FaceFormerBVHConverter.prototype.verticesToRotations = function(data) {
    // This would need specific implementation based on mesh topology
    console.warn('[FaceFormer BVH] Vertex to rotation conversion not fully implemented');
    return {};
};

FaceFormerBVHConverter.prototype.coefficientsToRotations = function(data) {
    // Convert expression coefficients to facial bone rotations
    const rotations = {};
    const coeffs = data.coefficients;
    
    if (coeffs.length >= 8) {
        rotations.jaw = {
            x: coeffs[0] * 20,  // Jaw open
            y: coeffs[1] * 10,  // Jaw side
            z: 0
        };
        
        rotations.leftEye = {
            x: coeffs[2] * 15,  // Eye up/down
            y: coeffs[3] * 12,  // Eye left/right
            z: 0
        };
        
        rotations.rightEye = {
            x: coeffs[4] * 15,
            y: coeffs[5] * 12,
            z: 0
        };
        
        rotations.leftEyebrow = {
            x: coeffs[6] * 10,
            y: 0,
            z: 0
        };
        
        rotations.rightEyebrow = {
            x: coeffs[7] * 10,
            y: 0,
            z: 0
        };
    }
    
    return rotations;
};

FaceFormerBVHConverter.prototype.rawDataToRotations = function(data) {
    // Simple fallback for unknown data formats
    return {
        head: { x: 0, y: 0, z: 0 },
        jaw: { x: Math.random() * 5, y: 0, z: 0 }  // Minimal animation
    };
};

FaceFormerBVHConverter.prototype.extractEyeVertices = function(vertices, side) {
    // Would need mesh topology information
    return [];
};

FaceFormerBVHConverter.prototype.extractMouthVertices = function(vertices) {
    // Would need mesh topology information
    return [];
};

FaceFormerBVHConverter.prototype.extractJawVertices = function(vertices) {
    // Would need mesh topology information
    return [];
};

FaceFormerBVHConverter.prototype.calculateEyebrowRotationFromLandmarks = function(eyebrowLandmarks, side) {
    if (!eyebrowLandmarks || eyebrowLandmarks.length < 3) {
        return { x: 0, y: 0, z: 0 };
    }
    
    // Calculate eyebrow raise/lower based on vertical position
    const center = this.calculateCenterPoint(eyebrowLandmarks);
    const leftmost = eyebrowLandmarks[0];
    const rightmost = eyebrowLandmarks[eyebrowLandmarks.length - 1];
    
    // Simple eyebrow movement calculation
    const browRaise = (center.y - ((leftmost.y + rightmost.y) / 2)) * 0.5;
    
    return {
        x: browRaise * 10,  // Up/down movement
        y: 0,
        z: 0
    };
};

FaceFormerBVHConverter.prototype.calculateCheekRotationFromLandmarks = function(cheekLandmarks, side) {
    if (!cheekLandmarks || cheekLandmarks.length < 2) {
        return { x: 0, y: 0, z: 0 };
    }
    
    // Simple cheek movement based on landmark displacement
    return {
        x: 0,
        y: 0,
        z: Math.random() * 2 - 1  // Subtle cheek movement
    };
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceFormerBVHConverter;
} else {
    window.FaceFormerBVHConverter = FaceFormerBVHConverter;
}
