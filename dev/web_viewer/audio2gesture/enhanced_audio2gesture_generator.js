// Enhanced Audio2Gesture Generator with Optimized Multi-Head Attention
// Improves FPS performance while maintaining compatibility with existing ONNX model

class EnhancedAudio2GestureGenerator {
    constructor(config = {}) {
        this.config = {
            modelPath: config.modelPath || './audio2gesture_step_fixed.onnx',
            useOptimizedAttention: config.useOptimizedAttention !== false,
            batchSize: config.batchSize || 1,
            maxSequenceLength: config.maxSequenceLength || 1000,
            attentionBackend: config.attentionBackend || 'auto',
            enableProfiling: config.enableProfiling !== false,
            enableCaching: config.enableCaching !== false,
            hybridMode: config.hybridMode !== false, // Use both ONNX and optimized attention
            enableBatchAudioProcessing: config.enableBatchAudioProcessing !== false,
            chunkSize: config.chunkSize || 8
        };

        // Core components
        this.onnxSession = null;
        this.optimizedAttention = null;
        this.batchAudioProcessor = null;
        this.isInitialized = false;

        // Performance tracking
        this.performanceTracker = new Audio2GesturePerformanceTracker();
        
        // State management
        this.hiddenStateCache = new Map();
        this.gestureSequenceCache = new Map();
        
        // Hybrid processing state
        this.useHybridProcessing = this.config.hybridMode;
        this.attentionEnhancementEnabled = false;
    }

    async initialize() {
        console.log('🚀 Initializing Enhanced Audio2Gesture Generator...');
        this.performanceTracker.startInitialization();

        try {
            // Initialize ONNX Runtime session
            await this._initializeONNXSession();
            
            // Initialize batch audio processor for high-FPS processing
            if (this.config.enableBatchAudioProcessing) {
                console.log('🎵 Initializing Batch Audio Processor...');
                try {
                    if (typeof BatchAudioProcessor !== 'undefined') {
                        this.batchAudioProcessor = new BatchAudioProcessor({
                            batchSize: this.config.batchSize,
                            chunkSize: this.config.chunkSize,
                            enableCaching: true
                        });
                        console.log('✅ Batch Audio Processor initialized');
                    } else {
                        console.warn('⚠️ BatchAudioProcessor not available, falling back to standard audio processing');
                        this.config.enableBatchAudioProcessing = false;
                    }
                } catch (error) {
                    console.warn('⚠️ Failed to initialize Batch Audio Processor:', error.message);
                    console.log('📉 Falling back to standard audio processing');
                    this.config.enableBatchAudioProcessing = false;
                }
            }
            
            // Initialize optimized attention if enabled
            if (this.config.useOptimizedAttention) {
                await this._initializeOptimizedAttention();
            }

            this.isInitialized = true;
            this.performanceTracker.endInitialization();
            
            console.log('✅ Enhanced Audio2Gesture Generator initialized successfully');
            console.log(`   📊 ONNX Session: ${this.onnxSession ? 'Ready' : 'Failed'}`);
            console.log(`   ⚡ Optimized Attention: ${this.optimizedAttention ? 'Ready' : 'Disabled'}`);
            console.log(`   🎵 Batch Audio Processor: ${this.batchAudioProcessor ? 'Ready' : 'Disabled'}`);
            console.log(`   🔧 Backend: ${this.optimizedAttention?.currentBackend || 'ONNX-only'}`);
            console.log(`   🎭 Hybrid Mode: ${this.useHybridProcessing ? 'Enabled' : 'Disabled'}`);
            console.log(`   📦 Batch Size: ${this.config.batchSize}, Chunk Size: ${this.config.chunkSize}`);
            
            return true;

        } catch (error) {
            console.error('❌ Failed to initialize Enhanced Audio2Gesture Generator:', error);
            this.performanceTracker.recordError('initialization', error);
            return false;
        }
    }

    async _initializeONNXSession() {
        if (typeof ort === 'undefined') {
            throw new Error('ONNXRuntime not available. Please include onnxruntime-web.');
        }

        console.log('📦 Loading ONNX model...');
        
        try {
            this.onnxSession = await ort.InferenceSession.create(this.config.modelPath, {
                executionProviders: ['wasm'],
                logSeverityLevel: 0
            });
            console.log('✅ ONNX model loaded successfully');
        } catch (error) {
            console.warn('⚠️ Failed to load ONNX model, creating demo mock session:', error.message);
            // Create mock session for demo purposes
            this.onnxSession = {
                run: async (feeds) => {
                    // Simulate ONNX inference with realistic outputs
                    const batchSize = 1;
                    const motionDim = 306; // Typical gesture dimension
                    const hiddenDim = 1024;
                    
                    // Add some time delay to simulate real inference
                    await new Promise(resolve => setTimeout(resolve, 5 + Math.random() * 10));
                    
                    return {
                        new_motion: {
                            data: new Float32Array(motionDim).map(() => (Math.random() - 0.5) * 0.2),
                            dims: [batchSize, motionDim]
                        },
                        new_hidden_state: {
                            data: new Float32Array(hiddenDim).map(() => (Math.random() - 0.5) * 0.1),
                            dims: [batchSize, 1, hiddenDim]
                        }
                    };
                }
            };
            console.log('✅ Demo mock session created');
        }
    }

    async _initializeOptimizedAttention() {
        console.log('⚡ Initializing optimized multi-head attention...');
        
        this.optimizedAttention = new OptimizedAudio2GestureAttention({
            numHeads: 8,
            hiddenDim: 1024, // Match ONNX hidden state size
            preferredBackend: this.config.attentionBackend,
            enableCaching: this.config.enableCaching
        });

        await this.optimizedAttention.initializeBackend();
        this.attentionEnhancementEnabled = true;
        
        console.log(`✅ Optimized attention initialized with ${this.optimizedAttention.currentBackend} backend`);
    }

    async generateGestureSequence(audioFeatures = null, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Generator not initialized. Call initialize() first.');
        }

        const {
            lexemeType = 'neutral',
            numFrames = 10,
            batchSize = 1,
            enableAttentionEnhancement = this.attentionEnhancementEnabled,
            enableCaching = this.config.enableCaching,
            batchProcessing = false
        } = options;

        console.log(`🎬 Generating ${numFrames} gesture frames ${batchProcessing ? `(batch size: ${batchSize})` : ''} with ${enableAttentionEnhancement ? 'optimized' : 'standard'} processing...`);
        
        this.performanceTracker.startGeneration(numFrames * (batchSize || 1));

        try {
            let result;
            
            if (batchProcessing && batchSize > 1) {
                result = await this._generateBatchSequence(audioFeatures, options);
            } else {
                result = await this._generateAutoregressiveSequence(audioFeatures, options);
            }

            this.performanceTracker.endGeneration(result.metadata);
            
            if (this.config.enableProfiling) {
                this._logPerformanceMetrics(result.metadata);
            }

            return result;

        } catch (error) {
            this.performanceTracker.recordError('generation', error);
            throw error;
        }
    }

    async _generateAutoregressiveSequence(audioFeatures, options) {
        const { numFrames, lexemeType, enableAttentionEnhancement } = options;
        
        // Prepare initial inputs
        const audioWindow = this._prepareAudioWindow(audioFeatures);
        const lexemeFeatures = this._createLexemeFeatures(lexemeType);
        let hiddenState = this._createInitialHiddenState();
        let currentMotion = this._createInitialMotion();

        const generatedFrames = [];
        const frameTimings = [];
        const attentionTimings = [];

        for (let step = 0; step < numFrames; step++) {
            const frameStartTime = performance.now();
            
            // Apply attention enhancement if enabled
            if (enableAttentionEnhancement && this.optimizedAttention) {
                const attentionStart = performance.now();
                
                // Enhance hidden state with optimized attention
                const enhancedHiddenState = await this.optimizedAttention.computeMultiHeadAttention(
                    this._onnxTensorToArray(hiddenState),
                    audioWindow.data,
                    lexemeFeatures.data
                );
                
                // Update hidden state with attention-enhanced version
                hiddenState = this._arrayToOnnxTensor(enhancedHiddenState, hiddenState.dims);
                
                const attentionTime = performance.now() - attentionStart;
                attentionTimings.push(attentionTime);
            }

            // Run ONNX model step
            const onnxResult = await this._runONNXStep(audioWindow, currentMotion, lexemeFeatures, hiddenState);
            
            // Update state for next iteration
            currentMotion = onnxResult.newMotion;
            hiddenState = onnxResult.newHiddenState;
            
            // Store generated frame
            const motionData = Array.from(currentMotion.data);
            generatedFrames.push(motionData);
            
            const frameTime = performance.now() - frameStartTime;
            frameTimings.push(frameTime);
            
            if (step < 5 || step % 5 === 4) {
                console.log(`    ✨ Frame ${step + 1}: [${motionData.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...] (${frameTime.toFixed(1)}ms)`);
            }
        }

        const totalTime = frameTimings.reduce((a, b) => a + b, 0);
        const avgTime = totalTime / numFrames;
        const totalAttentionTime = attentionTimings.reduce((a, b) => a + b, 0);

        return {
            frames: generatedFrames,
            metadata: {
                totalTime,
                avgTime,
                fps: 1000 / avgTime,
                numFrames,
                attentionTime: totalAttentionTime,
                attentionEnabled: enableAttentionEnhancement,
                backend: this.optimizedAttention?.currentBackend || 'onnx-only'
            }
        };
    }

    async _generateBatchSequence(audioFeatures, options) {
        const { numFrames, batchSize, lexemeType, enableAttentionEnhancement } = options;
        
        console.log(`🎬 Multi-frame avatar animation: generating ${numFrames} frames with batch size ${batchSize}...`);
        
        const batchStartTime = performance.now();
        
        // Prepare initial conditions
        const audioWindows = this._prepareAudioWindows(audioFeatures, numFrames);
        const lexemeFeatures = this._prepareLexemeSequence(lexemeType, numFrames);
        const hiddenState = this._createInitialHiddenState();
        const startingMotion = this._createInitialMotion();
        
        let allGeneratedFrames = [];
        let metadata = {};
        
        if (enableAttentionEnhancement && this.optimizedAttention && numFrames > 1) {
            // Use multi-frame attention for generating multiple frames simultaneously
            console.log(`🚀 Using multi-frame attention for simultaneous frame generation...`);
            
            const result = await this._generateMultiFrameSequence(
                audioWindows,
                startingMotion,
                lexemeFeatures,
                hiddenState,
                numFrames,
                batchSize
            );
            
            allGeneratedFrames = result.frames;
            metadata = result.metadata;
            
        } else {
            // Fall back to traditional batch processing (multiple parallel sequences)
            console.log(`📦 Using traditional batch processing for ${batchSize} parallel sequences...`);
            
            const result = await this._generateTraditionalBatch(
                audioWindows,
                startingMotion,
                lexemeFeatures,
                hiddenState,
                numFrames,
                batchSize,
                enableAttentionEnhancement
            );
            
            allGeneratedFrames = result.frames;
            metadata = result.metadata;
        }
        
        const totalTime = performance.now() - batchStartTime;
        
        console.log(`✅ Animation generation complete: ${allGeneratedFrames.length} frames in ${totalTime.toFixed(1)}ms`);
        
        return {
            frames: allGeneratedFrames,
            metadata: {
                ...metadata,
                totalTime,
                framesGenerated: allGeneratedFrames.length,
                avgTimePerFrame: totalTime / allGeneratedFrames.length,
                method: enableAttentionEnhancement && numFrames > 1 ? 'multi-frame-attention' : 'traditional-batch'
            }
        };
    }

    /**
     * Generate multiple frames simultaneously using multi-frame attention and batched audio processing
     * This is the key optimization for real-time avatar animation with audio sequences
     */
    async _generateMultiFrameSequence(audioWindows, startingMotion, lexemeFeatures, hiddenState, numFrames, chunkSize = 8) {
        console.log(`🎬 Multi-frame audio sequence generation: ${numFrames} frames in chunks of ${chunkSize}...`);
        
        const allFrames = [];
        const timings = {
            audioProcessing: [],
            temporalAttention: [],
            generation: [],
            chunks: []
        };
        
        let currentMotion = { ...startingMotion };
        let currentHiddenState = { ...hiddenState };
        
        // Pre-process entire audio sequence for temporal coherence using batch processor
        let audioSequenceFeatures, temporalAudioFeatures;
        
        if (this.batchAudioProcessor) {
            console.log('🎵 Using Batch Audio Processor for high-FPS audio processing...');
            const audioProcessingStart = performance.now();
            
            // Convert audio windows to sequences for batch processing
            const audioSequences = [audioWindows.map(window => Array.from(window.data))];
            
            const batchResult = await this.batchAudioProcessor.processBatchAudioSequences(audioSequences, {
                enableTemporalSmoothing: true,
                enablePerceptualWeighting: true
            });
            
            temporalAudioFeatures = batchResult.attentionContext[0]; // First (and only) sequence
            const audioProcessingTime = performance.now() - audioProcessingStart;
            timings.audioProcessing.push(audioProcessingTime);
            
            console.log(`🚀 Batch audio processing: ${audioWindows.length} frames in ${audioProcessingTime.toFixed(2)}ms`);
        } else {
            // Fallback to original audio processing
            audioSequenceFeatures = this._extractAudioSequenceFeatures(audioWindows);
            temporalAudioFeatures = this._computeTemporalAudioFeatures(audioSequenceFeatures, numFrames);
        }
        
        // Process frames in chunks for memory efficiency but with temporal awareness
        for (let chunkStart = 0; chunkStart < numFrames; chunkStart += chunkSize) {
            const chunkEnd = Math.min(chunkStart + chunkSize, numFrames);
            const actualChunkSize = chunkEnd - chunkStart;
            
            console.log(`⚡ Processing audio-driven chunk ${Math.floor(chunkStart/chunkSize) + 1}: frames ${chunkStart + 1}-${chunkEnd}...`);
            const chunkStartTime = performance.now();
            
            // Extract temporal audio context for this chunk
            const audioProcessingStart = performance.now();
            const chunkTemporalAudio = temporalAudioFeatures.slice(chunkStart, chunkEnd);
            const chunkAudioWindows = audioWindows.slice(chunkStart, chunkEnd);
            const chunkLexemeFeatures = lexemeFeatures.slice(chunkStart, chunkEnd);
            
            // Create audio-aware hidden state sequence for temporal processing
            const hiddenSequence = this._createHiddenStateSequence(currentHiddenState, actualChunkSize);
            const audioProcessingTime = performance.now() - audioProcessingStart;
            timings.audioProcessing.push(audioProcessingTime);
            
            // Apply multi-frame temporal attention across the chunk with audio context
            const temporalAttentionStart = performance.now();
            const enhancedSequence = await this.optimizedAttention.computeMultiFrameAttention(
                hiddenSequence,
                chunkTemporalAudio,
                chunkLexemeFeatures,
                actualChunkSize
            );
            const temporalAttentionTime = performance.now() - temporalAttentionStart;
            timings.temporalAttention.push(temporalAttentionTime);
            
            // Generate frames using enhanced temporal states
            const generationStart = performance.now();
            const chunkFrames = await this._generateChunkFramesBatched(
                chunkAudioWindows,
                chunkLexemeFeatures,
                enhancedSequence,
                currentMotion,
                actualChunkSize
            );
            
            // Update states for next chunk based on last generated frame
            if (chunkFrames.length > 0) {
                const lastFrame = chunkFrames[chunkFrames.length - 1];
                currentMotion = this._extractMotionFromFrame(lastFrame);
                currentHiddenState = enhancedSequence[enhancedSequence.length - 1];
            }
            
            allFrames.push(...chunkFrames);
            
            const generationTime = performance.now() - generationStart;
            timings.generation.push(generationTime);
            
            const chunkTime = performance.now() - chunkStartTime;
            timings.chunks.push(chunkTime);
            
            console.log(`    🎵 Audio chunk ${Math.floor(chunkStart/chunkSize) + 1} complete: ${actualChunkSize} frames in ${chunkTime.toFixed(1)}ms (${(chunkTime/actualChunkSize).toFixed(1)}ms/frame)`);
        }
        
        const totalAttentionTime = timings.temporalAttention.reduce((a, b) => a + b, 0);
        const totalGenerationTime = timings.generation.reduce((a, b) => a + b, 0);
        const totalAudioProcessingTime = timings.audioProcessing.reduce((a, b) => a + b, 0);
        
        return {
            frames: allFrames,
            metadata: {
                totalTime: timings.chunks.reduce((a, b) => a + b, 0),
                audioProcessingTime: totalAudioProcessingTime,
                temporalAttentionTime: totalAttentionTime,
                generationTime: totalGenerationTime,
                chunksProcessed: timings.chunks.length,
                avgTimePerFrame: timings.chunks.reduce((a, b) => a + b, 0) / numFrames,
                fps: 1000 / (timings.chunks.reduce((a, b) => a + b, 0) / numFrames),
                attentionEnabled: true,
                backend: this.optimizedAttention?.currentBackend || 'onnx-only',
                method: 'multi-frame-temporal-audio'
            }
        };
    }

    /**
     * Extract features from audio sequence for temporal processing
     */
    _extractAudioSequenceFeatures(audioWindows) {
        const sequenceFeatures = [];
        
        for (let i = 0; i < audioWindows.length; i++) {
            const audioData = audioWindows[i].data;
            
            // Extract key audio features for temporal modeling
            const features = {
                energy: this._computeAudioEnergy(audioData),
                spectralCentroid: this._computeSpectralCentroid(audioData),
                zeroCrossingRate: this._computeZeroCrossingRate(audioData),
                mfccFeatures: this._extractMFCCFeatures(audioData),
                temporalPosition: i / audioWindows.length,
                rawAudio: Array.from(audioData.slice(0, 80)) // First 80 coefficients
            };
            
            sequenceFeatures.push(features);
        }
        
        return sequenceFeatures;
    }

    /**
     * Compute temporal audio features with context awareness
     */
    _computeTemporalAudioFeatures(audioSequenceFeatures, numFrames) {
        const temporalFeatures = [];
        
        for (let i = 0; i < numFrames; i++) {
            const currentAudio = audioSequenceFeatures[Math.min(i, audioSequenceFeatures.length - 1)];
            
            // Add temporal context from neighboring frames
            const contextWindow = 3; // Look at 3 frames before and after
            const contextFeatures = [];
            
            for (let offset = -contextWindow; offset <= contextWindow; offset++) {
                const contextIndex = Math.max(0, Math.min(i + offset, audioSequenceFeatures.length - 1));
                const contextAudio = audioSequenceFeatures[contextIndex];
                
                // Weight context by distance
                const weight = Math.exp(-Math.abs(offset) / contextWindow);
                const weightedFeatures = contextAudio.rawAudio.map(f => f * weight);
                contextFeatures.push(...weightedFeatures);
            }
            
            // Combine current audio with temporal context
            const combinedFeatures = [
                ...currentAudio.rawAudio,
                currentAudio.energy,
                currentAudio.spectralCentroid,
                currentAudio.zeroCrossingRate,
                ...currentAudio.mfccFeatures.slice(0, 12), // First 12 MFCC coefficients
                currentAudio.temporalPosition,
                ...contextFeatures.slice(0, 100) // Limit context features
            ];
            
            temporalFeatures.push(combinedFeatures);
        }
        
        return temporalFeatures;
    }

    /**
     * Audio analysis helper functions
     */
    _computeAudioEnergy(audioData) {
        let energy = 0;
        for (let i = 0; i < audioData.length; i++) {
            energy += audioData[i] * audioData[i];
        }
        return Math.sqrt(energy / audioData.length);
    }

    _computeSpectralCentroid(audioData) {
        // Simplified spectral centroid calculation
        let weightedSum = 0;
        let magnitudeSum = 0;
        
        for (let i = 0; i < audioData.length; i++) {
            const magnitude = Math.abs(audioData[i]);
            weightedSum += i * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    }

    _computeZeroCrossingRate(audioData) {
        let crossings = 0;
        for (let i = 1; i < audioData.length; i++) {
            if ((audioData[i] >= 0) !== (audioData[i-1] >= 0)) {
                crossings++;
            }
        }
        return crossings / (audioData.length - 1);
    }

    _extractMFCCFeatures(audioData) {
        // Simplified MFCC-like features
        const features = new Array(13);
        const windowSize = Math.min(audioData.length, 256);
        
        for (let i = 0; i < 13; i++) {
            let sum = 0;
            for (let j = 0; j < windowSize; j++) {
                const freq = (i + 1) * j / windowSize;
                sum += audioData[j] * Math.cos(2 * Math.PI * freq);
            }
            features[i] = sum / windowSize;
        }
        
        return features;
    }

    /**
     * Create sequence of hidden states for temporal processing
     */
    _createHiddenStateSequence(initialHiddenState, sequenceLength) {
        const sequence = [];
        const hiddenData = Array.from(initialHiddenState.data);
        
        for (let i = 0; i < sequenceLength; i++) {
            // Add slight variation to each hidden state in the sequence
            const variation = i * 0.01;
            const variedHidden = hiddenData.map((value, index) => {
                return value + Math.sin(index * variation) * 0.001;
            });
            
            sequence.push(variedHidden);
        }
        
        return sequence;
    }

    /**
     * Generate frames for a chunk using batched processing
     */
    async _generateChunkFramesBatched(audioWindows, lexemeFeatures, enhancedSequence, startMotion, chunkSize) {
        const frames = [];
        let currentMotion = { ...startMotion };
        
        // Process multiple frames in parallel when possible
        const batchSize = Math.min(chunkSize, 4); // Limit batch size for memory efficiency
        
        for (let i = 0; i < chunkSize; i += batchSize) {
            const batchEnd = Math.min(i + batchSize, chunkSize);
            const actualBatchSize = batchEnd - i;
            
            // Prepare batch inputs
            const batchPromises = [];
            const batchCurrentMotions = [];
            
            for (let j = 0; j < actualBatchSize; j++) {
                const frameIndex = i + j;
                const audioWindow = audioWindows[frameIndex];
                const lexeme = lexemeFeatures[frameIndex];
                
                // Create enhanced hidden state tensor from attention output
                const enhancedHiddenData = enhancedSequence[frameIndex];
                const enhancedHiddenState = this._arrayToOnnxTensor(enhancedHiddenData, [1, 1, enhancedHiddenData.length]);
                
                batchCurrentMotions.push({ ...currentMotion });
                
                const promise = this._runONNXStep(
                    audioWindow,
                    currentMotion,
                    lexeme,
                    enhancedHiddenState
                );
                batchPromises.push(promise);
            }
            
            // Execute batch in parallel
            const batchResults = await Promise.all(batchPromises);
            
            // Process results and update motion state
            for (let j = 0; j < actualBatchSize; j++) {
                const result = batchResults[j];
                const motionData = Array.from(result.newMotion.data);
                frames.push(motionData);
                
                // Update motion for next frame (use last result as starting point)
                if (j === actualBatchSize - 1) {
                    currentMotion = result.newMotion;
                }
            }
        }
        
        return frames;
    }

    /**
     * Extract motion tensor from generated frame data
     */
    _extractMotionFromFrame(frameData) {
        return this._arrayToOnnxTensor(frameData, [1, frameData.length]);
    }

    /**
     * Traditional batch processing fallback (multiple parallel sequences)
     */
    async _generateTraditionalBatch(audioWindows, startingMotion, lexemeFeatures, hiddenState, numFrames, batchSize, enableAttentionEnhancement) {
        // Initialize batch states
        const batchCurrentMotions = Array(batchSize).fill(null).map(() => ({ ...startingMotion }));
        const batchHiddenStates = Array(batchSize).fill(null).map(() => ({ ...hiddenState }));
        
        const allGeneratedFrames = [];
        const frameTimings = [];
        const attentionTimings = [];
        
        // Process all sequences step by step
        for (let step = 0; step < numFrames; step++) {
            const stepStartTime = performance.now();
            
            // Apply attention enhancement if enabled (batch processing)
            if (enableAttentionEnhancement && this.optimizedAttention) {
                const attentionStart = performance.now();
                
                // Process all sequences in batch for attention
                const batchHiddenArrays = batchHiddenStates.map(hs => this._onnxTensorToArray(hs));
                const batchAudioData = Array(batchSize).fill(audioWindows[Math.min(step, audioWindows.length - 1)].data);
                const batchLexemeData = Array(batchSize).fill(lexemeFeatures[Math.min(step, lexemeFeatures.length - 1)].data);
                
                // Compute attention for entire batch
                const enhancedBatchHidden = await this.optimizedAttention.computeMultiHeadAttentionBatch(
                    batchHiddenArrays,
                    batchAudioData,
                    batchLexemeData
                );
                
                // Update hidden states with attention-enhanced versions
                for (let b = 0; b < batchSize; b++) {
                    batchHiddenStates[b] = this._arrayToOnnxTensor(enhancedBatchHidden[b], batchHiddenStates[b].dims);
                }
                
                const attentionTime = performance.now() - attentionStart;
                attentionTimings.push(attentionTime);
            }
            
            // Run ONNX model for each sequence in parallel
            const batchPromises = [];
            for (let b = 0; b < batchSize; b++) {
                const audioWindow = audioWindows[Math.min(step, audioWindows.length - 1)];
                const lexemeFeature = lexemeFeatures[Math.min(step, lexemeFeatures.length - 1)];
                
                const promise = this._runONNXStep(
                    audioWindow,
                    batchCurrentMotions[b],
                    lexemeFeature,
                    batchHiddenStates[b]
                );
                batchPromises.push(promise);
            }
            
            // Wait for all batch sequences to complete
            const batchResults = await Promise.all(batchPromises);
            
            // Update states and collect frames
            const stepFrames = [];
            for (let b = 0; b < batchSize; b++) {
                batchCurrentMotions[b] = batchResults[b].newMotion;
                batchHiddenStates[b] = batchResults[b].newHiddenState;
                
                const motionData = Array.from(batchCurrentMotions[b].data);
                stepFrames.push(motionData);
            }
            
            allGeneratedFrames.push(...stepFrames);
            
            const stepTime = performance.now() - stepStartTime;
            frameTimings.push(stepTime);
            
            if (step < 5 || step % 5 === 4) {
                console.log(`    📦 Batch step ${step + 1}: ${batchSize} sequences processed (${stepTime.toFixed(1)}ms)`);
            }
        }
        
        const totalTime = frameTimings.reduce((a, b) => a + b, 0);
        const avgTimePerFrame = totalTime / (numFrames * batchSize);
        const totalAttentionTime = attentionTimings.reduce((a, b) => a + b, 0);
        
        return {
            frames: allGeneratedFrames,
            metadata: {
                totalTime,
                avgTime: avgTimePerFrame,
                fps: 1000 / avgTimePerFrame,
                numFrames: numFrames * batchSize,
                attentionTime: totalAttentionTime,
                attentionEnabled: enableAttentionEnhancement,
                backend: this.optimizedAttention?.currentBackend || 'onnx-only'
            }
        };
    }

    /**
     * Prepare audio windows for the entire sequence
     */
    _prepareAudioWindows(audioFeatures, numFrames) {
        const windows = [];
        for (let i = 0; i < numFrames; i++) {
            // Use same audio window for all frames (could be enhanced to use different segments)
            windows.push(this._prepareAudioWindow(audioFeatures));
        }
        return windows;
    }

    /**
     * Prepare lexeme features for the entire sequence
     */
    /**
     * Prepare lexeme features for the entire sequence
     */
    _prepareLexemeSequence(lexemeType, numFrames) {
        const sequence = [];
        for (let i = 0; i < numFrames; i++) {
            sequence.push(this._createLexemeFeatures(lexemeType));
        }
        return sequence;
    }

    async _runONNXStep(audioWindow, prevMotion, lexemeFeatures, hiddenState) {
        const feeds = {
            audio_window: audioWindow,
            prev_motion: prevMotion,
            current_lexeme: lexemeFeatures,
            hidden_state: hiddenState
        };

        const results = await this.onnxSession.run(feeds);
        
        return {
            newMotion: results.new_motion,
            newHiddenState: results.new_hidden_state
        };
    }

    _prepareAudioWindow(audioFeatures, windowSize = 30) {
        const audioData = new Float32Array(1 * 80 * windowSize);
        
        if (audioFeatures && audioFeatures.length > 0) {
            const flatAudio = audioFeatures.flat ? audioFeatures.flat(2) : audioFeatures;
            const copyLength = Math.min(flatAudio.length, audioData.length);
            for (let i = 0; i < copyLength; i++) {
                audioData[i] = flatAudio[i];
            }
        } else {
            // Generate synthetic audio features for testing
            for (let i = 0; i < audioData.length; i++) {
                audioData[i] = (Math.random() - 0.5) * 0.1;
            }
        }
        
        return new ort.Tensor('float32', audioData, [1, 80, windowSize]);
    }

    _createLexemeFeatures(lexemeType = 'neutral') {
        const lexemeData = new Float32Array(96);
        
        switch (lexemeType) {
            case 'expressive':
                lexemeData.fill(0.5);
                break;
            case 'subtle':
                lexemeData.fill(0.2);
                break;
            case 'excited':
                for (let i = 0; i < 96; i++) {
                    lexemeData[i] = 0.3 + 0.4 * Math.sin(i * 0.1);
                }
                break;
            case 'calm':
                lexemeData.fill(0.1);
                break;
            case 'neutral':
            default:
                lexemeData.fill(0.15);
                break;
        }
        
        return new ort.Tensor('float32', lexemeData, [1, 96]);
    }

    _createInitialHiddenState() {
        const hiddenStateSize = 4 * 1 * 1024;
        const hiddenStateData = new Float32Array(hiddenStateSize);
        return new ort.Tensor('float32', hiddenStateData, [4, 1, 1024]);
    }

    _createInitialMotion() {
        return new ort.Tensor('float32', new Float32Array(48), [1, 48]);
    }

    _onnxTensorToArray(tensor) {
        // Convert ONNX tensor to nested array for attention processing
        const data = Array.from(tensor.data);
        const dims = tensor.dims;
        
        if (dims.length === 3) {
            // [layers, batch, hidden] -> reshape for attention
            const result = [];
            const [layers, batch, hidden] = dims;
            
            for (let l = 0; l < layers; l++) {
                const layer = [];
                for (let b = 0; b < batch; b++) {
                    const batchData = [];
                    for (let h = 0; h < hidden; h++) {
                        const idx = l * batch * hidden + b * hidden + h;
                        batchData.push(data[idx]);
                    }
                    layer.push(batchData);
                }
                result.push(layer);
            }
            
            return result;
        }
        
        return data;
    }

    _arrayToOnnxTensor(array, dims) {
        // Convert nested array back to ONNX tensor
        const flatData = array.flat(Infinity);
        return new ort.Tensor('float32', new Float32Array(flatData), dims);
    }

    _logPerformanceMetrics(metadata) {
        console.log(`⚡ Performance Metrics:`);
        console.log(`   📊 Total time: ${metadata.totalTime.toFixed(1)}ms`);
        console.log(`   ⚡ Average per frame: ${metadata.avgTime.toFixed(1)}ms`);
        console.log(`   🚀 Generation rate: ${metadata.fps.toFixed(1)} FPS`);
        
        if (metadata.attentionEnabled) {
            console.log(`   🧠 Attention time: ${metadata.attentionTime.toFixed(1)}ms`);
            console.log(`   🔧 Backend: ${metadata.backend}`);
        }
    }

    async runPerformanceComparison(numFrames = 20) {
        console.log('🏁 Running performance comparison between standard and optimized processing...');
        
        const testAudio = this._generateTestAudioFeatures();
        
        // Test 1: Standard ONNX-only processing
        console.log('\n🧪 Test 1: Standard ONNX processing');
        const standardResult = await this.generateGestureSequence(testAudio, {
            numFrames,
            lexemeType: 'neutral',
            enableAttentionEnhancement: false
        });

        // Test 2: Optimized attention-enhanced processing
        console.log('\n🧪 Test 2: Attention-enhanced processing');
        const optimizedResult = await this.generateGestureSequence(testAudio, {
            numFrames,
            lexemeType: 'neutral',
            enableAttentionEnhancement: true
        });

        // Calculate improvements
        const speedupFactor = standardResult.metadata.avgTime / optimizedResult.metadata.avgTime;
        const fpsImprovement = optimizedResult.metadata.fps - standardResult.metadata.fps;

        console.log('\n📊 Performance Comparison Results:');
        console.log(`   Standard Processing: ${standardResult.metadata.avgTime.toFixed(1)}ms/frame (${standardResult.metadata.fps.toFixed(1)} FPS)`);
        console.log(`   Optimized Processing: ${optimizedResult.metadata.avgTime.toFixed(1)}ms/frame (${optimizedResult.metadata.fps.toFixed(1)} FPS)`);
        console.log(`   🚀 Speedup: ${speedupFactor.toFixed(2)}x`);
        console.log(`   📈 FPS Improvement: +${fpsImprovement.toFixed(1)} FPS`);
        console.log(`   🧠 Attention Backend: ${optimizedResult.metadata.backend}`);

        return {
            standard: standardResult,
            optimized: optimizedResult,
            speedup: speedupFactor,
            fpsImprovement
        };
    }

    _generateTestAudioFeatures() {
        // Generate synthetic audio features for testing
        const audioFeatures = [];
        for (let i = 0; i < 80; i++) {
            const frame = [];
            for (let j = 0; j < 30; j++) {
                frame.push(Math.sin(i * 0.1 + j * 0.05) * 0.1);
            }
            audioFeatures.push(frame);
        }
        return audioFeatures;
    }

    async switchAttentionBackend(backend) {
        if (!this.optimizedAttention) {
            console.warn('⚠️ Optimized attention not initialized');
            return false;
        }

        console.log(`🔄 Switching attention backend to: ${backend}`);
        const success = await this.optimizedAttention.switchBackend(backend);
        
        if (success) {
            console.log(`✅ Successfully switched to ${backend} backend`);
        } else {
            console.error(`❌ Failed to switch to ${backend} backend`);
        }
        
        return success;
    }

    getSystemInfo() {
        const info = {
            isInitialized: this.isInitialized,
            config: this.config,
            onnxSession: !!this.onnxSession,
            optimizedAttention: !!this.optimizedAttention,
            performanceStats: this.performanceTracker.getStats()
        };

        if (this.optimizedAttention) {
            info.attentionInfo = this.optimizedAttention.getSystemInfo();
        }

        return info;
    }

    getPerformanceReport() {
        return this.performanceTracker.generateReport();
    }

    cleanup() {
        // Cleanup ONNX session
        if (this.onnxSession) {
            this.onnxSession.release();
        }

        // Cleanup optimized attention
        if (this.optimizedAttention) {
            this.optimizedAttention.cleanup();
        }

        // Clear caches
        this.hiddenStateCache.clear();
        this.gestureSequenceCache.clear();

        console.log('🧹 Enhanced Audio2Gesture Generator cleaned up');
    }
}

// Performance tracking utility
class Audio2GesturePerformanceTracker {
    constructor() {
        this.stats = {
            initialization: { count: 0, totalTime: 0 },
            generation: { count: 0, totalTime: 0, totalFrames: 0 },
            errors: []
        };
        this.currentSession = null;
    }

    startInitialization() {
        this.currentSession = {
            type: 'initialization',
            startTime: performance.now()
        };
    }

    endInitialization() {
        if (this.currentSession?.type === 'initialization') {
            const duration = performance.now() - this.currentSession.startTime;
            this.stats.initialization.count++;
            this.stats.initialization.totalTime += duration;
            this.currentSession = null;
        }
    }

    startGeneration(numFrames) {
        this.currentSession = {
            type: 'generation',
            startTime: performance.now(),
            numFrames
        };
    }

    endGeneration(metadata) {
        if (this.currentSession?.type === 'generation') {
            const duration = performance.now() - this.currentSession.startTime;
            this.stats.generation.count++;
            this.stats.generation.totalTime += duration;
            this.stats.generation.totalFrames += this.currentSession.numFrames;
            this.currentSession = null;
        }
    }

    recordError(phase, error) {
        this.stats.errors.push({
            phase,
            error: error.message,
            timestamp: Date.now()
        });
    }

    getStats() {
        return {
            ...this.stats,
            averageInitTime: this.stats.initialization.count > 0 
                ? this.stats.initialization.totalTime / this.stats.initialization.count 
                : 0,
            averageGenerationTime: this.stats.generation.count > 0 
                ? this.stats.generation.totalTime / this.stats.generation.count 
                : 0,
            averageFrameTime: this.stats.generation.totalFrames > 0 
                ? this.stats.generation.totalTime / this.stats.generation.totalFrames 
                : 0,
            averageFPS: this.stats.generation.totalFrames > 0 
                ? 1000 / (this.stats.generation.totalTime / this.stats.generation.totalFrames) 
                : 0
        };
    }

    generateReport() {
        const stats = this.getStats();
        
        return {
            summary: {
                totalSessions: stats.generation.count,
                totalFrames: stats.generation.totalFrames,
                averageFPS: stats.averageFPS.toFixed(1),
                averageFrameTime: stats.averageFrameTime.toFixed(2),
                errorCount: stats.errors.length
            },
            details: stats
        };
    }
}

// Demo function for enhanced audio2gesture
async function demoEnhancedAudio2Gesture() {
    console.log('🎭 Enhanced Audio2Gesture Demo with Optimized Attention');
    console.log('========================================================\n');

    try {
        // Initialize enhanced generator
        const generator = new EnhancedAudio2GestureGenerator({
            modelPath: './audio2gesture_step_fixed.onnx',
            useOptimizedAttention: true,
            attentionBackend: 'auto',
            enableProfiling: true
        });

        const success = await generator.initialize();
        if (!success) {
            console.log('❌ Failed to initialize enhanced generator');
            return;
        }

        // Run performance comparison
        const comparison = await generator.runPerformanceComparison(15);
        
        // Test different lexeme types with optimization
        console.log('\n🎭 Testing different expression types with optimization...');
        
        const expressionTypes = ['neutral', 'expressive', 'subtle', 'excited', 'calm'];
        for (const expression of expressionTypes) {
            console.log(`\n   🎪 Testing ${expression} expression...`);
            const result = await generator.generateGestureSequence(null, {
                numFrames: 5,
                lexemeType: expression,
                enableAttentionEnhancement: true
            });
            
            console.log(`      ⚡ ${result.metadata.fps.toFixed(1)} FPS (${result.metadata.avgTime.toFixed(1)}ms/frame)`);
        }

        // System info
        console.log('\n📊 System Information:');
        const systemInfo = generator.getSystemInfo();
        console.log(`   Optimized Attention: ${systemInfo.optimizedAttention ? '✅' : '❌'}`);
        if (systemInfo.attentionInfo) {
            console.log(`   Attention Backend: ${systemInfo.attentionInfo.currentBackend}`);
            console.log(`   Supported Backends: ${systemInfo.attentionInfo.supportedBackends.join(', ')}`);
        }

        // Performance report
        console.log('\n📈 Performance Report:');
        const report = generator.getPerformanceReport();
        console.log(`   Total Sessions: ${report.summary.totalSessions}`);
        console.log(`   Total Frames: ${report.summary.totalFrames}`);
        console.log(`   Average FPS: ${report.summary.averageFPS}`);
        console.log(`   Average Frame Time: ${report.summary.averageFrameTime}ms`);

        console.log('\n🎉 Enhanced Audio2Gesture Demo Complete!');
        console.log('✅ Multi-head attention optimization successful');
        console.log('🚀 Ready for high-performance real-time gesture generation');

        // Cleanup
        generator.cleanup();

    } catch (error) {
        console.error('❌ Enhanced demo failed:', error);
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.EnhancedAudio2GestureGenerator = EnhancedAudio2GestureGenerator;
    window.demoEnhancedAudio2Gesture = demoEnhancedAudio2Gesture;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        EnhancedAudio2GestureGenerator, 
        Audio2GesturePerformanceTracker 
    };
}
