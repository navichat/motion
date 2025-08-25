// Batch Audio Processor for High-FPS Audio2Gesture Generation
// Specialized for processing audio sequences in batches to maximize throughput

class BatchAudioProcessor {
    constructor(config = {}) {
        this.config = {
            batchSize: config.batchSize || 8,
            audioFrameSize: config.audioFrameSize || 2400, // 80 * 30
            mfccSize: config.mfccSize || 13,
            contextWindow: config.contextWindow || 3,
            enableCaching: config.enableCaching !== false,
            maxCacheSize: config.maxCacheSize || 100
        };
        
        this.audioCache = new Map();
        this.featureCache = new Map();
        this.processingStats = {
            totalBatches: 0,
            totalFrames: 0,
            totalTime: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
    }

    /**
     * Process multiple audio sequences in a single batch operation
     * This is the key optimization for high FPS audio2gesture
     */
    async processBatchAudioSequences(audioSequences, options = {}) {
        const batchStart = performance.now();
        const batchSize = audioSequences.length;
        
        console.log(`🎵 Processing batch of ${batchSize} audio sequences...`);
        
        try {
            // Pre-process all audio sequences for temporal features
            const temporalFeatures = await this._extractBatchTemporalFeatures(audioSequences);
            
            // Apply batch-level audio enhancements
            const enhancedFeatures = await this._applyBatchAudioEnhancements(temporalFeatures, options);
            
            // Generate audio-driven attention context
            const attentionContext = await this._generateBatchAttentionContext(enhancedFeatures, options);
            
            const batchTime = performance.now() - batchStart;
            this._updateProcessingStats(batchSize, batchTime);
            
            console.log(`✅ Batch audio processing complete: ${batchSize} sequences in ${batchTime.toFixed(2)}ms (${(batchTime/batchSize).toFixed(2)}ms per sequence)`);
            
            return {
                temporalFeatures,
                enhancedFeatures,
                attentionContext,
                metadata: {
                    batchSize,
                    processingTime: batchTime,
                    avgTimePerSequence: batchTime / batchSize,
                    cacheHitRate: this.processingStats.cacheHits / (this.processingStats.cacheHits + this.processingStats.cacheMisses)
                }
            };
            
        } catch (error) {
            console.error('❌ Batch audio processing failed:', error);
            throw error;
        }
    }

    /**
     * Extract temporal features from batch of audio sequences
     */
    async _extractBatchTemporalFeatures(audioSequences) {
        const batchFeatures = [];
        
        // Process sequences in parallel for maximum throughput
        const featurePromises = audioSequences.map(async (sequence, sequenceIndex) => {
            const cacheKey = this._generateSequenceCacheKey(sequence);
            
            if (this.config.enableCaching && this.featureCache.has(cacheKey)) {
                this.processingStats.cacheHits++;
                return this.featureCache.get(cacheKey);
            }
            
            this.processingStats.cacheMisses++;
            
            const sequenceFeatures = [];
            
            for (let frameIndex = 0; frameIndex < sequence.length; frameIndex++) {
                const audioFrame = sequence[frameIndex];
                
                // Extract comprehensive audio features for this frame
                const frameFeatures = {
                    // Basic audio properties
                    energy: this._computeAudioEnergy(audioFrame),
                    spectralCentroid: this._computeSpectralCentroid(audioFrame),
                    spectralBandwidth: this._computeSpectralBandwidth(audioFrame),
                    spectralRolloff: this._computeSpectralRolloff(audioFrame),
                    zeroCrossingRate: this._computeZeroCrossingRate(audioFrame),
                    
                    // Advanced features
                    mfccFeatures: this._extractMFCCFeatures(audioFrame),
                    spectralContrast: this._computeSpectralContrast(audioFrame),
                    chromaFeatures: this._extractChromaFeatures(audioFrame),
                    
                    // Temporal context
                    temporalPosition: frameIndex / sequence.length,
                    frameIndex: frameIndex,
                    sequenceIndex: sequenceIndex,
                    
                    // Raw audio sample (reduced for efficiency)
                    audioSample: this._downsampleAudio(audioFrame, 80)
                };
                
                // Add temporal context from neighboring frames
                frameFeatures.temporalContext = this._extractTemporalContext(
                    sequence, frameIndex, this.config.contextWindow
                );
                
                sequenceFeatures.push(frameFeatures);
            }
            
            // Cache the computed features
            if (this.config.enableCaching && this.featureCache.size < this.config.maxCacheSize) {
                this.featureCache.set(cacheKey, sequenceFeatures);
            }
            
            return sequenceFeatures;
        });
        
        return await Promise.all(featurePromises);
    }

    /**
     * Apply batch-level audio enhancements for better gesture generation
     */
    async _applyBatchAudioEnhancements(batchTemporalFeatures, options) {
        const enhanced = [];
        
        // Normalize features across the entire batch for consistency
        const batchStats = this._computeBatchStatistics(batchTemporalFeatures);
        
        for (const sequenceFeatures of batchTemporalFeatures) {
            const enhancedSequence = [];
            
            for (const frameFeatures of sequenceFeatures) {
                const enhanced_frame = {
                    ...frameFeatures,
                    
                    // Normalized features using batch statistics
                    normalizedEnergy: (frameFeatures.energy - batchStats.energy.mean) / batchStats.energy.std,
                    normalizedSpectralCentroid: (frameFeatures.spectralCentroid - batchStats.spectralCentroid.mean) / batchStats.spectralCentroid.std,
                    
                    // Enhanced MFCC features with batch normalization
                    enhancedMFCC: frameFeatures.mfccFeatures.map((mfcc, i) => {
                        const stat = batchStats.mfcc[i];
                        return (mfcc - stat.mean) / stat.std;
                    }),
                    
                    // Dynamic range compression for better attention
                    compressedFeatures: this._applyDynamicRangeCompression(frameFeatures),
                    
                    // Perceptual weighting based on human auditory system
                    perceptualWeights: this._computePerceptualWeights(frameFeatures)
                };
                
                enhancedSequence.push(enhanced_frame);
            }
            
            enhanced.push(enhancedSequence);
        }
        
        return enhanced;
    }

    /**
     * Generate attention context optimized for audio-driven animation
     */
    async _generateBatchAttentionContext(enhancedFeatures, options) {
        const batchContext = [];
        
        for (const sequenceFeatures of enhancedFeatures) {
            const sequenceContext = [];
            
            for (let i = 0; i < sequenceFeatures.length; i++) {
                const frameFeatures = sequenceFeatures[i];
                
                // Create rich attention context vector
                const contextVector = [
                    // Audio-driven motion cues
                    frameFeatures.normalizedEnergy * 2.0,        // Strong influence on motion intensity
                    frameFeatures.normalizedSpectralCentroid,    // Affects gesture precision
                    frameFeatures.spectralBandwidth * 0.5,      // Influences motion smoothness
                    frameFeatures.zeroCrossingRate,              // Affects gesture frequency
                    
                    // Enhanced MFCC features (first 8 coefficients)
                    ...frameFeatures.enhancedMFCC.slice(0, 8),
                    
                    // Temporal positioning and context
                    frameFeatures.temporalPosition,
                    Math.sin(frameFeatures.temporalPosition * 2 * Math.PI), // Cyclical encoding
                    Math.cos(frameFeatures.temporalPosition * 2 * Math.PI),
                    
                    // Perceptual importance weights
                    ...frameFeatures.perceptualWeights.slice(0, 5),
                    
                    // Inter-frame motion prediction
                    ...this._predictMotionCues(sequenceFeatures, i),
                    
                    // Reduced raw audio for direct neural processing
                    ...frameFeatures.audioSample.slice(0, 16)
                ];
                
                sequenceContext.push(contextVector);
            }
            
            batchContext.push(sequenceContext);
        }
        
        return batchContext;
    }

    /**
     * Audio feature extraction methods
     */
    _computeAudioEnergy(audioFrame) {
        let energy = 0;
        for (let i = 0; i < audioFrame.length; i++) {
            energy += audioFrame[i] * audioFrame[i];
        }
        return Math.sqrt(energy / audioFrame.length);
    }

    _computeSpectralCentroid(audioFrame) {
        let weightedSum = 0;
        let magnitudeSum = 0;
        
        for (let i = 0; i < audioFrame.length; i++) {
            const magnitude = Math.abs(audioFrame[i]);
            weightedSum += i * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    }

    _computeSpectralBandwidth(audioFrame) {
        const centroid = this._computeSpectralCentroid(audioFrame);
        let bandwidth = 0;
        let magnitudeSum = 0;
        
        for (let i = 0; i < audioFrame.length; i++) {
            const magnitude = Math.abs(audioFrame[i]);
            bandwidth += Math.pow(i - centroid, 2) * magnitude;
            magnitudeSum += magnitude;
        }
        
        return magnitudeSum > 0 ? Math.sqrt(bandwidth / magnitudeSum) : 0;
    }

    _computeSpectralRolloff(audioFrame, threshold = 0.85) {
        const magnitudes = audioFrame.map(x => Math.abs(x));
        const totalEnergy = magnitudes.reduce((sum, mag) => sum + mag * mag, 0);
        const thresholdEnergy = totalEnergy * threshold;
        
        let runningEnergy = 0;
        for (let i = 0; i < magnitudes.length; i++) {
            runningEnergy += magnitudes[i] * magnitudes[i];
            if (runningEnergy >= thresholdEnergy) {
                return i / magnitudes.length;
            }
        }
        
        return 1.0;
    }

    _computeZeroCrossingRate(audioFrame) {
        let crossings = 0;
        for (let i = 1; i < audioFrame.length; i++) {
            if ((audioFrame[i] >= 0) !== (audioFrame[i-1] >= 0)) {
                crossings++;
            }
        }
        return crossings / (audioFrame.length - 1);
    }

    _extractMFCCFeatures(audioFrame) {
        const features = new Array(this.config.mfccSize);
        const windowSize = Math.min(audioFrame.length, 512);
        
        for (let i = 0; i < this.config.mfccSize; i++) {
            let sum = 0;
            for (let j = 0; j < windowSize; j++) {
                const freq = (i + 1) * j / windowSize;
                sum += audioFrame[j] * Math.cos(2 * Math.PI * freq);
            }
            features[i] = sum / windowSize;
        }
        
        return features;
    }

    _computeSpectralContrast(audioFrame) {
        const contrast = [];
        const numBands = 6;
        const bandSize = Math.floor(audioFrame.length / numBands);
        
        for (let band = 0; band < numBands; band++) {
            const start = band * bandSize;
            const end = Math.min(start + bandSize, audioFrame.length);
            
            let maxMag = 0;
            let minMag = Infinity;
            
            for (let i = start; i < end; i++) {
                const mag = Math.abs(audioFrame[i]);
                maxMag = Math.max(maxMag, mag);
                minMag = Math.min(minMag, mag);
            }
            
            contrast.push(minMag > 0 ? Math.log10(maxMag / minMag) : 0);
        }
        
        return contrast;
    }

    _extractChromaFeatures(audioFrame) {
        const chromaSize = 12; // 12 semitones
        const chroma = new Array(chromaSize).fill(0);
        
        for (let i = 0; i < audioFrame.length; i++) {
            const chromaIndex = i % chromaSize;
            chroma[chromaIndex] += Math.abs(audioFrame[i]);
        }
        
        // Normalize
        const sum = chroma.reduce((a, b) => a + b, 0);
        return sum > 0 ? chroma.map(c => c / sum) : chroma;
    }

    _downsampleAudio(audioFrame, targetSize) {
        if (audioFrame.length <= targetSize) {
            return Array.from(audioFrame);
        }
        
        const ratio = audioFrame.length / targetSize;
        const downsampled = [];
        
        for (let i = 0; i < targetSize; i++) {
            const sourceIndex = Math.floor(i * ratio);
            downsampled.push(audioFrame[sourceIndex]);
        }
        
        return downsampled;
    }

    _extractTemporalContext(sequence, frameIndex, contextWindow) {
        const context = {
            pastFrames: [],
            futureFrames: [],
            velocities: [],
            accelerations: []
        };
        
        // Extract past and future frame features
        for (let offset = -contextWindow; offset <= contextWindow; offset++) {
            const index = frameIndex + offset;
            if (index >= 0 && index < sequence.length && index !== frameIndex) {
                const contextFrame = sequence[index];
                if (offset < 0) {
                    context.pastFrames.push(this._computeAudioEnergy(contextFrame));
                } else {
                    context.futureFrames.push(this._computeAudioEnergy(contextFrame));
                }
            }
        }
        
        // Compute velocities and accelerations for motion prediction
        if (frameIndex > 0) {
            const currentEnergy = this._computeAudioEnergy(sequence[frameIndex]);
            const pastEnergy = this._computeAudioEnergy(sequence[frameIndex - 1]);
            context.velocities.push(currentEnergy - pastEnergy);
            
            if (frameIndex > 1) {
                const pastPastEnergy = this._computeAudioEnergy(sequence[frameIndex - 2]);
                const pastVelocity = pastEnergy - pastPastEnergy;
                const currentVelocity = currentEnergy - pastEnergy;
                context.accelerations.push(currentVelocity - pastVelocity);
            }
        }
        
        return context;
    }

    _computeBatchStatistics(batchTemporalFeatures) {
        const allFeatures = batchTemporalFeatures.flat();
        
        const stats = {
            energy: this._computeStatistics(allFeatures.map(f => f.energy)),
            spectralCentroid: this._computeStatistics(allFeatures.map(f => f.spectralCentroid)),
            mfcc: []
        };
        
        // Compute statistics for each MFCC coefficient
        for (let i = 0; i < this.config.mfccSize; i++) {
            stats.mfcc.push(this._computeStatistics(allFeatures.map(f => f.mfccFeatures[i])));
        }
        
        return stats;
    }

    _computeStatistics(values) {
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        
        return { mean, std, variance };
    }

    _applyDynamicRangeCompression(frameFeatures) {
        const threshold = 0.7;
        const ratio = 4.0;
        const makeupGain = 1.2;
        
        const compressed = {
            energy: this._compressValue(frameFeatures.energy, threshold, ratio) * makeupGain,
            spectralCentroid: this._compressValue(frameFeatures.spectralCentroid, threshold, ratio) * makeupGain
        };
        
        return compressed;
    }

    _compressValue(value, threshold, ratio) {
        const normalizedValue = Math.abs(value);
        if (normalizedValue <= threshold) {
            return value;
        }
        
        const excessAmount = normalizedValue - threshold;
        const compressedExcess = excessAmount / ratio;
        const compressedValue = threshold + compressedExcess;
        
        return value >= 0 ? compressedValue : -compressedValue;
    }

    _computePerceptualWeights(frameFeatures) {
        // Weight features based on perceptual importance for gesture generation
        const weights = [
            Math.pow(frameFeatures.energy, 0.6),        // Energy is highly important but non-linear
            frameFeatures.spectralCentroid * 0.8,       // Spectral centroid moderately important
            frameFeatures.zeroCrossingRate * 0.4,       // ZCR less important
            Math.sqrt(frameFeatures.spectralBandwidth), // Bandwidth with square root scaling
            frameFeatures.spectralRolloff * 0.6         // Rolloff moderately important
        ];
        
        return weights;
    }

    _predictMotionCues(sequenceFeatures, currentIndex) {
        const cues = [];
        
        if (currentIndex > 0) {
            const current = sequenceFeatures[currentIndex];
            const previous = sequenceFeatures[currentIndex - 1];
            
            // Energy trend
            const energyTrend = current.energy - previous.energy;
            cues.push(energyTrend);
            
            // Spectral movement
            const spectralMovement = current.spectralCentroid - previous.spectralCentroid;
            cues.push(spectralMovement);
            
            // MFCC movement (first 3 coefficients)
            for (let i = 0; i < 3; i++) {
                const mfccMovement = current.mfccFeatures[i] - previous.mfccFeatures[i];
                cues.push(mfccMovement);
            }
        } else {
            // Fill with zeros for first frame
            cues.push(...new Array(5).fill(0));
        }
        
        return cues;
    }

    _generateSequenceCacheKey(sequence) {
        // Generate a simple hash of the sequence for caching
        let hash = 0;
        const samplePoints = Math.min(10, sequence.length);
        const step = Math.floor(sequence.length / samplePoints);
        
        for (let i = 0; i < samplePoints; i++) {
            const frameIndex = i * step;
            const frame = sequence[frameIndex];
            const frameSum = frame.reduce((sum, val) => sum + val, 0);
            hash = ((hash << 5) - hash + frameSum) & 0xffffffff;
        }
        
        return hash.toString(36);
    }

    _updateProcessingStats(batchSize, processingTime) {
        this.processingStats.totalBatches++;
        this.processingStats.totalFrames += batchSize;
        this.processingStats.totalTime += processingTime;
    }

    getProcessingStats() {
        const avgBatchTime = this.processingStats.totalTime / this.processingStats.totalBatches;
        const avgFrameTime = this.processingStats.totalTime / this.processingStats.totalFrames;
        const batchesPerSecond = 1000 / avgBatchTime;
        const framesPerSecond = 1000 / avgFrameTime;
        
        return {
            ...this.processingStats,
            avgBatchTime,
            avgFrameTime,
            batchesPerSecond,
            framesPerSecond,
            cacheHitRate: this.processingStats.cacheHits / (this.processingStats.cacheHits + this.processingStats.cacheMisses)
        };
    }

    clearCache() {
        this.audioCache.clear();
        this.featureCache.clear();
        console.log('🧹 Audio processing caches cleared');
    }

    clearStats() {
        this.processingStats = {
            totalBatches: 0,
            totalFrames: 0,
            totalTime: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.BatchAudioProcessor = BatchAudioProcessor;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = BatchAudioProcessor;
}
