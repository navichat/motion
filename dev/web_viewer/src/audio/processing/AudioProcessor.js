/**
 * AudioProcessor - Handles audio processing for VRM animation
 * Extracts features from audio for gesture and facial animation generation
 */

class AudioProcessor {
    constructor() {
        this.audioContext = null;
        this.initialized = false;
    }

    async initialize() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.initialized = true;
            console.log('✅ AudioProcessor initialized');
            return true;
        } catch (error) {
            console.error('❌ AudioProcessor initialization failed:', error);
            return false;
        }
    }

    /**
     * Extract audio features for gesture generation
     * @param {AudioBuffer} audioBuffer - The input audio buffer
     * @returns {Promise<Array>} - Extracted audio features
     */
    async extractFeatures(audioBuffer) {
        if (!this.initialized || !audioBuffer) {
            throw new Error('AudioProcessor not initialized or no audio buffer provided');
        }

        console.log('🎵 Extracting audio features...');
        
        try {
            // Get audio data
            const audioData = audioBuffer.getChannelData(0); // Get mono channel
            const sampleRate = audioBuffer.sampleRate;
            const duration = audioBuffer.duration;
            
            console.log(`Audio specs: ${duration.toFixed(2)}s, ${sampleRate}Hz, ${audioData.length} samples`);
            
            // Extract MFCC-like features
            const features = this.extractMFCCFeatures(audioData, sampleRate);
            
            // Extract rhythm features
            const rhythmFeatures = this.extractRhythmFeatures(audioData, sampleRate);
            
            // Extract energy features
            const energyFeatures = this.extractEnergyFeatures(audioData, sampleRate);
            
            // Combine features
            const combinedFeatures = this.combineFeatures(features, rhythmFeatures, energyFeatures);
            
            console.log(`✅ Extracted ${combinedFeatures.length} feature frames`);
            
            return combinedFeatures;
            
        } catch (error) {
            console.error('Feature extraction error:', error);
            throw error;
        }
    }

    /**
     * Extract MFCC-like features from audio
     */
    extractMFCCFeatures(audioData, sampleRate) {
        const frameSize = 1024;
        const hopSize = 512;
        const numFrames = Math.floor((audioData.length - frameSize) / hopSize) + 1;
        const numCoeffs = 13; // Number of MFCC coefficients
        
        const features = [];
        
        for (let i = 0; i < numFrames; i++) {
            const start = i * hopSize;
            const end = Math.min(start + frameSize, audioData.length);
            const frame = audioData.slice(start, end);
            
            // Apply window function (Hamming window)
            const windowed = this.applyHammingWindow(frame);
            
            // Compute FFT (simplified)
            const spectrum = this.computeSpectrum(windowed);
            
            // Extract mel-scale features
            const melFeatures = this.extractMelFeatures(spectrum, sampleRate);
            
            // Convert to MFCC-like coefficients
            const mfcc = this.computeMFCC(melFeatures, numCoeffs);
            
            features.push(mfcc);
        }
        
        return features;
    }

    /**
     * Extract rhythm and beat features
     */
    extractRhythmFeatures(audioData, sampleRate) {
        const frameSize = 2048;
        const hopSize = 1024;
        const numFrames = Math.floor((audioData.length - frameSize) / hopSize) + 1;
        
        const rhythmFeatures = [];
        
        for (let i = 0; i < numFrames; i++) {
            const start = i * hopSize;
            const end = Math.min(start + frameSize, audioData.length);
            const frame = audioData.slice(start, end);
            
            // Compute spectral centroid
            const spectralCentroid = this.computeSpectralCentroid(frame, sampleRate);
            
            // Compute zero crossing rate
            const zcr = this.computeZeroCrossingRate(frame);
            
            // Compute RMS energy
            const rms = this.computeRMS(frame);
            
            // Compute spectral rolloff
            const rolloff = this.computeSpectralRolloff(frame, sampleRate);
            
            rhythmFeatures.push([spectralCentroid, zcr, rms, rolloff]);
        }
        
        return rhythmFeatures;
    }

    /**
     * Extract energy and intensity features
     */
    extractEnergyFeatures(audioData, sampleRate) {
        const frameSize = 512;
        const hopSize = 256;
        const numFrames = Math.floor((audioData.length - frameSize) / hopSize) + 1;
        
        const energyFeatures = [];
        
        for (let i = 0; i < numFrames; i++) {
            const start = i * hopSize;
            const end = Math.min(start + frameSize, audioData.length);
            const frame = audioData.slice(start, end);
            
            // Compute short-time energy
            const energy = this.computeShortTimeEnergy(frame);
            
            // Compute spectral flux
            const flux = i > 0 ? this.computeSpectralFlux(frame, energyFeatures[i-1]) : 0;
            
            energyFeatures.push([energy, flux]);
        }
        
        return energyFeatures;
    }

    /**
     * Combine different feature types into a unified representation
     */
    combineFeatures(mfccFeatures, rhythmFeatures, energyFeatures) {
        const combinedFeatures = [];
        const maxLength = Math.max(mfccFeatures.length, rhythmFeatures.length, energyFeatures.length);
        
        for (let i = 0; i < maxLength; i++) {
            const mfcc = i < mfccFeatures.length ? mfccFeatures[i] : new Array(13).fill(0);
            const rhythm = i < rhythmFeatures.length ? rhythmFeatures[i] : new Array(4).fill(0);
            const energy = i < energyFeatures.length ? energyFeatures[i] : new Array(2).fill(0);
            
            // Combine all features into a single vector
            const combined = [...mfcc, ...rhythm, ...energy];
            combinedFeatures.push(combined);
        }
        
        return combinedFeatures;
    }

    // Utility functions for audio processing

    applyHammingWindow(frame) {
        const windowed = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const w = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (frame.length - 1));
            windowed[i] = frame[i] * w;
        }
        return windowed;
    }

    computeSpectrum(frame) {
        // Simplified FFT - in a real implementation, you'd use a proper FFT library
        const spectrum = new Float32Array(frame.length / 2);
        for (let i = 0; i < spectrum.length; i++) {
            let real = 0, imag = 0;
            for (let j = 0; j < frame.length; j++) {
                const angle = -2 * Math.PI * i * j / frame.length;
                real += frame[j] * Math.cos(angle);
                imag += frame[j] * Math.sin(angle);
            }
            spectrum[i] = Math.sqrt(real * real + imag * imag);
        }
        return spectrum;
    }

    extractMelFeatures(spectrum, sampleRate) {
        // Simplified mel-scale conversion
        const numMelBands = 26;
        const melFeatures = new Float32Array(numMelBands);
        
        for (let i = 0; i < numMelBands; i++) {
            const startBin = Math.floor(i * spectrum.length / numMelBands);
            const endBin = Math.floor((i + 1) * spectrum.length / numMelBands);
            
            let sum = 0;
            for (let j = startBin; j < endBin; j++) {
                sum += spectrum[j];
            }
            melFeatures[i] = sum / (endBin - startBin);
        }
        
        return melFeatures;
    }

    computeMFCC(melFeatures, numCoeffs) {
        // Simplified DCT for MFCC computation
        const mfcc = new Float32Array(numCoeffs);
        
        for (let i = 0; i < numCoeffs; i++) {
            let sum = 0;
            for (let j = 0; j < melFeatures.length; j++) {
                sum += Math.log(melFeatures[j] + 1e-10) * Math.cos(Math.PI * i * (2 * j + 1) / (2 * melFeatures.length));
            }
            mfcc[i] = sum;
        }
        
        return mfcc;
    }

    computeSpectralCentroid(frame, sampleRate) {
        const spectrum = this.computeSpectrum(frame);
        let numerator = 0, denominator = 0;
        
        for (let i = 0; i < spectrum.length; i++) {
            const freq = i * sampleRate / (2 * spectrum.length);
            numerator += freq * spectrum[i];
            denominator += spectrum[i];
        }
        
        return denominator > 0 ? numerator / denominator : 0;
    }

    computeZeroCrossingRate(frame) {
        let crossings = 0;
        for (let i = 1; i < frame.length; i++) {
            if ((frame[i] >= 0 && frame[i-1] < 0) || (frame[i] < 0 && frame[i-1] >= 0)) {
                crossings++;
            }
        }
        return crossings / frame.length;
    }

    computeRMS(frame) {
        let sum = 0;
        for (let i = 0; i < frame.length; i++) {
            sum += frame[i] * frame[i];
        }
        return Math.sqrt(sum / frame.length);
    }

    computeSpectralRolloff(frame, sampleRate) {
        const spectrum = this.computeSpectrum(frame);
        const totalEnergy = spectrum.reduce((sum, val) => sum + val, 0);
        const threshold = 0.85 * totalEnergy;
        
        let cumulativeEnergy = 0;
        for (let i = 0; i < spectrum.length; i++) {
            cumulativeEnergy += spectrum[i];
            if (cumulativeEnergy >= threshold) {
                return i * sampleRate / (2 * spectrum.length);
            }
        }
        
        return sampleRate / 2; // Nyquist frequency
    }

    computeShortTimeEnergy(frame) {
        let energy = 0;
        for (let i = 0; i < frame.length; i++) {
            energy += frame[i] * frame[i];
        }
        return energy / frame.length;
    }

    computeSpectralFlux(currentFrame, previousEnergy) {
        const currentEnergy = this.computeShortTimeEnergy(currentFrame);
        return Math.abs(currentEnergy - (Array.isArray(previousEnergy) ? previousEnergy[0] : previousEnergy));
    }

    /**
     * Convert audio buffer to format suitable for neural networks
     * @param {AudioBuffer} audioBuffer - Input audio buffer
     * @returns {Promise<Float32Array>} - Preprocessed audio data
     */
    async preprocessForNeuralNetwork(audioBuffer) {
        // Resample to 16kHz if needed
        const targetSampleRate = 16000;
        let audioData;
        
        if (audioBuffer.sampleRate !== targetSampleRate) {
            audioData = await this.resampleAudio(audioBuffer, targetSampleRate);
        } else {
            audioData = audioBuffer.getChannelData(0);
        }
        
        // Normalize audio
        const normalizedData = this.normalizeAudio(audioData);
        
        // Add pre-emphasis filter
        const preEmphasized = this.applyPreEmphasis(normalizedData);
        
        return preEmphasized;
    }

    async resampleAudio(audioBuffer, targetSampleRate) {
        // Create a new audio context with target sample rate
        const offlineContext = new OfflineAudioContext(1, audioBuffer.length * targetSampleRate / audioBuffer.sampleRate, targetSampleRate);
        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start(0);
        
        const resampledBuffer = await offlineContext.startRendering();
        return resampledBuffer.getChannelData(0);
    }

    normalizeAudio(audioData) {
        const maxVal = Math.max(...audioData.map(Math.abs));
        if (maxVal === 0) return audioData;
        
        const normalized = new Float32Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            normalized[i] = audioData[i] / maxVal;
        }
        return normalized;
    }

    applyPreEmphasis(audioData, alpha = 0.97) {
        const filtered = new Float32Array(audioData.length);
        filtered[0] = audioData[0];
        
        for (let i = 1; i < audioData.length; i++) {
            filtered[i] = audioData[i] - alpha * audioData[i - 1];
        }
        
        return filtered;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AudioProcessor };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.AudioProcessor = AudioProcessor;
}
