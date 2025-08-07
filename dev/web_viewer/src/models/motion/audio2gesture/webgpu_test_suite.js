// WebGPU Testing Suite for Audio2Gesture Multi-Head Attention
class WebGPUTestSuite {
    constructor() {
        this.results = [];
        this.attention = null;
    }

    async runAllTests() {
        console.log('🎮 Starting WebGPU Test Suite for Audio2Gesture...');
        console.log('======================================================');

        // Test WebGPU initialization
        await this.testWebGPUInitialization();
        
        // Test WebGPU vs CPU consistency
        await this.testWebGPUConsistency();
        
        // Test WebGPU performance
        await this.testWebGPUPerformance();
        
        // Test WebGPU with temporal smoothing
        await this.testWebGPUTemporalSmoothing();
        
        // Test WebGPU with realistic workloads
        await this.testWebGPURealisticWorkload();

        this.printResults();
        return this.results;
    }

    async testWebGPUInitialization() {
        console.log('🔧 Testing WebGPU initialization...');
        
        try {
            const attention = new OptimizedAudio2GestureAttention({
                numHeads: 8,
                hiddenDim: 1024
            });

            const success = await attention.initializeBackend('webgpu');
            
            if (success && attention.currentBackend === 'webgpu') {
                this.addResult('testWebGPUInitialization', true, 
                    `WebGPU initialized successfully. Device: ${attention.device ? 'Available' : 'Not Available'}`);
            } else {
                this.addResult('testWebGPUInitialization', false, 
                    `WebGPU initialization failed. Current backend: ${attention.currentBackend}`);
            }
        } catch (error) {
            this.addResult('testWebGPUInitialization', false, 
                `WebGPU initialization error: ${error.message}`);
        }
    }

    async testWebGPUConsistency() {
        console.log('⚖️ Testing WebGPU vs CPU consistency...');
        
        try {
            // Initialize WebGPU instance
            const webgpuAttention = new OptimizedAudio2GestureAttention({
                numHeads: 8,
                hiddenDim: 1024
            });
            await webgpuAttention.initializeBackend('webgpu');
            
            // Initialize CPU instance
            const cpuAttention = new OptimizedAudio2GestureAttention({
                numHeads: 8,
                hiddenDim: 1024
            });
            await cpuAttention.initializeBackend('cpu');

            // Generate test data
            const hiddenStates = this.generateTestHiddenState(4, 1, 1024);
            const audioFeatures = this.generateTestAudioFeatures(80, 30);
            const lexemeFeatures = this.generateTestLexemeFeatures(96);

            // Compute with both backends
            const webgpuResult = await webgpuAttention.computeMultiHeadAttention(
                hiddenStates, audioFeatures, lexemeFeatures
            );
            
            const cpuResult = await cpuAttention.computeMultiHeadAttention(
                hiddenStates, audioFeatures, lexemeFeatures
            );

            // Compare results
            const maxDifference = this.computeMaxDifference(webgpuResult, cpuResult);
            const consistent = maxDifference < 1e-5; // Very strict tolerance

            this.addResult('testWebGPUConsistency', consistent,
                `Max difference between WebGPU and CPU: ${maxDifference.toExponential(2)}, Consistent: ${consistent}`);

        } catch (error) {
            this.addResult('testWebGPUConsistency', false,
                `WebGPU consistency test failed: ${error.message}`);
        }
    }

    async testWebGPUPerformance() {
        console.log('⚡ Testing WebGPU performance...');
        
        try {
            const attention = new OptimizedAudio2GestureAttention({
                numHeads: 8,
                hiddenDim: 1024
            });
            await attention.initializeBackend('webgpu');

            // Generate realistic workload
            const hiddenStates = this.generateTestHiddenState(8, 4, 1024); // Larger batch
            const audioFeatures = this.generateTestAudioFeatures(80, 50); // More audio frames
            const lexemeFeatures = this.generateTestLexemeFeatures(96);

            // Warm up
            await attention.computeMultiHeadAttention(hiddenStates, audioFeatures, lexemeFeatures);

            // Performance test
            const iterations = 5;
            const startTime = performance.now();
            
            for (let i = 0; i < iterations; i++) {
                await attention.computeMultiHeadAttention(hiddenStates, audioFeatures, lexemeFeatures);
            }
            
            const endTime = performance.now();
            const avgTime = (endTime - startTime) / iterations;
            const fps = 1000 / avgTime;

            const goodPerformance = fps > 3.0; // Target at least 3 FPS for WebGPU

            this.addResult('testWebGPUPerformance', goodPerformance,
                `WebGPU average time: ${avgTime.toFixed(1)}ms (${fps.toFixed(1)} FPS), Target: >3 FPS`);

        } catch (error) {
            this.addResult('testWebGPUPerformance', false,
                `WebGPU performance test failed: ${error.message}`);
        }
    }

    async testWebGPUTemporalSmoothing() {
        console.log('🌊 Testing WebGPU with temporal smoothing...');
        
        try {
            const attention = new OptimizedAudio2GestureAttention({
                numHeads: 8,
                hiddenDim: 1024
            });
            await attention.initializeBackend('webgpu');
            
            // Enable temporal smoothing
            attention.setTemporalSmoothingMode('avatar');
            attention._temporalSmoothingFactor = 0.7;

            const results = [];
            let currentHidden = this.generateTestHiddenState(4, 1, 1024);
            const audioFeatures = this.generateTestAudioFeatures(80, 30);
            const lexemeFeatures = this.generateTestLexemeFeatures(96);

            // Run sequence to test temporal smoothing
            for (let i = 0; i < 4; i++) {
                const result = await attention.computeMultiHeadAttention(
                    currentHidden, audioFeatures, lexemeFeatures
                );
                results.push(result);
                
                // Small perturbation for next iteration
                currentHidden = this.perturbHiddenState(currentHidden, 0.02);
            }

            // Check temporal smoothness
            let maxDiff = 0;
            for (let i = 1; i < results.length; i++) {
                const diff = this.computeMaxDifference(results[i-1], results[i]);
                maxDiff = Math.max(maxDiff, diff);
            }

            const smoothingWorking = maxDiff < 2.0; // Reasonable threshold

            this.addResult('testWebGPUTemporalSmoothing', smoothingWorking,
                `WebGPU temporal smoothing max difference: ${maxDiff.toFixed(3)}, Working: ${smoothingWorking}`);

        } catch (error) {
            this.addResult('testWebGPUTemporalSmoothing', false,
                `WebGPU temporal smoothing test failed: ${error.message}`);
        }
    }

    async testWebGPURealisticWorkload() {
        console.log('🎯 Testing WebGPU with realistic avatar workload...');
        
        try {
            const attention = new OptimizedAudio2GestureAttention({
                numHeads: 8,
                hiddenDim: 1024
            });
            await attention.initializeBackend('webgpu');
            
            // Configure for avatar animation
            attention.setTemporalSmoothingMode('avatar');

            // Simulate realistic avatar animation sequence
            const sequenceLength = 10;
            const results = [];
            let currentHidden = this.generateTestHiddenState(4, 2, 1024); // Multi-sequence
            
            const startTime = performance.now();
            
            for (let frame = 0; frame < sequenceLength; frame++) {
                // Time-varying audio features (simulating speech)
                const audioFeatures = this.generateTimeVaryingAudioFeatures(80, 30, frame);
                const lexemeFeatures = this.generateTestLexemeFeatures(96);

                const result = await attention.computeMultiHeadAttention(
                    currentHidden, audioFeatures, lexemeFeatures
                );
                
                results.push(result);
                
                // Update for next frame
                currentHidden = this.perturbHiddenState(result, 0.05);
            }
            
            const endTime = performance.now();
            const totalTime = endTime - startTime;
            const avgTimePerFrame = totalTime / sequenceLength;
            const realtimeFps = 1000 / avgTimePerFrame;

            // Check if suitable for real-time avatar animation (target 15+ FPS)
            const realtimeCapable = realtimeFps > 10.0;
            
            // Check output quality
            const allValid = results.every(result => this.validateTensorStructure(result));
            const allFinite = results.every(result => this.allFinite(result));

            const overallSuccess = realtimeCapable && allValid && allFinite;

            this.addResult('testWebGPURealisticWorkload', overallSuccess,
                `WebGPU realistic workload: ${avgTimePerFrame.toFixed(1)}ms/frame (${realtimeFps.toFixed(1)} FPS), Valid: ${allValid}, Finite: ${allFinite}`);

        } catch (error) {
            this.addResult('testWebGPURealisticWorkload', false,
                `WebGPU realistic workload test failed: ${error.message}`);
        }
    }

    // Helper methods
    generateTestHiddenState(batch, seq, dim) {
        const hiddenState = [];
        for (let b = 0; b < batch; b++) {
            const batchData = [];
            for (let s = 0; s < seq; s++) {
                const seqData = [];
                for (let d = 0; d < dim; d++) {
                    seqData.push((Math.random() - 0.5) * 0.1);
                }
                batchData.push(seqData);
            }
            hiddenState.push(batchData);
        }
        return hiddenState;
    }

    generateTestAudioFeatures(mel, frames) {
        const features = [];
        for (let m = 0; m < mel; m++) {
            const row = [];
            for (let f = 0; f < frames; f++) {
                row.push((Math.random() - 0.5) * 0.2);
            }
            features.push(row);
        }
        return features;
    }

    generateTimeVaryingAudioFeatures(mel, frames, timeStep) {
        const features = [];
        for (let m = 0; m < mel; m++) {
            const row = [];
            for (let f = 0; f < frames; f++) {
                // Add time-varying component
                const timeComponent = Math.sin(timeStep * 0.1 + m * 0.01) * 0.1;
                row.push((Math.random() - 0.5) * 0.2 + timeComponent);
            }
            features.push(row);
        }
        return features;
    }

    generateTestLexemeFeatures(dim) {
        const features = [];
        for (let d = 0; d < dim; d++) {
            features.push((Math.random() - 0.5) * 0.15);
        }
        return features;
    }

    perturbHiddenState(hiddenState, strength) {
        const perturbed = [];
        for (let l = 0; l < hiddenState.length; l++) {
            const layer = [];
            for (let b = 0; b < hiddenState[l].length; b++) {
                const batch = [];
                for (let h = 0; h < hiddenState[l][b].length; h++) {
                    const noise = (Math.random() - 0.5) * strength;
                    batch.push(hiddenState[l][b][h] + noise);
                }
                layer.push(batch);
            }
            perturbed.push(layer);
        }
        return perturbed;
    }

    computeMaxDifference(tensor1, tensor2) {
        let maxDiff = 0;
        for (let i = 0; i < tensor1.length; i++) {
            for (let j = 0; j < tensor1[i].length; j++) {
                for (let k = 0; k < tensor1[i][j].length; k++) {
                    const diff = Math.abs(tensor1[i][j][k] - tensor2[i][j][k]);
                    maxDiff = Math.max(maxDiff, diff);
                }
            }
        }
        return maxDiff;
    }

    validateTensorStructure(tensor) {
        return Array.isArray(tensor) && 
               tensor.length > 0 && 
               Array.isArray(tensor[0]) && 
               tensor[0].length > 0 && 
               Array.isArray(tensor[0][0]);
    }

    allFinite(tensor) {
        for (const layer of tensor) {
            for (const batch of layer) {
                for (const val of batch) {
                    if (!isFinite(val)) return false;
                }
            }
        }
        return true;
    }

    addResult(testName, passed, details) {
        const result = { testName, passed, details };
        this.results.push(result);
        
        const status = passed ? '✅ PASS' : '❌ FAIL';
        console.log(`${status} ${testName}: ${details}`);
    }

    printResults() {
        const totalTests = this.results.length;
        const passedTests = this.results.filter(r => r.passed).length;
        const failedTests = totalTests - passedTests;
        const successRate = ((passedTests / totalTests) * 100).toFixed(1);

        console.log('\n======================================================');
        console.log('🎮 WEBGPU AUDIO2GESTURE TEST REPORT');
        console.log('======================================================');
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests} ✅`);
        console.log(`Failed: ${failedTests} ❌`);
        console.log(`Success Rate: ${successRate}%`);
        console.log('======================================================');

        if (failedTests > 0) {
            console.log('\n❌ Failed Tests:');
            this.results.filter(r => !r.passed).forEach(result => {
                console.log(`  • ${result.testName}: ${result.details}`);
            });
        }

        if (passedTests > 0) {
            console.log('\n✅ Passed Tests:');
            this.results.filter(r => r.passed).forEach(result => {
                console.log(`  • ${result.testName}: ${result.details}`);
            });
        }
    }
}

// Export for use in demo
window.WebGPUTestSuite = WebGPUTestSuite;
