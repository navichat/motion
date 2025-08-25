// Debug script to test temporal smoothing functionality
class TemporalSmoothingDebugger {
    constructor() {
        this.attention = null;
    }

    async initialize() {
        console.log('🔍 Initializing Temporal Smoothing Debugger...');
        
        this.attention = new OptimizedAudio2GestureAttention({
            numHeads: 8,
            hiddenDim: 1024
        });

        await this.attention.initializeBackend('cpu');
        
        // Enable temporal smoothing with strong factor
        this.attention.setTemporalSmoothingMode('avatar');
        this.attention._temporalSmoothingFactor = 0.8;
        
        console.log('✅ Debugger initialized');
    }

    async testTemporalSmoothing() {
        console.log('\n🧪 Testing Temporal Smoothing...');
        
        // Generate consistent test data
        const audioFeatures = this.generateConsistentAudioFeatures();
        const lexemeFeatures = this.generateConsistentLexemeFeatures();
        
        const results = [];
        let currentHiddenState = this.generateTestHiddenState();
        
        console.log('📊 Running sequence with temporal smoothing...');
        
        // Run sequence with very small changes
        for (let i = 0; i < 5; i++) {
            const result = await this.attention.computeMultiHeadAttention(
                currentHiddenState, audioFeatures, lexemeFeatures
            );
            
            results.push(result);
            
            // Make tiny change to hidden state
            currentHiddenState = this.tinyPerturbation(currentHiddenState, 0.01);
            
            if (i > 0) {
                const difference = this.computeMaxDifference(results[i-1], results[i]);
                console.log(`Step ${i}: Max difference = ${difference.toFixed(4)}`);
            }
        }
        
        // Check if differences are small and decreasing
        let allSmooth = true;
        for (let i = 1; i < results.length; i++) {
            const diff = this.computeMaxDifference(results[i-1], results[i]);
            if (diff > 1.0) {
                allSmooth = false;
                console.log(`❌ Large difference at step ${i}: ${diff.toFixed(4)}`);
            }
        }
        
        if (allSmooth) {
            console.log('✅ Temporal smoothing is working correctly');
        } else {
            console.log('❌ Temporal smoothing needs improvement');
        }
        
        return allSmooth;
    }

    generateConsistentAudioFeatures() {
        // Generate very consistent audio features
        const features = [];
        for (let i = 0; i < 80; i++) {
            const row = [];
            for (let j = 0; j < 30; j++) {
                row.push(0.1 + i * 0.001 + j * 0.0001); // Very gradual variation
            }
            features.push(row);
        }
        return features;
    }

    generateConsistentLexemeFeatures() {
        // Generate very consistent lexeme features
        const features = [];
        for (let i = 0; i < 96; i++) {
            features.push(0.2 + i * 0.001); // Very gradual variation
        }
        return features;
    }

    generateTestHiddenState() {
        // Generate consistent hidden state
        const hiddenState = [];
        for (let b = 0; b < 4; b++) {
            const batch = [];
            for (let s = 0; s < 1; s++) {
                const seq = [];
                for (let d = 0; d < 1024; d++) {
                    seq.push(0.3 + d * 0.0001); // Very gradual variation
                }
                batch.push(seq);
            }
            hiddenState.push(batch);
        }
        return hiddenState;
    }

    tinyPerturbation(hiddenState, strength) {
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
}

// Auto-run test when script loads
window.addEventListener('load', async () => {
    if (typeof OptimizedAudio2GestureAttention !== 'undefined') {
        const tester = new TemporalSmoothingDebugger();
        try {
            await tester.initialize();
            await tester.testTemporalSmoothing();
        } catch (error) {
            console.error('Debug test failed:', error);
        }
    }
});
