/**
 * Facial Expression System - Manages VRM character facial expressions
 * Responds to motion data and provides emotional animation
 */

class FacialExpressionSystem {
    constructor(vrmModel) {
        this.vrmModel = vrmModel;
        this.currentEmotion = 'neutral';
        this.emotionIntensity = 0.0;
        this.targetEmotion = 'neutral';
        this.targetIntensity = 0.0;
        this.blendSpeed = 0.05;
        this.motionAnalyzer = new MotionAnalyzer();
        
        this.expressions = {
            'neutral': { name: 'neutral', intensity: 0.0 },
            'happy': { name: 'happy', intensity: 0.0 },
            'sad': { name: 'sad', intensity: 0.0 },
            'angry': { name: 'angry', intensity: 0.0 },
            'surprised': { name: 'surprised', intensity: 0.0 },
            'excited': { name: 'happy', intensity: 0.0 },
            'focused': { name: 'neutral', intensity: 0.0 }
        };
        
        console.log('😊 FacialExpressionSystem initialized');
    }

    setEmotion(emotion, intensity = 1.0) {
        if (!this.vrmModel?.expressionManager) {
            console.warn('⚠️ VRM expression manager not available');
            return;
        }
        
        this.targetEmotion = emotion;
        this.targetIntensity = Math.max(0, Math.min(1, intensity));
        
        console.log(`🎭 Setting emotion: ${emotion} (${(intensity * 100).toFixed(1)}%)`);
    }

    updateFromMotion(frameData) {
        if (!frameData || frameData.length < 6) return;
        
        // Analyze motion characteristics
        const motionData = this.motionAnalyzer.analyzeFrame(frameData);
        
        // Map motion characteristics to emotions
        let targetEmotion = 'neutral';
        let targetIntensity = 0.2;
        
        if (motionData.energy > 0.7) {
            targetEmotion = 'excited';
            targetIntensity = Math.min(1.0, motionData.energy);
        } else if (motionData.energy > 0.4) {
            targetEmotion = 'happy';
            targetIntensity = Math.min(0.8, motionData.energy);
        } else if (motionData.energy < 0.1) {
            targetEmotion = 'sad';
            targetIntensity = 0.3;
        } else {
            targetEmotion = 'neutral';
            targetIntensity = 0.2;
        }
        
        // Check for sudden movements (surprise)
        if (motionData.suddenMovement > 0.5) {
            targetEmotion = 'surprised';
            targetIntensity = Math.min(0.9, motionData.suddenMovement);
        }
        
        this.setEmotion(targetEmotion, targetIntensity);
    }

    update() {
        if (!this.vrmModel?.expressionManager) return;
        
        // Smooth transition between emotions
        const emotionDiff = this.targetIntensity - this.emotionIntensity;
        if (Math.abs(emotionDiff) > 0.01) {
            this.emotionIntensity += emotionDiff * this.blendSpeed;
            
            // Reset all expressions
            this.resetAllExpressions();
            
            // Apply target expression
            this.applyExpression(this.targetEmotion, this.emotionIntensity);
            
            // Always maintain some neutral expression
            this.applyExpression('neutral', 0.1);
        }
        
        // Update current emotion
        if (this.targetEmotion !== this.currentEmotion) {
            this.currentEmotion = this.targetEmotion;
        }
    }

    resetAllExpressions() {
        if (!this.vrmModel?.expressionManager) return;
        
        const expressionNames = ['neutral', 'happy', 'sad', 'angry', 'surprised'];
        
        expressionNames.forEach(expr => {
            try {
                if (this.vrmModel.expressionManager.setValue) {
                    this.vrmModel.expressionManager.setValue(expr, 0);
                }
            } catch (error) {
                // Silently ignore missing expressions
            }
        });
    }

    applyExpression(emotion, intensity) {
        if (!this.vrmModel?.expressionManager) return;
        
        const expressionMap = {
            'neutral': 'neutral',
            'happy': 'happy',
            'excited': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'surprised': 'surprised',
            'focused': 'neutral'
        };
        
        const expressionName = expressionMap[emotion] || 'neutral';
        
        try {
            if (this.vrmModel.expressionManager.setValue) {
                this.vrmModel.expressionManager.setValue(expressionName, intensity);
            }
        } catch (error) {
            console.warn(`⚠️ Failed to set expression ${expressionName}:`, error);
        }
    }

    getCurrentEmotion() {
        return {
            emotion: this.currentEmotion,
            intensity: this.emotionIntensity
        };
    }

    setBlendSpeed(speed) {
        this.blendSpeed = Math.max(0.01, Math.min(1.0, speed));
    }

    // Manual expression control
    setHappiness(intensity) {
        this.setEmotion('happy', intensity);
    }

    setSadness(intensity) {
        this.setEmotion('sad', intensity);
    }

    setAnger(intensity) {
        this.setEmotion('angry', intensity);
    }

    setSurprise(intensity) {
        this.setEmotion('surprised', intensity);
    }

    setNeutral() {
        this.setEmotion('neutral', 0.2);
    }
}

/**
 * Motion Analyzer - Analyzes motion data for emotional content
 */
class MotionAnalyzer {
    constructor() {
        this.previousFrame = null;
        this.energyHistory = [];
        this.maxHistoryLength = 10;
    }

    analyzeFrame(frameData) {
        if (!frameData || frameData.length < 6) {
            return { energy: 0, suddenMovement: 0, smoothness: 1 };
        }
        
        let energy = 0;
        let suddenMovement = 0;
        let smoothness = 1;
        
        // Calculate motion energy
        for (let i = 6; i < frameData.length; i += 3) {
            const rotY = frameData[i] || 0;
            const rotX = frameData[i + 1] || 0;
            const rotZ = frameData[i + 2] || 0;
            
            const jointEnergy = Math.abs(rotY) + Math.abs(rotX) + Math.abs(rotZ);
            energy += jointEnergy;
        }
        
        // Normalize energy
        energy = energy / (frameData.length - 6) * 3;
        energy = Math.min(1.0, energy / 45.0); // Normalize to 0-1 range
        
        // Calculate sudden movement
        if (this.previousFrame) {
            let deltaSum = 0;
            for (let i = 6; i < Math.min(frameData.length, this.previousFrame.length); i++) {
                const delta = Math.abs(frameData[i] - this.previousFrame[i]);
                deltaSum += delta;
            }
            suddenMovement = Math.min(1.0, deltaSum / 100.0);
        }
        
        // Calculate smoothness
        this.energyHistory.push(energy);
        if (this.energyHistory.length > this.maxHistoryLength) {
            this.energyHistory.shift();
        }
        
        if (this.energyHistory.length > 1) {
            let variance = 0;
            const mean = this.energyHistory.reduce((a, b) => a + b) / this.energyHistory.length;
            for (const e of this.energyHistory) {
                variance += (e - mean) ** 2;
            }
            variance /= this.energyHistory.length;
            smoothness = 1.0 - Math.min(1.0, variance);
        }
        
        this.previousFrame = frameData.slice();
        
        return {
            energy: energy,
            suddenMovement: suddenMovement,
            smoothness: smoothness
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FacialExpressionSystem, MotionAnalyzer };
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.FacialExpressionSystem = FacialExpressionSystem;
    window.MotionAnalyzer = MotionAnalyzer;
}
