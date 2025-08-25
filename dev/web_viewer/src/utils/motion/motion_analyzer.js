/**
 * RSMT Motion Analysis Tools
 * 
 * Advanced analysis and debugging tools for motion data
 */

class MotionAnalyzer {
    constructor() {
        this.frameRate = 60;
        this.jointNames = [
            'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
            'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
            'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder',
            'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder',
            'RightArm', 'RightForeArm', 'RightHand'
        ];
    }
    
    /**
     * Analyze motion characteristics
     */
    analyzeMotion(frames) {
        if (!frames || frames.length === 0) {
            return null;
        }
        
        const analysis = {
            frameCount: frames.length,
            duration: frames.length / this.frameRate,
            velocity: this.calculateVelocity(frames),
            acceleration: this.calculateAcceleration(frames),
            rhythm: this.analyzeRhythm(frames),
            style: this.analyzeStyle(frames),
            footContacts: this.detectFootContacts(frames),
            energyProfile: this.calculateEnergyProfile(frames)
        };
        
        return analysis;
    }
    
    /**
     * Calculate velocity characteristics
     */
    calculateVelocity(frames) {
        if (frames.length < 2) return { average: 0, peak: 0, variance: 0 };
        
        const velocities = [];
        for (let i = 1; i < frames.length; i++) {
            let frameVel = 0;
            for (let j = 0; j < Math.min(frames[i].length, frames[i-1].length); j++) {
                const diff = frames[i][j] - frames[i-1][j];
                frameVel += diff * diff;
            }
            velocities.push(Math.sqrt(frameVel));
        }
        
        const average = velocities.reduce((a, b) => a + b, 0) / velocities.length;
        const peak = Math.max(...velocities);
        const variance = velocities.reduce((acc, vel) => acc + Math.pow(vel - average, 2), 0) / velocities.length;
        
        return { average, peak, variance };
    }
    
    /**
     * Calculate acceleration patterns
     */
    calculateAcceleration(frames) {
        if (frames.length < 3) return { average: 0, smoothness: 1 };
        
        const accelerations = [];
        for (let i = 2; i < frames.length; i++) {
            let frameAcc = 0;
            for (let j = 0; j < Math.min(frames[i].length, frames[i-1].length, frames[i-2].length); j++) {
                const vel1 = frames[i-1][j] - frames[i-2][j];
                const vel2 = frames[i][j] - frames[i-1][j];
                const acc = vel2 - vel1;
                frameAcc += acc * acc;
            }
            accelerations.push(Math.sqrt(frameAcc));
        }
        
        const average = accelerations.reduce((a, b) => a + b, 0) / accelerations.length;
        const smoothness = 1.0 / (1.0 + average * 0.01); // Lower acceleration = smoother
        
        return { average, smoothness };
    }
    
    /**
     * Analyze rhythmic patterns
     */
    analyzeRhythm(frames) {
        if (frames.length < 10) return { period: 0, regularity: 0 };
        
        // Analyze hip movement for basic rhythm detection
        const hipPos = frames.map(frame => frame[1] || 0); // Assuming hip Y is at index 1
        
        // Simple autocorrelation for period detection
        let bestPeriod = 0;
        let bestCorrelation = 0;
        
        for (let period = 10; period < frames.length / 2; period++) {
            let correlation = 0;
            let count = 0;
            
            for (let i = 0; i < frames.length - period; i++) {
                correlation += hipPos[i] * hipPos[i + period];
                count++;
            }
            
            correlation /= count;
            if (correlation > bestCorrelation) {
                bestCorrelation = correlation;
                bestPeriod = period;
            }
        }
        
        const regularity = Math.min(1.0, bestCorrelation / 100);
        
        return {
            period: bestPeriod / this.frameRate,
            regularity: Math.max(0, regularity)
        };
    }
    
    /**
     * Analyze motion style characteristics
     */
    analyzeStyle(frames) {
        const velocity = this.calculateVelocity(frames);
        const acceleration = this.calculateAcceleration(frames);
        
        // Classify energy level
        let energy = 'medium';
        if (velocity.average > 50) energy = 'high';
        else if (velocity.average < 20) energy = 'low';
        
        // Classify smoothness
        let smoothness = 'moderate';
        if (acceleration.smoothness > 0.8) smoothness = 'smooth';
        else if (acceleration.smoothness < 0.4) smoothness = 'jerky';
        
        // Estimate emotional content based on motion characteristics
        let emotion = 'neutral';
        if (velocity.average > 40 && acceleration.smoothness < 0.6) emotion = 'aggressive';
        else if (velocity.average < 25 && acceleration.smoothness > 0.7) emotion = 'calm';
        else if (velocity.variance > 1000) emotion = 'excited';
        
        return { energy, smoothness, emotion };
    }
    
    /**
     * Detect foot contact events
     */
    detectFootContacts(frames) {
        if (frames.length < 5) return [];
        
        const contacts = [];
        const leftFootIndex = 3 * 6 + 2; // Approximate left foot Y position
        const rightFootIndex = 8 * 6 + 2; // Approximate right foot Y position
        
        // Simple contact detection based on foot height threshold
        const threshold = -90; // Approximate ground level
        
        for (let i = 2; i < frames.length - 2; i++) {
            const leftFoot = frames[i][leftFootIndex] || 0;
            const rightFoot = frames[i][rightFootIndex] || 0;
            
            if (leftFoot < threshold) {
                contacts.push({ frame: i, foot: 'left', time: i / this.frameRate });
            }
            if (rightFoot < threshold) {
                contacts.push({ frame: i, foot: 'right', time: i / this.frameRate });
            }
        }
        
        return contacts;
    }
    
    /**
     * Calculate energy profile over time
     */
    calculateEnergyProfile(frames) {
        const windowSize = 10;
        const profile = [];
        
        for (let i = 0; i < frames.length - windowSize; i += 5) {
            const window = frames.slice(i, i + windowSize);
            const velocity = this.calculateVelocity(window);
            profile.push({
                time: i / this.frameRate,
                energy: velocity.average
            });
        }
        
        return profile;
    }
    
    /**
     * Compare two motions and calculate similarity
     */
    compareMotions(frames1, frames2) {
        if (!frames1 || !frames2 || frames1.length === 0 || frames2.length === 0) {
            return { similarity: 0, differences: [] };
        }
        
        const analysis1 = this.analyzeMotion(frames1);
        const analysis2 = this.analyzeMotion(frames2);
        
        // Calculate similarity metrics
        const velocitySim = 1 - Math.abs(analysis1.velocity.average - analysis2.velocity.average) / 100;
        const rhythmSim = 1 - Math.abs(analysis1.rhythm.period - analysis2.rhythm.period) / 2;
        const energySim = analysis1.style.energy === analysis2.style.energy ? 1 : 0.5;
        
        const similarity = (velocitySim + rhythmSim + energySim) / 3;
        
        const differences = [
            {
                metric: 'Velocity',
                value1: analysis1.velocity.average.toFixed(2),
                value2: analysis2.velocity.average.toFixed(2),
                similarity: velocitySim.toFixed(3)
            },
            {
                metric: 'Rhythm Period',
                value1: analysis1.rhythm.period.toFixed(2) + 's',
                value2: analysis2.rhythm.period.toFixed(2) + 's',
                similarity: rhythmSim.toFixed(3)
            },
            {
                metric: 'Energy Level',
                value1: analysis1.style.energy,
                value2: analysis2.style.energy,
                similarity: energySim.toFixed(3)
            }
        ];
        
        return { similarity: Math.max(0, similarity), differences };
    }
}

/**
 * Real-time Performance Monitor
 */
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            frameTime: [],
            renderTime: [],
            transitionTime: [],
            memoryUsage: []
        };
        this.maxSamples = 100;
    }
    
    recordFrameTime(time) {
        this.metrics.frameTime.push(time);
        if (this.metrics.frameTime.length > this.maxSamples) {
            this.metrics.frameTime.shift();
        }
    }
    
    recordRenderTime(time) {
        this.metrics.renderTime.push(time);
        if (this.metrics.renderTime.length > this.maxSamples) {
            this.metrics.renderTime.shift();
        }
    }
    
    recordTransitionTime(time) {
        this.metrics.transitionTime.push(time);
        if (this.metrics.transitionTime.length > this.maxSamples) {
            this.metrics.transitionTime.shift();
        }
    }
    
    getStats() {
        const avgFrameTime = this.average(this.metrics.frameTime);
        const avgRenderTime = this.average(this.metrics.renderTime);
        const avgTransitionTime = this.average(this.metrics.transitionTime);
        
        return {
            fps: avgFrameTime > 0 ? 1000 / avgFrameTime : 0,
            frameTime: avgFrameTime,
            renderTime: avgRenderTime,
            transitionTime: avgTransitionTime,
            performance: this.getPerformanceRating(avgFrameTime)
        };
    }
    
    average(arr) {
        return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    }
    
    getPerformanceRating(frameTime) {
        if (frameTime < 16.67) return 'Excellent (60+ FPS)';
        if (frameTime < 33.33) return 'Good (30+ FPS)';
        if (frameTime < 50) return 'Fair (20+ FPS)';
        return 'Poor (<20 FPS)';
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.MotionAnalyzer = MotionAnalyzer;
    window.PerformanceMonitor = PerformanceMonitor;
}
