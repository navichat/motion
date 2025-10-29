/**
 * Simple Animation Blender for ES6 modules
 * Handles blending between different animation states
 */

export class AnimationBlender {
    constructor(options = {}) {
        this.blendMode = options.blendMode || 'hierarchical';
        this.blendWeights = new Map();
        this.activeTracks = new Map();
        
        console.log('🎨 Animation Blender initialized');
    }

    /**
     * Blend animation data with weights
     * @param {Object} animationData - Animation data to blend
     * @param {Object} weights - Blending weights
     * @returns {Object} Blended animation data
     */
    blend(animationData, weights = {}) {
        if (!animationData) return null;

        // Apply hierarchical blending based on priorities
        const blended = this.applyHierarchicalBlending(animationData, weights);
        
        return blended;
    }

    /**
     * Apply hierarchical blending
     * @param {Object} data - Animation data
     * @param {Object} weights - Weights for blending
     * @returns {Object} Blended result
     */
    applyHierarchicalBlending(data, weights) {
        const result = { ...data };
        
        // Apply body weight
        if (weights.body && weights.body !== 1.0) {
            result.body = this.scaleAnimationData(result.body, weights.body);
        }
        
        // Apply upper body emphasis
        if (weights.upperBody && weights.upperBody !== 1.0) {
            result.upperBody = this.scaleAnimationData(result.upperBody, weights.upperBody);
        }
        
        // Apply facial weight
        if (weights.face && weights.face !== 1.0) {
            result.face = this.scaleAnimationData(result.face, weights.face);
        }
        
        return result;
    }

    /**
     * Scale animation data by weight
     * @param {Object} data - Animation data
     * @param {number} weight - Scale factor
     * @returns {Object} Scaled data
     */
    scaleAnimationData(data, weight) {
        if (!data || weight === 1.0) return data;
        
        const scaled = { ...data };
        
        // Scale numeric values
        for (const key in scaled) {
            if (typeof scaled[key] === 'number') {
                scaled[key] *= weight;
            } else if (scaled[key] && typeof scaled[key] === 'object') {
                scaled[key] = this.scaleAnimationData(scaled[key], weight);
            }
        }
        
        return scaled;
    }

    /**
     * Add animation track
     * @param {string} name - Track name
     * @param {Object} data - Animation data
     * @param {number} weight - Track weight
     */
    addTrack(name, data, weight = 1.0) {
        this.activeTracks.set(name, { data, weight });
        this.blendWeights.set(name, weight);
    }

    /**
     * Remove animation track
     * @param {string} name - Track name
     */
    removeTrack(name) {
        this.activeTracks.delete(name);
        this.blendWeights.delete(name);
    }

    /**
     * Update track weight
     * @param {string} name - Track name
     * @param {number} weight - New weight
     */
    setTrackWeight(name, weight) {
        if (this.blendWeights.has(name)) {
            this.blendWeights.set(name, weight);
        }
    }

    /**
     * Get current active tracks
     * @returns {Map} Active tracks
     */
    getActiveTracks() {
        return new Map(this.activeTracks);
    }

    /**
     * Clear all tracks
     */
    clear() {
        this.activeTracks.clear();
        this.blendWeights.clear();
    }
}

export default AnimationBlender;