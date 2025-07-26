// Simplified Pose Search Engine
class SimplePoseSearchEngine {
    constructor() {
        this.animationData = new Map();
    }

    async indexAnimations(animations) {
        for (const animation of animations) {
            this.animationData.set(animation.animationId, {
                poses: animation.poses,
                timestamps: animation.timestamps
            });
        }
    }

    async searchSimilarPoses(queryPose, targetAnimation, k = 5) {
        const targetData = this.animationData.get(targetAnimation);
        if (!targetData) return [];

        const distances = [];

        for (let i = 0; i < targetData.poses.length; i++) {
            const distance = this.computeDistance(queryPose, targetData.poses[i]);
            distances.push({
                distance,
                frameIndex: i,
                timestamp: targetData.timestamps[i]
            });
        }

        // Sort by distance and return top k
        distances.sort((a, b) => a.distance - b.distance);
        return distances.slice(0, k);
    }

    computeDistance(pose1, pose2) {
        let sum = 0;
        const length = Math.min(pose1.length, pose2.length);
        //console.log(`pose1.length = ${pose1.length} , pose2.length = ${pose2.length}`);

        // *** FIX: Start loop at index 6 to IGNORE root position/rotation ***
        // We only care about the similarity of the body's posture.
        for (let i = 6; i < length; i++) {
            const diff = (pose1[i] || 0) - (pose2[i] || 0);
            sum += diff * diff;
        }

        // Avoid division by zero if there are no rotational channels
        const rotationalChannels = Math.max(1, length - 6);
        //console.log(`rotationalChannels = ${rotationalChannels}`);

        return Math.sqrt(sum / rotationalChannels);
    }
}

export default SimplePoseSearchEngine;
