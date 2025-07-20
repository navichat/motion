async function smartTransition(targetAnimationId) {
    if (!this.searchEngine) {
        throw new Error('Search engine not initialized');
    }

    // Get current pose
    const currentPose = this.getCurrentPose();

    // Find best matching pose in target animation
    const results = await this.searchEngine.searchSimilarPoses(
        currentPose,
        targetAnimationId,
        5
    );

    if (results.length === 0) {
        throw new Error('No suitable transition point found');
    }

    const bestMatch = results[0];

    // Check if match is good enough
    if (bestMatch.distance > this.similarityThreshold) {
        print(`Warning: Best match distance ${bestMatch.distance.toFixed(4)} exceeds threshold ${this.similarityThreshold}`);
    }

    const sourcePose = this.getCurrentPose();

    const targetAnim = this.animations.get(targetAnimationId);

    if (!targetAnim) {
        throw new Error(`Animation not found: ${targetAnimationId}`);
    }

    print(`Transitioning to ${targetAnimationId} frame ${bestMatch.frameIndex} (distance: ${bestMatch.distance.toFixed(4)})`);

    // Returns  transition details
    return
        targetAnimationId,
        bestMatch.frameIndex,
        bestMatch.distance,
        results
    ;
}

export default smartTransition;
