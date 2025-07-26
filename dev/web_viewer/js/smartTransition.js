import getCurrentPose from './getCurrentPose.js'; // Direct import

async function smartTransition(timeline, targetAnimationId) {
    if (!timeline.searchEngine) {
        throw new Error('Search engine not initialized');
    }

    // Get current pose
    const currentPose = getCurrentPose(timeline);
    console.log(`currentPose.length = ${currentPose.length}`)

    // Find best matching pose in target animation
    const results = await timeline.searchEngine.searchSimilarPoses(
        currentPose,
        targetAnimationId,
        5
    );

    if (results.length === 0) {
        throw new Error('No suitable transition point found');
    }

    const bestMatch = results[0];

    // Check if match is good enough
    if (bestMatch.distance > timeline.similarityThreshold) {
        console.log(`Warning: Best match distance ${bestMatch.distance.toFixed(4)} exceeds threshold ${timeline.similarityThreshold}`);
    }

    const sourcePose = getCurrentPose(timeline);

    const targetAnim = timeline.animations.get(targetAnimationId);

    if (!targetAnim) {
        throw new Error(`Animation not found: ${targetAnimationId}`);
    }

    console.log(`Transitioning to ${targetAnimationId} frame ${bestMatch.frameIndex} (distance: ${bestMatch.distance.toFixed(4)})`);

    // Returns  transition details
    return [targetAnimationId, bestMatch.frameIndex, bestMatch.distance, results];
}

export default smartTransition;
