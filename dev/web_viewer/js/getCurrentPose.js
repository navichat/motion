function getCurrentPose(timeline) {
    const sourceAnim = timeline.animations.get(timeline.currentAnimationId);
    if (!sourceAnim) return new Float32Array(72).fill(0);

    // ✅ ALWAYS get the current pose from the playing animation
    const frameIndex = Math.floor(timeline.currentFrame);
    const nextFrameIndex = (frameIndex + 1) % sourceAnim.poses.length;
    const t_frame = timeline.currentFrame - frameIndex;

    // Interpolate between current and next frame for smooth playback
    const currentFramePose = sourceAnim.poses[frameIndex];
    const nextFramePose = sourceAnim.poses[nextFrameIndex];

    const sourcePose = new Float32Array(currentFramePose.length);
    for (let i = 0; i < currentFramePose.length; i++) {
        sourcePose[i] = currentFramePose[i] + (nextFramePose[i] - currentFramePose[i]) * t_frame;
    }

    if (!timeline.isTransitioning) {
        return sourcePose; // Return the smoothly interpolated pose for normal playback
    }

    // --- TRANSITION LOGIC WITH DYNAMIC SOURCE ---
    let t_blend = Math.min(timeline.transitionProgress, 1.0);

    // Apply smoothing
    t_blend = t_blend < 0.5 ? 8 * t_blend * t_blend * t_blend * t_blend : 1 - Math.pow(-2 * t_blend + 2, 4) / 2;

    const targetPose = timeline.transitionTarget.pose;
    const interpolatedPose = new Float32Array(sourcePose.length);

    // ✅ Use THREE.js for consistent quaternion operations
    //const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.177.0/build/three.module.js');
    // *** CRITICAL CHANGE: Get THREE from the bound 'timeline' context, not with await import ***
    const THREE = timeline.THREE;

    if (!THREE) {
        // Fallback to linear interpolation if THREE.js not available
        for (let i = 0; i < sourcePose.length; i++) {
            interpolatedPose[i] = (1 - t_blend) * (sourcePose[i] || 0) + t_blend * (targetPose[i] || 0);
        }
        return interpolatedPose;
    }

    const q1 = new THREE.Quaternion();
    const q2 = new THREE.Quaternion();
    const euler1 = new THREE.Euler();
    const euler2 = new THREE.Euler();
    const eulerResult = new THREE.Euler();

    // ✅ 1. Interpolate Root Position with slight arc for natural movement
    for (let i = 0; i < 3; i++) {
        if (i === 1) { // Y position - add slight arc
            const sourceY = sourcePose[i] || 0;
            const targetY = targetPose[i] || 0;
            const arcHeight = Math.abs(sourceY - targetY) * 0.1; // 10% of height difference
            const midY = Math.max(sourceY, targetY) + arcHeight;

            // Bezier curve through midpoint
            const u = 1 - t_blend;
            interpolatedPose[i] = u * u * sourceY + 2 * u * t_blend * midY + t_blend * t_blend * targetY;
        } else {
            interpolatedPose[i] = (1 - t_blend) * (sourcePose[i] || 0) + t_blend * (targetPose[i] || 0);
        }
    }

    // ✅ 2. Interpolate all rotations using proper quaternion SLERP
    for (let i = 3; i < sourcePose.length; i += 3) {
        if (i + 2 >= sourcePose.length) break;

        // ✅ Convert BVH Euler angles to quaternions (YXZ order is common in BVH)
        const sourceY = (sourcePose[i] || 0) * Math.PI / 180;
        const sourceX = (sourcePose[i + 1] || 0) * Math.PI / 180;
        const sourceZ = (sourcePose[i + 2] || 0) * Math.PI / 180;

        const targetY = (targetPose[i] || 0) * Math.PI / 180;
        const targetX = (targetPose[i + 1] || 0) * Math.PI / 180;
        const targetZ = (targetPose[i + 2] || 0) * Math.PI / 180;

        // Set Euler angles in YXZ order (common BVH order)
        euler1.set(sourceX, sourceY, sourceZ, 'YXZ');
        euler2.set(targetX, targetY, targetZ, 'YXZ');

        // Convert to quaternions
        q1.setFromEuler(euler1);
        q2.setFromEuler(euler2);

        // Perform SLERP
        q1.slerp(q2, t_blend);

        // Convert back to Euler angles
        eulerResult.setFromQuaternion(q1, 'YXZ');

        // Store back in BVH order (Y, X, Z)
        interpolatedPose[i]     = eulerResult.y * 180 / Math.PI; // Y
        interpolatedPose[i + 1] = eulerResult.x * 180 / Math.PI; // X
        interpolatedPose[i + 2] = eulerResult.z * 180 / Math.PI; // Z
    }

    return interpolatedPose;
}

export default getCurrentPose;
