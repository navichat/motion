/**
 * BVH Timeline Compositor Usage Examples
 * Demonstrates how to use the timeline system with different BVH animation sources
 */

// Example usage of the BVH Timeline Compositor
class BVHTimelineExample {
    constructor() {
        this.compositor = new BVHTimelineCompositor({
            frameRate: 30,
            blendMode: 'hierarchical',
            cacheSize: 1000
        });
        
        this.setupEventListeners();
        this.initializeExample();
    }
    
    setupEventListeners() {
        // Listen to timeline events
        this.compositor.on('frameUpdate', (data) => {
            console.log('Frame update:', data.timestamp);
            this.sendFrameToVRM(data.frame);
        });
        
        this.compositor.on('trackAdded', (data) => {
            console.log('Track added:', data.trackId);
        });
        
        this.compositor.on('clipAdded', (data) => {
            console.log('Clip added:', data.clip.id, 'to track:', data.trackId);
        });
    }
    
    async initializeExample() {
        // Create different types of tracks
        await this.setupBaseTracks();
        await this.addExampleClips();
        
        console.log('Timeline setup complete. Ready to play!');
    }
    
    async setupBaseTracks() {
        // 1. Base animation track (highest priority)
        this.compositor.addTrack('base_animations', {
            type: 'static',
            priority: 100,
            weight: 1.0,
            channels: 'body',
            blendMode: 'replace'
        });
        
        // 2. RSMT transition track
        this.compositor.addTrack('transitions', {
            type: 'rsmt',
            priority: 90,
            weight: 1.0,
            channels: 'body',
            blendMode: 'replace'
        });
        
        // 3. Audio2Gesture track for body gestures
        this.compositor.addTrack('body_gestures', {
            type: 'audio2gesture',
            priority: 80,
            weight: 0.7,
            channels: 'arms',
            blendMode: 'additive'
        });
        
        // 4. Faceformer track for facial animation
        this.compositor.addTrack('facial_animation', {
            type: 'faceformer',
            priority: 85,
            weight: 1.0,
            channels: 'face',
            blendMode: 'replace'
        });
        
        // 5. Neural network track for procedural animations
        this.compositor.addTrack('neural_gestures', {
            type: 'neural',
            priority: 70,
            weight: 0.5,
            channels: 'hands',
            blendMode: 'additive'
        });
        
        // 6. Background idle animation
        this.compositor.addTrack('idle_background', {
            type: 'static',
            priority: 10,
            weight: 0.3,
            channels: 'all',
            blendMode: 'additive',
            generator: this.generateIdleAnimation.bind(this)
        });
    }
    
    async addExampleClips() {
        // Add base walking animation
        this.compositor.addClip('base_animations', {
            id: 'walk_cycle',
            startTime: 0,
            duration: 5000, // 5 seconds
            bvhFile: 'animations/walk_cycle.bvh',
            loop: true,
            weight: 1.0
        });
        
        // Add gesture animation triggered by speech
        this.compositor.addClip('base_animations', {
            id: 'talking_gesture',
            startTime: 6000,
            duration: 3000,
            bvhFile: 'animations/talking_gesture.bvh',
            fadeIn: 200,
            fadeOut: 300,
            weight: 0.8
        });
        
        // Add RSMT transition between walk and gesture
        this.compositor.addClip('transitions', {
            id: 'walk_to_gesture_transition',
            startTime: 5500,
            duration: 1000,
            sourceClip: 'walk_cycle',
            targetClip: 'talking_gesture',
            transitionDuration: 500,
            parameters: {
                smoothness: 0.8,
                preserveRootMotion: true
            }
        });
        
        // Add audio-driven gesture
        this.compositor.addClip('body_gestures', {
            id: 'speech_gestures',
            startTime: 6000,
            duration: 8000,
            audioFile: 'audio/speech_sample.wav',
            parameters: {
                gestureIntensity: 0.7,
                gestureType: 'explanatory'
            },
            weight: 0.6
        });
        
        // Add facial animation synced to same audio
        this.compositor.addClip('facial_animation', {
            id: 'speech_lips',
            startTime: 6000,
            duration: 8000,
            audioFile: 'audio/speech_sample.wav',
            parameters: {
                visemeIntensity: 1.0,
                emotionalExpression: 'neutral'
            },
            weight: 1.0
        });
        
        // Add neural-generated hand gestures
        this.compositor.addClip('neural_gestures', {
            id: 'expressive_hands',
            startTime: 7000,
            duration: 4000,
            prompt: 'expressive hand gestures while explaining',
            parameters: {
                style: 'conversational',
                intensity: 0.5
            },
            weight: 0.4
        });
        
        // Background idle animation runs throughout
        this.compositor.addClip('idle_background', {
            id: 'breathing_idle',
            startTime: 0,
            duration: 20000, // 20 seconds
            loop: true,
            weight: 0.2
        });
    }
    
    // Custom generator for idle animation
    generateIdleAnimation(clip, localTime, weight) {
        const frame = this.compositor.getDefaultPose();
        
        // Subtle breathing motion
        const breathCycle = Math.sin(localTime * 0.004) * 0.02;
        if (frame[1]) { // Chest joint
            frame[1].position.y = breathCycle;
            frame[1].rotation.x = breathCycle * 0.5;
        }
        
        // Slight weight shift
        const weightShift = Math.sin(localTime * 0.001) * 0.01;
        if (frame[0]) { // Root joint
            frame[0].position.x = weightShift;
        }
        
        return frame;
    }
    
    // Integration with VRM/Three.js
    sendFrameToVRM(frame) {
        // This would integrate with your VRM system
        if (typeof window !== 'undefined' && window.VRMBVHAdapter) {
            window.VRMBVHAdapter.applyFrame(frame);
        }
        
        // Or send to Three.js scene
        if (this.vrm && this.vrm.humanoid) {
            this.applyFrameToVRM(frame);
        }
    }
    
    applyFrameToVRM(frame) {
        // Convert BVH frame to VRM humanoid pose
        for (const [jointIndex, jointData] of Object.entries(frame)) {
            const vrmBoneName = this.mapBVHJointToVRM(parseInt(jointIndex));
            if (vrmBoneName && this.vrm.humanoid.getBoneNode(vrmBoneName)) {
                const bone = this.vrm.humanoid.getBoneNode(vrmBoneName);
                
                // Apply rotation
                if (jointData.rotation) {
                    bone.rotation.set(
                        jointData.rotation.x * Math.PI / 180,
                        jointData.rotation.y * Math.PI / 180,
                        jointData.rotation.z * Math.PI / 180
                    );
                }
                
                // Apply position (usually only for root)
                if (jointData.position && jointIndex === '0') {
                    bone.position.set(
                        jointData.position.x,
                        jointData.position.y,
                        jointData.position.z
                    );
                }
            }
        }
    }
    
    mapBVHJointToVRM(jointIndex) {
        // Map BVH joint indices to VRM bone names
        const mapping = {
            0: 'hips',
            1: 'spine',
            2: 'chest',
            3: 'neck',
            4: 'head',
            5: 'leftShoulder',
            6: 'leftUpperArm',
            7: 'leftLowerArm',
            8: 'leftHand',
            9: 'rightShoulder',
            10: 'rightUpperArm',
            11: 'rightLowerArm',
            12: 'rightHand',
            13: 'leftUpperLeg',
            14: 'leftLowerLeg',
            15: 'leftFoot',
            16: 'rightUpperLeg',
            17: 'rightLowerLeg',
            18: 'rightFoot'
        };
        
        return mapping[jointIndex];
    }
    
    // Control methods
    async playTimeline(startTime = 0) {
        await this.compositor.setTime(startTime);
        this.compositor.play();
    }
    
    pauseTimeline() {
        this.compositor.pause();
    }
    
    stopTimeline() {
        this.compositor.stop();
    }
    
    async seekTo(timestamp) {
        await this.compositor.setTime(timestamp);
    }
    
    // Dynamic clip management
    async addSpeechClip(audioFile, startTime, duration) {
        // Add audio2gesture clip
        this.compositor.addClip('body_gestures', {
            id: `speech_gesture_${Date.now()}`,
            startTime: startTime,
            duration: duration,
            audioFile: audioFile,
            fadeIn: 200,
            fadeOut: 200,
            weight: 0.7
        });
        
        // Add corresponding facial animation
        this.compositor.addClip('facial_animation', {
            id: `speech_face_${Date.now()}`,
            startTime: startTime,
            duration: duration,
            audioFile: audioFile,
            weight: 1.0
        });
    }
    
    async addTransition(fromTime, toTime, duration) {
        this.compositor.addClip('transitions', {
            id: `transition_${Date.now()}`,
            startTime: fromTime,
            duration: duration,
            transitionDuration: duration,
            parameters: {
                smoothness: 0.9
            }
        });
    }
    
    // Track control
    muteTrack(trackId) {
        const track = this.compositor.tracks.get(trackId);
        if (track) {
            track.muted = true;
        }
    }
    
    unmuteTrack(trackId) {
        const track = this.compositor.tracks.get(trackId);
        if (track) {
            track.muted = false;
        }
    }
    
    setTrackWeight(trackId, weight) {
        const track = this.compositor.tracks.get(trackId);
        if (track) {
            track.weight = Math.max(0, Math.min(1, weight));
        }
    }
    
    // Export current timeline state
    exportTimeline() {
        return this.compositor.exportTimeline();
    }
    
    // Import timeline state
    importTimeline(data) {
        this.compositor.importTimeline(data);
    }
    
    // Get current status
    getStatus() {
        return {
            compositor: this.compositor.getStatus(),
            tracks: Array.from(this.compositor.tracks.entries()).map(([id, track]) => ({
                id,
                type: track.type,
                enabled: track.enabled,
                muted: track.muted,
                weight: track.weight,
                clipCount: track.clips.length
            }))
        };
    }
}

// Usage example
async function runExample() {
    const timeline = new BVHTimelineExample();
    
    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Start playing
    console.log('Starting timeline playback...');
    await timeline.playTimeline();
    
    // Add dynamic content during playback
    setTimeout(() => {
        timeline.addSpeechClip('audio/dynamic_speech.wav', 15000, 5000);
    }, 3000);
    
    // Control playback
    setTimeout(() => {
        timeline.pauseTimeline();
        console.log('Timeline paused');
    }, 10000);
    
    setTimeout(() => {
        timeline.seekTo(8000);
        timeline.playTimeline();
        console.log('Timeline resumed from 8 seconds');
    }, 12000);
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BVHTimelineExample;
} else if (typeof window !== 'undefined') {
    window.BVHTimelineExample = BVHTimelineExample;
    window.runBVHTimelineExample = runExample;
}
