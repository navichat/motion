/**
 * Animation Controller - Handles motion playback and transitions
 */

import * as THREE from 'three';

export class AnimationController {
    constructor(avatarInstance) {
        this.avatarInstance = avatarInstance;
        this.mixer = avatarInstance.mixer;
        this.currentAction = null;
        this.previousAction = null;
        this.animationQueue = [];
        this.isTransitioning = false;
        
        // Animation settings
        this.defaultTransitionDuration = 0.5;
        this.fadeInDuration = 0.3;
        this.fadeOutDuration = 0.3;
        
        // Event listeners
        this.eventListeners = new Map();
        
        console.log('Animation Controller initialized');
    }
    
    // Play a single animation
    playAnimation(animationData, options = {}) {
        const {
            loop = false,
            crossFade = true,
            fadeInDuration = this.fadeInDuration,
            fadeOutDuration = this.fadeOutDuration,
            timeScale = 1.0
        } = options;
        
        if (!this.mixer) {
            console.warn('No animation mixer available');
            return null;
        }
        
        let clip;
        
        // Handle different animation formats
        if (animationData.format === 'json') {
            clip = this.parseJSONAnimation(animationData);
        } else if (animationData.format === 'bvh') {
            clip = this.parseBVHAnimation(animationData);
        } else if (animationData instanceof THREE.AnimationClip) {
            clip = animationData;
        } else {
            console.error('Unsupported animation format');
            return null;
        }
        
        const action = this.mixer.clipAction(clip);
        action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
        action.timeScale = timeScale;
        
        if (crossFade && this.currentAction) {
            this.crossFadeToAction(action, fadeInDuration, fadeOutDuration);
        } else {
            this.switchToAction(action, fadeInDuration);
        }
        
        this.emit('animationStarted', { action, clip });
        
        return action;
    }
    
    // Cross-fade between animations
    crossFadeToAction(newAction, fadeInDuration = this.fadeInDuration, fadeOutDuration = this.fadeOutDuration) {
        if (this.currentAction && this.currentAction !== newAction) {
            this.previousAction = this.currentAction;
            this.currentAction.fadeOut(fadeOutDuration);
        }
        
        this.currentAction = newAction;
        this.currentAction.reset();
        this.currentAction.fadeIn(fadeInDuration);
        this.currentAction.play();
        
        this.isTransitioning = true;
        
        // Clear transition flag after fade completes
        setTimeout(() => {
            this.isTransitioning = false;
            this.emit('transitionComplete', { action: this.currentAction });
        }, Math.max(fadeInDuration, fadeOutDuration) * 1000);
    }
    
    // Immediate switch without fade
    switchToAction(newAction, fadeInDuration = 0) {
        if (this.currentAction) {
            this.currentAction.stop();
        }
        
        this.currentAction = newAction;
        this.currentAction.reset();
        
        if (fadeInDuration > 0) {
            this.currentAction.fadeIn(fadeInDuration);
        }
        
        this.currentAction.play();
    }
    
    // Queue multiple animations to play in sequence
    queueAnimations(animationList, options = {}) {
        this.animationQueue = [...animationList];
        
        if (!this.isTransitioning && this.animationQueue.length > 0) {
            this.playNextInQueue(options);
        }
    }
    
    playNextInQueue(options = {}) {
        if (this.animationQueue.length === 0) {
            this.emit('queueComplete');
            return;
        }
        
        const nextAnimation = this.animationQueue.shift();
        const action = this.playAnimation(nextAnimation, {
            ...options,
            loop: false
        });
        
        if (action) {
            // Set up listener for when this animation finishes
            const onFinished = () => {
                this.mixer.removeEventListener('finished', onFinished);
                setTimeout(() => this.playNextInQueue(options), 100);
            };
            
            this.mixer.addEventListener('finished', onFinished);
        }
    }
    
    // Parse JSON animation format (from chat interface)
    parseJSONAnimation(animationData) {
        if (!animationData.tracks) {
            console.error('Invalid JSON animation: missing tracks');
            return null;
        }
        
        const tracks = animationData.tracks.map(trackData => {
            const { name, type, times, values } = trackData;
            
            switch (type) {
                case 'vector':
                    return new THREE.VectorKeyframeTrack(name, times, values);
                case 'quaternion':
                    return new THREE.QuaternionKeyframeTrack(name, times, values);
                case 'number':
                    return new THREE.NumberKeyframeTrack(name, times, values);
                default:
                    console.warn(`Unknown track type: ${type}`);
                    return null;
            }
        }).filter(track => track !== null);
        
        const clip = new THREE.AnimationClip(
            animationData.name || 'Animation',
            animationData.duration || -1,
            tracks
        );
        
        return clip;
    }
    
    // Parse BVH animation format (from RSMT)
    parseBVHAnimation(animationData) {
        // This will be implemented in Phase 2 with RSMT integration
        console.log('BVH animation parsing will be implemented in Phase 2');
        
        // For now, return a placeholder
        return new THREE.AnimationClip('BVH_Placeholder', 1, []);
    }
    
    // Style transfer using RSMT (Phase 2)
    async applyStyleTransfer(sourceAnimation, targetStyle) {
        console.log('Style transfer will be implemented in Phase 2');
        
        // This will integrate with RSMT neural networks
        // For now, return the original animation
        return sourceAnimation;
    }
    
    // Generate transition between two animations (Phase 2)
    async generateTransition(sourceAnimation, targetAnimation, duration = 1.0) {
        console.log('Transition generation will be implemented in Phase 2');
        
        // This will use RSMT's neural networks
        // For now, return a simple cross-fade
        return {
            type: 'crossfade',
            duration: duration
        };
    }
    
    // Control playback
    pause() {
        if (this.currentAction) {
            this.currentAction.paused = true;
        }
    }
    
    resume() {
        if (this.currentAction) {
            this.currentAction.paused = false;
        }
    }
    
    stop() {
        if (this.currentAction) {
            this.currentAction.stop();
            this.currentAction = null;
        }
        this.clearQueue();
    }
    
    setSpeed(speed) {
        if (this.currentAction) {
            this.currentAction.timeScale = speed;
        }
    }
    
    getCurrentTime() {
        return this.currentAction ? this.currentAction.time : 0;
    }
    
    setCurrentTime(time) {
        if (this.currentAction) {
            this.currentAction.time = time;
        }
    }
    
    getDuration() {
        return this.currentAction ? this.currentAction.getClip().duration : 0;
    }
    
    clearQueue() {
        this.animationQueue = [];
    }
    
    // Event system
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.eventListeners.has(event)) {
            const listeners = this.eventListeners.get(event);
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }
    
    emit(event, ...args) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                callback(...args);
            });
        }
    }
    
    // Get current playback state
    getState() {
        return {
            isPlaying: this.currentAction && !this.currentAction.paused,
            isPaused: this.currentAction && this.currentAction.paused,
            isTransitioning: this.isTransitioning,
            currentTime: this.getCurrentTime(),
            duration: this.getDuration(),
            queueLength: this.animationQueue.length,
            timeScale: this.currentAction ? this.currentAction.timeScale : 1.0
        };
    }
    
    update(delta) {
        // Update animation mixer
        if (this.mixer) {
            this.mixer.update(delta);
        }
        
        // Emit update events
        this.emit('update', this.getState());
    }
    
    dispose() {
        if (this.mixer) {
            this.mixer.stopAllAction();
        }
        this.clearQueue();
        this.eventListeners.clear();
        console.log('Animation Controller disposed');
    }
}
