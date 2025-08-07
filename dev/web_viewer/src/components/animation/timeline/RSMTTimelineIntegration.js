/**
 * RSMT Timeline Integration
 * 
 * Integration layer connecting RSMT (Realtime Stylized Motion Transition) converter
 * with the BVH Timeline system for seamless animation transitions.
 * 
 * Features:
 * - Automatic animation library loading from assets folder
 * - Real-time transition triggering
 * - Timeline synchronization
 * - Transition queue management
 * - Smart pose matching and transition generation
 */

class RSMTTimelineIntegration {
    constructor(timeline, options = {}) {
        this.timeline = timeline;
        this.rsmt = new RSMTBVHConverter(options.rsmt || {});
        
        // Configuration
        this.config = {
            animationsPath: options.animationsPath || '/assets/animations/',
            autoLoadAnimations: options.autoLoadAnimations !== false,
            transitionTrackName: options.transitionTrackName || 'rsmt_transitions',
            transitionPriority: options.transitionPriority || 50,
            preloadTransitions: options.preloadTransitions || false,
            maxConcurrentTransitions: options.maxConcurrentTransitions || 3,
            ...options
        };
        
        // State management
        this.loadedAnimations = new Map();
        this.activeTransitions = new Map();
        this.transitionQueue = [];
        this.isProcessingQueue = false;
        
        // Timeline integration
        this.timelineTrack = null;
        this.lastTimelineTime = 0;
        this.isTimelineSynced = false;
        
        // Performance monitoring
        this.stats = {
            animationsLoaded: 0,
            transitionsGenerated: 0,
            queuedTransitions: 0,
            successfulTransitions: 0,
            failedTransitions: 0,
            averageTransitionTime: 0
        };
        
        // Event callbacks
        this.onTransitionStart = options.onTransitionStart || null;
        this.onTransitionComplete = options.onTransitionComplete || null;
        this.onTransitionFailed = options.onTransitionFailed || null;
        this.onAnimationLoaded = options.onAnimationLoaded || null;
        
        this.initialize();
    }
    
    /**
     * Initialize RSMT timeline integration
     */
    async initialize() {
        console.log('[RSMT Timeline] Initializing integration...');
        
        try {
            // Initialize RSMT converter
            await this.rsmt.initializeDeepPhase('./models/deepphase');
            
            // Set up timeline track for transitions
            this.setupTimelineTrack();
            
            // Auto-load animations if configured
            if (this.config.autoLoadAnimations) {
                await this.loadAnimationsFromAssets();
            }
            
            // Set up timeline synchronization
            this.setupTimelineSynchronization();
            
            console.log('[RSMT Timeline] Integration initialized successfully');
            
        } catch (error) {
            console.error('[RSMT Timeline] Initialization failed:', error);
            throw error;
        }
    }
    
    /**
     * Set up timeline track for RSMT transitions
     */
    setupTimelineTrack() {
        if (this.timeline.addTrack) {
            this.timelineTrack = this.timeline.addTrack(this.config.transitionTrackName, {
                type: 'rsmt',
                priority: this.config.transitionPriority,
                weight: 1.0,
                channels: 'all',
                blendMode: 'replace',
                generator: this.generateRSMTFrame.bind(this)
            });
            
            console.log(`[RSMT Timeline] Created track: ${this.config.transitionTrackName}`);
        } else {
            console.warn('[RSMT Timeline] Timeline does not support addTrack - using external integration');
        }
    }
    
    /**
     * Load animations from assets folder
     */
    async loadAnimationsFromAssets() {
        console.log('[RSMT Timeline] Loading animations from assets...');
        
        try {
            // Get list of animation files
            const animationFiles = await this.discoverAnimationFiles();
            
            // Load each animation
            const loadPromises = animationFiles.map(file => 
                this.loadAnimation(file.name, file.path)
            );
            
            const results = await Promise.allSettled(loadPromises);
            
            // Count successful loads
            const successful = results.filter(r => r.status === 'fulfilled').length;
            const failed = results.length - successful;
            
            console.log(`[RSMT Timeline] Loaded ${successful} animations (${failed} failed)`);
            this.stats.animationsLoaded = successful;
            
            return { successful, failed, total: results.length };
            
        } catch (error) {
            console.error('[RSMT Timeline] Failed to load animations from assets:', error);
            throw error;
        }
    }
    
    /**
     * Discover animation files in assets folder
     */
    async discoverAnimationFiles() {
        const animationFiles = [];
        
        try {
            // Try to fetch the animations directory listing
            // This is a simplified approach - in practice you might have a manifest file
            const commonAnimations = [
                'thanks-lady-bow.json',
                'idle.json',
                'walk.json',
                'run.json',
                'wave.json',
                'dance.json',
                'jump.json',
                'sit.json',
                'stand.json',
                'turn.json',
                'gesture.json',
                'talking.json'
            ];
            
            for (const filename of commonAnimations) {
                const path = `${this.config.animationsPath}${filename}`;
                
                try {
                    // Test if file exists by attempting to fetch it
                    const response = await fetch(path, { method: 'HEAD' });
                    if (response.ok) {
                        animationFiles.push({
                            name: filename.replace('.json', ''),
                            path: path,
                            filename: filename
                        });
                    }
                } catch (error) {
                    // File doesn't exist, continue
                    console.debug(`[RSMT Timeline] Animation file not found: ${filename}`);
                }
            }
            
            console.log(`[RSMT Timeline] Discovered ${animationFiles.length} animation files`);
            return animationFiles;
            
        } catch (error) {
            console.error('[RSMT Timeline] Failed to discover animation files:', error);
            return [];
        }
    }
    
    /**
     * Load a specific animation
     */
    async loadAnimation(animationName, animationPath) {
        try {
            console.log(`[RSMT Timeline] Loading animation: ${animationName}`);
            
            const animationData = await this.rsmt.loadAnimation(animationName, animationPath);
            this.loadedAnimations.set(animationName, {
                data: animationData,
                path: animationPath,
                loadedAt: Date.now()
            });
            
            if (this.onAnimationLoaded) {
                this.onAnimationLoaded(animationName, animationData);
            }
            
            console.log(`[RSMT Timeline] Animation loaded: ${animationName}`);
            return animationData;
            
        } catch (error) {
            console.error(`[RSMT Timeline] Failed to load animation ${animationName}:`, error);
            throw error;
        }
    }
    
    /**
     * Set up timeline synchronization
     */
    setupTimelineSynchronization() {
        if (this.timeline.on) {
            // Listen to timeline events
            this.timeline.on('timeUpdate', (time) => {
                this.lastTimelineTime = time;
                this.processTransitionQueue();
            });
            
            this.timeline.on('play', () => {
                console.log('[RSMT Timeline] Timeline playback started');
                this.isTimelineSynced = true;
            });
            
            this.timeline.on('pause', () => {
                console.log('[RSMT Timeline] Timeline playback paused');
                this.pauseActiveTransitions();
            });
            
            this.timeline.on('stop', () => {
                console.log('[RSMT Timeline] Timeline playback stopped');
                this.stopActiveTransitions();
            });
        }
    }
    
    /**
     * Request transition to target animation
     */
    async requestTransition(targetAnimationName, options = {}) {
        try {
            console.log(`[RSMT Timeline] Transition requested to: ${targetAnimationName}`);
            
            // Validate target animation
            if (!this.loadedAnimations.has(targetAnimationName)) {
                throw new Error(`Animation not loaded: ${targetAnimationName}`);
            }
            
            // Get current timeline state
            const currentTime = this.timeline.currentTime || this.lastTimelineTime;
            const currentFrame = await this.timeline.getFrameAtTime(currentTime);
            
            if (!currentFrame) {
                throw new Error('Cannot get current frame from timeline');
            }
            
            // Create transition request
            const transitionRequest = {
                id: `transition_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                targetAnimation: targetAnimationName,
                startTime: currentTime,
                currentFrame: currentFrame,
                options: {
                    duration: options.duration || 1.0,
                    delay: options.delay || 0,
                    style: options.style || 'natural',
                    priority: options.priority || 'normal',
                    blendMode: options.blendMode || 'replace',
                    ...options
                },
                status: 'queued',
                createdAt: Date.now()
            };
            
            // Add to queue
            this.transitionQueue.push(transitionRequest);
            this.stats.queuedTransitions++;
            
            // Process immediately if possible
            if (!this.isProcessingQueue) {
                await this.processTransitionQueue();
            }
            
            return transitionRequest;
            
        } catch (error) {
            console.error('[RSMT Timeline] Transition request failed:', error);
            this.stats.failedTransitions++;
            
            if (this.onTransitionFailed) {
                this.onTransitionFailed(targetAnimationName, error);
            }
            
            throw error;
        }
    }
    
    /**
     * Process transition queue
     */
    async processTransitionQueue() {
        if (this.isProcessingQueue || this.transitionQueue.length === 0) {
            return;
        }
        
        this.isProcessingQueue = true;
        
        try {
            while (this.transitionQueue.length > 0 && 
                   this.activeTransitions.size < this.config.maxConcurrentTransitions) {
                
                const transitionRequest = this.transitionQueue.shift();
                await this.processTransitionRequest(transitionRequest);
            }
        } catch (error) {
            console.error('[RSMT Timeline] Queue processing error:', error);
        } finally {
            this.isProcessingQueue = false;
        }
    }
    
    /**
     * Process individual transition request
     */
    async processTransitionRequest(request) {
        const startTime = performance.now();
        
        try {
            console.log(`[RSMT Timeline] Processing transition: ${request.id}`);
            
            request.status = 'processing';
            
            if (this.onTransitionStart) {
                this.onTransitionStart(request);
            }
            
            // Generate transition using RSMT
            const transition = await this.rsmt.generateTransition(
                request.currentFrame,
                request.targetAnimation,
                request.options
            );
            
            // Calculate timeline insertion point
            const insertionTime = request.startTime + (request.options.delay * 1000);
            
            // Create timeline clip
            const clipData = {
                id: request.id,
                startTime: insertionTime,
                duration: transition.duration * 1000, // Convert to ms
                type: 'rsmt_transition',
                rsmt: {
                    transition: transition,
                    targetAnimation: request.targetAnimation,
                    originalRequest: request
                },
                weight: request.options.weight || 1.0,
                blendMode: request.options.blendMode || 'replace',
                metadata: {
                    generatedBy: 'rsmt',
                    transitionId: request.id,
                    targetAnimation: request.targetAnimation,
                    quality: transition.quality
                }
            };
            
            // Add to timeline
            if (this.timeline.addClip) {
                this.timeline.addClip(this.config.transitionTrackName, clipData);
            }
            
            // Track active transition
            this.activeTransitions.set(request.id, {
                request: request,
                transition: transition,
                clip: clipData,
                startTime: insertionTime,
                endTime: insertionTime + (transition.duration * 1000)
            });
            
            // Update statistics
            const processingTime = performance.now() - startTime;
            this.updateTransitionStats(processingTime);
            
            request.status = 'active';
            console.log(`[RSMT Timeline] Transition activated: ${request.id} (${processingTime.toFixed(2)}ms)`);
            
            if (this.onTransitionComplete) {
                this.onTransitionComplete(request, transition);
            }
            
            // Schedule cleanup
            this.scheduleTransitionCleanup(request.id, transition.duration * 1000);
            
        } catch (error) {
            console.error(`[RSMT Timeline] Transition processing failed: ${request.id}`, error);
            
            request.status = 'failed';
            request.error = error;
            
            this.stats.failedTransitions++;
            
            if (this.onTransitionFailed) {
                this.onTransitionFailed(request, error);
            }
        }
    }
    
    /**
     * Generate RSMT frame for timeline
     */
    generateRSMTFrame(clip, localTime, weight) {
        try {
            if (!clip.rsmt || !clip.rsmt.transition) {
                return null;
            }
            
            const transition = clip.rsmt.transition;
            const frameIndex = Math.floor(localTime * this.rsmt.frameRate / 1000);
            
            if (frameIndex < 0 || frameIndex >= transition.frames.length) {
                return null;
            }
            
            const frame = transition.frames[frameIndex];
            
            // Apply weight to frame
            if (weight !== 1.0) {
                return this.applyWeightToFrame(frame, weight);
            }
            
            return frame;
            
        } catch (error) {
            console.error('[RSMT Timeline] Frame generation error:', error);
            return null;
        }
    }
    
    /**
     * Apply weight to frame for blending
     */
    applyWeightToFrame(frame, weight) {
        const weightedFrame = {
            ...frame,
            bones: {}
        };
        
        for (const [boneName, boneData] of Object.entries(frame.bones)) {
            weightedFrame.bones[boneName] = {
                position: {
                    x: boneData.position.x * weight,
                    y: boneData.position.y * weight,
                    z: boneData.position.z * weight
                },
                rotation: {
                    x: boneData.rotation.x * weight,
                    y: boneData.rotation.y * weight,
                    z: boneData.rotation.z * weight,
                    w: 1 - weight + boneData.rotation.w * weight
                }
            };
        }
        
        return weightedFrame;
    }
    
    /**
     * Schedule transition cleanup
     */
    scheduleTransitionCleanup(transitionId, duration) {
        setTimeout(() => {
            this.cleanupTransition(transitionId);
        }, duration + 1000); // Add 1 second buffer
    }
    
    /**
     * Clean up completed transition
     */
    cleanupTransition(transitionId) {
        if (this.activeTransitions.has(transitionId)) {
            const transition = this.activeTransitions.get(transitionId);
            console.log(`[RSMT Timeline] Cleaning up transition: ${transitionId}`);
            
            // Remove from timeline if possible
            if (this.timeline.removeClip) {
                this.timeline.removeClip(this.config.transitionTrackName, transitionId);
            }
            
            this.activeTransitions.delete(transitionId);
            this.stats.successfulTransitions++;
        }
    }
    
    /**
     * Pause active transitions
     */
    pauseActiveTransitions() {
        for (const [id, transition] of this.activeTransitions) {
            transition.paused = true;
        }
        console.log(`[RSMT Timeline] Paused ${this.activeTransitions.size} active transitions`);
    }
    
    /**
     * Stop active transitions
     */
    stopActiveTransitions() {
        for (const [id, transition] of this.activeTransitions) {
            this.cleanupTransition(id);
        }
        console.log('[RSMT Timeline] Stopped all active transitions');
    }
    
    /**
     * Preload common transitions
     */
    async preloadTransitions(animationPairs) {
        if (!this.config.preloadTransitions) return;
        
        console.log('[RSMT Timeline] Preloading transitions...');
        
        for (const { from, to } of animationPairs) {
            if (this.loadedAnimations.has(from) && this.loadedAnimations.has(to)) {
                try {
                    // Generate and cache transition
                    const fromAnimation = this.loadedAnimations.get(from);
                    const toAnimation = this.loadedAnimations.get(to);
                    
                    // Use a representative frame from the 'from' animation
                    const fromFrame = fromAnimation.data.frames[0];
                    
                    await this.rsmt.generateTransition(fromFrame, to, {
                        duration: 1.0 // Standard duration for preloading
                    });
                    
                    console.log(`[RSMT Timeline] Preloaded transition: ${from} -> ${to}`);
                    
                } catch (error) {
                    console.warn(`[RSMT Timeline] Failed to preload transition ${from} -> ${to}:`, error);
                }
            }
        }
    }
    
    /**
     * Utility methods
     */
    
    getLoadedAnimations() {
        return Array.from(this.loadedAnimations.keys());
    }
    
    getAnimationInfo(animationName) {
        const animation = this.loadedAnimations.get(animationName);
        if (!animation) return null;
        
        return {
            name: animationName,
            ...this.rsmt.getAnimationInfo(animationName),
            loadedAt: animation.loadedAt,
            path: animation.path
        };
    }
    
    getActiveTransitions() {
        return Array.from(this.activeTransitions.entries()).map(([id, transition]) => ({
            id,
            targetAnimation: transition.request.targetAnimation,
            status: transition.request.status,
            startTime: transition.startTime,
            endTime: transition.endTime,
            quality: transition.transition.quality
        }));
    }
    
    getQueuedTransitions() {
        return this.transitionQueue.map(req => ({
            id: req.id,
            targetAnimation: req.targetAnimation,
            status: req.status,
            createdAt: req.createdAt
        }));
    }
    
    updateTransitionStats(processingTime) {
        this.stats.transitionsGenerated++;
        
        const count = this.stats.transitionsGenerated;
        this.stats.averageTransitionTime = 
            (this.stats.averageTransitionTime * (count - 1) + processingTime) / count;
    }
    
    getStats() {
        return {
            ...this.stats,
            ...this.rsmt.getStats(),
            loadedAnimations: this.loadedAnimations.size,
            activeTransitions: this.activeTransitions.size,
            queuedTransitions: this.transitionQueue.length,
            isTimelineSynced: this.isTimelineSynced
        };
    }
    
    /**
     * Configuration methods
     */
    
    setTransitionDuration(duration) {
        this.rsmt.transitionDuration = duration;
    }
    
    setSimilarityThreshold(threshold) {
        this.rsmt.similarityThreshold = threshold;
    }
    
    setMaxConcurrentTransitions(max) {
        this.config.maxConcurrentTransitions = max;
    }
    
    /**
     * Advanced features
     */
    
    // Chain multiple transitions
    async chainTransitions(animationSequence, options = {}) {
        const results = [];
        let delay = 0;
        
        for (let i = 0; i < animationSequence.length; i++) {
            const animationName = animationSequence[i];
            const transitionOptions = {
                ...options,
                delay: delay
            };
            
            try {
                const result = await this.requestTransition(animationName, transitionOptions);
                results.push(result);
                
                // Add duration for next transition
                delay += (options.duration || 1.0) * 1000;
                
            } catch (error) {
                console.error(`[RSMT Timeline] Chain transition failed at ${animationName}:`, error);
                break;
            }
        }
        
        return results;
    }
    
    // Smart transition selection based on current pose
    async smartTransition(targetAnimationName, options = {}) {
        // Get current pose
        const currentTime = this.timeline.currentTime || this.lastTimelineTime;
        const currentFrame = await this.timeline.getFrameAtTime(currentTime);
        
        if (!currentFrame) {
            throw new Error('Cannot determine current pose for smart transition');
        }
        
        // Find best transition point
        const match = this.rsmt.findBestMatch(currentFrame, targetAnimationName);
        
        if (match && match.similarity > this.rsmt.similarityThreshold) {
            console.log(`[RSMT Timeline] Smart transition found good match (similarity: ${match.similarity.toFixed(3)})`);
            
            return this.requestTransition(targetAnimationName, {
                ...options,
                startFrame: match.frameIndex,
                quality: 'high'
            });
        } else {
            console.log('[RSMT Timeline] Smart transition using standard method');
            return this.requestTransition(targetAnimationName, options);
        }
    }
    
    /**
     * Cleanup and disposal
     */
    dispose() {
        // Stop active transitions
        this.stopActiveTransitions();
        
        // Clear queues
        this.transitionQueue = [];
        
        // Clear loaded animations
        this.loadedAnimations.clear();
        
        // Dispose RSMT converter
        if (this.rsmt) {
            this.rsmt.dispose();
        }
        
        // Remove timeline track
        if (this.timeline.removeTrack) {
            this.timeline.removeTrack(this.config.transitionTrackName);
        }
        
        console.log('[RSMT Timeline] Integration disposed');
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RSMTTimelineIntegration;
} else {
    // Browser global
    window.RSMTTimelineIntegration = RSMTTimelineIntegration;
}
