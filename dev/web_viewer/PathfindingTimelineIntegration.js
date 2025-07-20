/**
 * Pathfinding Timeline Integration
 * 
 * Integration layer connecting the PathfindingBVHPlanner with the BVH timeline
 * system. Provides seamless pathfinding-driven animation planning and execution.
 * 
 * Features:
 * - Real-time path planning and animation generation
 * - Timeline layer management for pathfinding
 * - Dynamic obstacle avoidance and replanning
 * - Animation backend coordination
 * - Interactive destination setting and path visualization
 */

class PathfindingTimelineIntegration {
    constructor(timeline, options = {}) {
        if (!timeline) {
            throw new Error('Timeline reference is required');
        }
        
        this.timeline = timeline;
        this.pathfinder = new PathfindingBVHPlanner(options.pathfinding || {});
        
        // Integration configuration
        this.config = {
            layerName: options.layerName || 'pathfinding',
            priority: options.priority || 5,
            autoReplan: options.autoReplan !== false,
            replanThreshold: options.replanThreshold || 2.0, // meters
            updateInterval: options.updateInterval || 100, // ms
            previewMode: options.previewMode || false,
            debugMode: options.debugMode || false
        };
        
        // Animation backend references
        this.backends = {
            audio2gesture: null,
            deepmimic: null,
            rsmt: null,
            faceformer: null
        };
        
        // Integration state
        this.state = {
            isInitialized: false,
            isPlanning: false,
            hasActivePath: false,
            currentDestination: null,
            activePlan: null,
            pathfindingLayer: null,
            lastUpdateTime: 0,
            planningQueue: []
        };
        
        // Event handling
        this.eventHandlers = new Map();
        this.bindTimelineEvents();
        
        // Performance monitoring
        this.performance = {
            planningTime: 0,
            executionTime: 0,
            replanCount: 0,
            frameDrops: 0,
            lastFrameTime: 0
        };
        
        console.log('[Pathfinding Integration] Initialized');
    }
    
    /**
     * Initialize pathfinding integration
     */
    async initialize(backends = {}, environment = {}) {
        try {
            console.log('[Pathfinding Integration] Initializing...');
            
            // Store backend references
            this.backends = { ...this.backends, ...backends };
            
            // Initialize pathfinder with backends
            const pathfindingBackends = {
                ...this.backends,
                timeline: this.timeline
            };
            
            await this.pathfinder.initialize(pathfindingBackends, environment);
            
            // Set up timeline layer
            await this.setupPathfindingLayer();
            
            // Start update loop if auto-update is enabled
            if (this.config.updateInterval > 0) {
                this.startUpdateLoop();
            }
            
            this.state.isInitialized = true;
            console.log('[Pathfinding Integration] Initialization complete');
            
            return true;
            
        } catch (error) {
            console.error('[Pathfinding Integration] Initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Set up pathfinding timeline layer
     */
    async setupPathfindingLayer() {
        const layer = {
            id: this.config.layerName,
            name: 'Pathfinding Movement',
            type: 'pathfinding',
            priority: this.config.priority,
            clips: [],
            isActive: true,
            blendMode: 'additive',
            metadata: {
                managedBy: 'pathfinding_integration',
                version: '1.0.0'
            }
        };
        
        // Add layer to timeline
        if (this.timeline.addLayer) {
            await this.timeline.addLayer(layer);
            this.state.pathfindingLayer = layer;
            console.log('[Pathfinding Integration] Timeline layer created');
        } else {
            console.warn('[Pathfinding Integration] Timeline does not support layers');
        }
    }
    
    /**
     * Plan path to destination
     */
    async planPathTo(destination, options = {}) {
        try {
            console.log('[Pathfinding Integration] Planning path to:', destination);
            
            this.state.isPlanning = true;
            this.state.currentDestination = destination;
            
            const startTime = performance.now();
            
            // Get current character state from timeline
            const currentTime = this.timeline.getCurrentTime();
            const currentState = await this.getCurrentCharacterState(currentTime);
            
            // Update pathfinder with current state
            this.pathfinder.updateCharacterState(
                currentState.position,
                currentState.orientation,
                currentState.velocity
            );
            
            // Plan path and generate animations
            const plan = await this.pathfinder.planPathToDestination(destination, {
                ...options,
                startTime: currentTime,
                frameRate: this.timeline.frameRate || 30
            });
            
            // Create timeline integration
            const timelineLayer = await this.integratePathWithTimeline(plan, options);
            
            // Store active plan
            this.state.activePlan = {
                ...plan,
                timelineLayer: timelineLayer,
                startTime: currentTime,
                destination: destination,
                options: options
            };
            
            this.state.hasActivePath = true;
            this.performance.planningTime = performance.now() - startTime;
            
            // Emit planning complete event
            this.emit('pathPlanned', {
                plan: this.state.activePlan,
                planningTime: this.performance.planningTime
            });
            
            console.log(`[Pathfinding Integration] Path planned in ${this.performance.planningTime.toFixed(2)}ms`);
            return this.state.activePlan;
            
        } catch (error) {
            console.error('[Pathfinding Integration] Path planning failed:', error);
            this.emit('planningError', { error, destination });
            throw error;
        } finally {
            this.state.isPlanning = false;
        }
    }
    
    /**
     * Integrate path plan with timeline
     */
    async integratePathWithTimeline(plan, options = {}) {
        // Clear existing pathfinding animations if requested
        if (options.clearExisting !== false) {
            await this.clearPathfindingAnimations();
        }
        
        // Create timeline layer from plan
        const timelineLayer = plan.timelineLayer;
        
        // Add clips to timeline
        for (const clip of timelineLayer.clips) {
            await this.addPathfindingClip(clip, options);
        }
        
        // Set up coordination with other animation backends
        await this.coordinateWithBackends(plan, options);
        
        console.log(`[Pathfinding Integration] Integrated ${timelineLayer.clips.length} clips`);
        return timelineLayer;
    }
    
    /**
     * Add pathfinding clip to timeline
     */
    async addPathfindingClip(clip, options = {}) {
        const timelineClip = {
            id: clip.id,
            trackName: this.config.layerName,
            startTime: clip.startTime,
            duration: clip.endTime - clip.startTime,
            frames: clip.frames,
            priority: this.config.priority,
            blendMode: options.blendMode || 'replace',
            metadata: {
                ...clip,
                source: 'pathfinding',
                animationType: clip.animationType
            }
        };
        
        // Add to timeline
        if (this.timeline.addClip) {
            await this.timeline.addClip(this.config.layerName, timelineClip);
        } else if (this.timeline.tracks && this.timeline.tracks[this.config.layerName]) {
            this.timeline.tracks[this.config.layerName].clips.push(timelineClip);
        }
        
        console.log(`[Pathfinding Integration] Added clip: ${clip.id}`);
    }
    
    /**
     * Coordinate pathfinding with other animation backends
     */
    async coordinateWithBackends(plan, options = {}) {
        const coordination = options.coordination || {};
        
        // Coordinate with Audio2Gesture for upper body gestures
        if (this.backends.audio2gesture && coordination.enableGestures !== false) {
            await this.coordinateWithAudio2Gesture(plan, coordination.gestures || {});
        }
        
        // Coordinate with FaceFormer for facial expressions
        if (this.backends.faceformer && coordination.enableFacial !== false) {
            await this.coordinateWithFaceFormer(plan, coordination.facial || {});
        }
        
        // Use DeepMimic for physics-based refinement
        if (this.backends.deepmimic && coordination.enablePhysics !== false) {
            await this.coordinateWithDeepMimic(plan, coordination.physics || {});
        }
        
        // Use RSMT for smooth transitions
        if (this.backends.rsmt && coordination.enableTransitions !== false) {
            await this.coordinateWithRSMT(plan, coordination.transitions || {});
        }
    }
    
    /**
     * Coordinate with Audio2Gesture for upper body animations
     */
    async coordinateWithAudio2Gesture(plan, options = {}) {
        console.log('[Pathfinding Integration] Coordinating with Audio2Gesture...');
        
        try {
            // Generate appropriate gestures for movement
            for (const segment of plan.animationSequence) {
                if (segment.animationType === 'walk' || segment.animationType === 'run') {
                    // Generate walking/running arm movements
                    const gestureData = await this.generateMovementGestures(segment, options);
                    
                    if (gestureData && this.backends.audio2gesture.addGestureSequence) {
                        await this.backends.audio2gesture.addGestureSequence({
                            startTime: segment.startTime,
                            duration: segment.duration,
                            gestureType: 'locomotion',
                            data: gestureData,
                            priority: 3 // Lower than pathfinding
                        });
                    }
                }
            }
            
        } catch (error) {
            console.warn('[Pathfinding Integration] Audio2Gesture coordination failed:', error);
        }
    }
    
    /**
     * Coordinate with FaceFormer for emotional expressions
     */
    async coordinateWithFaceFormer(plan, options = {}) {
        console.log('[Pathfinding Integration] Coordinating with FaceFormer...');
        
        try {
            // Generate appropriate facial expressions based on movement
            const expressions = this.generateMovementExpressions(plan, options);
            
            if (expressions.length > 0 && this.backends.faceformer.addExpressionSequence) {
                await this.backends.faceformer.addExpressionSequence({
                    expressions: expressions,
                    blendMode: 'additive',
                    priority: 2
                });
            }
            
        } catch (error) {
            console.warn('[Pathfinding Integration] FaceFormer coordination failed:', error);
        }
    }
    
    /**
     * Coordinate with DeepMimic for physics refinement
     */
    async coordinateWithDeepMimic(plan, options = {}) {
        console.log('[Pathfinding Integration] Coordinating with DeepMimic...');
        
        try {
            // Use DeepMimic to refine physics-based aspects
            if (this.backends.deepmimic.refinePhysics) {
                const refinedPlan = await this.backends.deepmimic.refinePhysics({
                    keyframes: plan.keyframes,
                    constraints: options.constraints || {},
                    quality: options.quality || 'high'
                });
                
                // Update timeline with refined physics
                if (refinedPlan) {
                    await this.applyPhysicsRefinement(refinedPlan);
                }
            }
            
        } catch (error) {
            console.warn('[Pathfinding Integration] DeepMimic coordination failed:', error);
        }
    }
    
    /**
     * Coordinate with RSMT for transition smoothing
     */
    async coordinateWithRSMT(plan, options = {}) {
        console.log('[Pathfinding Integration] Coordinating with RSMT...');
        
        try {
            // Use RSMT to smooth transitions between animation segments
            for (let i = 0; i < plan.animationSequence.length - 1; i++) {
                const currentSegment = plan.animationSequence[i];
                const nextSegment = plan.animationSequence[i + 1];
                
                if (currentSegment.animationType !== nextSegment.animationType) {
                    // Generate smooth transition
                    const transition = await this.backends.rsmt.generateTransition(
                        currentSegment.toKeyframe,
                        nextSegment.animationType,
                        {
                            duration: options.transitionDuration || 0.3,
                            quality: options.quality || 'high'
                        }
                    );
                    
                    if (transition) {
                        await this.addTransitionToTimeline(transition, currentSegment.endTime);
                    }
                }
            }
            
        } catch (error) {
            console.warn('[Pathfinding Integration] RSMT coordination failed:', error);
        }
    }
    
    /**
     * Real-time path monitoring and replanning
     */
    async startUpdateLoop() {
        const updateLoop = async () => {
            try {
                if (this.state.isInitialized && this.state.hasActivePath) {
                    await this.updatePathfinding();
                }
                
                // Schedule next update
                setTimeout(updateLoop, this.config.updateInterval);
                
            } catch (error) {
                console.error('[Pathfinding Integration] Update loop error:', error);
                setTimeout(updateLoop, this.config.updateInterval * 2); // Slower retry
            }
        };
        
        updateLoop();
        console.log('[Pathfinding Integration] Update loop started');
    }
    
    /**
     * Update pathfinding state and check for replanning
     */
    async updatePathfinding() {
        const currentTime = this.timeline.getCurrentTime();
        
        // Check if we need to replan
        if (this.config.autoReplan && await this.shouldReplan(currentTime)) {
            console.log('[Pathfinding Integration] Replanning path...');
            await this.replanPath();
            this.performance.replanCount++;
        }
        
        // Update character state
        await this.updateCharacterStateFromTimeline(currentTime);
        
        // Check for path completion
        if (await this.isPathComplete(currentTime)) {
            await this.handlePathCompletion();
        }
        
        this.state.lastUpdateTime = currentTime;
    }
    
    /**
     * Check if path replanning is needed
     */
    async shouldReplan(currentTime) {
        if (!this.state.activePlan) return false;
        
        // Get current character position
        const currentState = await this.getCurrentCharacterState(currentTime);
        
        // Check distance from planned path
        const plannedPosition = this.getPlannedPositionAtTime(currentTime);
        if (!plannedPosition) return false;
        
        const deviation = this.calculateDistance(currentState.position, plannedPosition);
        
        return deviation > this.config.replanThreshold;
    }
    
    /**
     * Replan current path
     */
    async replanPath() {
        if (!this.state.currentDestination) return;
        
        try {
            // Clear current plan
            await this.clearPathfindingAnimations();
            
            // Plan new path from current position
            await this.planPathTo(this.state.currentDestination, {
                ...this.state.activePlan?.options,
                isReplan: true
            });
            
            this.emit('pathReplanned', {
                destination: this.state.currentDestination,
                replanCount: this.performance.replanCount
            });
            
        } catch (error) {
            console.error('[Pathfinding Integration] Replanning failed:', error);
            this.emit('replanningError', { error });
        }
    }
    
    /**
     * Interactive destination setting
     */
    async setDestination(destination, options = {}) {
        console.log('[Pathfinding Integration] Setting destination:', destination);
        
        // Validate destination
        if (!this.isValidDestination(destination)) {
            throw new Error('Invalid destination: position is blocked or out of bounds');
        }
        
        // Cancel current planning if in progress
        if (this.state.isPlanning) {
            console.log('[Pathfinding Integration] Cancelling current planning...');
            // Implementation depends on pathfinder capabilities
        }
        
        // Plan path to new destination
        return await this.planPathTo(destination, options);
    }
    
    /**
     * Add obstacle to pathfinding
     */
    addObstacle(id, obstacle) {
        this.pathfinder.addObstacle(id, obstacle);
        
        // Trigger replanning if we have an active path
        if (this.state.hasActivePath && this.config.autoReplan) {
            setTimeout(() => this.replanPath(), 100);
        }
        
        this.emit('obstacleAdded', { id, obstacle });
    }
    
    /**
     * Remove obstacle from pathfinding
     */
    removeObstacle(id) {
        this.pathfinder.removeObstacle(id);
        this.emit('obstacleRemoved', { id });
    }
    
    /**
     * Preview path without executing
     */
    async previewPath(destination, options = {}) {
        const previewOptions = {
            ...options,
            preview: true,
            dryRun: true
        };
        
        try {
            const plan = await this.pathfinder.planPathToDestination(destination, previewOptions);
            
            this.emit('pathPreview', {
                destination,
                plan,
                isValid: plan && plan.path && plan.path.length > 0
            });
            
            return plan;
            
        } catch (error) {
            console.warn('[Pathfinding Integration] Path preview failed:', error);
            this.emit('pathPreview', {
                destination,
                plan: null,
                isValid: false,
                error
            });
            return null;
        }
    }
    
    /**
     * Utility methods
     */
    
    async getCurrentCharacterState(time) {
        // Get current frame from timeline
        const frame = await this.timeline.getFrameAtTime(time);
        
        if (!frame || !frame.bones || !frame.bones.hips) {
            // Return default state
            return {
                position: { x: 0, y: 0, z: 0 },
                orientation: { x: 0, y: 0, z: 0, w: 1 },
                velocity: { x: 0, y: 0, z: 0 }
            };
        }
        
        const hips = frame.bones.hips;
        
        // Calculate velocity from previous frame
        const prevFrame = await this.timeline.getFrameAtTime(time - 1/30); // Assume 30fps
        let velocity = { x: 0, y: 0, z: 0 };
        
        if (prevFrame && prevFrame.bones && prevFrame.bones.hips) {
            const dt = 1/30;
            velocity = {
                x: (hips.position.x - prevFrame.bones.hips.position.x) / dt,
                y: (hips.position.y - prevFrame.bones.hips.position.y) / dt,
                z: (hips.position.z - prevFrame.bones.hips.position.z) / dt
            };
        }
        
        return {
            position: hips.position,
            orientation: hips.rotation,
            velocity: velocity
        };
    }
    
    getPlannedPositionAtTime(time) {
        if (!this.state.activePlan || !this.state.activePlan.keyframes) return null;
        
        const keyframes = this.state.activePlan.keyframes;
        
        // Find surrounding keyframes
        for (let i = 0; i < keyframes.length - 1; i++) {
            const current = keyframes[i];
            const next = keyframes[i + 1];
            
            if (time >= current.time && time <= next.time) {
                // Interpolate position
                const t = (time - current.time) / (next.time - current.time);
                return {
                    x: current.position.x + (next.position.x - current.position.x) * t,
                    y: current.position.y + (next.position.y - current.position.y) * t,
                    z: current.position.z + (next.position.z - current.position.z) * t
                };
            }
        }
        
        return null;
    }
    
    calculateDistance(pos1, pos2) {
        return Math.sqrt(
            Math.pow(pos2.x - pos1.x, 2) +
            Math.pow(pos2.y - pos1.y, 2) +
            Math.pow(pos2.z - pos1.z, 2)
        );
    }
    
    isValidDestination(destination) {
        // Check if destination is within bounds and not blocked
        return !this.pathfinder.isPositionBlocked(destination);
    }
    
    async updateCharacterStateFromTimeline(time) {
        const state = await this.getCurrentCharacterState(time);
        this.pathfinder.updateCharacterState(
            state.position,
            state.orientation,
            state.velocity
        );
    }
    
    async isPathComplete(time) {
        if (!this.state.activePlan) return false;
        
        const endTime = this.state.activePlan.startTime + this.state.activePlan.estimatedDuration * 1000;
        return time >= endTime;
    }
    
    async handlePathCompletion() {
        console.log('[Pathfinding Integration] Path completed');
        
        this.state.hasActivePath = false;
        this.state.activePlan = null;
        this.state.currentDestination = null;
        
        this.emit('pathCompleted', {
            completionTime: this.timeline.getCurrentTime()
        });
    }
    
    async clearPathfindingAnimations() {
        if (this.state.pathfindingLayer) {
            // Clear clips from timeline layer
            if (this.timeline.clearLayer) {
                await this.timeline.clearLayer(this.config.layerName);
            } else if (this.timeline.tracks && this.timeline.tracks[this.config.layerName]) {
                this.timeline.tracks[this.config.layerName].clips = [];
            }
        }
    }
    
    generateMovementGestures(segment, options) {
        // Generate procedural arm movements for walking/running
        const gestureType = segment.animationType === 'run' ? 'running_arms' : 'walking_arms';
        
        return {
            type: gestureType,
            intensity: segment.animationType === 'run' ? 0.8 : 0.5,
            frequency: segment.animationType === 'run' ? 2.5 : 1.5,
            duration: segment.duration
        };
    }
    
    generateMovementExpressions(plan, options) {
        const expressions = [];
        
        // Add subtle expressions based on movement type
        for (const segment of plan.animationSequence) {
            let expression = 'neutral';
            
            if (segment.animationType === 'run' || segment.animationType === 'sprint') {
                expression = 'focused';
            } else if (segment.animationType === 'walk') {
                expression = 'calm';
            }
            
            expressions.push({
                startTime: segment.startTime,
                duration: segment.duration,
                expression: expression,
                intensity: 0.3
            });
        }
        
        return expressions;
    }
    
    async applyPhysicsRefinement(refinedPlan) {
        // Apply physics refinements to timeline
        console.log('[Pathfinding Integration] Applying physics refinement...');
        // Implementation would update timeline clips with refined data
    }
    
    async addTransitionToTimeline(transition, startTime) {
        const transitionClip = {
            id: `transition_${Date.now()}`,
            startTime: startTime,
            duration: transition.duration * 1000,
            frames: transition.frames,
            priority: this.config.priority + 1, // Higher priority
            blendMode: 'blend',
            metadata: {
                source: 'rsmt_transition',
                generatedBy: 'pathfinding_integration'
            }
        };
        
        await this.addPathfindingClip(transitionClip);
    }
    
    /**
     * Event handling
     */
    
    bindTimelineEvents() {
        // Bind to timeline events if available
        if (this.timeline.on) {
            this.timeline.on('positionChanged', (time) => {
                this.onTimelinePositionChanged(time);
            });
            
            this.timeline.on('playbackStopped', () => {
                this.onTimelinePlaybackStopped();
            });
        }
    }
    
    onTimelinePositionChanged(time) {
        // Update pathfinding state when timeline position changes
        if (this.state.hasActivePath) {
            this.updateCharacterStateFromTimeline(time);
        }
    }
    
    onTimelinePlaybackStopped() {
        // Handle timeline stopping
        console.log('[Pathfinding Integration] Timeline playback stopped');
    }
    
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }
    
    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index !== -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            for (const handler of handlers) {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`[Pathfinding Integration] Event handler error for ${event}:`, error);
                }
            }
        }
        
        if (this.config.debugMode) {
            console.log(`[Pathfinding Integration] Event: ${event}`, data);
        }
    }
    
    /**
     * Management methods
     */
    
    getStatus() {
        return {
            isInitialized: this.state.isInitialized,
            isPlanning: this.state.isPlanning,
            hasActivePath: this.state.hasActivePath,
            currentDestination: this.state.currentDestination,
            activePlan: this.state.activePlan ? {
                startTime: this.state.activePlan.startTime,
                duration: this.state.activePlan.estimatedDuration,
                keyframeCount: this.state.activePlan.keyframes?.length || 0
            } : null,
            performance: this.performance,
            pathfinderStats: this.pathfinder.getStats()
        };
    }
    
    getActivePlan() {
        return this.state.activePlan;
    }
    
    async cancelCurrentPlan() {
        if (this.state.hasActivePath) {
            await this.clearPathfindingAnimations();
            this.state.hasActivePath = false;
            this.state.activePlan = null;
            this.state.currentDestination = null;
            
            this.emit('pathCancelled', {
                cancelTime: this.timeline.getCurrentTime()
            });
            
            console.log('[Pathfinding Integration] Current plan cancelled');
        }
    }
    
    dispose() {
        // Clean up resources
        this.clearPathfindingAnimations();
        this.pathfinder.dispose();
        this.eventHandlers.clear();
        
        console.log('[Pathfinding Integration] Disposed');
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PathfindingTimelineIntegration };
} else {
    // Browser global
    window.PathfindingTimelineIntegration = PathfindingTimelineIntegration;
}
