/**
 * Pathfinding BVH Planner
 * 
 * Advanced pathfinding system that generates animation sequences to reach
 * destination points using available animation backends. Creates keyframe
 * sequences that can be integrated into the BVH timeline system.
 * 
 * Features:
 * - A* pathfinding with obstacle avoidance
 * - Animation backend integration (Audio2Gesture, DeepMimic, RSMT)
 * - Keyframe generation with timing and spatial constraints
 * - Multi-layer timeline integration
 * - Character locomotion planning
 * - Adaptive path refinement
 */

class PathfindingBVHPlanner {
    constructor(options = {}) {
        // Core configuration
        this.gridResolution = options.gridResolution || 0.5; // meters per grid cell
        this.maxPlanningDistance = options.maxPlanningDistance || 50; // meters
        this.characterRadius = options.characterRadius || 0.3; // meters
        this.maxStepHeight = options.maxStepHeight || 0.2; // meters
        
        // Movement constraints
        this.movementConfig = {
            maxSpeed: options.maxSpeed || 2.0, // m/s
            maxAcceleration: options.maxAcceleration || 3.0, // m/s²
            maxTurnRate: options.maxTurnRate || Math.PI, // rad/s
            preferredSpeed: options.preferredSpeed || 1.2, // m/s
            stoppingDistance: options.stoppingDistance || 0.1, // meters
            smoothingFactor: options.smoothingFactor || 0.8
        };
        
        // Animation backend references
        this.animationBackends = {
            audio2gesture: null,
            deepmimic: null,
            rsmt: null,
            timeline: null
        };
        
        // Pathfinding data structures
        this.navigationGrid = null;
        this.obstacles = new Map(); // obstacle_id -> obstacle_data
        this.waypoints = new Map(); // waypoint_id -> waypoint_data
        this.currentPath = null;
        this.activeKeyframes = [];
        
        // Locomotion animation library
        this.locomotionLibrary = {
            idle: { name: 'idle', speed: 0, blendWeight: 1.0 },
            walk: { name: 'walk', speed: 1.2, blendWeight: 1.0 },
            run: { name: 'run', speed: 3.0, blendWeight: 1.0 },
            turn_left: { name: 'turn_left', turnRate: Math.PI/2, blendWeight: 0.8 },
            turn_right: { name: 'turn_right', turnRate: -Math.PI/2, blendWeight: 0.8 },
            start_walking: { name: 'start_walking', transition: true, duration: 0.5 },
            stop_walking: { name: 'stop_walking', transition: true, duration: 0.3 },
            step_up: { name: 'step_up', elevation: 0.1, duration: 0.8 },
            step_down: { name: 'step_down', elevation: -0.1, duration: 0.6 }
        };
        
        // Planning state
        this.planningState = {
            currentPosition: { x: 0, y: 0, z: 0 },
            currentOrientation: { x: 0, y: 0, z: 0, w: 1 },
            currentVelocity: { x: 0, y: 0, z: 0 },
            targetPosition: null,
            targetOrientation: null,
            isPlanning: false,
            planningStartTime: 0
        };
        
        // Performance tracking
        this.stats = {
            pathsPlanned: 0,
            keyframesGenerated: 0,
            averagePlanningTime: 0,
            successfulPaths: 0,
            totalDistance: 0,
            cacheHits: 0
        };
        
        // Path cache for optimization
        this.pathCache = new Map();
        this.maxCacheSize = options.maxCacheSize || 100;
        
        console.log('[Pathfinding] BVH Planner initialized');
    }
    
    /**
     * Initialize pathfinding system with animation backends
     */
    async initialize(backends, environment = {}) {
        try {
            console.log('[Pathfinding] Initializing with backends...');
            
            // Store backend references
            this.animationBackends = { ...this.animationBackends, ...backends };
            
            // Initialize navigation grid
            await this.initializeNavigationGrid(environment);
            
            // Load locomotion animations
            await this.loadLocomotionAnimations();
            
            console.log('[Pathfinding] Initialization complete');
            return true;
            
        } catch (error) {
            console.error('[Pathfinding] Initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Initialize navigation grid for pathfinding
     */
    async initializeNavigationGrid(environment) {
        const bounds = environment.bounds || {
            minX: -25, maxX: 25,
            minZ: -25, maxZ: 25,
            minY: 0, maxY: 5
        };
        
        const width = Math.ceil((bounds.maxX - bounds.minX) / this.gridResolution);
        const height = Math.ceil((bounds.maxZ - bounds.minZ) / this.gridResolution);
        
        this.navigationGrid = {
            bounds: bounds,
            width: width,
            height: height,
            resolution: this.gridResolution,
            cells: new Array(width * height).fill(0), // 0 = free, 1 = blocked
            costs: new Array(width * height).fill(1.0), // movement cost multiplier
            metadata: environment.metadata || {}
        };
        
        console.log(`[Pathfinding] Navigation grid initialized: ${width}x${height}`);
        
        // Process initial obstacles from environment
        if (environment.obstacles) {
            for (const obstacle of environment.obstacles) {
                this.addObstacle(obstacle.id, obstacle);
            }
        }
    }
    
    /**
     * Load locomotion animations from backends
     */
    async loadLocomotionAnimations() {
        console.log('[Pathfinding] Loading locomotion animations...');
        
        // Load basic locomotion set
        const animationsToLoad = [
            'idle', 'walk', 'run', 'turn_left', 'turn_right',
            'start_walking', 'stop_walking', 'step_up', 'step_down'
        ];
        
        // Use RSMT backend to load locomotion library
        if (this.animationBackends.rsmt) {
            for (const animName of animationsToLoad) {
                try {
                    // Try to load from assets folder
                    const animPath = `/assets/locomotion/${animName}.json`;
                    await this.animationBackends.rsmt.loadAnimation(animName, animPath);
                    console.log(`[Pathfinding] Loaded locomotion: ${animName}`);
                } catch (error) {
                    console.warn(`[Pathfinding] Could not load ${animName}:`, error);
                    // Create procedural fallback
                    this.createProceduralLocomotion(animName);
                }
            }
        }
    }
    
    /**
     * Plan path to destination and generate keyframes
     */
    async planPathToDestination(destination, options = {}) {
        const startTime = performance.now();
        
        try {
            console.log('[Pathfinding] Planning path to:', destination);
            
            this.planningState.isPlanning = true;
            this.planningState.planningStartTime = startTime;
            this.planningState.targetPosition = destination;
            this.planningState.targetOrientation = options.targetOrientation;
            
            // Check cache first
            const cacheKey = this.generatePathCacheKey(
                this.planningState.currentPosition,
                destination,
                options
            );
            
            if (this.pathCache.has(cacheKey)) {
                console.log('[Pathfinding] Using cached path');
                this.stats.cacheHits++;
                const cachedResult = this.pathCache.get(cacheKey);
                return this.adaptCachedPath(cachedResult, options);
            }
            
            // Step 1: Find optimal path using A*
            const path = await this.findOptimalPath(
                this.planningState.currentPosition,
                destination,
                options
            );
            
            if (!path || path.length === 0) {
                throw new Error('No valid path found to destination');
            }
            
            // Step 2: Smooth and optimize path
            const smoothedPath = this.smoothPath(path, options);
            
            // Step 3: Generate timing keyframes
            const timedKeyframes = await this.generateTimedKeyframes(smoothedPath, options);
            
            // Step 4: Select appropriate animations for each segment
            const animationSequence = await this.planAnimationSequence(timedKeyframes, options);
            
            // Step 5: Generate BVH keyframes for timeline integration
            const bvhKeyframes = await this.generateBVHKeyframes(animationSequence, options);
            
            // Step 6: Create timeline layer
            const timelineLayer = this.createTimelineLayer(bvhKeyframes, options);
            
            const result = {
                path: smoothedPath,
                keyframes: timedKeyframes,
                animationSequence: animationSequence,
                bvhKeyframes: bvhKeyframes,
                timelineLayer: timelineLayer,
                totalDistance: this.calculatePathDistance(smoothedPath),
                estimatedDuration: timedKeyframes[timedKeyframes.length - 1]?.time || 0,
                planningTime: performance.now() - startTime
            };
            
            // Cache the result
            this.cachePathResult(cacheKey, result);
            
            // Update statistics
            this.updatePlanningStats(result);
            
            console.log(`[Pathfinding] Path planned in ${result.planningTime.toFixed(2)}ms`);
            return result;
            
        } catch (error) {
            console.error('[Pathfinding] Path planning failed:', error);
            throw error;
        } finally {
            this.planningState.isPlanning = false;
        }
    }
    
    /**
     * Find optimal path using A* algorithm
     */
    async findOptimalPath(start, goal, options = {}) {
        console.log('[Pathfinding] Running A* pathfinding...');
        
        const startGrid = this.worldToGrid(start);
        const goalGrid = this.worldToGrid(goal);
        
        // A* data structures
        const openSet = new PriorityQueue();
        const closedSet = new Set();
        const cameFrom = new Map();
        const gScore = new Map();
        const fScore = new Map();
        
        const startKey = this.gridToKey(startGrid);
        const goalKey = this.gridToKey(goalGrid);
        
        // Initialize start node
        gScore.set(startKey, 0);
        fScore.set(startKey, this.heuristic(startGrid, goalGrid));
        openSet.enqueue(startGrid, fScore.get(startKey));
        
        while (!openSet.isEmpty()) {
            const current = openSet.dequeue();
            const currentKey = this.gridToKey(current);
            
            // Check if we reached the goal
            if (currentKey === goalKey) {
                console.log('[Pathfinding] Path found!');
                return this.reconstructPath(cameFrom, current);
            }
            
            closedSet.add(currentKey);
            
            // Explore neighbors
            const neighbors = this.getNeighbors(current);
            
            for (const neighbor of neighbors) {
                const neighborKey = this.gridToKey(neighbor);
                
                if (closedSet.has(neighborKey)) continue;
                if (this.isBlocked(neighbor)) continue;
                
                const tentativeGScore = gScore.get(currentKey) + this.getMovementCost(current, neighbor);
                
                if (!gScore.has(neighborKey) || tentativeGScore < gScore.get(neighborKey)) {
                    cameFrom.set(neighborKey, current);
                    gScore.set(neighborKey, tentativeGScore);
                    fScore.set(neighborKey, tentativeGScore + this.heuristic(neighbor, goalGrid));
                    
                    if (!openSet.contains(neighbor)) {
                        openSet.enqueue(neighbor, fScore.get(neighborKey));
                    }
                }
            }
        }
        
        console.warn('[Pathfinding] No path found to destination');
        return null;
    }
    
    /**
     * Smooth path using spline interpolation
     */
    smoothPath(path, options = {}) {
        if (path.length < 3) return path;
        
        const smoothedPath = [path[0]]; // Keep start point
        const smoothingPasses = options.smoothingPasses || 2;
        
        for (let pass = 0; pass < smoothingPasses; pass++) {
            const tempPath = [...smoothedPath];
            
            for (let i = 1; i < path.length - 1; i++) {
                const prev = path[i - 1];
                const current = path[i];
                const next = path[i + 1];
                
                // Apply smoothing filter
                const smoothed = {
                    x: prev.x * 0.25 + current.x * 0.5 + next.x * 0.25,
                    y: prev.y * 0.25 + current.y * 0.5 + next.y * 0.25,
                    z: prev.z * 0.25 + current.z * 0.5 + next.z * 0.25
                };
                
                // Check if smoothed point is still valid
                if (!this.isPositionBlocked(smoothed)) {
                    tempPath.push(smoothed);
                } else {
                    tempPath.push(current); // Keep original if smoothed is blocked
                }
            }
            
            smoothedPath.splice(1, smoothedPath.length - 1, ...tempPath.slice(1));
        }
        
        smoothedPath.push(path[path.length - 1]); // Keep end point
        
        console.log(`[Pathfinding] Path smoothed: ${path.length} -> ${smoothedPath.length} points`);
        return smoothedPath;
    }
    
    /**
     * Generate timed keyframes with movement constraints
     */
    async generateTimedKeyframes(path, options = {}) {
        console.log('[Pathfinding] Generating timed keyframes...');
        
        const keyframes = [];
        let currentTime = options.startTime || 0;
        let currentVelocity = { ...this.planningState.currentVelocity };
        
        for (let i = 0; i < path.length; i++) {
            const point = path[i];
            const nextPoint = path[i + 1];
            
            // Calculate desired velocity and timing
            let segmentData = null;
            if (nextPoint) {
                segmentData = this.calculateSegmentTiming(point, nextPoint, currentVelocity);
                currentVelocity = segmentData.endVelocity;
            }
            
            const keyframe = {
                id: `keyframe_${i}`,
                time: currentTime,
                position: { ...point },
                velocity: { ...currentVelocity },
                orientation: this.calculateOrientation(point, nextPoint),
                movementType: this.classifyMovement(segmentData),
                constraints: {
                    maxSpeed: this.movementConfig.maxSpeed,
                    maxAcceleration: this.movementConfig.maxAcceleration
                },
                metadata: {
                    pathIndex: i,
                    isWaypoint: this.isWaypoint(point),
                    terrainType: this.getTerrainType(point)
                }
            };
            
            keyframes.push(keyframe);
            
            // Update time for next keyframe
            if (segmentData) {
                currentTime += segmentData.duration;
            }
        }
        
        console.log(`[Pathfinding] Generated ${keyframes.length} keyframes`);
        return keyframes;
    }
    
    /**
     * Plan animation sequence for keyframes
     */
    async planAnimationSequence(keyframes, options = {}) {
        console.log('[Pathfinding] Planning animation sequence...');
        
        const sequence = [];
        
        for (let i = 0; i < keyframes.length - 1; i++) {
            const currentKeyframe = keyframes[i];
            const nextKeyframe = keyframes[i + 1];
            
            // Determine required animation type
            const animationType = this.determineAnimationType(currentKeyframe, nextKeyframe);
            
            // Create animation segment
            const segment = {
                id: `segment_${i}`,
                startTime: currentKeyframe.time,
                endTime: nextKeyframe.time,
                duration: nextKeyframe.time - currentKeyframe.time,
                animationType: animationType,
                fromKeyframe: currentKeyframe,
                toKeyframe: nextKeyframe,
                animationData: await this.generateAnimationData(
                    animationType,
                    currentKeyframe,
                    nextKeyframe,
                    options
                ),
                blendSettings: {
                    blendIn: 0.1,
                    blendOut: 0.1,
                    priority: this.getAnimationPriority(animationType)
                }
            };
            
            sequence.push(segment);
        }
        
        console.log(`[Pathfinding] Animation sequence planned: ${sequence.length} segments`);
        return sequence;
    }
    
    /**
     * Generate BVH keyframes for timeline integration
     */
    async generateBVHKeyframes(animationSequence, options = {}) {
        console.log('[Pathfinding] Generating BVH keyframes...');
        
        const bvhKeyframes = [];
        
        for (const segment of animationSequence) {
            // Generate frames for this segment
            const frameCount = Math.ceil(segment.duration * (options.frameRate || 30));
            
            for (let frame = 0; frame < frameCount; frame++) {
                const progress = frame / Math.max(1, frameCount - 1);
                const time = segment.startTime + segment.duration * progress;
                
                // Interpolate between keyframes
                const position = this.interpolatePosition(
                    segment.fromKeyframe.position,
                    segment.toKeyframe.position,
                    progress
                );
                
                const orientation = this.interpolateOrientation(
                    segment.fromKeyframe.orientation,
                    segment.toKeyframe.orientation,
                    progress
                );
                
                // Generate BVH frame data
                const bvhFrame = await this.generateBVHFrame(
                    position,
                    orientation,
                    segment.animationData,
                    progress,
                    options
                );
                
                bvhKeyframes.push({
                    time: time,
                    frame: bvhFrame,
                    segment: segment.id,
                    metadata: {
                        animationType: segment.animationType,
                        progress: progress,
                        generatedBy: 'pathfinding'
                    }
                });
            }
        }
        
        console.log(`[Pathfinding] Generated ${bvhKeyframes.length} BVH keyframes`);
        return bvhKeyframes;
    }
    
    /**
     * Create timeline layer for integration
     */
    createTimelineLayer(bvhKeyframes, options = {}) {
        const layer = {
            id: options.layerId || `pathfinding_${Date.now()}`,
            name: options.layerName || 'Pathfinding Movement',
            type: 'pathfinding',
            priority: options.priority || 5, // Medium priority
            clips: [],
            metadata: {
                generatedBy: 'pathfinding',
                totalKeyframes: bvhKeyframes.length,
                startTime: bvhKeyframes[0]?.time || 0,
                endTime: bvhKeyframes[bvhKeyframes.length - 1]?.time || 0
            }
        };
        
        // Group keyframes into clips
        let currentClip = null;
        
        for (const keyframe of bvhKeyframes) {
            if (!currentClip || keyframe.segment !== currentClip.segmentId) {
                // Start new clip
                if (currentClip) {
                    layer.clips.push(currentClip);
                }
                
                currentClip = {
                    id: `clip_${keyframe.segment}`,
                    segmentId: keyframe.segment,
                    startTime: keyframe.time,
                    endTime: keyframe.time,
                    frames: [],
                    animationType: keyframe.metadata.animationType
                };
            }
            
            currentClip.frames.push(keyframe.frame);
            currentClip.endTime = keyframe.time;
        }
        
        if (currentClip) {
            layer.clips.push(currentClip);
        }
        
        console.log(`[Pathfinding] Timeline layer created with ${layer.clips.length} clips`);
        return layer;
    }
    
    /**
     * Add timeline layer to BVH timeline system
     */
    async addToTimeline(timelineLayer, options = {}) {
        if (!this.animationBackends.timeline) {
            throw new Error('Timeline backend not available');
        }
        
        try {
            // Add layer to timeline
            if (this.animationBackends.timeline.addLayer) {
                await this.animationBackends.timeline.addLayer(timelineLayer);
                console.log(`[Pathfinding] Layer added to timeline: ${timelineLayer.id}`);
                
                // Optionally start playback
                if (options.autoPlay) {
                    this.animationBackends.timeline.play();
                }
                
                return timelineLayer;
            } else {
                console.warn('[Pathfinding] Timeline does not support addLayer method');
                return timelineLayer;
            }
            
        } catch (error) {
            console.error('[Pathfinding] Failed to add layer to timeline:', error);
            throw error;
        }
    }
    
    /**
     * Update character position from timeline feedback
     */
    updateCharacterState(position, orientation, velocity) {
        this.planningState.currentPosition = { ...position };
        this.planningState.currentOrientation = { ...orientation };
        this.planningState.currentVelocity = { ...velocity };
        
        // Update active keyframes if needed
        this.updateActiveKeyframes();
    }
    
    /**
     * Obstacle and environment management
     */
    
    addObstacle(id, obstacle) {
        this.obstacles.set(id, obstacle);
        this.updateNavigationGrid(obstacle, true);
        console.log(`[Pathfinding] Obstacle added: ${id}`);
    }
    
    removeObstacle(id) {
        const obstacle = this.obstacles.get(id);
        if (obstacle) {
            this.obstacles.delete(id);
            this.updateNavigationGrid(obstacle, false);
            console.log(`[Pathfinding] Obstacle removed: ${id}`);
        }
    }
    
    addWaypoint(id, waypoint) {
        this.waypoints.set(id, waypoint);
        console.log(`[Pathfinding] Waypoint added: ${id}`);
    }
    
    removeWaypoint(id) {
        this.waypoints.delete(id);
        console.log(`[Pathfinding] Waypoint removed: ${id}`);
    }
    
    /**
     * Utility methods
     */
    
    worldToGrid(worldPos) {
        const bounds = this.navigationGrid.bounds;
        return {
            x: Math.floor((worldPos.x - bounds.minX) / this.gridResolution),
            z: Math.floor((worldPos.z - bounds.minZ) / this.gridResolution)
        };
    }
    
    gridToWorld(gridPos) {
        const bounds = this.navigationGrid.bounds;
        return {
            x: bounds.minX + gridPos.x * this.gridResolution + this.gridResolution / 2,
            y: 0, // Will be adjusted based on terrain
            z: bounds.minZ + gridPos.z * this.gridResolution + this.gridResolution / 2
        };
    }
    
    gridToKey(gridPos) {
        return `${gridPos.x},${gridPos.z}`;
    }
    
    heuristic(pos1, pos2) {
        // Manhattan distance with diagonal movement
        const dx = Math.abs(pos1.x - pos2.x);
        const dz = Math.abs(pos1.z - pos2.z);
        return Math.max(dx, dz) + (Math.sqrt(2) - 1) * Math.min(dx, dz);
    }
    
    getNeighbors(gridPos) {
        const neighbors = [];
        const directions = [
            { x: -1, z: -1 }, { x: 0, z: -1 }, { x: 1, z: -1 },
            { x: -1, z: 0 },                   { x: 1, z: 0 },
            { x: -1, z: 1 },  { x: 0, z: 1 },  { x: 1, z: 1 }
        ];
        
        for (const dir of directions) {
            const neighbor = {
                x: gridPos.x + dir.x,
                z: gridPos.z + dir.z
            };
            
            if (this.isValidGridPosition(neighbor)) {
                neighbors.push(neighbor);
            }
        }
        
        return neighbors;
    }
    
    isValidGridPosition(gridPos) {
        return gridPos.x >= 0 && gridPos.x < this.navigationGrid.width &&
               gridPos.z >= 0 && gridPos.z < this.navigationGrid.height;
    }
    
    isBlocked(gridPos) {
        if (!this.isValidGridPosition(gridPos)) return true;
        
        const index = gridPos.z * this.navigationGrid.width + gridPos.x;
        return this.navigationGrid.cells[index] === 1;
    }
    
    isPositionBlocked(worldPos) {
        const gridPos = this.worldToGrid(worldPos);
        return this.isBlocked(gridPos);
    }
    
    getMovementCost(from, to) {
        const isDiagonal = from.x !== to.x && from.z !== to.z;
        const baseCost = isDiagonal ? Math.sqrt(2) : 1;
        
        const index = to.z * this.navigationGrid.width + to.x;
        const terrainCost = this.navigationGrid.costs[index];
        
        return baseCost * terrainCost;
    }
    
    calculateSegmentTiming(from, to, currentVelocity) {
        const distance = Math.sqrt(
            Math.pow(to.x - from.x, 2) +
            Math.pow(to.y - from.y, 2) +
            Math.pow(to.z - from.z, 2)
        );
        
        const direction = {
            x: (to.x - from.x) / distance,
            y: (to.y - from.y) / distance,
            z: (to.z - from.z) / distance
        };
        
        const currentSpeed = Math.sqrt(
            currentVelocity.x * currentVelocity.x +
            currentVelocity.y * currentVelocity.y +
            currentVelocity.z * currentVelocity.z
        );
        
        const targetSpeed = Math.min(this.movementConfig.preferredSpeed, this.movementConfig.maxSpeed);
        const maxAccel = this.movementConfig.maxAcceleration;
        
        // Simple motion planning
        const accelerationTime = Math.abs(targetSpeed - currentSpeed) / maxAccel;
        const constantSpeedDistance = Math.max(0, distance - 0.5 * maxAccel * accelerationTime * accelerationTime);
        const constantSpeedTime = constantSpeedDistance / targetSpeed;
        
        const totalTime = accelerationTime + constantSpeedTime;
        
        return {
            duration: totalTime,
            distance: distance,
            startVelocity: currentVelocity,
            endVelocity: {
                x: direction.x * targetSpeed,
                y: direction.y * targetSpeed,
                z: direction.z * targetSpeed
            },
            maxSpeed: targetSpeed,
            direction: direction
        };
    }
    
    classifyMovement(segmentData) {
        if (!segmentData) return 'idle';
        
        const speed = Math.sqrt(
            segmentData.endVelocity.x * segmentData.endVelocity.x +
            segmentData.endVelocity.y * segmentData.endVelocity.y +
            segmentData.endVelocity.z * segmentData.endVelocity.z
        );
        
        if (speed < 0.1) return 'idle';
        if (speed < 1.5) return 'walk';
        if (speed < 3.0) return 'run';
        return 'sprint';
    }
    
    calculateOrientation(from, to) {
        if (!to) return { x: 0, y: 0, z: 0, w: 1 };
        
        const direction = {
            x: to.x - from.x,
            z: to.z - from.z
        };
        
        const angle = Math.atan2(direction.x, direction.z);
        
        return {
            x: 0,
            y: Math.sin(angle / 2),
            z: 0,
            w: Math.cos(angle / 2)
        };
    }
    
    determineAnimationType(currentKeyframe, nextKeyframe) {
        const movementType = nextKeyframe.metadata?.movementType || 'walk';
        
        // Add transition logic
        if (currentKeyframe.metadata?.movementType !== movementType) {
            return `transition_to_${movementType}`;
        }
        
        return movementType;
    }
    
    async generateAnimationData(animationType, fromKeyframe, toKeyframe, options) {
        // Use appropriate backend to generate animation
        if (this.animationBackends.rsmt && this.locomotionLibrary[animationType]) {
            try {
                const transition = await this.animationBackends.rsmt.generateTransition(
                    { bones: {} }, // Simplified for now
                    animationType,
                    {
                        duration: toKeyframe.time - fromKeyframe.time,
                        style: options.animationStyle || 'natural'
                    }
                );
                return transition;
            } catch (error) {
                console.warn(`[Pathfinding] RSMT animation failed for ${animationType}:`, error);
            }
        }
        
        // Fallback to procedural generation
        return this.generateProceduralAnimation(animationType, fromKeyframe, toKeyframe);
    }
    
    async generateBVHFrame(position, orientation, animationData, progress, options) {
        // Generate BVH frame based on position, orientation, and animation data
        const frame = {
            frameNumber: Math.floor(progress * 100), // Simplified
            time: 0, // Will be set by caller
            bones: {
                hips: {
                    position: position,
                    rotation: orientation
                }
                // Add other bones based on animationData
            },
            metadata: {
                source: 'pathfinding',
                animationData: animationData ? 'backend' : 'procedural'
            }
        };
        
        // Apply animation data if available
        if (animationData && animationData.frames) {
            const animFrame = animationData.frames[Math.floor(progress * (animationData.frames.length - 1))];
            if (animFrame && animFrame.bones) {
                frame.bones = { ...frame.bones, ...animFrame.bones };
            }
        }
        
        return frame;
    }
    
    generateProceduralAnimation(animationType, fromKeyframe, toKeyframe) {
        // Simple procedural animation generation
        return {
            type: 'procedural',
            animationType: animationType,
            frames: [
                { bones: {}, time: fromKeyframe.time },
                { bones: {}, time: toKeyframe.time }
            ],
            quality: 0.6
        };
    }
    
    createProceduralLocomotion(animName) {
        console.log(`[Pathfinding] Creating procedural locomotion: ${animName}`);
        // Create basic procedural animation data
        this.locomotionLibrary[animName] = {
            ...this.locomotionLibrary[animName],
            isProcedural: true,
            frames: this.generateBasicLocomotionFrames(animName)
        };
    }
    
    generateBasicLocomotionFrames(animName) {
        // Generate basic frame data for locomotion
        const frameCount = 30; // 1 second at 30fps
        const frames = [];
        
        for (let i = 0; i < frameCount; i++) {
            frames.push({
                frameNumber: i,
                time: i / 30,
                bones: {
                    hips: {
                        position: { x: 0, y: 0, z: 0 },
                        rotation: { x: 0, y: 0, z: 0, w: 1 }
                    }
                },
                metadata: { source: 'procedural', animType: animName }
            });
        }
        
        return frames;
    }
    
    interpolatePosition(pos1, pos2, t) {
        return {
            x: pos1.x + (pos2.x - pos1.x) * t,
            y: pos1.y + (pos2.y - pos1.y) * t,
            z: pos1.z + (pos2.z - pos1.z) * t
        };
    }
    
    interpolateOrientation(ori1, ori2, t) {
        // Simple linear interpolation for quaternions (should use SLERP in production)
        return {
            x: ori1.x + (ori2.x - ori1.x) * t,
            y: ori1.y + (ori2.y - ori1.y) * t,
            z: ori1.z + (ori2.z - ori1.z) * t,
            w: ori1.w + (ori2.w - ori1.w) * t
        };
    }
    
    reconstructPath(cameFrom, current) {
        const path = [];
        const currentKey = this.gridToKey(current);
        let node = current;
        
        while (node) {
            path.unshift(this.gridToWorld(node));
            const nodeKey = this.gridToKey(node);
            node = cameFrom.get(nodeKey);
        }
        
        return path;
    }
    
    calculatePathDistance(path) {
        let distance = 0;
        for (let i = 1; i < path.length; i++) {
            const prev = path[i - 1];
            const curr = path[i];
            distance += Math.sqrt(
                Math.pow(curr.x - prev.x, 2) +
                Math.pow(curr.y - prev.y, 2) +
                Math.pow(curr.z - prev.z, 2)
            );
        }
        return distance;
    }
    
    generatePathCacheKey(start, end, options) {
        const startKey = `${start.x.toFixed(1)},${start.y.toFixed(1)},${start.z.toFixed(1)}`;
        const endKey = `${end.x.toFixed(1)},${end.y.toFixed(1)},${end.z.toFixed(1)}`;
        const optionsKey = JSON.stringify(options);
        return `${startKey}_${endKey}_${optionsKey}`;
    }
    
    cachePathResult(cacheKey, result) {
        if (this.pathCache.size >= this.maxCacheSize) {
            // Remove oldest entry
            const firstKey = this.pathCache.keys().next().value;
            this.pathCache.delete(firstKey);
        }
        this.pathCache.set(cacheKey, result);
    }
    
    adaptCachedPath(cachedResult, options) {
        // Adapt cached path to current conditions
        return {
            ...cachedResult,
            adaptedAt: Date.now(),
            cacheHit: true
        };
    }
    
    updateNavigationGrid(obstacle, isAdding) {
        // Update grid cells based on obstacle
        const bounds = this.navigationGrid.bounds;
        
        const minX = Math.max(0, Math.floor((obstacle.minX - bounds.minX) / this.gridResolution));
        const maxX = Math.min(this.navigationGrid.width - 1, Math.ceil((obstacle.maxX - bounds.minX) / this.gridResolution));
        const minZ = Math.max(0, Math.floor((obstacle.minZ - bounds.minZ) / this.gridResolution));
        const maxZ = Math.min(this.navigationGrid.height - 1, Math.ceil((obstacle.maxZ - bounds.minZ) / this.gridResolution));
        
        for (let z = minZ; z <= maxZ; z++) {
            for (let x = minX; x <= maxX; x++) {
                const index = z * this.navigationGrid.width + x;
                this.navigationGrid.cells[index] = isAdding ? 1 : 0;
                
                if (obstacle.cost !== undefined) {
                    this.navigationGrid.costs[index] = isAdding ? obstacle.cost : 1.0;
                }
            }
        }
    }
    
    updateActiveKeyframes() {
        // Update active keyframes based on current position
        // This would be called by the timeline system
    }
    
    isWaypoint(position) {
        // Check if position is near a defined waypoint
        for (const waypoint of this.waypoints.values()) {
            const distance = Math.sqrt(
                Math.pow(position.x - waypoint.x, 2) +
                Math.pow(position.z - waypoint.z, 2)
            );
            if (distance < 1.0) return true;
        }
        return false;
    }
    
    getTerrainType(position) {
        // Determine terrain type at position
        return 'normal'; // Simplified
    }
    
    getAnimationPriority(animationType) {
        const priorities = {
            idle: 1,
            walk: 3,
            run: 4,
            sprint: 5,
            transition_to_walk: 2,
            transition_to_run: 3
        };
        return priorities[animationType] || 3;
    }
    
    updatePlanningStats(result) {
        this.stats.pathsPlanned++;
        this.stats.successfulPaths++;
        this.stats.totalDistance += result.totalDistance;
        this.stats.keyframesGenerated += result.keyframes.length;
        
        const count = this.stats.pathsPlanned;
        this.stats.averagePlanningTime = 
            (this.stats.averagePlanningTime * (count - 1) + result.planningTime) / count;
    }
    
    /**
     * Public API methods
     */
    
    getStats() {
        return {
            ...this.stats,
            loadedAnimations: Object.keys(this.locomotionLibrary).length,
            obstacles: this.obstacles.size,
            waypoints: this.waypoints.size,
            cacheSize: this.pathCache.size,
            gridSize: `${this.navigationGrid.width}x${this.navigationGrid.height}`,
            isPlanning: this.planningState.isPlanning
        };
    }
    
    getCurrentState() {
        return { ...this.planningState };
    }
    
    clearCache() {
        this.pathCache.clear();
        console.log('[Pathfinding] Path cache cleared');
    }
    
    dispose() {
        this.pathCache.clear();
        this.obstacles.clear();
        this.waypoints.clear();
        this.activeKeyframes = [];
        
        console.log('[Pathfinding] Planner disposed');
    }
}

/**
 * Priority Queue implementation for A*
 */
class PriorityQueue {
    constructor() {
        this.items = [];
    }
    
    enqueue(item, priority) {
        const queueElement = { item, priority };
        let added = false;
        
        for (let i = 0; i < this.items.length; i++) {
            if (queueElement.priority < this.items[i].priority) {
                this.items.splice(i, 0, queueElement);
                added = true;
                break;
            }
        }
        
        if (!added) {
            this.items.push(queueElement);
        }
    }
    
    dequeue() {
        if (this.isEmpty()) return null;
        return this.items.shift().item;
    }
    
    isEmpty() {
        return this.items.length === 0;
    }
    
    contains(item) {
        return this.items.some(queueElement => 
            queueElement.item.x === item.x && queueElement.item.z === item.z
        );
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PathfindingBVHPlanner, PriorityQueue };
} else {
    // Browser global
    window.PathfindingBVHPlanner = PathfindingBVHPlanner;
    window.PriorityQueue = PriorityQueue;
}
