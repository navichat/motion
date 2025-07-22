/**
 * BVH Timeline DeepMimic Integration
 * Seamlessly integrates DeepMimic motion generation with BVHTimeline
 */

class BVHTimelineDeepMimicIntegration {
    constructor(timeline, deepMimicGenerator, options = {}) {
        this.timeline = timeline;
        this.generator = deepMimicGenerator;
        
        // Integration settings
        this.settings = {
            defaultTrack: options.defaultTrack || 'deepmimic',
            bufferAhead: options.bufferAhead || 2.0, // seconds
            transitionDuration: options.transitionDuration || 0.5, // seconds
            autoCleanup: options.autoCleanup !== false,
            maxClips: options.maxClips || 10
        };
        
        // Active motion sessions
        this.activeSessions = new Map();
        
        // Motion templates for common actions
        this.motionTemplates = this.createMotionTemplates();
        
        console.log('BVH Timeline DeepMimic Integration initialized');
    }
    
    /**
     * Create motion templates for common actions
     */
    createMotionTemplates() {
        return {
            idle: {
                model: 'walk',
                targetMotion: { speed: 0, direction: [0, 0, 0] },
                duration: 10.0,
                loop: true
            },
            
            walk: {
                model: 'walk',
                targetMotion: { speed: 1.0, direction: [0, 0, 1] },
                duration: 4.0,
                loop: true
            },
            
            run: {
                model: 'run',
                targetMotion: { speed: 2.0, direction: [0, 0, 1] },
                duration: 3.0,
                loop: true
            },
            
            jump: {
                model: 'jump',
                targetMotion: { speed: 0.5, direction: [0, 1, 0] },
                duration: 2.0,
                loop: false
            },
            
            dance: {
                model: 'dance_a',
                targetMotion: { speed: 0, rhythmic: true },
                duration: 8.0,
                loop: true
            },
            
            combat: {
                model: 'punch',
                targetMotion: { speed: 0, aggressive: true },
                duration: 1.5,
                loop: false
            },
            
            acrobatic: {
                model: 'backflip',
                targetMotion: { speed: 0, direction: [0, 0, -1] },
                duration: 2.5,
                loop: false
            }
        };
    }
    
    /**
     * Add a motion clip using a template
     */
    async addMotionFromTemplate(templateName, startTime, options = {}) {
        const template = this.motionTemplates[templateName];
        if (!template) {
            throw new Error(`Motion template '${templateName}' not found. Available: ${Object.keys(this.motionTemplates).join(', ')}`);
        }
        
        const clipOptions = {
            ...template,
            ...options,
            startTime: startTime
        };
        
        return await this.addMotionClip(clipOptions);
    }
    
    /**
     * Add a DeepMimic motion clip to the timeline
     */
    async addMotionClip(options = {}) {
        const {
            model = 'walk',
            duration = 5.0,
            startTime = 0,
            trackName = this.settings.defaultTrack,
            targetMotion = null,
            weight = 1.0,
            blendMode = 'replace',
            loop = false,
            priority = 1,
            clipId = null
        } = options;
        
        console.log(`Adding DeepMimic motion clip: ${model} at time ${startTime}s`);
        
        try {
            // Generate the motion clip
            const motionClip = await this.generator.generateMotionClip({
                duration,
                model,
                targetMotion,
                clipId
            });
            
            // Create BVH clip for timeline
            const bvhClip = new BVHClip({
                id: motionClip.id,
                type: 'deepmimic_generated',
                startTime: startTime,
                duration: duration,
                weight: weight,
                blendMode: blendMode,
                loop: loop,
                generator: this.createFrameGenerator(motionClip),
                metadata: {
                    ...motionClip.metadata,
                    priority: priority,
                    templateUsed: options.templateUsed || null
                }
            });
            
            // Add to timeline
            const clipId_final = this.timeline.addClip(trackName, bvhClip);
            
            // Store in active sessions
            this.activeSessions.set(clipId_final, {
                clip: bvhClip,
                trackName: trackName,
                addedAt: Date.now(),
                motionData: motionClip
            });
            
            // Cleanup old clips if needed
            if (this.settings.autoCleanup) {
                this.cleanupOldClips();
            }
            
            console.log(`Added motion clip ${clipId_final} to track '${trackName}'`);
            return clipId_final;
            
        } catch (error) {
            console.error('Failed to add motion clip:', error);
            throw error;
        }
    }
    
    /**
     * Create frame generator function for BVH clip
     */
    createFrameGenerator(motionClip) {
        return async (time, frameIndex) => {
            // Calculate local time within the clip
            const localTime = time % motionClip.duration;
            const frameInClip = Math.floor(localTime * this.generator.frameRate);
            
            // Return pre-generated frame if available
            if (frameInClip < motionClip.frames.length) {
                const frame = motionClip.frames[frameInClip];
                // Update timestamp to match current time
                return {
                    ...frame,
                    time: time
                };
            }
            
            // Fallback to default frame if out of range
            return this.generator.createDefaultBVHFrame(time);
        };
    }
    
    /**
     * Add real-time motion generation
     */
    async addRealTimeMotion(options = {}) {
        const {
            model = 'walk',
            trackName = this.settings.defaultTrack + '_realtime',
            targetMotion = null,
            weight = 1.0,
            blendMode = 'replace',
            duration = null // null = infinite
        } = options;
        
        console.log(`Starting real-time DeepMimic motion: ${model}`);
        
        // Switch to requested model
        await this.generator.switchModel(model);
        
        // Create real-time generator
        const realtimeGenerator = this.generator.createTimelineGenerator({
            model,
            targetMotion,
            resetState: true
        });
        
        // Create BVH clip for real-time generation
        const bvhClip = new BVHClip({
            id: `realtime_${model}_${Date.now()}`,
            type: 'deepmimic_realtime',
            startTime: this.timeline.currentTime,
            duration: duration || 999999, // Very long duration for infinite motion
            weight: weight,
            blendMode: blendMode,
            generator: realtimeGenerator,
            metadata: {
                model: model,
                realtime: true,
                targetMotion: targetMotion
            }
        });
        
        // Add to timeline
        const clipId = this.timeline.addClip(trackName, bvhClip);
        
        // Store in active sessions
        this.activeSessions.set(clipId, {
            clip: bvhClip,
            trackName: trackName,
            addedAt: Date.now(),
            realtime: true,
            model: model
        });
        
        console.log(`Started real-time motion ${clipId} on track '${trackName}'`);
        return clipId;
    }
    
    /**
     * Transition between different motions smoothly\n     */\n    async transitionToMotion(newMotionOptions, options = {}) {\n        const {\n            transitionDuration = this.settings.transitionDuration,\n            trackName = this.settings.defaultTrack,\n            blendDuringTransition = true\n        } = options;\n        \n        const currentTime = this.timeline.currentTime;\n        \n        // Add transition period if blending is enabled\n        if (blendDuringTransition && transitionDuration > 0) {\n            // Create transition clip\n            const transitionClip = await this.createTransitionClip(\n                currentTime,\n                transitionDuration,\n                newMotionOptions\n            );\n            \n            // Add transition\n            const transitionId = this.timeline.addClip(trackName + '_transition', transitionClip);\n            \n            // Schedule new motion after transition\n            setTimeout(() => {\n                this.addMotionClip({\n                    ...newMotionOptions,\n                    startTime: currentTime + transitionDuration\n                });\n            }, transitionDuration * 1000);\n            \n            return transitionId;\n        } else {\n            // Direct transition\n            return await this.addMotionClip({\n                ...newMotionOptions,\n                startTime: currentTime\n            });\n        }\n    }\n    \n    /**\n     * Create transition clip between motions\n     */\n    async createTransitionClip(startTime, duration, targetMotionOptions) {\n        const transitionGenerator = async (time, frameIndex) => {\n            const progress = (time - startTime) / duration;\n            const blendFactor = this.easeInOutCubic(Math.min(1, Math.max(0, progress)));\n            \n            // Generate frame with transition blending\n            const frame = await this.generator.generateFrame(time, {\n                targetMotion: targetMotionOptions.targetMotion,\n                transitionBlend: blendFactor\n            });\n            \n            return frame;\n        };\n        \n        return new BVHClip({\n            id: `transition_${Date.now()}`,\n            type: 'deepmimic_transition',\n            startTime: startTime,\n            duration: duration,\n            weight: 1.0,\n            blendMode: 'weighted',\n            generator: transitionGenerator,\n            metadata: {\n                transitionTo: targetMotionOptions.model || 'unknown',\n                transitionDuration: duration\n            }\n        });\n    }\n    \n    /**\n     * Easing function for smooth transitions\n     */\n    easeInOutCubic(t) {\n        return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;\n    }\n    \n    /**\n     * Add motion sequence with multiple parts\n     */\n    async addMotionSequence(sequence, startTime = null) {\n        const baseTime = startTime || this.timeline.currentTime;\n        const clipIds = [];\n        let currentTime = baseTime;\n        \n        for (const motionPart of sequence) {\n            const clipId = await this.addMotionClip({\n                ...motionPart,\n                startTime: currentTime\n            });\n            \n            clipIds.push(clipId);\n            currentTime += motionPart.duration || 5.0;\n        }\n        \n        console.log(`Added motion sequence: ${clipIds.length} clips starting at ${baseTime}s`);\n        return clipIds;\n    }\n    \n    /**\n     * Create a complex motion sequence from description\n     */\n    async createMotionFromDescription(description, options = {}) {\n        const { duration = 10.0, startTime = null } = options;\n        \n        // Simple motion parsing - could be enhanced with NLP\n        const motionSequence = this.parseMotionDescription(description, duration);\n        \n        return await this.addMotionSequence(motionSequence, startTime);\n    }\n    \n    /**\n     * Parse motion description into sequence\n     */\n    parseMotionDescription(description, totalDuration) {\n        const desc = description.toLowerCase();\n        const sequence = [];\n        \n        // Simple keyword-based parsing\n        if (desc.includes('walk') && desc.includes('run')) {\n            sequence.push(\n                { ...this.motionTemplates.walk, duration: totalDuration * 0.5 },\n                { ...this.motionTemplates.run, duration: totalDuration * 0.5 }\n            );\n        } else if (desc.includes('dance')) {\n            sequence.push({ ...this.motionTemplates.dance, duration: totalDuration });\n        } else if (desc.includes('jump') || desc.includes('acrobat')) {\n            sequence.push({ ...this.motionTemplates.acrobatic, duration: totalDuration });\n        } else if (desc.includes('fight') || desc.includes('combat')) {\n            sequence.push({ ...this.motionTemplates.combat, duration: totalDuration });\n        } else if (desc.includes('run')) {\n            sequence.push({ ...this.motionTemplates.run, duration: totalDuration });\n        } else if (desc.includes('walk')) {\n            sequence.push({ ...this.motionTemplates.walk, duration: totalDuration });\n        } else {\n            // Default to idle\n            sequence.push({ ...this.motionTemplates.idle, duration: totalDuration });\n        }\n        \n        return sequence;\n    }\n    \n    /**\n     * Update target motion for real-time clips\n     */\n    updateTargetMotion(clipId, targetMotion) {\n        const session = this.activeSessions.get(clipId);\n        if (session && session.realtime) {\n            // Update the target motion parameters\n            session.clip.metadata.targetMotion = targetMotion;\n            console.log(`Updated target motion for clip ${clipId}`);\n        }\n    }\n    \n    /**\n     * Get motion control interface for real-time manipulation\n     */\n    getMotionController(clipId) {\n        const session = this.activeSessions.get(clipId);\n        if (!session) {\n            throw new Error(`Clip ${clipId} not found in active sessions`);\n        }\n        \n        return {\n            updateSpeed: (speed) => {\n                if (session.clip.metadata.targetMotion) {\n                    session.clip.metadata.targetMotion.speed = speed;\n                }\n            },\n            \n            updateDirection: (direction) => {\n                if (session.clip.metadata.targetMotion) {\n                    session.clip.metadata.targetMotion.direction = direction;\n                }\n            },\n            \n            updateWeight: (weight) => {\n                session.clip.weight = weight;\n            },\n            \n            stop: () => {\n                this.removeClip(clipId);\n            },\n            \n            getStatus: () => ({\n                isActive: this.timeline.tracks[session.trackName]?.getActiveClipsAtTime(this.timeline.currentTime)\n                    .some(clip => clip.id === clipId) || false,\n                trackName: session.trackName,\n                model: session.model,\n                addedAt: session.addedAt\n            })\n        };\n    }\n    \n    /**\n     * Remove a motion clip\n     */\n    removeClip(clipId) {\n        const session = this.activeSessions.get(clipId);\n        if (session) {\n            this.timeline.removeClip(session.trackName, clipId);\n            this.activeSessions.delete(clipId);\n            console.log(`Removed motion clip ${clipId}`);\n            return true;\n        }\n        return false;\n    }\n    \n    /**\n     * Clear all motion clips\n     */\n    clearAllMotions() {\n        for (const [clipId, session] of this.activeSessions) {\n            this.timeline.removeClip(session.trackName, clipId);\n        }\n        this.activeSessions.clear();\n        console.log('Cleared all DeepMimic motion clips');\n    }\n    \n    /**\n     * Cleanup old clips to manage memory\n     */\n    cleanupOldClips() {\n        if (this.activeSessions.size <= this.settings.maxClips) return;\n        \n        // Sort by age and remove oldest\n        const sortedSessions = Array.from(this.activeSessions.entries())\n            .sort((a, b) => a[1].addedAt - b[1].addedAt);\n        \n        const toRemove = sortedSessions.slice(0, sortedSessions.length - this.settings.maxClips);\n        \n        for (const [clipId] of toRemove) {\n            this.removeClip(clipId);\n        }\n        \n        if (toRemove.length > 0) {\n            console.log(`Cleaned up ${toRemove.length} old motion clips`);\n        }\n    }\n    \n    /**\n     * Get statistics about active motions\n     */\n    getMotionStats() {\n        const stats = {\n            activeClips: this.activeSessions.size,\n            realtimeClips: 0,\n            generatedClips: 0,\n            tracks: new Set(),\n            models: new Set(),\n            totalDuration: 0\n        };\n        \n        for (const [clipId, session] of this.activeSessions) {\n            stats.tracks.add(session.trackName);\n            \n            if (session.realtime) {\n                stats.realtimeClips++;\n                stats.models.add(session.model);\n            } else {\n                stats.generatedClips++;\n                stats.models.add(session.motionData?.metadata?.model || 'unknown');\n                stats.totalDuration += session.clip.duration;\n            }\n        }\n        \n        stats.tracks = Array.from(stats.tracks);\n        stats.models = Array.from(stats.models);\n        \n        return stats;\n    }\n    \n    /**\n     * Export motion data for analysis or caching\n     */\n    exportMotionData(clipId) {\n        const session = this.activeSessions.get(clipId);\n        if (!session) {\n            throw new Error(`Clip ${clipId} not found`);\n        }\n        \n        return {\n            clipId: clipId,\n            trackName: session.trackName,\n            addedAt: session.addedAt,\n            clip: {\n                id: session.clip.id,\n                type: session.clip.type,\n                startTime: session.clip.startTime,\n                duration: session.clip.duration,\n                metadata: session.clip.metadata\n            },\n            motionData: session.motionData,\n            performance: this.generator.getPerformanceStats()\n        };\n    }\n    \n    /**\n     * Get available motion templates\n     */\n    getAvailableTemplates() {\n        return Object.keys(this.motionTemplates);\n    }\n    \n    /**\n     * Get available models from generator\n     */\n    getAvailableModels() {\n        return this.generator.getAvailableModels();\n    }\n    \n    /**\n     * Dispose and cleanup\n     */\n    dispose() {\n        this.clearAllMotions();\n        this.activeSessions.clear();\n        console.log('BVH Timeline DeepMimic Integration disposed');\n    }\n}\n\n// Export for use in modules or global scope\nif (typeof module !== 'undefined' && module.exports) {\n    module.exports = BVHTimelineDeepMimicIntegration;\n} else if (typeof window !== 'undefined') {\n    window.BVHTimelineDeepMimicIntegration = BVHTimelineDeepMimicIntegration;\n}
