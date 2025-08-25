/**
 * TimelineComposer.js
 *
 * Manages and composites multiple BVHTimeline instances into a single, coherent
 * animation stream. Handles blending, layering, and transitions between timelines.
 */
class TimelineComposer {
    constructor() {
        this.activeTimelines = new Map(); // Map<timelineId, { timeline: BVHTimeline, weight: number, startTime: number, endTime: number }>
        this.currentTimelineId = null;
        this.transitionDuration = 0.5; // Default transition duration in seconds
    }

    /**
     * Adds a BVHTimeline to the composer.
     * @param {BVHTimeline} timeline - The BVHTimeline instance to add.
     * @param {number} [weight=1.0] - The initial blending weight for this timeline (0.0 to 1.0).
     * @param {number} [startTime=0] - The time in the composer's global timeline when this timeline becomes active.
     * @param {number} [endTime=Infinity] - The time in the composer's global timeline when this timeline becomes inactive.
     */
    addTimeline(timeline, weight = 1.0, startTime = 0, endTime = Infinity) {
        if (this.activeTimelines.has(timeline.id)) {
            console.warn(`Timeline with ID ${timeline.id} already added.`);
            return;
        }
        this.activeTimelines.set(timeline.id, { timeline, weight, startTime, endTime });
        console.log(`Added timeline ${timeline.id}.`);
    }

    /**
     * Sets the currently active timeline and initiates a transition if necessary.
     * @param {string} timelineId - The ID of the timeline to activate.
     * @param {number} [transitionDuration] - Optional: Duration of the transition in seconds. Uses default if not provided.
     */
    setActiveTimeline(timelineId, transitionDuration = this.transitionDuration) {
        if (!this.activeTimelines.has(timelineId)) {
            console.error(`Timeline with ID ${timelineId} not found.`);
            return;
        }

        if (this.currentTimelineId === timelineId) {
            console.log(`Timeline ${timelineId} is already active.`);
            return;
        }

        const newTimelineInfo = this.activeTimelines.get(timelineId);
        console.log(`Setting active timeline to ${timelineId}.`);

        if (this.currentTimelineId) {
            // Initiate transition from current to new timeline
            const oldTimelineInfo = this.activeTimelines.get(this.currentTimelineId);
            // For a real implementation, you'd manage weights over time for smooth blending.
            // This is a simplified immediate switch for the placeholder.
            oldTimelineInfo.weight = 0;
            newTimelineInfo.weight = 1;
            console.log(`Transitioning from ${this.currentTimelineId} to ${timelineId} over ${transitionDuration}s.`);
        } else {
            // No current timeline, just activate the new one
            newTimelineInfo.weight = 1;
        }
        this.currentTimelineId = timelineId;
    }

    /**
     * Gets the composited BVH frame data at a given global time.
     * This method would handle blending multiple timelines based on their weights.
     * For simplicity, this placeholder just returns the frame from the active timeline.
     * @param {number} globalTime - The current global time in seconds.
     * @returns {Object|null} The composited BVH frame data.
     */
    getCompositedFrameAtTime(globalTime) {
        if (!this.currentTimelineId) {
            return null;
        }

        const activeInfo = this.activeTimelines.get(this.currentTimelineId);
        if (!activeInfo || activeInfo.weight === 0) {
            return null;
        }

        // In a full implementation, you'd iterate through all active timelines,
        // get their frames at the current time (adjusted for their start/end times),
        // and blend them based on their weights.
        // For now, we just get the frame from the single active timeline.
        const frame = activeInfo.timeline.getFrameAtTime(globalTime - activeInfo.startTime);
        return frame; // Assuming frame is already BVH data
    }

    /**
     * Removes a timeline from the composer.
     * @param {string} timelineId - The ID of the timeline to remove.
     */
    removeTimeline(timelineId) {
        if (this.activeTimelines.has(timelineId)) {
            if (this.currentTimelineId === timelineId) {
                this.currentTimelineId = null; // Clear current if it's being removed
                console.warn(`Removed active timeline ${timelineId}. No timeline is currently active.`);
            }
            this.activeTimelines.delete(timelineId);
            console.log(`Removed timeline ${timelineId}.`);
        }
    }

    /**
     * Sets the default transition duration for setActiveTimeline.
     * @param {number} duration - The new default transition duration in seconds.
     */
    setTransitionDuration(duration) {
        if (duration >= 0) {
            this.transitionDuration = duration;
        }
    }
}

export default TimelineComposer;