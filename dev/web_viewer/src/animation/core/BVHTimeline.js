/**
 * BVHTimeline.js
 *
 * Represents an independent sequence of BVH frames, allowing for time-based indexing
 * and management of animation data.
 */
class BVHTimeline {
    constructor(id) {
        this.id = id || `timeline-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        this.frames = []; // Array of { time: number, bvhData: Object } BVH frame objects
        this.duration = 0; // Total duration of the timeline in seconds
        this.frameRate = 30; // Default frame rate, can be adjusted
    }

    /**
     * Adds a BVH frame to the timeline.
     * @param {Object} bvhData - The BVH frame data.
     * @param {number} [time] - Optional: The specific time for this frame. If not provided, appends to end.
     */
    addFrame(bvhData, time = null) {
        const newFrame = { bvhData: bvhData };
        if (time !== null) {
            newFrame.time = time;
            // Insert in sorted order if time is specified
            let inserted = false;
            for (let i = 0; i < this.frames.length; i++) {
                if (this.frames[i].time > time) {
                    this.frames.splice(i, 0, newFrame);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                this.frames.push(newFrame);
            }
        } else {
            // Append to the end, calculate time based on frameRate
            newFrame.time = this.frames.length > 0 ? this.frames[this.frames.length - 1].time + (1 / this.frameRate) : 0;
            this.frames.push(newFrame);
        }
        this._updateDuration();
    }

    /**
     * Retrieves the BVH frame closest to the given time.
     * @param {number} time - The time in seconds.
     * @returns {Object|null} The BVH frame data or null if no frames.
     */
    getFrameAtTime(time) {
        if (this.frames.length === 0) {
            return null;
        }
        // Simple linear search for now, can be optimized with binary search for large timelines
        for (let i = 0; i < this.frames.length; i++) {
            if (this.frames[i].time >= time) {
                return this.frames[i].bvhData;
            }
        }
        return this.frames[this.frames.length - 1].bvhData; // Return last frame if time is beyond duration
    }

    /**
     * Updates the total duration of the timeline.
     * @private
     */
    _updateDuration() {
        if (this.frames.length > 0) {
            this.duration = this.frames[this.frames.length - 1].time;
        } else {
            this.duration = 0;
        }
    }

    /**
     * Clears all frames from the timeline.
     */
    clear() {
        this.frames = [];
        this.duration = 0;
    }

    /**
     * Gets the total number of frames in the timeline.
     * @returns {number}
     */
    getFrameCount() {
        return this.frames.length;
    }

    /**
     * Sets the frame rate for calculating frame times when not explicitly provided.
     * @param {number} rate - Frames per second.
     */
    setFrameRate(rate) {
        if (rate > 0) {
            this.frameRate = rate;
        }
    }
}

export default BVHTimeline;