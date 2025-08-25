// Minimal BVHTimeline implementation for chunked animation segments
// Focus: appendChunk, addSegment, clear, snapshot (lightweight)

class BVHTimeline {
	constructor({ framerate = 30 } = {}) {
		this.framerate = framerate;
		this.version = 0;
		// tracks: { [trackId]: { clips: Array<Segment> } }
		this.tracks = Object.create(null);
	}

	ensureTrack(trackId) {
		if (!this.tracks[trackId]) {
			this.tracks[trackId] = { clips: [] };
		}
		return this.tracks[trackId];
	}

	addSegment(segment, options = {}) {
		const trackId = segment.track || options.trackId || 'default';
		const track = this.ensureTrack(trackId);
		track.clips.push({ ...segment, track: trackId });
		this.version++;
		return this.version;
	}

	// Adapter-friendly: accepts { t0, dt, frames }
	appendChunk(trackId, chunk, options = {}) {
		const { t0, dt, frames } = chunk || {};
		if (typeof t0 !== 'number' || typeof dt !== 'number' || !Array.isArray(frames)) {
			throw new Error('Invalid chunk format');
		}
		const segment = {
			startTime: t0,
			duration: dt,
			frames,
			track: trackId,
			meta: options.meta || {}
		};
		return this.addSegment(segment);
	}

	clear(trackId, fromTime = -Infinity) {
		const track = this.tracks[trackId];
		if (!track) return this.version;
		track.clips = track.clips.filter(s => s.startTime + s.duration <= fromTime);
		this.version++;
		return this.version;
	}

	// Very lightweight snapshot: returns all frames in [t0, t1)
	snapshot(t0, t1) {
		const out = [];
		for (const trackId of Object.keys(this.tracks)) {
			const { clips } = this.tracks[trackId];
			for (const seg of clips) {
				const segEnd = seg.startTime + seg.duration;
				if (segEnd <= t0 || seg.startTime >= t1) continue;
				for (const f of seg.frames) {
					if (f.time >= t0 && f.time < t1) out.push(f);
				}
			}
		}
		return out.sort((a, b) => a.time - b.time);
	}
}

// UMD-style exposure for browser tests that use page.addInitScript
if (typeof window !== 'undefined') {
	window.BVHTimeline = BVHTimeline;
}
if (typeof module !== 'undefined' && module.exports) {
	module.exports = { BVHTimeline };
}

