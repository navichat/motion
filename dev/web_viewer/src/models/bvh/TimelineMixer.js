// Minimal TimelineMixer to compose frames from multiple BVHTimelines

class TimelineMixer {
	constructor(rules = {}) {
		this.rules = rules;
	}

	// Compose a very simple pose at time t by taking the most recent frame per channel
	compose(timelines, t, rules = this.rules) {
		const pose = new Map();
		for (const tl of timelines) {
			// Find the latest frame <= t across all tracks
			let best = null;
			for (const trackId of Object.keys(tl.tracks)) {
				const clips = tl.tracks[trackId].clips;
				for (const seg of clips) {
					for (const f of seg.frames) {
						if (f.time <= t && (!best || f.time > best.time)) best = f;
					}
				}
			}
			if (best && best.channels) {
				for (const [bone, value] of (best.channels instanceof Map ? best.channels : Object.entries(best.channels))) {
					// Simplified: last-writer-wins
					if (best.channels instanceof Map) {
						pose.set(bone, value);
					} else {
						pose.set(bone, value[1]);
					}
				}
			}
		}
		return pose;
	}
}

if (typeof window !== 'undefined') {
	window.TimelineMixer = TimelineMixer;
}
if (typeof module !== 'undefined' && module.exports) {
	module.exports = { TimelineMixer };
}

