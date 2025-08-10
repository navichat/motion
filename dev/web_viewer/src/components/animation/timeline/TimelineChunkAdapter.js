/**
 * TimelineChunkAdapter: append FramesChunk to BVHTimeline tracks with simple fades.
 */
const timelineNs = (typeof require !== 'undefined' && typeof window === 'undefined')
  ? require('./BVHTimeline')
  : (typeof window !== 'undefined' ? window : {});
const BVHTimeline = timelineNs.BVHTimeline || timelineNs;
const BVHClip = timelineNs.BVHClip || (typeof window !== 'undefined' ? window.BVHClip : undefined);

class TimelineChunkAdapter {
  constructor(timeline) {
    this.timeline = timeline || new BVHTimeline();
  }

  appendChunk(trackName, chunk, opts = {}) {
    const { t0, dt, frames } = chunk;
    const fadeInMs = opts.fadeInMs ?? 120;
    const weight = opts.weight ?? 1.0;

    // Frame generator shared by raw and BVHClip paths
    const generator = async (localTime, frameIndex) => {
      const idx = Math.min(frames.length - 1, Math.max(0, Math.floor(localTime * 30)));
      const base = frames[idx] || { motionData: [], metadata: {} };
      const frame = { ...base, metadata: { ...(base.metadata || {}), ...(opts.meta || {}) } };
      if (fadeInMs > 0) {
        const w = Math.min(1, (localTime * 1000) / fadeInMs);
        frame.metadata.weightEnvelope = w;
      }
      return frame;
    };

    // If a prebuilt clip is provided, honor it directly
    if (opts.clip && typeof opts.clip === 'object') {
      if (this.timeline && typeof this.timeline.addClip === 'function') {
        return this.timeline.addClip(trackName, opts.clip);
      }
    }

    // Prefer constructing a BVHClip when available to satisfy getFrameAtTime
    let clip;
    if (typeof BVHClip === 'function') {
      clip = new BVHClip({
        type: 'generated',
        startTime: t0,
        duration: dt,
        weight,
        blendMode: opts.blendMode || 'replace',
        generator,
        cacheEnabled: false,
        metadata: opts.meta || {}
      });
    } else {
      // Fallback raw clip (used by minimal timelines/tests)
      clip = {
        type: 'generated',
        startTime: t0,
        duration: dt,
        weight,
        blendMode: opts.blendMode || 'replace',
        generator,
        meta: opts.meta || {}
      };
    }

    if (this.timeline && typeof this.timeline.addClip === 'function') {
      return this.timeline.addClip(trackName, clip);
    }
    // Fallback: minimal timeline with tracks[trackName].clips
    const track = (this.timeline.tracks[trackName] ||= { clips: [] });
    clip.id = clip.id || `${trackName}_clip_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;
    track.clips.push(clip);
    return clip.id;
  }

  /**
   * Clear future clips from a track starting at a cutoff time.
   *
   * Semantics:
   * - For minimal timelines (no removeClip API), a clip is removed when (startTime + duration) < fromTime is false,
   *   i.e., it will keep clips that fully end before the cutoff and remove any clip that overlaps or starts after it.
   * - For richer timelines exposing removeClip, the same cutoff logic is applied by filtering candidates first.
   *
   * Tip: Use the timeline's currentTime as fromTime to preserve what has already played while dropping future content.
   */
  clearFrom(trackName, fromTime) {
    // If the underlying timeline exposes its own clearFrom, delegate to it
    if (this.timeline && typeof this.timeline.clearFrom === 'function') {
      try { this.timeline.clearFrom(trackName, fromTime); } catch {}
      return;
    }
    // Remove clips that overlap the cutoff (keep those that end strictly before fromTime)
    const track = this.timeline.tracks[trackName];
    if (!track) return;
    if (typeof this.timeline.removeClip === 'function') {
      const toRemove = track.clips.filter(c => (c.startTime + c.duration) >= fromTime);
      for (const clip of toRemove) this.timeline.removeClip(trackName, clip.id);
      return;
    }
    // Fallback: mutate clips directly
    track.clips = track.clips.filter(c => (c.startTime + c.duration) < fromTime);
  }
}

// Always expose to window when available
if (typeof window !== 'undefined') {
  window.TimelineChunkAdapter = TimelineChunkAdapter;
  try { window.TimelineChunkAdapter.TimelineChunkAdapter = TimelineChunkAdapter; } catch {}
}
// Also support CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TimelineChunkAdapter };
}
