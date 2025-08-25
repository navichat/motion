/**
 * BVHClipLibrary
 * Lightweight helper to load BVH files and create static clips for BVHTimeline.
 *
 * Contract:
 * - loadStaticClip(url, opts) => { type:'static', startTime, duration, weight, blendMode, bvhData, metadata }
 * - addStaticClip(timeline, trackName, url, startTime, opts) => clipId
 */

(function factory(root, mod) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = mod();
  } else {
    root.BVHClipLibrary = mod();
  }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  class BVHClipLibrary {
    constructor(opts = {}) {
      this.defaultBlendMode = opts.defaultBlendMode || 'replace';
      this.defaultWeight = opts.defaultWeight ?? 1.0;
    }

    async fetchText(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed to fetch BVH: ${url} (${res.status})`);
      return await res.text();
    }

    // Minimal duration extraction: parses 'Frames:' and 'Frame Time:' lines
    static parseDurationFromBVHText(bvhText, fallbackFPS = 30) {
      let frames = 0;
      let frameTime = 1 / fallbackFPS;
      const lines = bvhText.split(/\r?\n/);
      for (const line of lines) {
        if (line.startsWith('Frames:')) {
          const v = parseInt(line.split(':')[1].trim());
          if (!Number.isNaN(v)) frames = v;
        } else if (line.startsWith('Frame Time:')) {
          const v = parseFloat(line.split(':')[1].trim());
          if (!Number.isNaN(v)) frameTime = v;
        }
      }
      return frames * frameTime;
    }

    async loadStaticClip(url, opts = {}) {
      const bvhText = await this.fetchText(url);
      const duration = BVHClipLibrary.parseDurationFromBVHText(bvhText, opts.fps || 30);
      // Prefer a BVHClip instance if available from the full timeline module
      const BVHClipCtor = (typeof window !== 'undefined' && (window.BVHClip || (window.BVHTimeline && window.BVHTimeline.BVHClip)))
        ? (window.BVHClip || (window.BVHTimeline && window.BVHTimeline.BVHClip))
        : undefined;
      if (typeof BVHClipCtor === 'function') {
        return new BVHClipCtor({
          type: 'static',
          startTime: opts.startTime || 0,
          duration: opts.duration || duration || 0,
          weight: opts.weight ?? this.defaultWeight,
          blendMode: opts.blendMode || this.defaultBlendMode,
          loop: !!opts.loop,
          bvhData: bvhText,
          metadata: { sourceUrl: url, kind: 'bvh_static', ...opts.metadata }
        });
      }
      // Fallback plain object clip (sufficient for minimal timeline path)
      return {
        type: 'static',
        startTime: opts.startTime || 0,
        duration: opts.duration || duration || 0,
        weight: opts.weight ?? this.defaultWeight,
        blendMode: opts.blendMode || this.defaultBlendMode,
        loop: !!opts.loop,
        bvhData: bvhText,
        metadata: { sourceUrl: url, kind: 'bvh_static', ...opts.metadata }
      };
    }

    async addStaticClip(timeline, trackName, url, startTime = 0, opts = {}) {
      const clip = await this.loadStaticClip(url, { ...opts, startTime });
      if (!timeline || typeof timeline.addClip !== 'function') {
        throw new Error('Timeline missing addClip(trackName, clip)');
      }
      return timeline.addClip(trackName, clip);
    }
  }

  return { BVHClipLibrary };
});
