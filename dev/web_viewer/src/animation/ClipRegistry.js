// ClipRegistry: manage BVH clips (saved or generated) with metadata and quick lookup
// UMD export: window.ClipRegistry or module.exports
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.ClipRegistry = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class ClipRegistry {
    constructor() {
      this._clips = new Map();
    }
    // meta: { track, priority, boneMask?: string[], fadeInMs?: number, fadeOutMs?: number, duration?: number }
    add(name, chunkOrTimeline, meta = {}) {
      if (!name) throw new Error('Clip name required');
      this._clips.set(name, { name, data: chunkOrTimeline, meta: { ...meta } });
      return this;
    }
    has(name) { return this._clips.has(name); }
    get(name) { return this._clips.get(name); }
    list() { return Array.from(this._clips.values()).map(x => ({ name: x.name, meta: x.meta })); }
    remove(name) { this._clips.delete(name); }

    // Load a BVH asset into the registry using BVHClipLibrary (browser-only)
    async addBVH(name, url, meta = {}, opts = {}) {
      const root = (typeof window !== 'undefined') ? window : {};
      const libNs = root.BVHClipLibrary || null;
      const BVHClipLibrary = libNs && (libNs.BVHClipLibrary || libNs);
      if (!BVHClipLibrary) throw new Error('BVHClipLibrary not available in this environment');
      const lib = new BVHClipLibrary();
      const clip = await lib.loadStaticClip(url, { loop: !!meta.loop, metadata: { ...(meta || {}) } });
      this.add(name, clip, meta);
      return clip;
    }

    // Load entries from a JSON manifest.
    // Accepts an array of { name, meta, data? } or an object map { name: { meta, data? } }.
    // This does NOT fetch external BVH assets; it only registers metadata and optional precomputed chunks.
    loadFromManifest(manifest) {
      if (!manifest) return this;
      const entries = Array.isArray(manifest)
        ? manifest
        : Object.entries(manifest).map(([name, obj]) => ({ name, ...obj }));
      for (const ent of entries) {
        if (!ent || !ent.name) continue;
        // Support simple { url } to be fetched later by the caller/demo
        if (ent.url) {
          this.add(ent.name, { url: ent.url }, ent.meta || {});
        } else {
          this.add(ent.name, ent.data || null, ent.meta || {});
        }
      }
      return this;
    }
  }
  return { ClipRegistry };
});
