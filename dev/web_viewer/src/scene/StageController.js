// StageController: manage interactable classroom props and simple actions (stub)
// UMD export: window.StageController or module.exports
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.StageController = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class StageController {
    constructor(sceneBuilder, options = {}) {
      this.sceneBuilder = sceneBuilder;
      this.actions = [];
      this.registry = options.registry || null;
      // Basic default mapping from high-level intents to clip names
      this.intentMap = Object.assign({
        pointAt: 'point',
        wave: 'wave',
        writeOn: 'write', // may fallback if missing
        pickUp: 'pickup'  // may fallback if missing
      }, options.intentMap || {});
    }
    setClipRegistry(registry) { this.registry = registry; }
    // Return clip name for an intent, or null if none
    mapIntentToClipName(intent, params = {}) {
      return this.intentMap[intent] || null;
    }
    // Resolve intent to a registry entry { name, data, meta } or null
    mapIntentToClip(intent, params = {}) {
      if (!this.registry) return null;
      const name = this.mapIntentToClipName(intent, params);
      if (!name) return null;
      const entry = this.registry.get(name);
      if (!entry) return null;
      return entry;
    }
    // High-level action hook; returns a structured event with optional clip metadata
    perform(action, params = {}) {
      const clipEntry = this.mapIntentToClip(action, params);
      const evt = { action, params, t: Date.now(), clip: clipEntry ? { name: clipEntry.name, meta: clipEntry.meta } : null };
      this.actions.push(evt);
      return evt;
    }
    getRecent(n = 5) { return this.actions.slice(-n); }
  }
  return { StageController };
});
