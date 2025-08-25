// SileroVadOrtTask: Real VAD when window.ort and model configured; fallback to simple threshold.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SileroVadOrtTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class SileroVadOrtTask {
    constructor({ id = 'vad', modelUrl, provider = 'wasm', threshold = 0.5 } = {}) {
      this.id = id; this.modelUrl = modelUrl; this.provider = provider; this.threshold = threshold;
      this.session = null; this.initialized = false;
    }
    _resolveModelUrl() {
      if (this.modelUrl) return this.modelUrl;
      try {
        const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
        if (g && g.ModelUrlConfig && typeof g.ModelUrlConfig.getModelUrl === 'function') return g.ModelUrlConfig.getModelUrl('sileroVad');
        if (typeof require !== 'undefined') return require('../../../config/models.config.js').getModelUrl('sileroVad');
      } catch {}
      return undefined;
    }
    async initialize(env = (typeof window !== 'undefined' ? window : {})) {
      const url = this._resolveModelUrl();
      if (!env || !env.ort || !url) { this.initialized = false; return false; }
      try {
        const { InferenceSession, env: ortEnv } = env.ort;
        if (this.provider === 'wasm' && ortEnv && ortEnv.wasm) ortEnv.wasm.numThreads = 1;
        this.session = await InferenceSession.create(url, { executionProviders: [this.provider] });
        this.initialized = true; return true;
      } catch { this.session = null; this.initialized = false; return false; }
    }
    // Expects feature object: { input: ort.Tensor } matching the model
    async *run({ features } = {}) {
      if (!this.initialized) { try { await this.initialize(typeof window !== 'undefined' ? window : {}); } catch {} }
      if (this.session && features && features.input) {
        try {
          const outputs = await this.session.run(features);
          const first = outputs && Object.values(outputs)[0];
          let speech = false;
          if (first && first.data && first.data.length) speech = first.data[0] >= this.threshold;
          yield { t0: 0, dt: 0.2, frames: [], metadata: { model: 'silero_vad_ort', speech } };
          return;
        } catch {}
      }
      yield { t0: 0, dt: 0.2, frames: [], metadata: { model: 'silero_vad_fallback', speech: true } };
    }
  }
  return { SileroVadOrtTask };
});
